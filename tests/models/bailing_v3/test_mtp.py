"""Ling-3.2 MTP head (``--mtp on``) vs the reference's MTP layer on the NVFP4A16 tiny checkpoint.

The goldens' ``mtp64`` is HF's ``BailingMoeV3MTPLayer`` output (f64, ``output_mtp_hidden_states``): row ``i``
from ``model.norm``'s output at ``i`` and the embedding of token ``i + 1``. The head runs teacher-forced on
bf16 copies of ``final64`` at HF's positions; the relay and the lm_head input are that output.
"""

from __future__ import annotations

import pytest
import torch

from .common import ModelRig, attach_tiny_experts, build_tiny_model, golden, needs_golden, rel, tiny_ckpt, tiny_config, use_tp

pytestmark = [needs_golden, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

CKPT, GOLD = "tiny_nvfp4_ckpt", "tiny_nvfp4.pt"


@pytest.fixture(scope="module")
def rig():
    from freetoken.distributed import try_get_tp_info

    if try_get_tp_info() is None:
        use_tp(0, 1)
    mp = pytest.MonkeyPatch()
    mp.setenv("FREETOKEN_ENABLE_MTP", "1")
    try:
        ckpt = tiny_ckpt(CKPT)
        config = tiny_config(ckpt=ckpt)
        config, model = build_tiny_model(config, ckpt)
    finally:
        mp.undo()
    from freetoken.layers import iter_moe_layers

    stack = [layer for layer in iter_moe_layers(model) if not getattr(layer, "state_dict_experts", False)]
    from freetoken.moe.expert_banks import attach_resident_banks, load_expert_banks
    from freetoken.layers.quantization.method import finalize_quant

    finalize_quant(model)
    banks = load_expert_banks(ckpt, config, method=stack[0].quant_method, device=torch.device("cuda"), parallel=False, resident=True)
    attach_resident_banks(stack, banks)
    model.prepare_for_runtime()
    return config, model


def test_mtp_slab_only_with_the_head():
    off = tiny_config(ckpt=tiny_ckpt(CKPT))
    full = next(g for g in off.attention_groups if g.name == "full")
    assert not off.mtp_enabled and off.num_layers not in full.layer_ids


def test_head_experts_are_packed_nvfp4_banks(rig):
    from freetoken.layers.quantization import QuantKind

    config, model = rig
    ex = model.mtp.layer.mlp.experts
    assert ex.quant_method.kind is QuantKind.NVFP4 and ex._expert_banks is not None and ex.gate is None
    assert config.mtp_enabled and config.num_layers in next(g for g in config.attention_groups if g.name == "full").layer_ids


def _mtp_rows(config, model, monkeypatch, h: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
    from types import SimpleNamespace

    r = ModelRig(config, model, monkeypatch)
    n = h.shape[0]
    req = SimpleNamespace(table_idx=0, device_len=n, extend_len=n, cached_len=0)
    batch = SimpleNamespace(
        phase="prefill", is_prefill=True, is_decode=False, size=1, reqs=[req], padded_reqs=[req],
        positions=torch.arange(n, device="cuda"), out_loc=r.page_table[0, :n].long(), features=None, mm_embeds=None,
    )
    r.ctx.batch = batch
    r.ctx.attn_backend.prepare_metadata(batch)
    embed = model.model.embed_tokens.forward
    out, sample = model.mtp.draft_step(embed, h.to("cuda", torch.bfloat16), nxt.to("cuda"), batch)
    assert out is sample
    return out.float().cpu()


@pytest.mark.parametrize("T", [17, 316])
def test_head_vs_reference(rig, monkeypatch, T):
    """Within 3x the reference's own bf16 error on the MTP layer's attention plus MoE (HF's bf16 modules on bf16
    copies of the f64 inputs; a flipped near-tie route moves a row by a whole expert), and the draft argmax agrees
    with the f64 head's on nearly every row. Measured: 0.006 / 0.054 at T=17 / 316 (HF modules 0.18 / 0.032)."""
    config, model = rig
    s = golden(GOLD)["seqs"][T]
    ids, layer = s["ids"][0], s["layers"][config.num_layers]
    got = _mtp_rows(config, model, monkeypatch, s["final64"][:-1], ids[1:])
    want = s["mtp64"][:-1]
    ref_err = rel(layer["attn.out_hf"], layer["attn.out64"]) + rel(layer["mlp.out_hf"], layer["mlp.out64"])
    assert rel(got, want) <= 3.0 * ref_err, f"head {rel(got, want):.3e} vs HF modules {ref_err:.3e}"
    w = model.lm_head.weight.float().cpu()
    agree = ((got @ w.t()).argmax(-1) == (want.float() @ w.t()).argmax(-1)).float().mean().item()
    assert agree >= 0.9, agree


def test_head_input_order_matters(rig, monkeypatch):
    """The check above sees a swapped eh_proj input (``cat([h, e])``)."""
    config, model = rig
    s = golden(GOLD)["seqs"][17]
    ids = s["ids"][0]
    real = model.mtp.eh_proj.forward
    monkeypatch.setattr(model.mtp.eh_proj, "forward", lambda x: real(torch.cat(x.chunk(2, -1)[::-1], -1)))
    got = _mtp_rows(config, model, monkeypatch, s["final64"][:-1], ids[1:])
    assert rel(got, s["mtp64"][:-1]) > 1.0

