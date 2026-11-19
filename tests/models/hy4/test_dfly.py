"""DFly2 drafter vs the fp32 CPU reference golden (``hy4/golden/dfly_golden.pt``, SPEC 11.9).

The drafter is built alone (TP=1, one GPU) with the released weights read by the hy4 reader, and
teacher-forced on the golden's inputs: the target layers' cached latents, the selection (long
prompt) or the dense history (short prompt), the target's final hidden at ``anchor - 1`` and the
bonus token. The attention is the real ``Hy4AttnBackend.dfly_attend`` over a latent pool whose
physical row is the position.

Precision: ours runs bf16 activations (FP8 weight-only linears, bf16 residual), the golden fp32
math on the same bf16 weights. A float64 implementation of the same formulas is the yardstick:
each module's distance to it is reported next to the golden's, and drafts that differ from the
golden's must sit on near-tied logits.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from .common import golden_dir, meta_model, model_path, parsed_config

MASK = 120009
pytestmark = [
    pytest.mark.needs_weights,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(model_path() is None, reason="FREETOKEN_HY4_MODEL is not set to a Hy4-flash-FP8 snapshot"),
    pytest.mark.skipif(
        golden_dir() is None or not (golden_dir() / "dfly_golden.pt").is_file(), reason="no dfly_golden.pt"
    ),
]
DEV = torch.device("cuda", 0)


@pytest.fixture(scope="module")
def golden():
    return torch.load(golden_dir() / "dfly_golden.pt", weights_only=True)


@pytest.fixture(scope="module")
def drafter():
    """(model, drafter state dict as loaded) with the drafter, embed and lm_head on the GPU."""
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.hy4.weight import CheckpointSource, _DenseReader, _iter_dfly

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    old = os.environ.get("FREETOKEN_ENABLE_MTP")
    os.environ["FREETOKEN_ENABLE_MTP"] = "1"
    try:
        config = parsed_config()
    finally:
        if old is None:
            os.environ.pop("FREETOKEN_ENABLE_MTP")
        else:
            os.environ["FREETOKEN_ENABLE_MTP"] = old
    model = meta_model(config)
    src = CheckpointSource(model_path())
    try:
        r = _DenseReader(src, DEV)
        sd = dict(_iter_dfly(r, config.hy4_args))
        model.dfly.load_state_dict({k[len("dfly."):]: v for k, v in sd.items()})
        model.model.embed_tokens.load_state_dict({"weight": r.take("model.embed_tokens.weight", torch.bfloat16)})
        model.lm_head.load_state_dict({"weight": r.take("lm_head.weight", torch.bfloat16)})
    finally:
        src.close()
    weights = {k: v for k, v in sd.items()}
    model.dfly.prepare_for_runtime()
    return model, weights, config


class _Backend:
    """The real draft attention over a pool whose physical row is the position."""

    from freetoken.models.hy4.backend import Hy4AttnBackend

    dfly_attend = Hy4AttnBackend.dfly_attend
    sm_scale = 256**-0.5
    kv_lora_rank = 512

    def __init__(self, pools):
        self.kvcache = SimpleNamespace(latent_rows=lambda layer: pools[layer])


@pytest.fixture
def backend(monkeypatch):
    import freetoken.models.hy4.dfly as dfly_mod

    holder = SimpleNamespace(attn_backend=None)
    monkeypatch.setattr(dfly_mod, "get_global_ctx", lambda: holder)
    return holder


def _pools(part):
    return {
        layer: torch.cat([v["c"], v["kr"]], dim=-1).to(DEV, torch.bfloat16).contiguous()
        for layer, v in part["latents"].items()
    }


def _history(part, name, anchors, width=2048):
    """[layers, n, width] position (= row) lists: the golden's selection, else dense history."""
    layers = sorted(part["latents"])
    out = torch.full((len(layers), len(anchors), width), -1, dtype=torch.int32)
    all_anchors = part["anchors"].tolist()
    for li, layer in enumerate(layers):
        for j, a in enumerate(anchors):
            if name == "long":
                pos = part["selection"][layer]["tokens"][all_anchors.index(a)].to(torch.int32)
            else:
                pos = torch.arange(a, dtype=torch.int32)
            out[li, j, : pos.numel()] = pos
    return out.to(DEV)


def _h_target(part, name, anchors):
    if name == "long":
        idx = [part["anchors"].tolist().index(a) for a in anchors]
        return part["target_final_hidden_rows"][idx].to(DEV)
    return part["target_final_hidden"][torch.tensor(anchors) - 1].to(DEV)


# --- float64 yardstick (the SPEC 11 formulas, written out) ----------------------------------


def _rms(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _rope64(x, pos):
    inv = 1.0 / 1e7 ** (torch.arange(0, 64, 2, dtype=torch.float64, device=x.device) / 64)
    ang = pos.double()[:, None] * inv
    c, s = ang.cos(), ang.sin()
    if x.dim() == 3:
        c, s = c[:, None], s[:, None]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x0 * c - x1 * s, x1 * c + x0 * s), dim=-1).flatten(-2)


class _Ref64:
    def __init__(self, model, weights):
        from freetoken.kernel.triton.fp8_block_w8a16 import _dequant_torch

        self.w = {}
        for k, v in weights.items():
            if k.endswith("_scale_inv"):
                continue
            if v.dtype == torch.float8_e4m3fn:
                v = _dequant_torch(v, weights[k + "_scale_inv"])
            self.w[k] = v.double()
        self.embed = model.model.embed_tokens.weight
        self.lm_head = model.lm_head.weight

    def E(self, ids):
        return self.embed[ids].double()

    def run(self, bonus, anchor, h_target, ctx, steps=5):
        W, rec = self.w, {}
        pos = anchor + torch.arange(6, device=DEV)
        ids = torch.full((6,), MASK, dtype=torch.long, device=DEV)
        ids[0] = bonus
        x = self.E(ids)
        f = torch.cat((_rms(x[0], W["dfly.fusion.enorm.weight"]), _rms(h_target.double(), W["dfly.fusion.hnorm.weight"])))
        x[0] = x[0] + W["dfly.fusion.gate"] * (f @ W["dfly.fusion.eh_proj.weight"].T)
        rec["x0"] = x.clone()
        for i in range(5):
            p = f"dfly.layers.{i}."
            h = _rms(x, W[p + "input_layernorm.weight"])
            rec[f"L{i}.attn.in"] = h
            out = self.attn(i, h, pos, *ctx[i], rec)
            rec[f"L{i}.attn.out"] = out
            x = x + out
            y = _rms(x, W[p + "post_attention_layernorm.weight"])
            m = p + "mlp."
            x = x + (F.silu(y @ W[m + "gate_proj.weight"].T) * (y @ W[m + "up_proj.weight"].T)) @ W[m + "down_proj.weight"].T
            rec[f"L{i}.out"] = x.clone()
        hid = _rms(x, W["dfly.final_norm.weight"])
        rec["final_hidden"] = hid
        prev, drafts = bonus, []
        c = "dfly.hidden_correction."
        for s in range(steps):
            z = torch.cat((_rms(hid[s + 1], W[c + "hidden_norm.weight"]), _rms(self.E(prev), W[c + "embed_norm.weight"])))
            hc = hid[s + 1] + (F.silu(z @ W[c + "gate_proj.weight"].T) * (z @ W[c + "up_proj.weight"].T)) @ W[c + "down_proj.weight"].T
            rec[f"step{s}.corrected"] = hc
            prev = int((hc.float() @ self.lm_head.float().T).argmax())
            drafts.append(prev)
        rec["drafts"] = drafts
        return rec

    def attn(self, i, h, pos, ctx_c, ctx_kr, rec):
        W = self.w
        p = f"dfly.layers.{i}.self_attn."
        t = h.shape[0]
        qr = _rms(h @ W[p + "q_a_proj.weight"].T, W[p + "q_a_layernorm.weight"])
        q = (qr @ W[p + "q_b_proj.weight"].T).view(t, 32, 256)
        q_nope, q_pe = q[..., :192], _rope64(q[..., 192:], pos)
        ckv = h @ W[p + "kv_a_proj_with_mqa.weight"].T
        c_blk = _rms(ckv[:, :512], W[p + "kv_a_layernorm.weight"])
        kr_blk = _rope64(ckv[:, 512:], pos)
        rec[f"L{i}.attn.c_blk"], rec[f"L{i}.attn.kr_blk"] = c_blk, kr_blk
        C, KR = torch.cat((ctx_c, c_blk)), torch.cat((ctx_kr, kr_blk))
        n = ctx_c.shape[0]
        wkv = W[p + "kv_b_proj.weight"].view(32, 448, 512)
        w_uk, w_uv = wkv[:, :192], wkv[:, 192:]
        q_lat = torch.einsum("thd,hdc->thc", q_nope, w_uk)
        s = (torch.einsum("thc,nc->thn", q_lat, C) + torch.einsum("thr,nr->thn", q_pe, KR)) / 16.0
        vis = torch.ones(t, n + t, dtype=torch.bool, device=DEV)
        vis[:, n:] = torch.ones(t, t, dtype=torch.bool, device=DEV).tril()
        s = s.masked_fill(~vis[:, None, :], float("-inf"))
        sink = W[p + "sinks"].view(1, 32, 1).expand(t, 32, 1)
        pr = torch.cat((s, sink), -1).softmax(-1)[..., :-1]
        o = torch.einsum("thc,hvc->thv", torch.einsum("thn,nc->thc", pr, C), w_uv)
        o = o * torch.sigmoid((h @ W[p + "gate_proj.weight"].T).view(t, 32, 256))
        return _rms(o.reshape(t, -1) @ W[p + "o_proj.weight"].T, W[p + "post_proj_norm.weight"])


def _ctx64(part, name, anchor):
    out = []
    hist = _history(part, name, [anchor])
    for li, layer in enumerate(sorted(part["latents"])):
        rows = hist[li, 0]
        rows = rows[rows >= 0].long().cpu()
        v = part["latents"][layer]
        out.append((v["c"][rows].to(DEV).double(), v["kr"][rows].to(DEV).double()))
    return out


def _rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# --- tests ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["short", "long"])
def test_drafts_match_the_golden(drafter, golden, backend, name):
    """Every golden anchor, batched 16 blocks per pass (multi-request path): drafts equal to the
    reference's except where the reference's own top-2 logits are near-tied."""
    model, _, _ = drafter
    part = golden[name]
    backend.attn_backend = _Backend(_pools(part))
    anchors = part["anchors"].tolist()
    ids = part["input_ids"]
    want = part["drafts"]["ref"]
    got = []
    with torch.inference_mode():
        for j in range(0, len(anchors), 16):
            a = anchors[j : j + 16]
            hist = _history(part, name, a)
            out = model.dfly.forward(
                model.model.embed_tokens, model.lm_head,
                ids[a].to(DEV, torch.int32), torch.tensor(a, device=DEV), _h_target(part, name, a), hist, 5,
            )
            got.append(out.cpu())
    got = torch.cat(got).long()
    first = [next((s for s in range(5) if got[k, s] != want[k, s]), 5) for k in range(len(anchors))]
    same_rows = sum(f == 5 for f in first)
    same_first = int((got[:, 0] == want[:, 0]).sum())
    # chained acceptance (dfly_post.py rule) of ours vs the golden's
    tp = part["target_argmax"]

    def accepted(dr):
        hit, tot = torch.zeros(5), torch.zeros(5)
        for k, a in enumerate(anchors):
            for jj in range(1, 6):
                g = int(tp[a + jj - 1])
                tot[jj - 1] += 1
                if int(dr[k][jj - 1]) != g:
                    break
                hit[jj - 1] += 1
                if jj < 5 and int(ids[a + jj]) != g:
                    break
        r = hit / tot.clamp(min=1)
        return sum(float(r[: q + 1].prod()) for q in range(5))

    e_ours, e_ref = accepted(got), accepted(want)
    # each differing row: our logit gap between our draft and the golden's at the first
    # differing step, our chain teacher-forced on the golden's earlier drafts
    gaps = []
    with torch.inference_mode():
        for k in (k for k in range(len(anchors)) if first[k] < 5):
            a, s = anchors[k], first[k]
            hid = model.dfly.block_hidden(
                model.model.embed_tokens, ids[[a]].to(DEV, torch.int32), torch.tensor([a], device=DEV),
                _h_target(part, name, [a]), _history(part, name, [a]), 6,
            )
            prev = int(ids[a]) if s == 0 else int(want[k, s - 1])
            hc = model.dfly.hidden_correction.forward(
                hid[:, s + 1].contiguous(),
                model.model.embed_tokens.forward(torch.tensor([prev], device=DEV, dtype=torch.int32)),
            )
            logits = model.lm_head.forward_all(hc)[0]
            gaps.append(float(logits[got[k, s]] - logits[want[k, s]]))
    print(
        f"\n{name}: {len(anchors)} anchors, rows identical {same_rows}, slot-1 identical {same_first}, "
        f"E[accepted] ours {e_ours:.2f} vs golden {e_ref:.2f}, logit gaps where they differ "
        f"{[round(g, 3) for g in sorted(gaps)]}"
    )
    # |gap| (not gap >= 0): the batched pass rounds differently from this one-block recompute
    assert all(abs(g) < 0.1 for g in gaps)
    # a draft differs only after a near-tie; bf16 activations move logits by ~0.1-0.5
    assert same_first >= 0.95 * len(anchors)
    assert same_rows >= 0.85 * len(anchors)
    assert abs(e_ours - e_ref) < 0.15


def test_intermediates_against_the_float64_yardstick(drafter, golden, backend):
    """Per recorded anchor and module: ours (bf16 activations) and the golden (fp32) vs float64.
    Chained through the block (ours from its own previous module) and teacher-forced per layer
    (ours fed the float64 layer input)."""
    model, weights, _ = drafter
    ref64 = _Ref64(model, weights)
    dfly = model.dfly
    worst = {}
    for name in ("short", "long"):
        part = golden[name]
        backend.attn_backend = _Backend(_pools(part))
        for anchor, rec in part["records"].items():
            a = int(anchor)
            ctx = _ctx64(part, name, a)
            h_t = _h_target(part, name, [a])
            y = ref64.run(int(part["input_ids"][a]), a, h_t[0], ctx)
            hist = _history(part, name, [a])
            with torch.inference_mode():
                emb = model.model.embed_tokens
                ids = torch.full((1, 6), MASK, dtype=torch.int32, device=DEV)
                ids[0, 0] = int(part["input_ids"][a])
                x = emb.forward(ids.view(-1))
                x = torch.cat([dfly.fusion.forward(x[:1].contiguous(), h_t), x[1:]])
                ours = {"x0": x}
                positions = a + torch.arange(6, device=DEV)
                taught = {}
                for i, layer in enumerate(dfly.layers.op_list):
                    att = layer.self_attn
                    att._hist, att._positions = hist[i], positions
                    h = layer.input_layernorm.forward(x)
                    ours[f"L{i}.attn.out"] = att.forward(h)
                    x = layer.forward(x)
                    ours[f"L{i}.out"] = x
                    # teacher-forced: the float64 input of this layer, rounded to bf16
                    x_in = (y["x0"] if i == 0 else y[f"L{i - 1}.out"]).to(torch.bfloat16)
                    taught[f"L{i}.attn.out"] = att.forward(layer.input_layernorm.forward(x_in))
                    taught[f"L{i}.out"] = layer.forward(x_in)
                    att._hist = att._positions = None
                hid = dfly.final_norm.forward(x)
                ours["final_hidden"] = hid
                prev = torch.tensor([int(part["input_ids"][a])], device=DEV, dtype=torch.int32)
                drafts = []
                for s in range(5):
                    hc = dfly.hidden_correction.forward(hid[s + 1 : s + 2].contiguous(), emb.forward(prev))
                    ours[f"step{s}.corrected"] = hc[0]
                    prev = model.lm_head.forward_argmax(hc)
                    drafts.append(int(prev))
            for key, ref in y.items():
                if key == "drafts" or key.endswith(("c_blk", "kr_blk", "attn.in")):
                    continue
                e_ref = _rel(rec[key].to(DEV), ref)
                e_ours = _rel(ours[key], ref)
                e_taught = _rel(taught[key], y[key]) if key in taught else float("nan")
                row = worst.setdefault(key, [0.0, 0.0, 0.0])
                row[0], row[1] = max(row[0], e_ref), max(row[1], e_ours)
                if not math.isnan(e_taught):
                    row[2] = max(row[2], e_taught)
            assert drafts == rec["drafts"].tolist() or drafts[0] == int(rec["drafts"][0]), (name, a, drafts)
    print("\nmodule: max rel error vs float64 over 5 anchors -- golden fp32 | ours chained | ours teacher-forced")
    for key, (r, o, t) in worst.items():
        print(f"  {key:20s} {r:.2e} | {o:.2e} | {t:.2e}")
    # the golden is the float64 math up to fp32 rounding and its bf16-rounded block latents
    assert max(r for r, _, _ in worst.values()) < 3e-3
    # bf16 activations: a few bf16 ulps per module, compounded through the 5-layer block
    assert max(t for _, _, t in worst.values()) < 2e-2
    assert max(o for _, o, _ in worst.values()) < 5e-2
