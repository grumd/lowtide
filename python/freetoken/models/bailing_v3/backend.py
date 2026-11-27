"""Ling-3.2's MLA backend: the DSA backend's dense identity-selection path (Ling has no indexer) over the
``MLAKVCache`` latent rows, plus the speculative verify surface (``--ngram-spec``, MTP).

A verify window is ``n`` requests x ``t`` rows. Every row is a decode query at its own position over its
request's position-ordered page-table row, with a single decode step's split, so each row reduces in the
order that step would; the shapes are static, so a captured verify replays it with staged addressing.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from freetoken.attention.dsa import DSAAttnBackend, DSAMetadata
from freetoken.core import get_global_ctx

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


class BailingV3AttnBackend(DSAAttnBackend):
    def __init__(self, config: "ModelConfig") -> None:
        super().__init__(config)
        self._graph_verify: dict | None = None

    def _model_args(self, config: "ModelConfig"):
        args = config.bailing_args
        assert args is not None, "bailing_v3 backend needs ModelConfig.bailing_args"
        return args

    def mla_forward(self, q_nope, q_pe, c_kv, k_rope, layer_id, batch, indexer_inputs=None) -> torch.Tensor:
        md = batch.attn_metadata
        if md.is_decode or not getattr(batch, "mtp_verify", False):
            return super().mla_forward(q_nope, q_pe, c_kv, k_rope, layer_id, batch, indexer_inputs)
        if md.rows is None:
            self._stage_eager_verify(md, batch)
        self.kvcache.store_kv(c_kv, k_rope, batch.out_loc, layer_id)
        return self._verify(md, layer_id, torch.cat([q_nope, q_pe], dim=-1))

    def _verify(self, md: DSAMetadata, layer_id: int, q_cat: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.glm_dsa_sparse import _split_count, glm_dsa_sparse_attn
        from freetoken.kvcache.format import NONE

        total, h, d = q_cat.shape
        t = md.verify_t
        n = total // t
        q_pos = md.kvlen.view(n, 1) - t + torch.arange(t, device=self.device, dtype=torch.int32)
        width = md.rows.shape[-1]
        sel = md.rows.unsqueeze(1).expand(n, t, width).reshape(total, 1, width)
        kv_format = getattr(self.kvcache, "kv_format", NONE)
        scale = self.kvcache.latent_scale(layer_id) if kv_format.quantized else None
        o = glm_dsa_sparse_attn(
            q_cat.view(total, 1, h, d), self.kvcache.latent_rows(layer_id), sel, self.sm_scale,
            counts=(q_pos + 1).reshape(total, 1), d_v=self.kv_lora_rank,
            force_splits=_split_count(1, 1, h, width, q_cat.device), kv_format=kv_format, scale=scale,
        )
        return o.view(total, h, self.kv_lora_rank)

    def _stage_eager_verify(self, md: DSAMetadata, batch: "Batch") -> None:
        """An eager verify: the addressing ``stage_verify_round`` stages for a captured one."""
        qo = md.qo_indptr_cpu
        t = int(qo[1] - qo[0])
        assert bool(((qo[1:] - qo[:-1]) == t).all()), "a verify window has the same width for every request"
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        idx = torch.tensor([r.table_idx for r in reqs], dtype=torch.int64).to(self.device)
        md.rows = get_global_ctx().page_table.index_select(0, idx)
        md.kvlen = md.kv_len_cpu.to(self.device, torch.int32)
        md.verify_t = t

    # ----- captured verify (engine/verify_graph.py) ----------------------------------------
    def init_capture_verify(self, n_max: int, t: int) -> None:
        width = get_global_ctx().page_table.shape[1]
        self._graph_verify = {
            "n": n_max, "t": t,
            "rows": torch.full((n_max, width), -1, dtype=torch.int32, device=self.device),
            "kvlen": torch.full((n_max,), t, dtype=torch.int32, device=self.device),
        }

    def set_verify_n(self, n: int) -> None:
        self._graph_verify["n"] = n

    def make_verify_metadata(self) -> DSAMetadata:
        st = self._graph_verify
        n, t = st["n"], st["t"]
        qo = torch.arange(n + 1, dtype=torch.int32) * t
        md = DSAMetadata(
            is_decode=False, last_indices=(qo[1:] - 1).to(self.device), qo_indptr_cpu=qo,
            kv_len_cpu=torch.full((n,), t, dtype=torch.int32), rows=st["rows"][:n], kvlen=st["kvlen"][:n],
        )
        md.verify_t = t
        return md

    def stage_verify_round(self, table_idx, kvlen, pad_table_idx: int) -> None:
        st = self._graph_verify
        n, t = st["n"], st["t"]
        idx = [int(x) for x in table_idx] + [int(pad_table_idx)] * (n - len(table_idx))
        kv = [int(x) for x in kvlen] + [t] * (n - len(table_idx))
        st["rows"][:n].copy_(get_global_ctx().page_table.index_select(0, torch.tensor(idx).to(self.device)))
        st["kvlen"][:n].copy_(torch.tensor(kv, dtype=torch.int32).to(self.device))

    def verify_scratch(self):
        return nullcontext()


__all__ = ["BailingV3AttnBackend"]
