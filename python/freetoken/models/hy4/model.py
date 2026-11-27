"""Hy4-flash model tree: decoder layers with identity hyper-connections, Engram on four layers.

Per layer (SPEC section 2): ``X += engram(X)`` on Engram layers, then
``attn_hc -> input_layernorm -> self_attn`` and ``ffn_hc -> post_attention_layernorm -> mlp``,
each sublayer written back into the ``hc_mult`` bf16 streams. The streams collapse through
``hc_head`` and ``norm`` before the fp32 ``lm_head``.

A sublayer's write-back ``X_j += post_j * y`` is fused into the next HC (the next sublayer's
pre, or ``hc_head``), so each layer hands its last ``(y, post)`` on. Engram layers apply the
pending write-back alone first, because the Engram add sits between it and ``attn_hc``.
Engram rows come from the disk store attached by ``load_host_tables`` (zeros when none is).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.layers.mhc import hc_expand, hc_post
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Hy4Attention
from .engram import EngramRows, Hy4Engram, Hy4EngramStore, engram_mask
from .hc import Hy4HyperConnection, Hy4HyperHead
from .mlp import Hy4DenseMLP
from .moe import Hy4LatentMoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Hy4DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args = config.hy4_args
        self._layer_id = layer_id
        self._has_engram = args.engram_slot(layer_id) is not None
        if self._has_engram:
            self.engram = Hy4Engram(config, layer_id, prefix=f"{prefix}.engram")
        self.attn_hc = Hy4HyperConnection(config)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.self_attn = Hy4Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.ffn_hc = Hy4HyperConnection(config)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.norm_eps)
        if layer_id >= config.first_k_dense_replace:
            self.mlp: BaseOP = Hy4LatentMoE(config, layer_id, prefix=f"{prefix}.mlp")
        else:
            self.mlp = Hy4DenseMLP(
                args.hidden_size, args.intermediate_size, args.norm_eps,
                quant_config=config.quant, prefix=f"{prefix}.mlp",
            )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        streams: torch.Tensor,
        y: torch.Tensor | None,
        post: torch.Tensor | None,
        engram: Tuple[EngramRows, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``streams`` [T, n, H] with the previous layer's pending ``(y, post)``; returns the
        streams and this layer's pending MLP write-back. ``engram`` is the forward's
        ``(rows, mask)``."""
        if self._has_engram:
            if post is not None:
                streams = hc_post(y, streams, post)
                y = post = None
            rows, mask = engram
            streams = self.engram.forward(streams, rows.get(self._engram_index), mask)
        hc = self.attn_hc.forward(streams, y, post)
        y = self.self_attn.forward(self.input_layernorm.forward(hc.x))
        hc = self.ffn_hc.forward(hc.streams, y, hc.post)
        y = self.mlp.forward(self.post_attention_layernorm.forward(hc.x))
        return hc.streams, y, hc.post


class Hy4Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self._n = config.hy4_args.hc_mult
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = OPList(
            [Hy4DecoderLayer(config, i, prefix=f"{prefix}.layers.{i}") for i in range(config.num_layers)]
        )
        self.hc_head = Hy4HyperHead(config)
        self.norm = RMSNorm(size=config.hidden_size, eps=config.rms_norm_eps)
        self._args = config.hy4_args
        # ids, not the layers: op walks (offload attach, state_dict) recurse into lists and
        # would visit the Engram layers twice
        self._engram_layer_ids = tuple(layer._layer_id for layer in self.layers.op_list if layer._has_engram)
        # table order of the store: the Engram layers present, ascending (engram_sources' order)
        for i, layer in enumerate(self._engram_layers):
            layer._engram_index = i
        self._engram_store: Hy4EngramStore | None = None
        # Debug tap, off by default: tap(name, tensor) sees each layer's completed streams
        # ("L{i}") and the final hidden ("hidden"); a tensor it returns replaces that layer's streams.
        self.tap = None

    @property
    def _engram_layers(self) -> list[Hy4DecoderLayer]:
        return [self.layers.op_list[i] for i in self._engram_layer_ids]

    def attach_engram_store(self, store: Hy4EngramStore | None) -> None:
        if store is not None:
            assert store.layers == [layer._layer_id for layer in self._engram_layers], store.layers
        self._engram_store = store

    def forward(self, input_ids: torch.Tensor, batch=None) -> torch.Tensor:
        streams = hc_expand(self.embed_tokens.forward(input_ids), self._n)
        engram = None
        if self._engram_layers:
            args = self._args
            rows = EngramRows(
                self._engram_store, input_ids.shape[0], len(self._engram_layers),
                args.engram_num_heads, args.engram_head_dim, streams.device, streams.dtype,
            )
            engram = (rows, engram_mask(args, input_ids))
        y = post = None
        for layer in self.layers.op_list:
            streams, y, post = layer.forward(streams, y, post, engram)
            if self.tap is not None:
                done = streams if post is None else hc_post(y, streams, post)
                forced = self.tap(f"L{layer._layer_id}", done)
                if forced is not None:
                    streams, y, post = forced, None, None
        out = self.norm.forward(self.hc_head.forward(streams, y, post).x)
        if self.tap is not None:
            self.tap("hidden", out)
        return out


class Hy4LMHead(ParallelLMHead):
    """fp32 logits, as the reference's fp32 lm_head. bf16 x bf16 products are exact in fp32,
    so a bf16 GEMM accumulating and storing in fp32 equals the fp32 GEMM up to summation order."""

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        w = self.tied_embedding.weight if self.tied_embedding is not None else getattr(self, "weight", None)
        if self.bias is None and w is not None and w.dtype == x.dtype and w.dtype in (torch.bfloat16, torch.float16):
            if x.is_cuda:
                if (
                    x.shape[0] == 1
                    and x.is_contiguous()
                    and x.dtype == torch.bfloat16
                    and w.stride(1) == 1
                ):
                    from freetoken.kernel.triton.gemv import gemv

                    return gemv(x, w, out_dtype=torch.float32)
                return torch.mm(x, w.t(), out_dtype=torch.float32)
            return F.linear(x.float(), w.float())
        return super()._logits(x).float()

    def _gather_vocab(self, logits: torch.Tensor) -> torch.Tensor:
        # pynccl moves no fp32: gather the fp32 rows as raw bytes (the rank order is kept)
        if self.tp_size == 1 or logits.dtype != torch.float32:
            return super()._gather_vocab(logits)
        t, v = logits.shape
        raw = self._comm.all_gather(logits.contiguous().view(torch.uint8))
        full = raw.view(torch.float32).view(self.tp_size, t, v).permute(1, 0, 2).reshape(t, self.tp_size * v)
        return full[:, : self.num_embeddings]


class Hy4ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self._config = config
        self.model = Hy4Model(config)
        self.lm_head = Hy4LMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        self.dfly = None
        if config.hy4_args.dfly_enabled:
            from .dfly import Hy4DFly

            self.dfly = Hy4DFly(config)
        self._dfly_tap = None
        self._dfly_graphs = None
        self._draft_vocab = None

    # ----- DFly2 drafter --------------------------------------------------------------------
    def dfly_drafter(self):
        """The drafter when this serve builds it (``--mtp on``), else None."""
        return self.dfly

    def set_draft_vocab(self, dv) -> None:
        self._draft_vocab = dv

    def setup_draft_buffers(self, rows: int, slots: int, device: torch.device) -> None:
        """Before any graph capture: the static target-row export the drafter reads (``rows``
        = the most rows one forward hands over, ``slots`` = request table rows)."""
        if self.dfly is None:
            return
        from .dfly import DFlyTap

        args = self._config.hy4_args
        self._dfly_tap = DFlyTap(
            self.dfly.target_layers, args.hidden_size, args.candidate_topk, rows, slots, device
        )
        get_global_ctx().attn_backend.dfly_tap = self._dfly_tap

    def dfly_keep(self, slots: torch.Tensor, rows: torch.Tensor) -> None:
        self._dfly_tap.keep(slots, rows)

    def capture_draft_graphs(self, bs_list, max_k: int, pad_slot: int, device, stream, pool) -> None:
        """With the verify graphs: the draft for every batch size and draft count 1..``max_k``."""
        if self.dfly is None:
            return
        from .dfly import DFlyGraphs

        self._dfly_graphs = DFlyGraphs(
            self._dfly_eager, bs_list, range(1, max_k + 1), pad_slot, device, stream, pool,
            draft_sampled=self._dfly_eager_sampled, draft_vocab=self._draft_vocab,
        )

    def dfly_draft(
        self,
        slots: torch.Tensor,
        bonus: torch.Tensor,
        anchor: torch.Tensor,
        k: int,
        *,
        temp: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
        u: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``k`` drafts per request from the state kept in table rows ``slots``, each block
        starting with the pending token ``bonus`` at position ``anchor``. When ``temp`` is given
        (non-greedy batch), draws from the draft head's truncated distribution and returns
        ``(drafts [n, k], cand_ids [n, k, DRAFT_K], cand_q [n, k, DRAFT_K])``; otherwise returns
        ``(drafts [n, k], None, None)`` with zero sampling overhead."""
        dv = self._draft_vocab
        use_dv = dv is not None and dv.active
        graphs = self._dfly_graphs
        if temp is not None:
            assert top_p is not None and top_k is not None
            if u is None:
                from freetoken.engine.draft_sample import DRAFT_K

                u = torch.rand((slots.shape[0], k, DRAFT_K), device=slots.device)
            if graphs is not None and graphs.can_use(slots.shape[0], k, sampled=True, use_dv=use_dv):
                return graphs.run_sampled(slots, bonus, anchor, k, temp, top_p, top_k, u, use_dv=use_dv)
            return self._dfly_eager_sampled(
                slots, bonus, anchor, k, temp, top_p, top_k, u, draft_vocab=dv if use_dv else None
            )
        if graphs is not None and graphs.can_use(slots.shape[0], k, sampled=False, use_dv=use_dv):
            return graphs.run(slots, bonus, anchor, k, use_dv=use_dv), None, None
        return self._dfly_eager(slots, bonus, anchor, k, draft_vocab=dv if use_dv else None), None, None

    def _dfly_eager(
        self, slots: torch.Tensor, bonus: torch.Tensor, anchor: torch.Tensor, k: int, *, draft_vocab=None
    ) -> torch.Tensor:
        tap = self._dfly_tap
        return self.dfly.forward(
            self.model.embed_tokens, self.lm_head, bonus, anchor,
            tap.hidden.index_select(0, slots), tap.sel.index_select(1, slots), k,
            hist_cnt=tap.cnt.index_select(1, slots), draft_vocab=draft_vocab,
        )

    def _dfly_eager_sampled(
        self, slots: torch.Tensor, bonus: torch.Tensor, anchor: torch.Tensor, k: int,
        temp: torch.Tensor, top_p: torch.Tensor, top_k: torch.Tensor, u: torch.Tensor, *, draft_vocab=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tap = self._dfly_tap
        return self.dfly.forward_sampled(
            self.model.embed_tokens, self.lm_head, bonus, anchor,
            tap.hidden.index_select(0, slots), tap.sel.index_select(1, slots),
            tap.cnt.index_select(1, slots), k, temp, top_p, top_k, u, draft_vocab=draft_vocab,
        )

    def _export_hidden(self, out: torch.Tensor) -> None:
        tap = self._dfly_tap
        if tap is None:
            return
        batch = get_global_ctx().batch
        if batch.is_prefill and not getattr(batch, "mtp_verify", False):
            out = out[batch.attn_metadata.get_last_indices(batch.size)]
        tap.put_hidden(out)

    def load_host_tables(self, engine_config) -> int:
        """Open the Engram row tables in place (page-cached reads; only staging is pinned) and
        hook the store's host staging around every dispatch."""
        model = self.model
        if not model._engram_layers or getattr(engine_config, "use_dummy_weight", False):
            return 0
        from .weight import engram_table_sources

        sources = engram_table_sources(engine_config.model_path, self._config)
        store = Hy4EngramStore(
            sources,
            [layer.engram.lookup_table for layer in model._engram_layers],
            self._config.hy4_args,
            max_graph_rows=max(256, getattr(engine_config, "cuda_graph_max_bs", 0) or 0),
            max_extend_tokens=engine_config.max_extend_tokens,
        )
        model.attach_engram_store(store)
        self.forward_host_ctx = store.forward_host_ctx
        return 0

    def prepare_for_runtime(self) -> None:
        """Post-load, pre-KV-sizing: repack every layer's kv_b for the absorbed MLA and free
        the checkpoint layout (glm_moe_dsa precedent)."""
        for layer in self.model.layers.op_list:
            layer.self_attn.prepare_for_runtime()
        if self.dfly is not None:
            self.dfly.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        self._export_hidden(output)
        return self.lm_head.forward(output)

    def forward_logits_all(self) -> torch.Tensor:
        """Every forwarded row's fp32 logits [T, vocab] (speculative verify), no last-row selection."""
        output = self.model.forward(get_global_ctx().batch.input_ids)
        self._export_hidden(output)
        return self.lm_head.forward_all(output)

    def select_verify_width(self, t: int) -> None:
        """After a captured verify of width ``t`` (no Python ran): its stash is the one to commit."""
        get_global_ctx().attn_backend.select_verify(t)

    def verify_graph_host_fill(self, batch) -> None:
        """Before a captured verify replays: stage its Engram rows into the graph buffers (the
        graph WAITs on the fill's flag; the eager path does this in forward_host_ctx)."""
        store = self.model._engram_store
        if store is None:
            return
        from freetoken.models.qwen4_exp.ple_disk import _context

        pad = store.table.eos_token_id
        runs = [
            torch.cat((
                torch.tensor(_context(r.input_ids, r.cached_len, pad), dtype=torch.int64),
                r.input_ids[r.cached_len : r.device_len].to(torch.int64),
            ))
            for r in batch.padded_reqs
        ]
        store.table.fill(runs, graph=True)

    def commit_mtp_verify(self, pool, slots, lens) -> None:
        """After a speculative verify: the only non-KV state a verify defers is the indexer's
        tail ring (``lens`` [n] device int, rows kept per request; no recurrent pool)."""
        get_global_ctx().attn_backend.commit_verify(lens)


__all__ = ["Hy4ForCausalLM", "Hy4Model", "Hy4DecoderLayer", "Hy4LMHead"]
