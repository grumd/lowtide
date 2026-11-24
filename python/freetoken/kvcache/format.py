from __future__ import annotations

from dataclasses import dataclass

import torch

ROW = -1  # one scale per row


@dataclass(frozen=True)
class KVFormat:
    """How one K or V row is stored in the KV pool; the encoding lives in the ``kv_rows.py`` arms of the same name."""

    name: str  # --kv-cache-dtype value and the kernels' KV_FMT
    elem_dtype: torch.dtype | None  # logical element type; None follows the compute dtype
    elem_store_dtype: torch.dtype | None  # dtype of the element tensor; None follows the compute dtype
    scale_dtype: torch.dtype | None = None  # logical scale type; None means no scale tensor
    scale_store_dtype: torch.dtype | None = None  # dtype of the scale tensor
    scale_group: int = ROW  # channels per scale
    rope_unquantized: bool = False  # MLA: keep trailing k_rope in compute dtype (bf16)

    def __post_init__(self) -> None:
        if (self.elem_dtype is None) != (self.elem_store_dtype is None):
            raise ValueError(f"{self.name}: elem_dtype and elem_store_dtype must be set together")
        if (self.scale_dtype is None) != (self.scale_store_dtype is None):
            raise ValueError(f"{self.name}: scale_dtype and scale_store_dtype must be set together")
        if self.scale_dtype is not None and self.scale_dtype.itemsize != self.scale_store_dtype.itemsize:
            raise ValueError(f"{self.name}: scale_store_dtype must have the size of scale_dtype")

    @property
    def quantized(self) -> bool:
        return self.elem_dtype is not None

    def elem_tensor_dtype(self, compute_dtype: torch.dtype) -> torch.dtype:
        return compute_dtype if self.elem_store_dtype is None else self.elem_store_dtype

    def elem_dim(self, head_dim: int) -> int:
        """Elements of the element tensor per row."""
        if self.elem_dtype is None:
            return head_dim
        row_bytes = head_dim * self.elem_dtype.itemsize
        if row_bytes % self.elem_store_dtype.itemsize:
            raise ValueError(f"{self.name} cannot store a {head_dim}-wide row")
        return row_bytes // self.elem_store_dtype.itemsize

    def scale_dim(self, head_dim: int) -> int:
        """Scales per row."""
        if self.scale_dtype is None:
            return 0
        if self.scale_group == ROW:
            return 1
        if head_dim % self.scale_group:
            raise ValueError(f"{self.name} needs head_dim divisible by {self.scale_group}, got {head_dim}")
        return head_dim // self.scale_group

    def row_bytes(self, head_dim: int, compute_dtype: torch.dtype, segments: int = 1) -> int:
        """Bytes of one K or V row split into ``segments`` separately scaled parts, its scales included."""
        elem_bytes = self.elem_dim(head_dim) * self.elem_tensor_dtype(compute_dtype).itemsize
        if self.scale_dtype is None:
            return elem_bytes
        return elem_bytes + segments * self.scale_dim(head_dim) * self.scale_store_dtype.itemsize

    # -- MLA latent row helpers (ckv [d_v] + k_rope [d_r]) ----------------------------------
    def mla_row_elems(self, d_v: int, d_r: int, compute_dtype: torch.dtype = torch.bfloat16) -> int:
        """Elements of the row buffer per token: for ``rope_unquantized`` formats, ``ckv`` is stored
        in ``elem_store_dtype`` (1 B/elem) and ``k_rope`` stays raw ``compute_dtype`` (2 B/elem)."""
        if not self.quantized:
            return d_v + d_r
        ckv_bytes = d_v * self.elem_dtype.itemsize
        rope_bytes = d_r * (compute_dtype.itemsize if self.rope_unquantized else self.elem_dtype.itemsize)
        total_bytes = ckv_bytes + rope_bytes
        if total_bytes % self.elem_store_dtype.itemsize:
            raise ValueError(f"{self.name} cannot store MLA row ({d_v}+{d_r})")
        return total_bytes // self.elem_store_dtype.itemsize

    def mla_scale_dim(self, d_v: int, d_r: int = 0) -> int:
        """Number of fp32 scales per MLA latent row."""
        if self.scale_dtype is None:
            return 0
        n_ckv = self.scale_dim(d_v)
        if self.rope_unquantized or d_r == 0:
            return n_ckv
        return n_ckv + (1 if self.scale_group == ROW else self.scale_dim(d_r))

    def mla_row_bytes(self, d_v: int, d_r: int, compute_dtype: torch.dtype) -> int:
        """Total bytes per MLA latent row (data + scales)."""
        if not self.quantized:
            return (d_v + d_r) * compute_dtype.itemsize
        data_bytes = self.mla_row_elems(d_v, d_r, compute_dtype) * self.elem_store_dtype.itemsize
        scale_bytes = self.mla_scale_dim(d_v, d_r) * self.scale_store_dtype.itemsize
        return data_bytes + scale_bytes


NONE = KVFormat("none", elem_dtype=None, elem_store_dtype=None)
FP8_MLA = KVFormat(
    "fp8_mla",
    elem_dtype=torch.float8_e4m3fn,
    elem_store_dtype=torch.uint8,
    scale_dtype=torch.float32,
    scale_store_dtype=torch.float32,
    scale_group=128,
    rope_unquantized=True,
)
INT8_MLA = KVFormat(
    "int8_mla",
    elem_dtype=torch.int8,
    elem_store_dtype=torch.int8,
    scale_dtype=torch.float32,
    scale_store_dtype=torch.float32,
    scale_group=128,
    rope_unquantized=True,
)

KV_FORMATS: dict[str, KVFormat] = {
    fmt.name: fmt for fmt in (NONE, FP8_MLA, INT8_MLA)
}
# Formats supported by MLA/DSA pools (group-128 quantized ckv + unquantized bf16 k_rope).
ROW_SCALE_FORMATS = frozenset({NONE.name, FP8_MLA.name, INT8_MLA.name})


def resolve_kv_format(kv_cache_dtype: str) -> KVFormat:
    """The format of a --kv-cache-dtype value; "auto" never quantizes."""
    if kv_cache_dtype == "auto":
        return NONE
    try:
        return KV_FORMATS[kv_cache_dtype]
    except KeyError:
        raise ValueError(f"unknown --kv-cache-dtype {kv_cache_dtype!r}; one of auto, {', '.join(KV_FORMATS)}") from None
