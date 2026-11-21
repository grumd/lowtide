// W4A16 (AutoGPTQ int4) experts: the banks the GPU streams (moe/wna16_cpu.h has the layout).
// qweight [K/8, N] int32, code j of word w -> k = 8w + j; qzeros [groups, N/8] int32 holding
// zero - 1; fp16 scales [groups, N]. N is contiguous, so like mxfp4 a work item owns 64 output
// columns and streams K. Group g covers the band's k in [128g - k_off, 128(g+1) - k_off).
//
// (code - zero) is an integer in [-16, 14]: a 32-entry bf16 table turns it into bf16 exactly, and
// vdpbf16ps multiplies it with a pair of bf16 activations (xp[4w + p] = x[8w + p] | x[8w + p + 4]
// << 16) into fp32 sums. Subtracting the zero point first avoids the cancellation of folding it
// out afterwards (sum code * x - zero * sum x, the GPU kernel's form). AVX-512 BF16 only. An exact
// fp64 variant was 100% correctly rounded but 4-5% slower end to end, with no KLD difference.
#include <cmath>
#include <vector>

#include <torch/extension.h>

#include "../format.h"

namespace freetoken::cpu_moe {
namespace {

constexpr int W4_COLS = 64;
static_assert(W4_COLS <= kMaxRowBlock, "the w4a16 tile must fit the executor's row block");

inline int w4_group_lo(int g, int k_off) { return std::max(0, g * 128 - k_off); }
inline int w4_group_hi(int g, int k_off, int K) { return std::min(K, (g + 1) * 128 - k_off); }

inline void w4_pack_pairs(const bf16_t* x, int K, uint32_t* xp) {
  for (int w = 0; w < K / 8; ++w)
    for (int p = 0; p < 4; ++p) xp[4 * w + p] = uint32_t(x[8 * w + p]) | (uint32_t(x[8 * w + p + 4]) << 16);
}

#if CPU_MOE_X86 && defined(CPU_MOE_HAS_AVX512BF16)
#define CPU_MOE_HAS_W4A16 1
CPU_MOE_TARGET("avx512f,avx512bw,avx512bf16,fma,f16c")
void w4a16_cols64(const int32_t* qw, const int32_t* qz, const uint16_t* sc, int N, int n0, int K, int k_off,
                  const uint32_t* xp, float* out) {
  constexpr int V = W4_COLS / 16;
  const __m512i nib = _mm512_set1_epi32(15), one = _mm512_set1_epi32(1), nib16 = _mm512_set1_epi16(15);
  const __m512i zidx = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
  const __m512i zsh = _mm512_set_epi32(28, 24, 20, 16, 12, 8, 4, 0, 28, 24, 20, 16, 12, 8, 4, 0);
  alignas(64) uint16_t tbl[32];
  for (int i = 0; i < 32; ++i) tbl[i] = f32_to_bf16(float(i - 16));
  const __m512i lut = _mm512_load_si512(tbl);
  const int groups = (K + k_off + 127) / 128;
  __m512 o[V];
  for (int v = 0; v < V; ++v) o[v] = _mm512_setzero_ps();
  for (int g = 0; g < groups; ++g) {
    const int k_lo = w4_group_lo(g, k_off), k_hi = w4_group_hi(g, k_off, K);
    if (k_hi <= k_lo) continue;
    __m512i off[V];
    __m512 acc[V];
    for (int v = 0; v < V; ++v) {
      const int32_t* zr = qz + (size_t)g * (N / 8) + (n0 + 16 * v) / 8;
      const __m512i zw = _mm512_permutexvar_epi32(zidx, _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                         0, 0, zr[1], zr[0]));
      const __m512i zp = _mm512_add_epi32(_mm512_and_si512(_mm512_srlv_epi32(zw, zsh), nib), one);
      const __m512i o32 = _mm512_sub_epi32(_mm512_set1_epi32(16), zp);  // index = code + 16 - (zero + 1)
      off[v] = _mm512_or_si512(o32, _mm512_slli_epi32(o32, 16));
      acc[v] = _mm512_setzero_ps();
    }
    for (int w = k_lo / 8; w < k_hi / 8; ++w) {
      const int32_t* row = qw + (size_t)w * N + n0;
      __m512i q[V];
      for (int v = 0; v < V; ++v) q[v] = _mm512_loadu_si512(row + 16 * v);
      for (int p = 0; p < 4; ++p) {
        const __m512bh xb = (__m512bh)_mm512_set1_epi32(xp[4 * w + p]);
        for (int v = 0; v < V; ++v) {
          const __m512i idx = _mm512_add_epi16(_mm512_and_si512(_mm512_srli_epi16(q[v], 4 * p), nib16), off[v]);
          acc[v] = _mm512_dpbf16_ps(acc[v], (__m512bh)_mm512_permutexvar_epi16(idx, lut), xb);
        }
      }
    }
    for (int v = 0; v < V; ++v)
      o[v] = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sc + (size_t)g * N + n0 + 16 * v))),
                             acc[v], o[v]);
  }
  for (int v = 0; v < V; ++v) _mm512_storeu_ps(out + 16 * v, o[v]);
}
#endif

// Pass 1 computes the gate and up columns of a 64-unit block over K = H, then silu(gate) * up
// rounded to bf16, the GPU's one store point before the down GEMV. Pass 2 returns each route's
// down columns over K = I; the executor adds them in route order in fp32 and rounds once (or
// leaves fp32 for the GPU to add its routes).
class W4a16Format final : public MoeFormat {
 public:
  W4a16Format(const MoeParams& p, const FormatArgs& a) : MoeFormat(p) {
#ifndef CPU_MOE_HAS_W4A16
    throw std::runtime_error("w4a16 CPU MoE needs a compiler with AVX-512 BF16 (GCC >= 10)");
#else
    if (!__builtin_cpu_supports("avx512bf16") || !__builtin_cpu_supports("avx512bw"))
      throw std::runtime_error("w4a16 CPU MoE needs AVX-512 BF16");
#endif
    // the GPU path it shares the decode with: silu(gate) * up, router weight after down
    if (p.act != ACT_SILU) throw std::runtime_error("w4a16 CPU MoE computes silu(gate) * up only");
    if (p.H % 128 != 0 || p.I % W4_COLS != 0 || p.H % W4_COLS != 0)
      throw std::runtime_error("w4a16 CPU MoE requires H % 128 == 0 and I, H multiples of 64");
    gate_up_ = a.bank("gate_up");
    gu_zero_ = a.bank("gate_up_zero");
    gu_scale_ = a.bank("gate_up_scale");
    down_ = a.bank("down");
    dn_zero_ = a.bank("down_zero");
    dn_scale_ = a.bank("down_scale");
    // a TP rank's down band starts mid-group (balanced bands); see wna16_down_k_off
    k_off_ = static_cast<int>(a.option("k_off", 0));
    dn_groups_ = static_cast<int>(a.option("down_groups", (p.I + k_off_ + 127) / 128));
    if (k_off_ < 0 || k_off_ % 8 != 0 || dn_groups_ != (p.I + k_off_ + 127) / 128)
      throw std::invalid_argument("w4a16 down band: k_off must be a non-negative multiple of 8 and "
                                  "down_groups == ceil((I + k_off) / 128)");
    xp_.resize(static_cast<size_t>(p.max_tokens) * (p.H / 2));
    gp_.resize(static_cast<size_t>(p.max_tokens) * p.top_k * (p.I / 2));
  }

  std::string isa_name() const override { return "avx512bf16(w4a16)"; }
  int row_block() const override { return W4_COLS; }

  void prep_input(int tok, const bf16_t* x) override { w4_pack_pairs(x, p.H, xp_.data() + (size_t)tok * (p.H / 2)); }
  bool preps_inter() const override { return true; }
  void prep_inter(int64_t r, bf16_t* g) override { w4_pack_pairs(g, p.I, gp_.data() + (size_t)r * (p.I / 2)); }

  void gate_up(int layer, int e, int tok, int i0, int /*i1*/, const bf16_t* /*x*/, float w_in, bf16_t* g) const override {
#ifdef CPU_MOE_HAS_W4A16
    const int H = p.H, I = p.I, N1 = 2 * I;
    const int32_t* qw = static_cast<const int32_t*>(tbl_at(gate_up_, layer)) + (size_t)e * (H / 8) * N1;
    const int32_t* qz = static_cast<const int32_t*>(tbl_at(gu_zero_, layer)) + (size_t)e * (H / 128) * (N1 / 8);
    const uint16_t* sc = static_cast<const uint16_t*>(tbl_at(gu_scale_, layer)) + (size_t)e * (H / 128) * N1;
    const uint32_t* xp = xp_.data() + (size_t)tok * (H / 2);
    float gate[W4_COLS], up[W4_COLS];
    w4a16_cols64(qw, qz, sc, N1, i0, H, 0, xp, gate);
    w4a16_cols64(qw, qz, sc, N1, I + i0, H, 0, xp, up);
    // the GPU's expression (wna16_decode._reduce_silu_mul_kernel)
    for (int c = 0; c < W4_COLS; ++c) {
      const float gt = gate[c] * w_in;
      g[i0 + c] = f32_to_bf16(gt / (1.0f + std::exp(-gt)) * (up[c] * w_in));
    }
#else
    (void)layer, (void)e, (void)tok, (void)i0, (void)w_in, (void)g;
#endif
  }

  void down(int layer, int h0, int /*h1*/, const DownRoute* routes, int n, float* out) const override {
#ifdef CPU_MOE_HAS_W4A16
    const int H = p.H, I = p.I;
    const int32_t* qw_l = static_cast<const int32_t*>(tbl_at(down_, layer));
    const int32_t* qz_l = static_cast<const int32_t*>(tbl_at(dn_zero_, layer));
    const uint16_t* sc_l = static_cast<const uint16_t*>(tbl_at(dn_scale_, layer));
    for (int j = 0; j < n; ++j) {
      const int e = routes[j].e;
      w4a16_cols64(qw_l + (size_t)e * (I / 8) * H, qz_l + (size_t)e * dn_groups_ * (H / 8),
                   sc_l + (size_t)e * dn_groups_ * H, H, h0, I, k_off_, gp_.data() + (size_t)routes[j].r * (I / 2),
                   out + (size_t)j * W4_COLS);
    }
#else
    (void)layer, (void)h0, (void)routes, (void)n, (void)out;
#endif
  }

 private:
  const uint64_t* gate_up_ = nullptr;
  const uint64_t* gu_zero_ = nullptr;
  const uint64_t* gu_scale_ = nullptr;
  const uint64_t* down_ = nullptr;
  const uint64_t* dn_zero_ = nullptr;
  const uint64_t* dn_scale_ = nullptr;
  int k_off_ = 0, dn_groups_ = 0;
  AlignedVector<uint32_t> xp_, gp_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_w4a16_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<W4a16Format>(p, a);
}

#ifdef CPU_MOE_HAS_W4A16
// Tests: the W4A16 column kernel over a whole [K/8, N] bank, x bf16 [K] -> fp32 [N].
torch::Tensor w4a16_gemv(torch::Tensor qw, torch::Tensor qz, torch::Tensor sc, torch::Tensor x, int64_t k_off) {
  TORCH_CHECK(__builtin_cpu_supports("avx512bf16"), "needs AVX-512 BF16");
  TORCH_CHECK(qw.dim() == 2 && qw.scalar_type() == at::kInt && qw.is_contiguous(), "qw: int32 [K/8, N]");
  TORCH_CHECK(qz.scalar_type() == at::kInt && qz.is_contiguous(), "qz: int32 [groups, N/8]");
  TORCH_CHECK(sc.scalar_type() == at::kHalf && sc.is_contiguous(), "sc: fp16 [groups, N]");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "x: bf16 [K]");
  const int K = static_cast<int>(qw.size(0) * 8), N = static_cast<int>(qw.size(1));
  TORCH_CHECK(N % W4_COLS == 0 && x.numel() == K, "N % 64 == 0 and x [K]");
  std::vector<uint32_t> xp(K / 2);
  w4_pack_pairs(reinterpret_cast<const bf16_t*>(x.data_ptr<at::BFloat16>()), K, xp.data());
  auto out = torch::empty({N}, torch::kFloat);
  for (int n0 = 0; n0 < N; n0 += W4_COLS)
    w4a16_cols64(qw.data_ptr<int32_t>(), qz.data_ptr<int32_t>(), reinterpret_cast<const uint16_t*>(sc.data_ptr<at::Half>()),
                 N, n0, K, static_cast<int>(k_off), xp.data(), out.data_ptr<float>() + n0);
  return out;
}
bool w4a16_available() { return true; }
#else
torch::Tensor w4a16_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t) {
  throw std::runtime_error("w4a16 needs AVX-512 BF16 at build time");
}
bool w4a16_available() { return false; }
#endif

}  // namespace freetoken::cpu_moe
