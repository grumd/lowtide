// QSA KV tiering (--kv-tier): the K/V of every position lives in pinned, device-mapped host
// memory in the pool's own paged layout; VRAM holds a cache of indexer blocks (index_ratio
// tokens, all KV heads) per layer. See kvcache/qsa_pool.py and attention/qsa_sparse.py.
//
//   store      write K/V rows to the host copy, and to the cache slot when the block is resident
//   resolve    make every block a decode selection names resident (one 1024-thread block: hits
//              stamp the epoch and the clock bit, misses are claimed once, a CLOCK sweep picks one
//              victim per miss that this call does not use, the map is re-pointed), then write
//              the selection as cache cells (slot * ratio + token % ratio)
//   copy_rows  copy rows by index, host -> VRAM, zero-copy (the misses; prefill page staging)
//   mark_pages which pages of a staging plan a prefill chunk's selection reads
//
// All pointers arrive as int64 addresses (host pointers must be device-mapped), plus one device
// tensor that names the device and therefore the stream. Every kernel is graph-capturable.
#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace kv_tier {

constexpr int kResolveThreads = 1024;
// ctl[16] per layer: [0] epoch [1] clock hand [2] misses of the last call [3] error flags;
// u64 at [4] misses, [6] lookups, [8] calls; [10] slots in use
constexpr int kCtlInts = 16;
constexpr int kCtlActive = 10;  // slots in use, 0 = all

struct Selection {
  const int32_t* indices;  // [rows, width] logical token positions, -1 padded
  int64_t idx_stride;
  int32_t width;
  int32_t rows;
  const int32_t* block_table;  // [num_requests, bt_width] physical page ids
  int64_t bt_stride;
  int32_t bt_width;
  int32_t num_requests;
  const int32_t* token_to_req;  // [rows]
  int32_t num_pages;            // pages of the pool (main's num_cache_blocks)
  int32_t page_size;
  int32_t ratio;
};

struct Map {
  int32_t* map;  // [n_blocks] physical block -> slot, -1 (or -2 while claimed)
  int32_t* slot_block;
  int32_t* slot_stamp;
  int32_t* slot_ref;
  int32_t* ctl;
  int32_t* miss_block;  // [n_slots]
  int32_t* miss_slot;   // [n_slots]
  int32_t n_slots;
};

// The physical block of a selected token, or -1 where main's attention kernel masks it.
__device__ __forceinline__ int64_t phys_block(const Selection& s, int row, int token) {
  if (token < 0) return -1;
  const int req = s.token_to_req[row];
  if (req < 0 || req >= s.num_requests) return -1;
  const int lp = token / s.page_size;
  if (lp >= s.bt_width) return -1;
  const int pp = s.block_table[(int64_t)req * s.bt_stride + lp];
  if (pp < 0 || pp >= s.num_pages) return -1;
  return (int64_t)pp * (s.page_size / s.ratio) + (token % s.page_size) / s.ratio;
}

// Block-wide exclusive prefix sum of one int per thread; `total` is the block's sum.
__device__ int block_scan(int v, int* warp_sums, int& total) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) warp_sums[w] = x;
  __syncthreads();
  if (w == 0) {
    int t = warp_sums[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, t, o);
      if (lane >= o) t += y;
    }
    warp_sums[lane] = t;
  }
  __syncthreads();
  total = warp_sums[31];
  const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
  __syncthreads();
  return excl;
}

// `out` (optional, [rows, out_stride]): the selection as cache cells, -1 where the token is masked
__global__ void __launch_bounds__(kResolveThreads) resolve_kernel(Map m, Selection s, int32_t* out,
                                                                  int64_t out_stride) {
  __shared__ int s_nmiss, s_lookups, s_cut;
  __shared__ int warp_sums[32];
  const int epoch = m.ctl[0] + 1;
  if (threadIdx.x == 0) {
    s_nmiss = 0;
    s_lookups = 0;
  }
  __syncthreads();
  // 1. one lookup per (row, block): the expanded selection lists a block's tokens back to back.
  //    A token whose block is resident gets its cache cell here: this pass stamps the slot, so
  //    the sweep below cannot take it, and the map entry of a resident block does not change
  //    before step 3. With every block resident, the usual decode step, that is the whole launch.
  int lookups = 0;
  for (int r = 0; r < s.rows; ++r) {
    const int32_t* row = s.indices + (int64_t)r * s.idx_stride;
    for (int i = threadIdx.x; i < s.width; i += kResolveThreads) {
      const int t = row[i];
      const int64_t b = phys_block(s, r, t);
      if (b < 0) {
        if (out != nullptr) out[(int64_t)r * out_stride + i] = -1;
        continue;
      }
      const int sl = m.map[b];
      if (out != nullptr && sl >= 0) out[(int64_t)r * out_stride + i] = sl * s.ratio + t % s.ratio;
      if (i > 0) {
        const int prev = row[i - 1];
        if (prev >= 0 && prev / s.ratio == t / s.ratio) continue;
      }
      ++lookups;
      if (sl >= 0) {
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
      } else if (sl == -1 && atomicCAS(&m.map[b], -1, -2) == -1) {
        const int k = atomicAdd(&s_nmiss, 1);
        if (k < m.n_slots) {
          m.miss_block[k] = (int32_t)b;
        } else {
          m.map[b] = -1;
          atomicOr(&m.ctl[3], 1);
        }
      }
    }
  }
  if (lookups > 0) atomicAdd(&s_lookups, lookups);
  __syncthreads();
  // 2. one victim per miss, CLOCK from the hand: a slot this call uses (stamp == epoch) is never
  //    taken; a referenced slot loses its bit as the hand passes and is taken on a later pass
  //    ctl[10] > 0 narrows the cache to its first ctl[10] slots (an elastic pool smaller than the cache)
  const int act = m.ctl[kCtlActive];
  const int n = act > 0 && act < m.n_slots ? act : m.n_slots;
  const int need = min(s_nmiss, n);
  int hand = m.ctl[1], got = 0;
  for (int scanned = 0; got < need && scanned < 3 * n; scanned += kResolveThreads) {
    const int j = (int)(((int64_t)hand + threadIdx.x) % n);
    const bool mine = m.slot_stamp[j] == epoch;
    const bool cand = !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
    int total = 0;
    const int rank = block_scan(cand ? 1 : 0, warp_sums, total);
    const int want = need - got;
    if (threadIdx.x == 0) s_cut = kResolveThreads;
    __syncthreads();
    if (cand && rank == want - 1) s_cut = threadIdx.x + 1;  // the hand stops past the last slot taken
    __syncthreads();
    const int cut = s_cut;
    if (cand && rank < want) {
      m.miss_slot[got + rank] = j;
      m.slot_stamp[j] = epoch;  // a sweep that wraps around must not take it twice
    } else if (threadIdx.x < cut && !mine) {
      m.slot_ref[j] = 0;
    }
    got += total < want ? total : want;
    hand = (int)(((int64_t)hand + cut) % n);
    __syncthreads();
  }
  // 3. re-point the map; copy_rows fills the slots
  const int placed = got < need ? got : need;
  for (int k = threadIdx.x; k < need; k += kResolveThreads) {
    const int b = m.miss_block[k];
    if (k >= placed) {
      m.map[b] = -1;
      continue;
    }
    const int sl = m.miss_slot[k];
    const int old = m.slot_block[sl];
    if (old >= 0) m.map[old] = -1;
    m.slot_block[sl] = b;
    m.slot_stamp[sl] = epoch;
    m.slot_ref[sl] = 1;
    m.map[b] = sl;
  }
  if (threadIdx.x == 0) {
    m.ctl[0] = epoch;
    m.ctl[1] = hand;
    m.ctl[2] = placed;
    if (placed < s_nmiss) m.ctl[3] |= 2;
    auto* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
    c[0] += (unsigned long long)placed;
    c[1] += (unsigned long long)s_lookups;
    c[2] += 1ull;
  }
  if (out == nullptr || s_nmiss == 0) return;
  __syncthreads();
  // 4. after a miss: the selection in cache cells once more, now that every block it names has
  //    a slot
  for (int r = 0; r < s.rows; ++r) {
    const int32_t* row = s.indices + (int64_t)r * s.idx_stride;
    for (int i = threadIdx.x; i < s.width; i += kResolveThreads) {
      const int t = row[i];
      const int64_t b = phys_block(s, r, t);
      int v = -1;
      if (b >= 0) {
        const int sl = m.map[b];
        if (sl >= 0) {
          v = sl * s.ratio + t % s.ratio;
        } else {
          atomicOr(&m.ctl[3], 4);  // a selected block that is not resident: never with a legal cache size
        }
      }
      out[(int64_t)r * out_stride + i] = v;
    }
  }
}

struct Rows {
  const uint8_t* src[2];
  uint8_t* dst[2];
  int64_t row_bytes;
  const int32_t* src_rows;
  const int32_t* dst_rows;  // null: row k goes to row k
  const int32_t* count;     // null: n rows
  int32_t n;
};

// grid (G, 2): blockIdx.y picks K or V; each block copies whole rows, 4 loads in flight per thread
__global__ void copy_rows_kernel(Rows r) {
  const int n = r.count != nullptr ? *r.count : r.n;
  const int a = blockIdx.y;
  const int64_t vec = r.row_bytes / 16;
  for (int k = blockIdx.x; k < n; k += gridDim.x) {
    const int64_t sr = r.src_rows[k];
    const int64_t dr = r.dst_rows != nullptr ? r.dst_rows[k] : k;
    const uint4* src = reinterpret_cast<const uint4*>(r.src[a] + sr * r.row_bytes);
    uint4* dst = reinterpret_cast<uint4*>(r.dst[a] + dr * r.row_bytes);
    for (int64_t i = threadIdx.x; i < vec; i += 4 * blockDim.x) {
      uint4 v[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int64_t j = i + u * blockDim.x;
        if (j < vec) v[u] = src[j];
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int64_t j = i + u * blockDim.x;
        if (j < vec) dst[j] = v[u];
      }
    }
  }
}

// used[row] = 1 for every row of a staging plan the selection reads a token of. `s.block_table`
// is the plan's table here, (request, logical page) -> staged row, and `s.num_pages` its rows.
__global__ void mark_pages_kernel(Selection s, int32_t* used) {
  const int r = blockIdx.y;
  const int req = s.token_to_req[r];
  if (req < 0 || req >= s.num_requests) return;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < s.width; i += gridDim.x * blockDim.x) {
    const int t = s.indices[(int64_t)r * s.idx_stride + i];
    if (t < 0) continue;
    const int lp = t / s.page_size;
    if (lp >= s.bt_width) continue;
    const int row = s.block_table[(int64_t)req * s.bt_stride + lp];
    if (row >= 0 && row < s.num_pages) used[row] = 1;
  }
}

struct Store {
  const uint8_t* k;
  const uint8_t* v;
  int64_t k_stride;  // bytes between input rows
  int64_t v_stride;
  const void* out_loc;
  int32_t loc64;
  uint8_t* host[2];
  uint8_t* cache[2];
  const int32_t* map;
  int64_t row_bytes;
  int32_t ratio;
  int32_t length;
};

// one warp per token
__global__ void store_kernel(Store p) {
  const int64_t w = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  if (w >= p.length) return;
  const int64_t pos = p.loc64 ? static_cast<const int64_t*>(p.out_loc)[w]
                              : static_cast<const int32_t*>(p.out_loc)[w];
  const int sl = p.map[pos / p.ratio];
  const int64_t vec = p.row_bytes / 16;
  const int64_t cell = (int64_t)sl * p.ratio + pos % p.ratio;
#pragma unroll
  for (int a = 0; a < 2; ++a) {
    const uint4* src = reinterpret_cast<const uint4*>((a ? p.v : p.k) + w * (a ? p.v_stride : p.k_stride));
    uint4* host = reinterpret_cast<uint4*>(p.host[a] + pos * p.row_bytes);
    uint4* cache = reinterpret_cast<uint4*>(p.cache[a] + cell * p.row_bytes);
    for (int64_t i = lane; i < vec; i += 32) {
      const uint4 x = src[i];
      host[i] = x;
      if (sl >= 0) cache[i] = x;
    }
  }
}

inline DLDevice device_of(const tvm::ffi::TensorView& anchor) {
  const DLDevice d = anchor.device();
  host::RuntimeCheck(d.device_type == kDLCUDA, "kv_tier: the anchor tensor must be on the GPU");
  return d;
}

inline void check_aligned(int64_t addr, int64_t bytes, const char* what) {
  host::RuntimeCheck(addr % 16 == 0 && bytes % 16 == 0, "kv_tier: ", what, " is not 16-byte aligned");
}

void store(tvm::ffi::TensorView anchor, int64_t k, int64_t v, int64_t k_stride, int64_t v_stride,
           int64_t out_loc, int64_t loc64, int64_t host_k, int64_t host_v, int64_t cache_k,
           int64_t cache_v, int64_t map, int64_t row_bytes, int64_t ratio, int64_t length) {
  if (length <= 0) return;
  check_aligned(k, k_stride, "k");
  check_aligned(v, v_stride, "v");
  check_aligned(host_k, row_bytes, "host k");
  check_aligned(host_v, row_bytes, "host v");
  Store p{};
  p.k = reinterpret_cast<const uint8_t*>(k);
  p.v = reinterpret_cast<const uint8_t*>(v);
  p.k_stride = k_stride;
  p.v_stride = v_stride;
  p.out_loc = reinterpret_cast<const void*>(out_loc);
  p.loc64 = (int32_t)loc64;
  p.host[0] = reinterpret_cast<uint8_t*>(host_k);
  p.host[1] = reinterpret_cast<uint8_t*>(host_v);
  p.cache[0] = reinterpret_cast<uint8_t*>(cache_k);
  p.cache[1] = reinterpret_cast<uint8_t*>(cache_v);
  p.map = reinterpret_cast<const int32_t*>(map);
  p.row_bytes = row_bytes;
  p.ratio = (int32_t)ratio;
  p.length = (int32_t)length;
  constexpr int kThreads = 256;
  const auto blocks = (unsigned)host::div_ceil(length * 32, (int64_t)kThreads);
  host::LaunchKernel(blocks, kThreads, device_of(anchor))(store_kernel, p);
}

Selection make_selection(int64_t indices, int64_t idx_stride, int64_t width, int64_t rows,
                         int64_t block_table, int64_t bt_stride, int64_t bt_width,
                         int64_t num_requests, int64_t token_to_req, int64_t num_pages,
                         int64_t page_size, int64_t ratio) {
  Selection s{};
  s.indices = reinterpret_cast<const int32_t*>(indices);
  s.idx_stride = idx_stride;
  s.width = (int32_t)width;
  s.rows = (int32_t)rows;
  s.block_table = reinterpret_cast<const int32_t*>(block_table);
  s.bt_stride = bt_stride;
  s.bt_width = (int32_t)bt_width;
  s.num_requests = (int32_t)num_requests;
  s.token_to_req = reinterpret_cast<const int32_t*>(token_to_req);
  s.num_pages = (int32_t)num_pages;
  s.page_size = (int32_t)page_size;
  s.ratio = (int32_t)ratio;
  return s;
}

// state: int64 [7] device addresses of map, slot_block, slot_stamp, slot_ref, ctl, miss_block, miss_slot
void resolve(tvm::ffi::TensorView anchor, int64_t indices, int64_t idx_stride, int64_t width,
             int64_t rows, int64_t block_table, int64_t bt_stride, int64_t bt_width,
             int64_t num_requests, int64_t token_to_req, int64_t num_pages, int64_t page_size,
             int64_t ratio, int64_t map, int64_t slot_block, int64_t slot_stamp, int64_t slot_ref,
             int64_t ctl, int64_t miss_block, int64_t miss_slot, int64_t n_slots, int64_t out,
             int64_t out_stride) {
  if (rows <= 0) return;
  // one sweep window looks at kResolveThreads consecutive slots: with fewer slots two threads
  // would see the same slot and could both take it
  host::RuntimeCheck(n_slots >= kResolveThreads, "kv_tier: the block cache needs >= ",
                     kResolveThreads, " slots, got ", n_slots);
  Map m{};
  m.map = reinterpret_cast<int32_t*>(map);
  m.slot_block = reinterpret_cast<int32_t*>(slot_block);
  m.slot_stamp = reinterpret_cast<int32_t*>(slot_stamp);
  m.slot_ref = reinterpret_cast<int32_t*>(slot_ref);
  m.ctl = reinterpret_cast<int32_t*>(ctl);
  m.miss_block = reinterpret_cast<int32_t*>(miss_block);
  m.miss_slot = reinterpret_cast<int32_t*>(miss_slot);
  m.n_slots = (int32_t)n_slots;
  const Selection s = make_selection(indices, idx_stride, width, rows, block_table, bt_stride,
                                     bt_width, num_requests, token_to_req, num_pages, page_size,
                                     ratio);
  host::LaunchKernel(1, kResolveThreads, device_of(anchor))(resolve_kernel, m, s,
                                                            reinterpret_cast<int32_t*>(out), out_stride);
}

void copy_rows(tvm::ffi::TensorView anchor, int64_t src_k, int64_t src_v, int64_t dst_k,
               int64_t dst_v, int64_t row_bytes, int64_t src_rows, int64_t dst_rows,
               int64_t count, int64_t n, int64_t grid, int64_t threads) {
  check_aligned(src_k, row_bytes, "copy source k");
  check_aligned(src_v, row_bytes, "copy source v");
  check_aligned(dst_k, row_bytes, "copy destination k");
  check_aligned(dst_v, row_bytes, "copy destination v");
  if (grid <= 0) return;
  Rows r{};
  r.src[0] = reinterpret_cast<const uint8_t*>(src_k);
  r.src[1] = reinterpret_cast<const uint8_t*>(src_v);
  r.dst[0] = reinterpret_cast<uint8_t*>(dst_k);
  r.dst[1] = reinterpret_cast<uint8_t*>(dst_v);
  r.row_bytes = row_bytes;
  r.src_rows = reinterpret_cast<const int32_t*>(src_rows);
  r.dst_rows = reinterpret_cast<const int32_t*>(dst_rows);
  r.count = reinterpret_cast<const int32_t*>(count);
  r.n = (int32_t)n;
  host::LaunchKernel(dim3((unsigned)grid, 2), (unsigned)threads, device_of(anchor))(copy_rows_kernel, r);
}

void mark_pages(tvm::ffi::TensorView anchor, int64_t indices, int64_t idx_stride, int64_t width,
                int64_t rows, int64_t block_table, int64_t bt_stride, int64_t bt_width,
                int64_t num_requests, int64_t token_to_req, int64_t num_pages, int64_t page_size,
                int64_t ratio, int64_t used) {
  if (rows <= 0) return;
  const Selection s = make_selection(indices, idx_stride, width, rows, block_table, bt_stride,
                                     bt_width, num_requests, token_to_req, num_pages, page_size,
                                     ratio);
  constexpr int kThreads = 256;
  const auto gx = (unsigned)host::div_ceil(width, (int64_t)kThreads);
  host::LaunchKernel(dim3(gx, (unsigned)rows), kThreads, device_of(anchor))(
      mark_pages_kernel, s, reinterpret_cast<int32_t*>(used));
}

}  // namespace kv_tier
