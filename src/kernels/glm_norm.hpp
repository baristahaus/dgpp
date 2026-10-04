#pragma once
// GLM decoder-path elementwise kernels (M4 chunk 6, DESIGN §7.5): the
// two-rounding RMSNorm and the mHC stream initialization.
//
// The norm reproduces the transformers Glm5NextTextRMSNorm choreography
// bit-for-bit: fp32 interior, then
//     u  = bf16(x * rsqrt(mean(x^2) + eps))     (round 1)
//     y  = bf16(w * u)                          (round 2 — bf16 multiply)
// The generic rmsnorm_bf16 (kernels.hpp) rounds ONCE and serves the toy
// doll; the GLM path must not mix the two.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// y[rows, dim] = two-rounding RMSNorm of x[rows, dim] with weight[dim].
// bf16 io, fp32 accumulation, deterministic (fixed reduction order).
void glm_rmsnorm_bf16(const void* x, const void* weight, void* y, int rows,
                      int dim, float eps, cudaStream_t stream);

// Initializes the mHC residual streams: streams[t, s, h] = embed[tokens[t],
// h] for all s < hc_mult (the reference's expand of the embedding). hc_mult
// is pinned to 4 by the config parser (kernel smem layouts assume it).
// streams: bf16 [num_tokens, 4, hidden]; embed_table: bf16 [vocab, hidden].
void glm_embed_bcast_streams(const void* embed_table, const int64_t* tokens,
                             void* streams, int num_tokens, int hidden,
                             cudaStream_t stream);

// ---- MTP draft block (DESIGN §9) -------------------------------------------
// out[t] = [rmsnorm(embed[tokens[t]]) * enorm | rmsnorm(hidden_cache[pos_t])
// * hnorm], bf16 [rows, 2*hidden] — the eh_proj input. pos_t = positions[t]
// (device) or first_pos + t when positions is null; a negative position is
// a padding row and reads position 0. Both norms are the two-rounding GLM
// norm above.
void glm_mtp_input_bf16(const void* embed_table, const int64_t* tokens,
                        const void* hidden_cache, const int64_t* positions,
                        int64_t first_pos, const void* enorm, const void* hnorm,
                        void* out, int rows, int hidden, float eps,
                        cudaStream_t stream);

// Request-indexed fixed-batch form. hidden_cache starts at slot 0 and
// request_cache_stride is its bf16-element distance between slots;
// request_ids[t] chooses the cache for row t. Negative positions are
// deterministic zero padding rows (both halves are zero).
void glm_mtp_input_bf16_batched(
    const void* embed_table, const int64_t* tokens, const void* hidden_cache,
    int64_t request_cache_stride, const int32_t* request_ids,
    const int64_t* positions, const void* enorm, const void* hnorm, void* out,
    int rows, int hidden, float eps, cudaStream_t stream);

// x[n] += y[n] in bf16 (fp32 add, one rounding): the plain pre-norm
// residual the draft block uses (the main stack's residual is mHC's).
void glm_residual_add_bf16(void* x, const void* y, int64_t n,
                           cudaStream_t stream);

// cache[positions[t], :] = rows[t, :] for t < num_rows (bf16 rows of
// `hidden`): the per-position hidden cache the draft block reads from.
void glm_rows_scatter_bf16(const void* rows, const int64_t* positions,
                           void* cache, int num_rows, int hidden,
                           cudaStream_t stream);

// Request-indexed scatter into cache[request_ids[t], positions[t], :].
// Padding rows (negative positions) do not touch the cache.
void glm_rows_scatter_bf16_batched(
    const void* rows, const int32_t* request_ids, const int64_t* positions,
    void* cache, int64_t request_cache_stride, int num_rows, int hidden,
    cudaStream_t stream);

}  // namespace dgpp
