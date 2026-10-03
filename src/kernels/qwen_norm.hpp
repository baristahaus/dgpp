#pragma once
// Qwen3.8-Flash-Next norm kernels (Q3, 2026-09-09; docs/qwen38_flash_next_plan.md
// §1.2, §1.3, §1.5). The family's norms are the zero-centered form: fp32
// interior, y = bf16(x * rsqrt(mean(x^2) + eps) * (1 + w)) — one rounding
// (transformers Qwen3_5RMSNorm), unlike GLM's two-rounding norm. The GDN
// output norm keeps the reference's three roundings (Qwen3NextRMSNormGated:
// bf16 after the normalization, after the plain-weight multiply, after the
// sigmoid gate). Deterministic reductions (fixed order) everywhere.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// y[rows, dim] = rmsnorm(x) * (1 + w); bf16 io, fp32 inside, one rounding.
void qwen_rmsnorm_bf16(const void* x, const void* weight, void* y, int64_t rows,
                       int dim, float eps, cudaStream_t stream);

// The grouped form over a row of `groups` x `group_dim` (the 4-branch hyper
// state, group_dim = hidden): every group normalized on its own, the weight
// [groups * group_dim] indexed by the flat position. bf16 io.
void qwen_group_rmsnorm_bf16(const void* x, const void* weight, void* y,
                             int64_t rows, int groups, int group_dim, float eps,
                             cudaStream_t stream);

// The GDN output norm: y = bf16(bf16(bf16(x * rstd) * w) * act(gate)),
// one row per (token, head) of width dim; w is the plain weight (init 1).
// act is sigmoid, or silu(gate) = g * sigmoid(g) when swish_gate (the
// dense form's F.gated_rms_norm_zx act_fn="swish" knob).
void gdn_gated_rmsnorm_bf16(const void* x, const void* gate, const void* weight,
                            void* y, int64_t rows, int dim, float eps,
                            cudaStream_t stream, bool swish_gate = false);

}  // namespace dgpp
