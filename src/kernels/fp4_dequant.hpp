#pragma once
// NVFP4 -> BF16 dequantization, the fp8 bridge's fp4 twin
// (kernels/fp8_dequant.hpp): the dense MLP's prefill bridge under
// engine.dense_weights = "nvfp4" (docs/qwen38_dual_spark.md).
//   out[n, k] = bf16( e2m1(payload[n, k]) * e4m3(scales[n, k/16]) / global )
// The kernels' contract (kernels/fp4_gemv.hpp): the device global is the
// encoder recipe's 1 / weight_scale_2. One round to BF16 at the end — the
// same single-rounding shape as the fp8 bridge, so the bf16 GEMM that
// consumes it matches the fp4 GEMV to the format's precision.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// Enqueues the dequant for a [rows, cols] NVFP4 matrix (cols % 16 == 0,
// the GlmFp4Matrix layout: e2m1 pairs [rows, cols/2], e4m3 block scales
// [rows, cols/16], one device fp32 global). Async on `stream`.
void launch_fp4_dequant(const uint8_t* payload, const uint8_t* scales,
                        const float* global_scale, uint16_t* out_bf16, int64_t rows,
                        int64_t cols, cudaStream_t stream);

}  // namespace dgpp
