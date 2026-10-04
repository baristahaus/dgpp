#pragma once
// Block-scaled E4M3 -> BF16 dequantization (DESIGN §4 contract):
//   out[n, k] = decode_e4m3(payload[n, k]) * scales[n / 128][k / 128]
// This is the transient bridge for interfaces that still consume BF16 (the M3
// DSA layer weights); native block-scaled GEMM consumption is the M4
// scale-aware GEMM's job. Kept as a standalone kernel because parity tests
// and the loader both need the exact same rounding (decode-multiply, then a
// single round-to-nearest-even BF16 conversion).
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// Enqueues the dequant for a [rows, cols] E4M3 payload with an F32 scale
// grid in the grid launchers' log2 form: [ceil(rows / 2^rs),
// ceil(cols / 2^cs)] row-major (defaults 7/7: the checkpoint's 128 x 128
// blocks). Ragged tail blocks (rows or cols not a multiple of the block)
// are handled; out-of-block elements never read their scale. Async on
// `stream`.
void launch_fp8_dequant_blocks(const uint8_t* payload, const float* scales,
                               uint16_t* out_bf16, int64_t rows, int64_t cols,
                               cudaStream_t stream, int rs = 7, int cs = 7);

}  // namespace dgpp
