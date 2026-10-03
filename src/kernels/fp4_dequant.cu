#include "kernels/fp4_dequant.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp {
namespace {

constexpr int64_t kFp4Group = 16;
constexpr int kThreads = 256;

// One thread per output element, the fp8 bridge's shape: two u8 reads and
// one bf16 write per element, memory-bound by construction. The bridge
// runs once per prefill chunk, never in a hot loop.
__global__ void fp4_dequant_kernel(const uint8_t* __restrict__ payload,
                                   const uint8_t* __restrict__ scales,
                                   const float* __restrict__ global_scale,
                                   uint16_t* __restrict__ out, int64_t rows,
                                   int64_t cols) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  const int64_t total = rows * cols;
  if (i >= total) return;
  const int64_t n = i / cols;
  const int64_t k = i - n * cols;
  const uint8_t byte = payload[n * (cols / 2) + k / 2];
  const uint8_t code = (k & 1) ? static_cast<uint8_t>(byte >> 4)
                               : static_cast<uint8_t>(byte & 0xFu);
  const uint8_t sc = scales[n * (cols / kFp4Group) + k / kFp4Group];
  const float v = fp4_e2m1_bits_to_float(code) * fp8_e4m3_bits_to_float(sc) /
                  *global_scale;
  out[i] = float_to_bf16_bits(v);
}

}  // namespace

void launch_fp4_dequant(const uint8_t* payload, const uint8_t* scales,
                        const float* global_scale, uint16_t* out_bf16,
                        int64_t rows, int64_t cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0)
    throw std::invalid_argument("fp4_dequant: rows and cols must be positive");
  if (cols % kFp4Group != 0)
    throw std::invalid_argument("fp4_dequant: cols must be a multiple of 16");
  if (!payload || !scales || !global_scale || !out_bf16)
    throw std::invalid_argument("fp4_dequant: null pointer");
  const int64_t total = rows * cols;
  const int64_t blocks = (total + kThreads - 1) / kThreads;
  if (blocks > 2147483647LL)
    throw std::runtime_error("fp4_dequant: grid too large");
  fp4_dequant_kernel<<<static_cast<int>(blocks), kThreads, 0, stream>>>(
      payload, scales, global_scale, out_bf16, rows, cols);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
