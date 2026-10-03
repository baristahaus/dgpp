#include "kernels/qwen_vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"

namespace {
template <class T>
struct Managed {
  T* p = nullptr;
  explicit Managed(size_t n) { DGPP_CUDA_OK(cudaMallocManaged(&p, n * sizeof(T))); }
  ~Managed() { cudaFree(p); }
};

DGPP_TEST(qwen_vision_layernorm_gelu_cuda_reference) {
  // PyTorch 2.14 CUDA goldens: F.layer_norm(x, (1152,), w, b, 1e-6),
  // followed separately by F.gelu(..., approximate="tanh") and F.gelu(...),
  // all BF16 tensors. The LayerNorm and the tanh approximation are
  // bit-stable across the engine's arches, but the exact (erf) GELU's
  // negative tail is not: for inputs below -4.7 the output is ~1e-8, and
  // nvcc contracts v*0.5f*(1 + erff(...)) into different instruction
  // sequences for sm_120a and sm_121a, so tail elements straddle BF16
  // rounding points differently (the deep tail flushes to -0 on one arch
  // and keeps a subnormal on the other). Both arches' digests are recorded
  // — the golden for erf (exact) GELU rides the build arch (CMakeLists' DGPP_CUDA_ARCH).
#if defined(DGPP_CUDA_ARCH) && DGPP_CUDA_ARCH == 120
  constexpr uint64_t kExactDigest = 0xff941ffcd9874396ull;  // sm_120a (RTX 5070 Ti)
#else
  constexpr uint64_t kExactDigest = 0x0057a450751a6f7dull;  // sm_121a (GB10)
#endif
  constexpr int rows = 16, dim = 1152;
  Managed<uint16_t> x(rows * dim), weight(dim), bias(dim), out(rows * dim), exact(rows * dim);
  uint32_t state = 337;
  const auto value = [&] {
    state = state * 1664525u + 1013904223u;
    const float mantissa = static_cast<float>(int((state >> 8) % 251) - 125) / 128;
    return dgpp::float_to_bf16_bits(std::ldexp(mantissa, int(state % 9) - 4));
  };
  for (int i = 0; i < rows * dim; ++i) x.p[i] = value();
  for (int i = 0; i < dim; ++i) weight.p[i] = value();
  for (int i = 0; i < dim; ++i) bias.p[i] = value();
  const auto digest = [](const uint16_t* data) {
    uint64_t hash = 14695981039346656037ull;
    for (int i = 0; i < rows * dim; ++i) {
      hash = (hash ^ (data[i] & 255)) * 1099511628211ull;
      hash = (hash ^ (data[i] >> 8)) * 1099511628211ull;
    }
    return hash;
  };
  dgpp::qwen_vision_layernorm(x.p, weight.p, bias.p, out.p, rows, dim, 1e-6f, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  if (digest(out.p) != 0x8058bc51f45a4ab7ull)
    throw std::runtime_error("Qwen LayerNorm differs from the CUDA reference");
  DGPP_CUDA_OK(cudaMemcpy(exact.p, out.p, rows * dim * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
  dgpp::qwen_vision_gelu_exact(exact.p, rows * dim, nullptr);
  dgpp::qwen_vision_gelu(out.p, rows * dim, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  if (digest(exact.p) != kExactDigest)
    throw std::runtime_error("Qwen merger GELU differs from the CUDA reference");
  if (digest(out.p) != 0xc1b2ea6c08967880ull)
    throw std::runtime_error("Qwen GELU differs from the CUDA reference");
}

DGPP_TEST(qwen_vision_softmax_matches_host_across_patch_counts) {
  // Includes the smallest harness image, the serving floor, low detail and
  // the 1024-token ceiling. Run this test under racecheck as well: numerical
  // agreement alone cannot establish that the shared reductions are safe.
  constexpr int rows = 16;
  const float scale = 1.0f / std::sqrt(72.0f);
  for (int n : {4, 64, 256, 1024, 2048, 2052, 4096}) {
    Managed<float> scores(rows * n);
    Managed<uint16_t> probs(rows * n);
    for (int i = 0; i < rows * n; ++i)
      scores.p[i] = static_cast<float>((i * 17) % 127) * 0.1237f - 8.1234f;
    dgpp::qwen_vision_softmax(scores.p, probs.p, rows, n, scale, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    for (int row = 0; row < rows; ++row) {
      const float* s = scores.p + row * n;
      const double maximum = *std::max_element(s, s + n);
      std::vector<double> expected(n);
      double sum = 0;
      for (int j = 0; j < n; ++j) {
        expected[j] = std::exp((static_cast<double>(s[j]) - maximum) * scale);
        sum += expected[j];
      }
      for (int j = 0; j < n; ++j) {
        const double want = expected[j] / sum;
        const float actual = dgpp::bf16_bits_to_float(probs.p[row * n + j]);
        if (!std::isfinite(actual) || std::abs(actual - want) > want * 0.004)
          throw std::runtime_error("Qwen vision softmax differs from the FP64 host reference");
      }
    }
  }
}
}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 2;
  return dgpp::test::run_all();
}
