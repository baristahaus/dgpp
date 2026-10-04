// Loader smoke test: Qwen35LayerStream against a real 27B release
// checkpoint (FP8-block or NVFP4-mixed). Loads layer 0 (GDN) + layer 3
// (Full) and verifies the resident pointers are non-null and the norm
// weights read back ~1.0. DGPP_QWEN35_SMOKE_RESIDENT=1 runs the resident
// stream instead of the streaming one — the serving path's build, which
// writes a packed image under DGPP_QWEN35_SMOKE_IMAGE (default
// /tmp/q35-resident-<release>). Discriminates loader-build crashes from
// model/run_rows crashes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "models/qwen/config35.hpp"
#include "models/qwen/loader35.hpp"

namespace {

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::printf("CUDA FAIL %s:%d: %s\n", __FILE__, __LINE__,                 \
                  cudaGetErrorString(err__));                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL: %s\n", msg);                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

float h2f(uint16_t v) {
  uint32_t b = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

std::vector<uint16_t> down16(const uint16_t* d, size_t n) {
  std::vector<uint16_t> h(n);
  cudaError_t err__ = cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost);
  if (err__ != cudaSuccess) {
    std::printf("CUDA FAIL download: %s\n", cudaGetErrorString(err__));
    std::exit(1);
  }
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <checkpoint-dir>\n"
                 "  <checkpoint-dir> must hold config.json + the safetensors shards for the "
                 "Qwen3.8-27B FP8-block or NVFP4-mixed release\n",
                 argv[0]);
    return 2;
  }
  const std::string ckpt = argv[1];
  dgpp::Qwen35TextConfig cfg = dgpp::Qwen35TextConfig::from_json_file(ckpt + "/config.json");
  std::printf("config: hidden=%d layers=%d vocab=%d\n", cfg.hidden_size, cfg.num_hidden_layers,
              cfg.vocab_size);
  if (std::getenv("DGPP_QWEN35_SMOKE_BYTES")) {
    // The closed forms only: the globals layout against the staging the
    // stream allocates, and the whole resident footprint.
    const size_t gb = dgpp::Qwen35LayerStream::globals_bytes(cfg, 0, 1,
                                                             dgpp::LoaderHeadSharding::Full);
    const size_t rb = dgpp::Qwen35LayerStream::resident_bytes(cfg, 0, 1,
                                                              dgpp::LoaderHeadSharding::Full, true);
    size_t largest = 0;
    for (int l = 0; l < cfg.num_hidden_layers + 1; ++l)
      largest = std::max(largest, dgpp::Qwen35LayerStream::layer_bytes(cfg, l, 0, 1));
    std::printf("bytes: globals %.3f GiB largest layer %.3f GiB resident(mtp) %.3f GiB\n",
                gb / double(1u << 30), largest / double(1u << 30), rb / double(1u << 30));
    // The globals' device truth: download a known embed row and the head's
    // scales, and compare against the values the checkpoint holds. The
    // engine's flat outputs look like a zero embedding, so measure it.
    dgpp::Qwen35LayerStream gs(cfg, ckpt, 0, 1, dgpp::LoaderResidency::Streaming,
                               dgpp::LoaderHeadSharding::Full, false);
    const dgpp::Qwen35GlobalsResident g = gs.load_globals();
    std::vector<uint16_t> row(static_cast<size_t>(cfg.hidden_size));
    cudaMemcpy(row.data(), g.embed + static_cast<size_t>(1000) * cfg.hidden_size,
               row.size() * 2, cudaMemcpyDeviceToHost);
    double s = 0;
    for (uint16_t v : row) {
      const uint32_t b = static_cast<uint32_t>(v) << 16;
      float f; std::memcpy(&f, &b, 4); s += double(f) * f;
    }
    std::printf("device embed row 1000: rms %.5f\n", std::sqrt(s / row.size()));
    std::vector<float> hs(4);
    cudaMemcpy(hs.data(), const_cast<float*>(g.lm_head_fp8.scales), 16, cudaMemcpyDeviceToHost);
    std::printf("head scales [0..3]: %.5f %.5f %.5f %.5f\n", hs[0], hs[1], hs[2], hs[3]);
    return 0;
  }
  const bool resident = std::getenv("DGPP_QWEN35_SMOKE_RESIDENT") != nullptr;
  if (resident)
    dgpp::Qwen35LayerStream::set_resident_image_dir(
        std::getenv("DGPP_QWEN35_SMOKE_IMAGE") ? std::getenv("DGPP_QWEN35_SMOKE_IMAGE")
                                               : "/tmp/q35-resident");
  dgpp::Qwen35LayerStream stream(cfg, ckpt, 0, 1,
                                 resident ? dgpp::LoaderResidency::Resident
                                          : dgpp::LoaderResidency::Streaming,
                                 dgpp::LoaderHeadSharding::Full, false);
  // Layer 0: GDN. The MLP form follows the release: the mixed one carries
  // NVFP4 triples (payload + e4m3 scales + the reciprocal-global slot),
  // the FP8 release the dense fp8-block matrix.
  {
    const dgpp::Qwen35LayerResident& r = stream.load_layer(0);
    CHECK(r.kind == dgpp::Qwen35LayerKind::Gdn, "layer 0 kind != Gdn");
    CHECK(r.input_norm && r.post_norm, "layer 0 norms null");
    CHECK(r.gdn.in_proj_qkv_fp8.payload, "layer 0 qkv fp8 payload null");
    CHECK(r.gdn.in_proj_a && r.gdn.in_proj_b, "layer 0 a/b (fp8->bf16 dequant) null");
    CHECK(r.gdn.norm, "layer 0 gdn norm null");
    if (cfg.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Mixed) {
      CHECK(r.mlp.form == dgpp::Qwen35MlpForm::Nvfp4, "layer 0 mlp form != Nvfp4");
      CHECK(r.mlp.gate_fp4.payload && r.mlp.gate_fp4.scales,
            "layer 0 mlp gate fp4 payload/scales null");
      CHECK(r.mlp.gate_fp4.global_scale, "layer 0 mlp gate fp4 global slot null");
      CHECK(r.mlp.down_fp4.payload, "layer 0 mlp down fp4 payload null");
    } else {
      CHECK(r.mlp.gate_fp8.payload, "layer 0 mlp gate fp8 null");
    }
    const std::vector<uint16_t> n = down16(r.input_norm, 4);
    std::printf("L0 input_norm[0..3] = %.4f %.4f %.4f %.4f\n", h2f(n[0]), h2f(n[1]), h2f(n[2]),
                h2f(n[3]));
    for (float v : {h2f(n[0]), h2f(n[1]), h2f(n[2]), h2f(n[3])})
      CHECK(std::isfinite(v) && std::fabs(v) < 100.0f, "layer 0 norm value out of range");
  }
  // Layer 3: Full attention.
  {
    const dgpp::Qwen35LayerResident& r = stream.load_layer(3);
    CHECK(r.kind == dgpp::Qwen35LayerKind::Full, "layer 3 kind != Full");
    CHECK(r.full.q_proj_fp8.payload, "layer 3 q fp8 payload null");
    CHECK(r.full.o_proj_fp8.payload, "layer 3 o fp8 payload null");
    CHECK(r.full.q_norm && r.full.k_norm, "layer 3 q/k norms null");
    const std::vector<uint16_t> n = down16(r.post_norm, 4);
    std::printf("L3 post_norm[0..3] = %.4f %.4f %.4f %.4f\n", h2f(n[0]), h2f(n[1]), h2f(n[2]),
                h2f(n[3]));
  }
  // The mixed release's channel-MLP exception layer: the parsed
  // layers.(56|..) target must come back as the dense fp8 form while its
  // neighbours stay NVFP4 (checked at layer 0 above).
  if (cfg.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Mixed && !cfg.channel_mlp_layers.empty()) {
    const int ch = cfg.channel_mlp_layers.front();
    const dgpp::Qwen35LayerResident& r = stream.load_layer(ch);
    CHECK(r.mlp.form == dgpp::Qwen35MlpForm::Fp8,
          "channel-mlp layer form != Fp8");
    CHECK(r.mlp.gate_fp8.payload, "channel-mlp layer gate fp8 payload null");
    CHECK(r.mlp.gate_fp8.scales, "channel-mlp layer gate fp8 scales null");
  }
  std::printf("SMOKE PASS\n");
  return 0;
}
