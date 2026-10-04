#pragma once
// GLM-4.7 weight loader using ResidentLayerStream. The family supplies
// tensor bindings, TP geometry and modelopt NVFP4 builders, including
// load-time requantization of the BF16 draft experts. The shared stream
// owns allocations, staging, resident images, byte accounting and digests.
//
// Placement (every slice a formula in the world size W):
//   attention: 96/W query heads (q_proj rows and bias), 8/W kv heads
//              (k/v rows and biases; a kv head shared by W/8 ranks at
//              W > 8), o_proj packed columns; q/k norms replicated.
//   dense MLP: gate/up rows and down columns at 12288/W (NVFP4 triples,
//              the modelopt scale convention: global = 1 / weight_scale_2).
//   MoE:       router and its bias replicated; every routed expert and the
//              shared expert sliced on the intermediate dim at 1536/W
//              (gate/up rows, down columns, 16-block aligned).
//   draft:     as a main layer, its BF16 experts and shared expert
//              requantized to NVFP4 at load (loaders/nvfp4_quant.hpp);
//              enorm/hnorm/eh_proj/shared_head.norm replicated; the
//              embedding / head copies verified against the globals and
//              never loaded.
//   globals:   embed replicated (a row gather), the final norm, lm_head
//              vocab-sharded under VocabSharded.
//
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/resident_stream.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"
#include "models/glm4/binding.hpp"
#include "models/glm4/config.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

using Glm4Residency = LoaderResidency;
using Glm4HeadSharding = LoaderHeadSharding;
using Glm4ReplicatedDigest = ReplicatedDigest;

struct Glm4AttnResident {
  const uint16_t* q_proj = nullptr;  // BF16 [lh * d, hidden]
  const uint16_t* k_proj = nullptr;  // BF16 [lkv * d, hidden]
  const uint16_t* v_proj = nullptr;  // BF16 [lkv * d, hidden]
  const uint16_t* o_proj = nullptr;  // BF16 [hidden, lh * d] (packed columns)
  const uint16_t* q_bias = nullptr;  // BF16 [lh * d] (null without attention_bias)
  const uint16_t* k_bias = nullptr;  // BF16 [lkv * d]
  const uint16_t* v_bias = nullptr;  // BF16 [lkv * d]
  const uint16_t* q_norm = nullptr;  // BF16 [d] (null without use_qk_norm)
  const uint16_t* k_norm = nullptr;  // BF16 [d]
  int local_heads = 0;
  int head_begin = 0;
  int local_kv_heads = 0;
  int kv_head_begin = 0;
};

struct Glm4DenseMlpResident {
  GlmFp4Matrix gate, up, down;  // gate/up [I/W, hidden], down [hidden, I/W]
  int64_t local_inter = 0;
};

struct Glm4MoeResident {
  const uint16_t* router = nullptr;     // BF16 [E, hidden]
  const float* router_bias = nullptr;   // F32 [E]
  GlmFp4Matrix shared[3];               // gate, up [S/W, hidden]; down [hidden, S/W]
  std::vector<GlmFp4Matrix> experts;    // [E * 3]: gate, up, down per expert (inter-sliced)
  int64_t local_inter = 0;              // I/W
  int64_t local_shared_inter = 0;       // S/W
  const GlmFp4Matrix& expert(int e, int i) const {
    return experts[static_cast<size_t>(e) * 3 + i];
  }
};

struct Glm4LayerResident {
  int layer = -1;
  bool moe = false;
  const uint16_t* input_norm = nullptr;   // BF16 [hidden]
  const uint16_t* post_norm = nullptr;    // BF16 [hidden]
  Glm4AttnResident attn;
  Glm4DenseMlpResident dense;  // dense layers
  Glm4MoeResident moe_w;       // MoE layers (the draft too)
  // The draft layer's head tensors (null on main layers).
  const uint16_t* enorm = nullptr;             // BF16 [hidden]
  const uint16_t* hnorm = nullptr;             // BF16 [hidden]
  const uint16_t* eh_proj = nullptr;           // BF16 [hidden, 2 * hidden]
  const uint16_t* shared_head_norm = nullptr;  // BF16 [hidden]
  size_t bytes = 0;
};

struct Glm4GlobalsResident {
  const uint16_t* embed = nullptr;       // BF16 [vocab, hidden]
  const uint16_t* final_norm = nullptr;  // BF16 [hidden]
  const uint16_t* lm_head = nullptr;     // BF16 [lm_vocab_count, hidden]
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  size_t bytes = 0;
};

// The local TP geometry at (rank, world): every slice bound the builders
// and the views use, in one place.
struct Glm4LocalGeometry {
  int world = 1, rank = 0;
  int local_heads = 0, head_begin = 0;
  int local_kv_heads = 0, kv_head_begin = 0;
  int64_t local_inter = 0;         // moe_intermediate_size / W
  int64_t local_shared_inter = 0;  // shared_expert_inter() / W
  int64_t local_dense_inter = 0;   // intermediate_size / W
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  static Glm4LocalGeometry from_config(const Glm4TextConfig& cfg, int rank, int world,
                                       Glm4HeadSharding head);
};

// The family behind the shared stream (loaders/resident_stream.hpp).
struct Glm4LoaderFamily {
  using Config = Glm4TextConfig;
  using Expected = Glm4ExpectedTensor;
  using LayerResident = Glm4LayerResident;
  using GlobalsResident = Glm4GlobalsResident;
  using Geometry = Glm4LocalGeometry;
  using PresentMap = std::unordered_map<std::string, Glm4TensorDesc>;
  struct Builder;  // models/glm4/loader.cpp
  static const char* who() { return "glm4 loader"; }
  static uint64_t loader_format(const Config&) { return 1; }
  static int max_layer(const Config& c) { return c.num_hidden_layers + (c.mtp_layer() >= 0 ? 1 : 0); }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) {
    return glm4_expected_layer_tensors(c, layer);
  }
  static std::vector<Expected> global_table(const Config& c) { return glm4_expected_global_tensors(c); }
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config& c, const LoaderTensorMap& tensors);
  static bool digest_included(const Expected& e);
  static bool discard_after_pack(const Expected& e);
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors,
                            LayerBump& bump, GlobalsResident& out, uint64_t& source_bytes,
                            uint64_t& verbatim_bytes, LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t globals_side_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t extra_resident_bytes(const Config&, int, int) { return 0; }
  static size_t min_staging_bytes() { return 0; }
  static void after_restore(const Config&, int, const LoaderTensorMap&, LayerResident&) {}
};

extern template class ResidentLayerStream<Glm4LoaderFamily>;

class Glm4LayerStream : public ResidentLayerStream<Glm4LoaderFamily> {
 public:
  Glm4LayerStream(const Glm4TextConfig& cfg, const std::string& checkpoint_dir, int rank = 0,
                  int world = 1, Glm4Residency residency = Glm4Residency::Streaming,
                  Glm4HeadSharding head = Glm4HeadSharding::Full, bool resident_mtp = false);
  ~Glm4LayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }
};

}  // namespace dgpp
