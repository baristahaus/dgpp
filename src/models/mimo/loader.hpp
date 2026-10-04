#pragma once
// MiMo-V2.6-Flash weight loader using ResidentLayerStream. The family
// supplies tensor bindings, TP geometry and the release's own formats —
// fp8 block-128 dense matrices, MXFP4 experts, BF16 o_proj / router /
// head — untouched. The shared stream owns allocations, staging, resident
// images, byte accounting and digests.
//
// Placement (every slice a formula in the world size W; W divides the
// qkv_proj chunk count, num_key_value_heads = 4: worlds 1, 2, 4):
//   attention: the fused qkv_proj is pre-sharded in 4 chunks, chunk c =
//              [Q heads 16c..16c+15 | K kv heads of chunk c | V ...] with
//              its fp8 scale grid tiled per chunk; rank r holds chunks
//              [4r/W, 4(r+1)/W) — its 64/W query heads, 4/W global or 8/W
//              sliding-window kv heads — stacked as ONE fp8 matrix, each
//              chunk but the last padded to a 128-row multiple so every
//              chunk's scale rows re-anchor exactly (zero payload rows,
//              never read); o_proj packed columns (64/W x 128), packable
//              (bf12); the sink bias's 64/W heads widened to fp32.
//   dense MLP: gate/up rows and down columns at 16384/W (fp8, the scale
//              grid re-anchored at the 128-aligned slice start).
//   MoE:       router and its bias replicated; every routed expert sliced
//              on the intermediate dim at 2048/W (MXFP4 gate/up rows, down
//              columns on 32-block boundaries).
//   draft:     as a main sliding-window layer with the dense MLP;
//              enorm/hnorm/eh_proj/final_layernorm replicated (eh_proj
//              packable).
//   globals:   embed replicated (a row gather), the final norm, lm_head
//              vocab-sharded under VocabSharded (packable).
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
#include "models/mimo/binding.hpp"
#include "models/mimo/config.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

using MimoResidency = LoaderResidency;
using MimoHeadSharding = LoaderHeadSharding;
using MimoReplicatedDigest = ReplicatedDigest;

struct MimoAttnResident {
  // fp8 [(chunks - 1) * chunk_stride + chunk_rows, hidden], the rank's
  // chunks of the fused projection (scale grid [chunks * scale rows per
  // chunk, hidden / 128]).
  GlmQuantMatrix qkv;
  int chunks = 0;
  int64_t chunk_rows = 0;    // real rows per chunk
  int64_t chunk_stride = 0;  // padded rows per chunk (ceil(chunk_rows / 128) * 128)
  int q_per_chunk = 0;
  int kv_per_chunk = 0;
  const uint16_t* o_proj = nullptr;  // BF16 [hidden, lh * 128] (packed columns)
  const float* sink = nullptr;       // F32 [lh] (null without a sink)
  int local_heads = 0;
  int head_begin = 0;
  int local_kv_heads = 0;
  int kv_head_begin = 0;
  bool swa = false;
};

struct MimoDenseMlpResident {
  GlmQuantMatrix gate, up, down;  // gate/up [I/W, hidden], down [hidden, I/W]
  int64_t local_inter = 0;
};

struct MimoMoeResident {
  const uint16_t* router = nullptr;     // BF16 [E, hidden]
  const float* router_bias = nullptr;   // F32 [E]
  std::vector<GlmFp4Matrix> experts;    // [E * 3]: gate, up, down per expert (MXFP4, inter-sliced)
  int64_t local_inter = 0;              // I/W
  const GlmFp4Matrix& expert(int e, int i) const {
    return experts[static_cast<size_t>(e) * 3 + i];
  }
};

struct MimoLayerResident {
  int layer = -1;
  bool moe = false;
  bool swa = false;
  const uint16_t* input_norm = nullptr;   // BF16 [hidden]
  const uint16_t* post_norm = nullptr;    // BF16 [hidden] (the draft's pre_mlp_layernorm)
  MimoAttnResident attn;
  MimoDenseMlpResident dense;  // dense layers (the draft too)
  MimoMoeResident moe_w;       // MoE layers
  // The draft layer's head tensors (null on main layers).
  const uint16_t* enorm = nullptr;       // BF16 [hidden]
  const uint16_t* hnorm = nullptr;       // BF16 [hidden]
  const uint16_t* eh_proj = nullptr;     // BF16 [hidden, 2 * hidden]
  const uint16_t* final_norm = nullptr;  // BF16 [hidden] (final_layernorm: before the shared head)
  size_t bytes = 0;
};

struct MimoGlobalsResident {
  const uint16_t* embed = nullptr;       // BF16 [vocab, hidden]
  const uint16_t* final_norm = nullptr;  // BF16 [hidden]
  const uint16_t* lm_head = nullptr;     // BF16 [lm_vocab_count, hidden]
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  size_t bytes = 0;
};

// The local TP geometry at (rank, world): every slice bound the builders
// and the views use, in one place.
struct MimoLocalGeometry {
  int world = 1, rank = 0;
  int chunks = 0, chunk_begin = 0;       // the rank's qkv_proj chunks
  int local_heads = 0, head_begin = 0;   // query heads
  int local_kv_heads = 0, kv_head_begin = 0;          // the global layers'
  int local_swa_kv_heads = 0, swa_kv_head_begin = 0;  // the sliding-window layers'
  int64_t local_inter = 0;         // moe_intermediate_size / W
  int64_t local_dense_inter = 0;   // intermediate_size / W
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  int kv_heads_of(const MimoTextConfig& cfg, int layer) const {
    return cfg.is_swa_layer(layer) ? local_swa_kv_heads : local_kv_heads;
  }
  static MimoLocalGeometry from_config(const MimoTextConfig& cfg, int rank, int world,
                                       MimoHeadSharding head);
};

// The family behind the shared stream (loaders/resident_stream.hpp).
struct MimoLoaderFamily {
  using Config = MimoTextConfig;
  using Expected = MimoExpectedTensor;
  using LayerResident = MimoLayerResident;
  using GlobalsResident = MimoGlobalsResident;
  using Geometry = MimoLocalGeometry;
  using PresentMap = std::unordered_map<std::string, MimoTensorDesc>;
  struct Builder;  // models/mimo/loader.cpp
  static const char* who() { return "mimo loader"; }
  static uint64_t loader_format(const Config&) { return 1; }
  static int max_layer(const Config& c) { return c.num_hidden_layers + c.mtp_layers_loaded; }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) {
    return mimo_expected_layer_tensors(c, layer);
  }
  static std::vector<Expected> global_table(const Config& c) { return mimo_expected_global_tensors(c); }
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config&, const LoaderTensorMap&) {}
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

extern template class ResidentLayerStream<MimoLoaderFamily>;

class MimoLayerStream : public ResidentLayerStream<MimoLoaderFamily> {
 public:
  MimoLayerStream(const MimoTextConfig& cfg, const std::string& checkpoint_dir, int rank = 0,
                  int world = 1, MimoResidency residency = MimoResidency::Streaming,
                  MimoHeadSharding head = MimoHeadSharding::Full, bool resident_mtp = false);
  ~MimoLayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }
};

}  // namespace dgpp
