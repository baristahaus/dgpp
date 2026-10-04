#pragma once
// Full GLM-5.3 weight loader using ResidentLayerStream (2026-09-12,
// docs/glm53_plan.md G3). The family supplies tensor bindings, TP geometry
// and the packed-int builders, including the load-time requantization of
// the BF16 draft experts (plan D5) and the bf16 bridge of kv_b (plan D6).
// The shared stream owns allocations, staging, resident images, byte
// accounting and digests.
//
// Placement (every slice a formula in the world size W, plan §2):
//   attention: q_a and kv_a replicated as ONE fused [q_lora + kv_lora +
//              rope, hidden] projection (rows are independent in every
//              format, so the two checkpoint matrices stack); q_b rows and
//              kv_b rows per head block (64/W heads); o_proj packed
//              columns; the latent norms replicated. Packed on the packed
//              layers (int8 g64), BF16 on the dense layers and the draft;
//              kv_b BF16 everywhere (dequantized at load where packed).
//   indexer:   replicated (wq_b, wk, weights_proj, k_norm) on the layers
//              that own one.
//   dense MLP: BF16 gate/up rows and down columns at 12288/W.
//   MoE:       router and its bias replicated; every routed expert and the
//              shared expert sliced on the intermediate dim at 2048/W
//              (gate/up rows, down columns on a 64-group boundary): int4
//              and int8 triples on the packed layers, requantized from the
//              draft's BF16 (routed int4, shared int8).
//   draft:     as a main layer; enorm/hnorm/eh_proj/shared_head.norm
//              replicated.
//   globals:   embed replicated (a row gather), the final norm, lm_head
//              vocab-sharded under VocabSharded.
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
#include "models/glm_dsa/binding.hpp"
#include "models/glm_dsa/config.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

using GlmDsaResidency = LoaderResidency;
using GlmDsaHeadSharding = LoaderHeadSharding;
using GlmDsaReplicatedDigest = ReplicatedDigest;

struct GlmDsaAttnResident {
  // The fused [q_a | kv_a] projection [q_lora + kv_lora + rope, hidden]:
  // BF16 (dense layers, the draft) or packed (the packed layers).
  const uint16_t* qkv_a = nullptr;
  GlmPackedMatrix qkv_a_packed;
  const uint16_t* q_aln = nullptr;   // BF16 [q_lora]
  const uint16_t* kv_aln = nullptr;  // BF16 [kv_lora]
  const uint16_t* q_b = nullptr;     // BF16 [lh * (nope + rope), q_lora]
  GlmPackedMatrix q_b_packed;
  const uint16_t* kv_b = nullptr;    // BF16 [lh * (nope + v), kv_lora] (the bridge where packed)
  const uint16_t* o_proj = nullptr;  // BF16 [hidden, lh * v] (packed columns)
  GlmPackedMatrix o_proj_packed;
  // The indexer (null on shared layers).
  const uint16_t* wq_b = nullptr;      // BF16 [index_heads * 128, q_lora]
  const uint16_t* wk = nullptr;        // BF16 [128, hidden]
  const uint16_t* wp = nullptr;        // BF16 [index_heads, hidden]
  const uint16_t* k_norm_w = nullptr;  // BF16 [128]
  const uint16_t* k_norm_b = nullptr;  // BF16 [128]
  int local_heads = 0;
  int head_begin = 0;
  bool packed() const { return qkv_a_packed.packed != nullptr; }
  bool owns_indexer() const { return wq_b != nullptr; }
};

struct GlmDsaDenseMlpResident {
  const uint16_t* gate = nullptr;  // BF16 [I/W, hidden]
  const uint16_t* up = nullptr;    // BF16 [I/W, hidden]
  const uint16_t* down = nullptr;  // BF16 [hidden, I/W] (packed columns)
  int64_t local_inter = 0;
};

struct GlmDsaMoeResident {
  const uint16_t* router = nullptr;      // BF16 [E, hidden]
  const float* router_bias = nullptr;    // F32 [E]
  GlmPackedMatrix shared[3];             // gate, up [S/W, hidden]; down [hidden, S/W] (int8)
  std::vector<GlmPackedMatrix> experts;  // [E * 3]: gate, up, down per expert (inter-sliced)
  int64_t local_inter = 0;               // I/W
  int64_t local_shared_inter = 0;        // S/W
  const GlmPackedMatrix& expert(int e, int i) const {
    return experts[static_cast<size_t>(e) * 3 + i];
  }
};

struct GlmDsaLayerResident {
  int layer = -1;
  bool moe = false;
  const uint16_t* input_norm = nullptr;  // BF16 [hidden]
  const uint16_t* post_norm = nullptr;   // BF16 [hidden]
  GlmDsaAttnResident attn;
  GlmDsaDenseMlpResident dense;  // dense layers
  GlmDsaMoeResident moe_w;       // MoE layers (the draft too)
  // The draft layer's head tensors (null on main layers).
  const uint16_t* enorm = nullptr;             // BF16 [hidden]
  const uint16_t* hnorm = nullptr;             // BF16 [hidden]
  const uint16_t* eh_proj = nullptr;           // BF16 [hidden, 2 * hidden]
  const uint16_t* shared_head_norm = nullptr;  // BF16 [hidden]
  size_t bytes = 0;
};

struct GlmDsaGlobalsResident {
  const uint16_t* embed = nullptr;       // BF16 [embed_vocab_count, hidden]: the whole table, or this rank's rows
  int embed_vocab_begin = 0;             // the first row held (0 unless vocab-sharded)
  int embed_vocab_count = 0;             // rows held (the vocabulary unless vocab-sharded)
  const uint16_t* final_norm = nullptr;  // BF16 [hidden]
  const uint16_t* lm_head = nullptr;     // BF16 [lm_vocab_count, hidden]
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  size_t bytes = 0;
};

// The local TP geometry at (rank, world): every slice bound the builders
// and the views use, in one place.
struct GlmDsaLocalGeometry {
  int world = 1, rank = 0;
  int local_heads = 0, head_begin = 0;
  int64_t local_inter = 0;         // moe_intermediate_size / W
  int64_t local_shared_inter = 0;  // shared_expert_inter() / W
  int64_t local_dense_inter = 0;   // intermediate_size / W
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  // The embedding rows this rank holds: the whole table, or — under
  // GlmDsaLayerStream::embed_vocab_sharded() at world > 1 — the lm head's
  // vocabulary slice (the rows are gathered per token and summed by one
  // fold, so the ranks see the full rows bitwise).
  int embed_vocab_begin = 0, embed_vocab_count = 0;
  static GlmDsaLocalGeometry from_config(const GlmDsaTextConfig& cfg, int rank, int world,
                                        GlmDsaHeadSharding head);
};

// The family behind the shared stream (loaders/resident_stream.hpp).
struct GlmDsaLoaderFamily {
  using Config = GlmDsaTextConfig;
  using Expected = GlmDsaExpectedTensor;
  using LayerResident = GlmDsaLayerResident;
  using GlobalsResident = GlmDsaGlobalsResident;
  using Geometry = GlmDsaLocalGeometry;
  using PresentMap = std::unordered_map<std::string, GlmDsaTensorDesc>;
  struct Builder;  // models/glm_dsa/loader.cpp
  static const char* who() { return "glm_dsa loader"; }
  static uint64_t loader_format(const Config&) { return 1; }
  static int max_layer(const Config& c) { return c.num_hidden_layers + (c.mtp_layer() >= 0 ? 1 : 0); }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) {
    return glm_dsa_expected_layer_tensors(c, layer);
  }
  static std::vector<Expected> global_table(const Config& c) { return glm_dsa_expected_global_tensors(c); }
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

extern template class ResidentLayerStream<GlmDsaLoaderFamily>;

class GlmDsaLayerStream : public ResidentLayerStream<GlmDsaLoaderFamily> {
 public:
  GlmDsaLayerStream(const GlmDsaTextConfig& cfg, const std::string& checkpoint_dir, int rank = 0,
                    int world = 1, GlmDsaResidency residency = GlmDsaResidency::Streaming,
                    GlmDsaHeadSharding head = GlmDsaHeadSharding::Full, bool resident_mtp = false);
  ~GlmDsaLayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();
  // The deployment's `engine.embed_sharding` ("vocab": each rank holds
  // its lm-head slice of the embedding, −1.33 GiB per rank at world 4,
  // one fold per lookup; "replicated": the whole table on every rank).
  // Process-wide like the Qwen knobs; set before the plan and the load.
  static void set_embed_vocab_sharded(bool on);
  static bool embed_vocab_sharded();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }
};

}  // namespace dgpp
