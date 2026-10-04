// Full-model forward parity (M4 chunk 6, DESIGN §7.5).
//
// Modes:
//   (none)                      standalone kernel checks (norm choreography,
//                               stream init, topk tie-break) — always safe
//   --write-fixture DIR         write a synthetic mini-checkpoint whose
//                               weight MAGNITUDES suit a forward pass (the
//                               loader test's fixture targets byte-exactness
//                               with full-range values; saturating every
//                               sigmoid would test nothing)
//   --checkpoint-dir DIR
//   --dump-file FILE            run GlmDiagnosticModel over the fixture and
//                               compare against the pure-python full-stack
//                               reference dump: final hidden state (ulp
//                               budgets with cancellation floor), per-token
//                               top-k logits, and per-layer routing decisions.
//
// The ctest chain is: glm_forward_fixture (this binary, --write-fixture)
// -> glm_forward_generate (tools/glm_reference_dump.py gen-pure) ->
// glm_forward_test (this binary, --checkpoint-dir/--dump-file). The python
// side reads the same checkpoint this test writes, so the tensor table has
// exactly one source (the C++ binding table).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/glm_norm.hpp"
#include "loaders/minijson.hpp"
#include "models/glm/binding.hpp"
#include "models/glm/dump.hpp"
#include "models/glm/forward.hpp"

#include "glm_fixture.hpp"

namespace fs = std::filesystem;

namespace {

// The fixture machinery moved to glm_fixture.hpp (shared with the M5 TP
// parity test, which generates its own geometry with identical values).
using glmfx::Rng;
using glmfx::seed_for;

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::GlmExpectedTensor;
using dgpp::GlmWeightClass;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// ---- mini config (forward-suitable geometry; see file header) ----------

const char* kMiniJson = R"json({
  "hidden_size": 128, "vocab_size": 96, "num_hidden_layers": 6,
  "rms_norm_eps": 1e-5, "tie_word_embeddings": false,
  "hidden_act": "silu", "swiglu_limit": 7.5,
  "layer_types": ["linear_attention", "linear_attention",
                  "deepseek_sparse_attention", "linear_attention",
                  "deepseek_sparse_attention", "linear_attention"],
  "mlp_layer_types": ["dense", "dense", "sparse", "sparse", "sparse", "sparse"],
  "indexer_types": ["full", "full", "full", "full", "full", "full"],
  "first_k_dense_replace": 2,
  "linear_attn_config": {
    "num_heads": 2, "head_dim": 32, "short_conv_kernel_size": 4,
    "gate_lower_bound": -3.5,
    "kda_layers": [0, 1, 3, 5], "full_attn_layers": [2, 4]
  },
  "num_attention_heads": 4, "q_lora_rank": 32, "kv_lora_rank": 64,
  "qk_nope_head_dim": 32, "qk_rope_head_dim": 0, "v_head_dim": 32,
  "mla_use_nope": true,
  "index_n_heads": 32, "index_head_dim": 128, "index_kpool": 4,
  "index_topk": 32, "index_kpool_compress": true,
  "index_kpool_always_select_tail": true, "indexer_rope_interleave": true,
  "intermediate_size": 200, "moe_intermediate_size": 64,
  "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
  "scoring_func": "sigmoid", "topk_method": "noaux_tc",
  "norm_topk_prob": true, "routed_scaling_factor": 1.5,
  "n_group": 1, "topk_group": 1, "moe_router_dtype": "float32",
  "mhc": true, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
  "num_nextn_predict_layers": 1
})json";

dgpp::GlmTextConfig mini_config() {
  auto parsed = dgpp::minijson::parse(kMiniJson);
  return dgpp::GlmTextConfig::parse(parsed.root);
}

void write_fixture(const std::string& dir) {
  glmfx::write_fixture(mini_config(), kMiniJson, dir);
}

// ---- standalone kernel checks -------------------------------------------

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu)
                         : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

DGPP_TEST(glm_rmsnorm_two_rounding_matches_reference_choreography) {
  // The two-rounding choreography (u = bf16(x*rstd); y = bf16(w*u)) differs
  // from single-round bf16(w*x*rstd) on a quarter-ish of random elements —
  // the exact regression this test pins (someone swapping in the generic
  // rmsnorm_bf16). fp32-vs-double rstd flips at most a couple of elements.
  const int rows = 8, dim = 256;
  Rng rng(12345);
  std::vector<uint16_t> x(static_cast<size_t>(rows) * dim),
      w(dim), got(static_cast<size_t>(rows) * dim);
  for (auto& v : x) v = float_to_bf16_bits(1.0f * rng.normal3());
  for (auto& v : w) v = float_to_bf16_bits(0.9f + 0.2f * rng.unit());

  uint16_t *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_x, x.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_w, w.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_y, got.size() * 2));
  std::memcpy(d_x, x.data(), x.size() * 2);
  std::memcpy(d_w, w.data(), w.size() * 2);
  cudaStream_t s;
  DGPP_CUDA_OK(cudaStreamCreate(&s));
  dgpp::glm_rmsnorm_bf16(d_x, d_w, d_y, rows, dim, 1e-6f, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  DGPP_CUDA_OK(cudaMemcpy(got.data(), d_y, got.size() * 2,
                          cudaMemcpyDeviceToHost));

  size_t mismatch_two_round = 0, mismatch_one_round = 0, differs_one_round = 0;
  for (int r = 0; r < rows; ++r) {
    double ssq = 0;
    for (int i = 0; i < dim; ++i) {
      const double v = bf16_bits_to_float(x[static_cast<size_t>(r) * dim + i]);
      ssq += v * v;
    }
    const double rstd = 1.0 / std::sqrt(ssq / dim + 1e-6);
    for (int i = 0; i < dim; ++i) {
      const double xv = bf16_bits_to_float(x[static_cast<size_t>(r) * dim + i]);
      const double wv = bf16_bits_to_float(w[i]);
      const uint16_t u_bits =
          float_to_bf16_bits(static_cast<float>(xv * rstd));
      const uint16_t two_round = float_to_bf16_bits(
          wv * static_cast<double>(bf16_bits_to_float(u_bits)));
      const uint16_t one_round =
          float_to_bf16_bits(wv * xv * rstd);
      const uint16_t kernel_out = got[static_cast<size_t>(r) * dim + i];
      if (kernel_out != two_round) ++mismatch_two_round;
      if (kernel_out != one_round) ++mismatch_one_round;
      if (two_round != one_round) ++differs_one_round;
    }
  }
  require(mismatch_two_round <= 4,
          "two-round mismatches: " + std::to_string(mismatch_two_round));
  require(differs_one_round >= rows * dim / 8,
          "fixture failed to separate the choreographies");
  require(mismatch_one_round >= rows * dim / 8,
          "kernel output matches the SINGLE-round choreography");

  cudaFree(d_x);
  cudaFree(d_w);
  cudaFree(d_y);
  cudaStreamDestroy(s);
  std::printf(
      "[ OK ] rmsnorm two-rounding: %zu/%zu differ from single-round, "
      "%zu exceed 0 ulp vs double host\n",
      differs_one_round, size_t(rows) * dim, mismatch_two_round);
}

DGPP_TEST(glm_embed_bcast_streams_copies_all_four) {
  const int vocab = 8, hidden = 16, tokens = 5;
  Rng rng(777);
  std::vector<uint16_t> table(static_cast<size_t>(vocab) * hidden);
  for (auto& v : table) v = float_to_bf16_bits(0.5f * rng.normal3());
  std::vector<int64_t> ids = {3, 0, 7, 7, 1};

  uint16_t* d_table = nullptr;
  int64_t* d_ids = nullptr;
  uint16_t* d_streams = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_table, table.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_ids, ids.size() * 8));
  DGPP_CUDA_OK(cudaMallocManaged(
      &d_streams, static_cast<size_t>(tokens) * 4 * hidden * 2));
  std::memcpy(d_table, table.data(), table.size() * 2);
  std::memcpy(d_ids, ids.data(), ids.size() * 8);
  cudaStream_t s;
  DGPP_CUDA_OK(cudaStreamCreate(&s));
  dgpp::glm_embed_bcast_streams(d_table, d_ids, d_streams, tokens, hidden, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  for (int t = 0; t < tokens; ++t)
    for (int stream = 0; stream < 4; ++stream)
      for (int h = 0; h < hidden; ++h)
        require(d_streams[(static_cast<size_t>(t) * 4 + stream) * hidden + h] ==
                    table[static_cast<size_t>(ids[t]) * hidden + h],
                "stream copy mismatch");
  cudaFree(d_table);
  cudaFree(d_ids);
  cudaFree(d_streams);
  cudaStreamDestroy(s);
}

DGPP_TEST(topk_breaks_ties_by_lowest_id) {
  // values: idx 0..4 = [5, 5, 3, 5, 1]; k=2 -> ids [0, 1].
  const std::vector<float> logits = {5.0f, 5.0f, 3.0f, 5.0f, 1.0f};
  const auto got = dgpp::GlmDiagnosticModel::topk(logits, 1, 5, 2);
  require(got.size() == 1 && got[0].size() == 2, "topk shape");
  require(got[0][0].first == 0 && got[0][0].second == 5.0f, "topk[0]");
  require(got[0][1].first == 1 && got[0][1].second == 5.0f, "topk[1]");
}

// ---- dump-parity runner --------------------------------------------------

struct UlpStats {
  long over_soft = 0, hard = 0, total = 0;
  double max_ulps = 0, l2 = 0;
};

UlpStats compare_hidden(const std::vector<uint16_t>& got,
                        const std::vector<uint16_t>& want, double max_abs,
                        int soft, int hard) {
  UlpStats st;
  st.total = static_cast<long>(got.size());
  double sum_d2 = 0, sum_o2 = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const int u = bf16_ulps(got[i], want[i]);
    const double d = bf16_bits_to_float(got[i]) - bf16_bits_to_float(want[i]);
    sum_d2 += d * d;
    sum_o2 += std::pow(bf16_bits_to_float(want[i]), 2);
    if (u > soft) ++st.over_soft;
    if (u > hard) ++st.hard;
    if (u > st.max_ulps && std::abs(bf16_bits_to_float(want[i])) >
                               1e-3 * max_abs)
      st.max_ulps = u;  // cancellation floor: tiny outputs measure fp32
                        // noise in ulps of a magnitude nothing uses
  }
  st.l2 = std::sqrt(sum_d2) / std::sqrt(sum_o2 + 1e-30);
  return st;
}

int run_dump_parity(const std::string& checkpoint_dir,
                    const std::string& dump_path) {
  const dgpp::GlmDumpFile dump = dgpp::GlmDumpFile::load(dump_path);
  const dgpp::GlmTextConfig cfg = dgpp::GlmTextConfig::from_json_file(
      (fs::path(checkpoint_dir) / "config.json").string());
  require(cfg.hidden_size == dump.hidden() &&
              cfg.vocab_size == dump.vocab() &&
              cfg.num_hidden_layers == dump.num_layers(),
          "dump config disagrees with checkpoint config");
  require(dump.token_count() <= 4096, "fixture token count sanity");

  const std::vector<int64_t> tokens(dump.tokens(),
                                    dump.tokens() + dump.token_count());
  const int T = static_cast<int>(tokens.size());

  dgpp::GlmDiagnosticModel model(cfg, checkpoint_dir, T,
                                 128 /*dsa slots: one block*/);
  const dgpp::GlmDiagnosticModel::Outputs out = model.forward(tokens);

  // Determinism: a second run replays bitwise (no cross-call state).
  const dgpp::GlmDiagnosticModel::Outputs out2 = model.forward(tokens);
  require(out.final_hidden_bits == out2.final_hidden_bits &&
              out.logits == out2.logits,
          "forward is not deterministic across calls");

  // ---- final hidden -------------------------------------------------
  double max_abs = 0;
  {
    const uint16_t* fh = dump.final_hidden();
    const size_t n = static_cast<size_t>(T) * dump.hidden();
    for (size_t i = 0; i < n; ++i)
      max_abs = std::max(max_abs,
                         static_cast<double>(std::abs(bf16_bits_to_float(fh[i]))));
  }
  const UlpStats hidden = compare_hidden(
      out.final_hidden_bits,
      std::vector<uint16_t>(
          dump.final_hidden(),
          dump.final_hidden() + static_cast<size_t>(T) * dump.hidden()),
      max_abs, /*soft=*/16, /*hard=*/128);
  // Budgets anchored to the DSA attention's module-level tolerance (M3
  // kept-row drift ~3.5e-3) propagating through six layers: soft outliers
  // are the tail of that noise; hard outliers are chained bf16 boundary
  // flips (the mHC update chain's documented 1e-6-rate events).
  require(static_cast<double>(hidden.over_soft) / hidden.total < 0.02,
          "final hidden soft ulp budget");
  require(static_cast<double>(hidden.hard) / hidden.total < 0.005,
          "final hidden hard ulp budget");
  require(hidden.l2 < 0.01, "final hidden l2 budget");

  // ---- logits: top-1 exact, top-8 set overlap ------------------------
  const auto got_top = dgpp::GlmDiagnosticModel::topk(
      out.logits, T, cfg.vocab_size, dump.top_k());
  const int32_t* ref_ids = dump.topk_ids();
  const float* ref_vals = dump.topk_logits();
  int top1_mismatch = 0, set_miss = 0;
  double min_margin = 1e9;
  for (int t = 0; t < T; ++t) {
    const int32_t ref_top1 = ref_ids[static_cast<size_t>(t) * dump.top_k()];
    const float ref_v1 = ref_vals[static_cast<size_t>(t) * dump.top_k()];
    const float ref_v2 =
        ref_vals[static_cast<size_t>(t) * dump.top_k() + 1];
    min_margin = std::min(min_margin,
                          std::abs(ref_v1 - ref_v2) /
                              (std::abs(ref_v1) + 1e-30));
    if (got_top[t][0].first != ref_top1) ++top1_mismatch;
    std::vector<int32_t> got_ids, ref_set;
    for (int k = 0; k < dump.top_k(); ++k) {
      got_ids.push_back(got_top[t][k].first);
      ref_set.push_back(ref_ids[static_cast<size_t>(t) * dump.top_k() + k]);
    }
    std::sort(got_ids.begin(), got_ids.end());
    std::sort(ref_set.begin(), ref_set.end());
    std::vector<int32_t> inter;
    std::set_intersection(got_ids.begin(), got_ids.end(), ref_set.begin(),
                          ref_set.end(), std::back_inserter(inter));
    set_miss += dump.top_k() - static_cast<int>(inter.size());
    for (int k = 0; k < static_cast<int>(got_top[t].size()); ++k)
      require(std::abs(got_top[t][k].second) < 1e4, "logit magnitude sanity");
  }
  require(top1_mismatch == 0,
          "top-1 token mismatch on " + std::to_string(top1_mismatch) +
              " rows (min ref margin " + std::to_string(min_margin) + ")");
  require(set_miss <= T, "top-8 set overlap (missed " +
                             std::to_string(set_miss) + " ids total)");

  // ---- routing decisions ---------------------------------------------
  require(out.routes.size() == dump.route_layers().size(),
          "route layer count");
  size_t route_pos = 0;
  int weight_over = 0, id_mismatch = 0;
  double max_w_rel = 0;
  for (size_t li = 0; li < dump.route_layers().size(); ++li) {
    const auto& rl = dump.route_layers()[li];
    const auto& got_route = out.routes[li];
    require(got_route.layer_idx == static_cast<uint32_t>(rl.layer_idx) &&
                got_route.top_k == static_cast<uint32_t>(rl.top_k) &&
                got_route.tokens == static_cast<uint64_t>(rl.tokens),
            "route layer metadata");
    const int32_t* ref_ids_l = dump.route_ids() + route_pos;
    const float* ref_w_l = dump.route_weights() + route_pos;
    for (int64_t t = 0; t < rl.tokens; ++t)
      for (int k = 0; k < rl.top_k; ++k) {
        const size_t idx = static_cast<size_t>(t) * rl.top_k + k;
        if (got_route.ids[idx] != ref_ids_l[idx]) ++id_mismatch;
        const double rel =
            std::abs(got_route.weights[idx] - ref_w_l[idx]) /
            (std::abs(ref_w_l[idx]) + 1e-30);
        max_w_rel = std::max(max_w_rel, rel);
        // Stack-level: the router's input carries the DSA attention's
        // module tolerance (~3e-3 hidden drift), so weights inherit ~1e-3
        // relative noise; 1e-5 is a module-level budget, not a stack one.
        if (rel > 2e-3) ++weight_over;
      }
    route_pos += static_cast<size_t>(rl.tokens) * rl.top_k;
  }
  require(id_mismatch == 0,
          "route id sets mismatch on " + std::to_string(id_mismatch) +
              " slots");
  require(weight_over == 0, "route weights exceed 1e-5 relative");

  std::printf(
      "[ OK ] forward parity (%d tokens, %d layers): hidden max %.0f ulps "
      "(%ld/%ld over soft, l2=%.2g); top-1 exact on %d rows (min margin "
      "%.2g); top-8 set misses %d; routes exact on %d MoE layers (max "
      "weight rel %.2g)\n",
      T, cfg.num_hidden_layers, hidden.max_ulps, hidden.over_soft,
      hidden.total, hidden.l2, T, min_margin, set_miss,
      static_cast<int>(out.routes.size()), max_w_rel);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  // --write-fixture is pure CPU (config + safetensors bytes): it must
  // succeed on GPU-less boxes so the ctest generate step always has its
  // input. Everything else needs a device.
  std::string checkpoint_dir, dump_file;
  bool wrote_fixture = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--write-fixture") == 0 && i + 1 < argc) {
      write_fixture(argv[++i]);
      wrote_fixture = true;
    } else if (std::strcmp(argv[i], "--checkpoint-dir") == 0 && i + 1 < argc) {
      checkpoint_dir = argv[++i];
    } else if (std::strcmp(argv[i], "--dump-file") == 0 && i + 1 < argc) {
      dump_file = argv[++i];
    } else {
      std::fprintf(stderr, "glm_forward_test: unknown argument %s\n", argv[i]);
      return 1;
    }
  }
  if (wrote_fixture) return 0;
  if (!checkpoint_dir.empty() != !dump_file.empty()) {
    std::fprintf(stderr,
                 "glm_forward_test: --checkpoint-dir and --dump-file go "
                 "together\n");
    return 1;
  }

  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  const int rc = dgpp::test::run_all();
  if (rc != 0) return rc;
  if (!dump_file.empty()) {
    try {
      return run_dump_parity(checkpoint_dir, dump_file);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "glm_forward_test: %s\n", e.what());
      return 1;
    }
  }
  return 0;
}
