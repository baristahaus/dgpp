// The Qwen3.8-27B (qwen3_5) world-1 forward against its reference (#84,
// 2026-10-03).
//
// Modes:
//   --write-fixture DIR          the tiny synthetic checkpoint (tests/cuda/qwen35_fixture.hpp)
//   --smoke DIR                  Qwen35Model over the fixture: finite outputs, bitwise repeat
//   --checkpoint-dir DIR
//   --dump-file FILE             Qwen35Model against tools/qwen35_reference_dump.py's numpy
//                                double reference: every layer's residual, the final
//                                read, the per-token top-k logits, the draft block's rows
//   --engine-states FILE         (with --dump-file) write the engine's layer residuals and
//                                post-final-norm hidden for the teacher-forced reference
//   --relaxed                    the end-to-end budgets
//
// The ctest chain: qwen35_forward_fixture -> qwen35_forward_smoke, and
// qwen35_forward_fixture -> qwen35_forward_generate (python, end to end) ->
// qwen35_forward_test_e2e (--relaxed, writes the engine's states) ->
// qwen35_forward_generate_teacher (python, --teacher: every reference layer
// fed the engine's own input) -> qwen35_forward_test (strict).
//
// Budgets: the MiMo / GLM-4.7 gates' (16 / 128 bf16 ulps soft / hard with a
// 2 % RMS cancellation floor, l2 1 %, 0.5 % hard elements; top-1 exact
// outside certified near ties) on the teacher-forced dump, where each
// layer's error is one layer's worth of the fp32 order against the
// reference's doubles (the GDN recurrence is the chunked kernel against the
// scalar form; the attention the tile chain against the same chain in
// doubles). The end-to-end run is the relaxed yardstick (l2 3 %, 2 % hard):
// there is no routing here, so it has no flip exemption.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/model35.hpp"
#include "qwen35_fixture.hpp"

namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::Qwen35Model;
using dgpp::Qwen35TextConfig;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

std::vector<int64_t> smoke_tokens(const Qwen35TextConfig& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  if (n > 9) t[9] = cfg.eos_token_ids.empty() ? 0 : cfg.eos_token_ids[0];
  return t;
}

// Per row: the k largest logits, ties to the lower id.
std::vector<std::vector<std::pair<int, float>>> topk(const std::vector<float>& logits, int rows, int vocab, int k) {
  std::vector<std::vector<std::pair<int, float>>> out(static_cast<size_t>(rows));
  for (int r = 0; r < rows; ++r) {
    const float* row = logits.data() + static_cast<size_t>(r) * vocab;
    std::vector<int> ids(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) ids[static_cast<size_t>(i)] = i;
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                      [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
    for (int i = 0; i < k; ++i) out[static_cast<size_t>(r)].emplace_back(ids[static_cast<size_t>(i)], row[ids[static_cast<size_t>(i)]]);
  }
  return out;
}

Qwen35Model make_model(const Qwen35TextConfig& cfg, const std::string& dir, int T, bool mtp) {
  return Qwen35Model(cfg, dir, /*max_tokens=*/std::max(256, T), /*max_cache_tokens=*/512,
                     dgpp::LoaderResidency::Streaming, /*boundary=*/nullptr, /*rank=*/0, /*world=*/1,
                     /*max_requests=*/1, /*decode_rows=*/8, mtp);
}

Qwen35Model make_resident_model(const Qwen35TextConfig& cfg, const std::string& dir, int T, bool mtp) {
  return Qwen35Model(cfg, dir, /*max_tokens=*/std::max(256, T), /*max_cache_tokens=*/512,
                     dgpp::LoaderResidency::Resident, /*boundary=*/nullptr, /*rank=*/0, /*world=*/1,
                     /*max_requests=*/1, /*decode_rows=*/8, mtp);
}

// ---- the reference dump ------------------------------------------------------
struct Dump {
  struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
  };
  std::vector<uint8_t> bytes;
  std::map<std::string, Tensor> tensors;
  int hidden = 0, vocab = 0, num_layers = 0, top_k = 0;
  int64_t token_count = 0;
  static Dump load(const std::string& path) {
    Dump d;
    std::ifstream f(path, std::ios::binary);
    require(f.good(), "dump: cannot open " + path);
    d.bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    require(d.bytes.size() >= 16 && std::memcmp(d.bytes.data(), "DGPPQW35", 8) == 0, "dump: bad magic");
    uint32_t version = 0, header_len = 0;
    std::memcpy(&version, d.bytes.data() + 8, 4);
    std::memcpy(&header_len, d.bytes.data() + 12, 4);
    require(version == 1, "dump: unsupported version");
    require(16 + static_cast<size_t>(header_len) <= d.bytes.size(), "dump: header exceeds file");
    const std::string_view header(reinterpret_cast<const char*>(d.bytes.data() + 16), header_len);
    dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(header);
    const dgpp::minijson::Value& root = parsed.root;
    const dgpp::minijson::Value& cfg = root.at("config");
    d.hidden = static_cast<int>(cfg.at("hidden").as_int());
    d.vocab = static_cast<int>(cfg.at("vocab").as_int());
    d.num_layers = static_cast<int>(cfg.at("num_layers").as_int());
    d.top_k = static_cast<int>(cfg.at("top_k").as_int());
    d.token_count = cfg.at("tokens").as_int();
    const size_t payload_base = 16 + header_len;
    for (const auto& m : root.at("tensors").members()) {
      Tensor t;
      t.dtype = std::string(m.value.at("dtype").as_string());
      for (const auto& dim : m.value.at("shape").items()) t.shape.push_back(dim.as_int());
      const uint64_t offset = m.value.at("offset").as_int();
      t.nbytes = static_cast<size_t>(m.value.at("nbytes").as_int());
      require(payload_base + offset + t.nbytes <= d.bytes.size(), "dump: tensor exceeds payload");
      t.data = d.bytes.data() + payload_base + offset;
      d.tensors[m.key] = t;
    }
    return d;
  }
  const Tensor& tensor(const std::string& name) const {
    auto it = tensors.find(name);
    require(it != tensors.end(), "dump: missing tensor " + name);
    return it->second;
  }
};

struct Stats {
  double l2 = 0, max_ulps = 0;
  long soft = 0, hard = 0, total = 0;
};

// bf16 ulps with a cancellation floor at 2 % of the reference's RMS, the
// soft/hard counts and the relative l2.
Stats compare_bf16(const uint16_t* got, const uint16_t* want, size_t n, int soft, int hard) {
  Stats s;
  s.total = static_cast<long>(n);
  double rms = 0;
  for (size_t i = 0; i < n; ++i) rms += std::pow(bf16_bits_to_float(want[i]), 2);
  rms = std::sqrt(rms / std::max<size_t>(n, 1));
  const double floor_abs = 0.02 * rms;
  double sum_d2 = 0, sum_o2 = 0;
  for (size_t i = 0; i < n; ++i) {
    const double g = bf16_bits_to_float(got[i]), w = bf16_bits_to_float(want[i]);
    int u = bf16_ulps(got[i], want[i]);
    if (std::fabs(g - w) <= floor_abs) u = 0;
    s.max_ulps = std::max(s.max_ulps, static_cast<double>(u));
    sum_d2 += (g - w) * (g - w);
    sum_o2 += w * w;
    if (u > soft) ++s.soft;
    if (u > hard) ++s.hard;
  }
  s.l2 = std::sqrt(sum_d2) / std::sqrt(sum_o2 + 1e-30);
  return s;
}

int run_smoke(const std::string& dir) {
  const Qwen35TextConfig cfg = Qwen35TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const bool mixed = cfg.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Mixed;
  // Under the mixed form walk 200 tokens: past the 128-row PT / bridge
  // thresholds, so the resident leg's PT views and the streaming leg's
  // grid dequant both take their wide-row arms. DGPP_Q35_SMOKE_T caps the
  // walk for real checkpoints, whose per-row buffers at 200 rows exceed a
  // 16 GB card's room for the model's own pools.
  int T = mixed ? 200 : 72;  // past two 32-token attention tiles
  if (const char* tv = std::getenv("DGPP_Q35_SMOKE_T"); tv && *tv)
    T = std::max(1, std::min(T, std::atoi(tv)));
  const std::vector<int64_t> tokens = smoke_tokens(cfg, T);
  Qwen35Model model = make_model(cfg, dir, T, false);
  const Qwen35Model::Outputs out = model.forward(tokens, true);
  // Value summary for real-checkpoint runs: per-layer residual scale and the
  // logits' spread — zeros here mean the walk or the head produced nothing.
  {
    auto l2 = [](const std::vector<uint16_t>& v) {
      double s = 0;
      for (size_t i = 0; i < v.size() && i < 4096; ++i) {
        const uint32_t b = static_cast<uint32_t>(v[i]) << 16;
        float f; std::memcpy(&f, &b, 4); s += double(f) * f;
      }
      return std::sqrt(s);
    };
    if (!out.layer_states.empty()) {
      std::printf("[val] hidden l2: L0 %.4f  Lmid %.4f  last %.4f\n", l2(out.layer_states[0]),
                  l2(out.layer_states[out.layer_states.size() / 2]), l2(out.layer_states.back()));
      const auto& st0 = out.layer_states[0];
      std::printf("[val] L0 row0 first 12 raw bf16:");
      for (size_t i = 0; i < 12 && i < st0.size(); ++i) std::printf(" %04x", st0[i]);
      std::printf("\n[val] L0 row1@%zu first 6 raw:", (size_t)cfg.hidden_size);
      for (size_t i = cfg.hidden_size; i < cfg.hidden_size + 6 && i < st0.size(); ++i)
        std::printf(" %04x", st0[i]);
      std::printf("\n");
    }
    {
      double sh = 0;
      for (uint16_t v : out.final_hidden_bits) {
        const uint32_t b = static_cast<uint32_t>(v) << 16;
        float f; std::memcpy(&f, &b, 4); sh += double(f) * f;
      }
      std::printf("[val] final hidden rms %.4g over %zu\n",
                  std::sqrt(sh / double(std::max<size_t>(1, out.final_hidden_bits.size()))),
                  out.final_hidden_bits.size());
      std::printf("[val] row0 logits first 6:");
      for (int i = 0; i < 6; ++i) std::printf(" %.4g", out.logits[static_cast<size_t>(i)]);
      std::printf("\n[val] tokens:");
      for (int i = 0; i < 6 && i < T; ++i) std::printf(" %lld", (long long)tokens[static_cast<size_t>(i)]);
      std::printf("\n");
    }
    double mx = -1e30, mn = 1e30, sum = 0, sum2 = 0;
    int argmax = 0;
    const float* last = out.logits.data() + (out.logits.size() - cfg.vocab_size);
    for (int64_t i = 0; i < cfg.vocab_size; ++i) {
      if (last[i] > mx) { mx = last[i]; argmax = static_cast<int>(i); }
      if (last[i] < mn) mn = last[i];
      sum += last[i]; sum2 += double(last[i]) * last[i];
    }
    const float mean = float(sum / cfg.vocab_size);
    std::printf("[val] last-row logits: max %.3f (argmax %d) min %.3f std %.3f\n", mx, argmax, mn,
                std::sqrt(sum2 / cfg.vocab_size - mean * mean));
    std::printf("[val] globals %.3f GiB resident %.3f GiB\n",
                dgpp::Qwen35LayerStream::globals_bytes(cfg, 0, 1, dgpp::LoaderHeadSharding::Full) /
                    double(1u << 30),
                dgpp::Qwen35LayerStream::resident_bytes(cfg, 0, 1, dgpp::LoaderHeadSharding::Full,
                                                       false) / double(1u << 30));
  }
  require(out.layer_states.size() == static_cast<size_t>(cfg.num_hidden_layers), "smoke: layer captures");
  for (size_t l = 0; l < out.layer_states.size(); ++l) {
    double rms = 0, mx = 0;
    for (uint16_t v : out.layer_states[l]) {
      const float f = bf16_bits_to_float(v);
      require(std::isfinite(f), "smoke: non-finite residual at layer " + std::to_string(l));
      rms += static_cast<double>(f) * f;
      mx = std::max(mx, static_cast<double>(std::fabs(f)));
    }
    rms = std::sqrt(rms / static_cast<double>(out.layer_states[l].size()));
    std::printf("[ .. ] layer %zu: h rms %.4g max %.4g\n", l, rms, mx);
  }
  // A flat logits row is a dead head, not a finite one: the zero buffer
  // passes isfinite and repeats deterministically, so check the spread.
  {
    const float* lr = out.logits.data() + (out.logits.size() - cfg.vocab_size);
    float lmax = lr[0], lmin = lr[0];
    for (int64_t i = 1; i < cfg.vocab_size; ++i) { lmax = std::max(lmax, lr[i]); lmin = std::min(lmin, lr[i]); }
    require(lmax - lmin > 1e-3f, "smoke: the last row's logits are flat (no head signal)");
  }
  require(out.logits.size() == static_cast<size_t>(T) * cfg.vocab_size, "smoke: every row's logits");
  for (float v : out.logits) require(std::isfinite(v), "smoke: non-finite logit");
  const Qwen35Model::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "smoke: forward is not deterministic across calls");
  const auto top = topk(out.logits, T, cfg.vocab_size, 4);
  std::printf("[ .. ] smoke: %d tokens, %d layers, top-1 of the last row %d (%.4g); deterministic\n", T,
              cfg.num_hidden_layers, top.back()[0].first, top.back()[0].second);
  // The draft: teacher-forced rows, finite + deterministic. Under the
  // mixed form this is the BF16 MTP (its own dense_mlp branch).
  if (cfg.mtp_layer() >= 0) {
    Qwen35Model draft = make_model(cfg, dir, T, true);
    const Qwen35Model::Outputs d1 = draft.mtp_forward(tokens);
    require(d1.logits.size() == static_cast<size_t>(T - 1) * cfg.vocab_size, "smoke: draft logits rows");
    for (float v : d1.logits) require(std::isfinite(v), "smoke: non-finite draft logit");
    const Qwen35Model::Outputs d2 = draft.mtp_forward(tokens);
    require(d2.logits == d1.logits, "smoke: draft forward is not deterministic");
    std::printf("[ .. ] smoke: draft ok (%d rows), deterministic\n", T - 1);
  }
  // Under the mixed form, the resident leg with the PT recipe on: the
  // channel/bf16 attention requants at boot, the PT views at 200 rows,
  // the resident fp4 / channel-bridge / bf16-dense arms in one model.
  if (mixed) {
    // The resident leg needs the whole release plus the PT scratch on one
    // card: skip it when the card cannot hold both (the streaming leg above
    // is the config this box's 16 GiB cards serve).
    const double need_gib =
        double(dgpp::Qwen35LayerStream::resident_bytes(cfg, 0, 1, dgpp::LoaderHeadSharding::Full,
                                                       true) +
               dgpp::Qwen35LayerStream::globals_bytes(cfg, 0, 1, dgpp::LoaderHeadSharding::Full)) /
            1073741824.0 +
        3.0;  // + the PT scratch and runtime buffers
    size_t free_b = 0, total_b = 0;
    DGPP_CUDA_OK(cudaMemGetInfo(&free_b, &total_b));
    if (need_gib > static_cast<double>(free_b) / 1073741824.0 - 1.0) {
      std::printf("[ .. ] smoke: resident + PT recipe SKIPPED (needs ~%.1f GiB, the card holds %.1f)\n",
                  need_gib, static_cast<double>(total_b) / 1073741824.0);
    } else {
    Qwen35Model::set_prefill_fp8_per_tensor(true);
    Qwen35Model resident = make_resident_model(cfg, dir, T, true);
    Qwen35Model::set_prefill_fp8_per_tensor(false);
    const Qwen35Model::Outputs r1 = resident.forward(tokens, true);
    require(r1.logits.size() == static_cast<size_t>(T) * cfg.vocab_size, "smoke: resident logits rows");
    for (float v : r1.logits) require(std::isfinite(v), "smoke: non-finite resident logit");
    const Qwen35Model::Outputs r2 = resident.forward(tokens, false);
    require(r2.logits == r1.logits, "smoke: resident forward is not deterministic");
    const Qwen35Model::Outputs rd1 = resident.mtp_forward(tokens);
    for (float v : rd1.logits) require(std::isfinite(v), "smoke: non-finite resident draft logit");
    const Qwen35Model::Outputs rd2 = resident.mtp_forward(tokens);
    require(rd2.logits == rd1.logits, "smoke: resident draft forward is not deterministic");
    std::printf("[ .. ] smoke: resident + PT recipe ok (attn channel/bf16 slots), deterministic\n");
    }
  }
  std::printf("[ OK ] qwen35_forward_smoke%s\n", mixed ? " (nvfp4 mixed)" : "");
  return 0;
}

int run_dump_parity(const std::string& dir, const std::string& dump_path, const std::string& states_path,
                    bool relaxed) {
  const Dump dump = Dump::load(dump_path);
  const Qwen35TextConfig cfg = Qwen35TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  require(cfg.hidden_size == dump.hidden && cfg.vocab_size == dump.vocab && cfg.num_hidden_layers == dump.num_layers,
          "dump config disagrees with the checkpoint config");
  const Dump::Tensor& tok = dump.tensor("tokens");
  require(tok.dtype == "I64", "dump: tokens dtype");
  const int T = static_cast<int>(dump.token_count);
  std::vector<int64_t> tokens(static_cast<size_t>(T));
  std::memcpy(tokens.data(), tok.data, static_cast<size_t>(T) * 8);

  Qwen35Model model = make_model(cfg, dir, T, false);
  const Qwen35Model::Outputs out = model.forward(tokens, true);
  const Qwen35Model::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "forward is not deterministic across calls");
  const int H = cfg.hidden_size;
  if (!states_path.empty()) {
    // The engine's layer residuals for the teacher-forced reference
    // (tools/qwen35_reference_dump.py --teacher): int32 L, T, H then bf16
    // rows per layer, the post-final-norm hidden after them.
    std::ofstream f(states_path, std::ios::binary);
    const int32_t hdr[3] = {cfg.num_hidden_layers, T, H};
    f.write(reinterpret_cast<const char*>(hdr), 12);
    for (const auto& st : out.layer_states)
      f.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size() * 2));
    f.write(reinterpret_cast<const char*>(out.final_hidden_bits.data()),
            static_cast<std::streamsize>(out.final_hidden_bits.size() * 2));
  }

  bool ok = true;
  const double l2_budget = relaxed ? 0.03 : 0.01, hard_budget = relaxed ? 0.02 : 0.005;
  // ---- per-layer residual states ---------------------------------------------
  const Dump::Tensor& ls = dump.tensor("layer_states");
  require(ls.dtype == "BF16" && ls.shape.size() == 3 && ls.shape[0] == cfg.num_hidden_layers && ls.shape[1] == T &&
              ls.shape[2] == H,
          "dump: layer_states shape");
  require(out.layer_states.size() == static_cast<size_t>(cfg.num_hidden_layers), "the engine captured every layer");
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const uint16_t* want = reinterpret_cast<const uint16_t*>(ls.data) + static_cast<size_t>(l) * T * H;
    const uint16_t* got = out.layer_states[static_cast<size_t>(l)].data();
    const Stats s = compare_bf16(got, want, static_cast<size_t>(T) * H, 16, 128);
    std::printf("[ .. ] layer %d (%s) h: l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", l,
                cfg.layers[static_cast<size_t>(l)] == dgpp::Qwen35LayerKind::Gdn ? "gdn" : "full", s.l2, s.max_ulps,
                s.soft, s.hard, s.total);
    std::vector<std::pair<double, int>> rows;
    for (int t = 0; t < T; ++t) {
      const Stats r = compare_bf16(got + static_cast<size_t>(t) * H, want + static_cast<size_t>(t) * H,
                                   static_cast<size_t>(H), 16, 128);
      rows.emplace_back(r.l2, t);
    }
    std::sort(rows.rbegin(), rows.rend());
    std::printf("[ .. ]   worst rows:");
    for (int i = 0; i < 4 && i < T; ++i)
      std::printf(" t%d %.3g", rows[static_cast<size_t>(i)].second, rows[static_cast<size_t>(i)].first);
    std::printf("\n");
    if (s.hard > 0 && s.hard <= 8) {
      double rms = 0;
      for (size_t i = 0; i < static_cast<size_t>(T) * H; ++i) rms += std::pow(bf16_bits_to_float(want[i]), 2);
      rms = std::sqrt(rms / (static_cast<double>(T) * H));
      for (size_t i = 0, shown = 0; i < static_cast<size_t>(T) * H && shown < 8; ++i) {
        const float g = bf16_bits_to_float(got[i]), w = bf16_bits_to_float(want[i]);
        if (bf16_ulps(got[i], want[i]) > 128 && std::fabs(g - w) > 0.02 * rms) {
          std::printf("[ .. ]   hard element t%zu h%zu: got %g want %g\n", i / H, i % H, g, w);
          ++shown;
        }
      }
    }
    if (s.total > 0 && (s.l2 > l2_budget || static_cast<double>(s.hard) / s.total > hard_budget)) ok = false;
  }
  // ---- the final read ---------------------------------------------------------
  const Dump::Tensor& fh = dump.tensor("final_hidden");
  require(fh.dtype == "BF16" && fh.shape.size() == 2 && fh.shape[0] == T && fh.shape[1] == H, "dump: final_hidden shape");
  const Stats fs_ = compare_bf16(out.final_hidden_bits.data(), reinterpret_cast<const uint16_t*>(fh.data),
                                 static_cast<size_t>(T) * H, 16, 128);
  std::printf("[ .. ] final hidden: l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", fs_.l2, fs_.max_ulps, fs_.soft,
              fs_.hard, fs_.total);
  if (fs_.total > 0 && (fs_.l2 > l2_budget || static_cast<double>(fs_.hard) / fs_.total > hard_budget)) ok = false;
  // ---- logits: top-1 (near ties certified), top-k overlap --------------------
  const Dump::Tensor& tid = dump.tensor("topk_ids");
  const Dump::Tensor& tlog = dump.tensor("topk_logits");
  const int k = dump.top_k;
  require(tid.dtype == "I32" && tlog.dtype == "F32" && tid.shape[0] == T && tid.shape[1] == k, "dump: topk shape");
  const int32_t* ref_ids = reinterpret_cast<const int32_t*>(tid.data);
  const float* ref_vals = reinterpret_cast<const float*>(tlog.data);
  const auto got = topk(out.logits, T, cfg.vocab_size, k);
  int top1_hard = 0, top1_soft = 0, set_miss = 0;
  double max_rel = 0;
  for (int t = 0; t < T; ++t) {
    const int32_t* rid = ref_ids + static_cast<size_t>(t) * k;
    const float* rval = ref_vals + static_cast<size_t>(t) * k;
    const double margin = std::fabs(rval[0] - rval[1]) / (std::fabs(rval[0]) + 1e-30);
    if (got[static_cast<size_t>(t)][0].first != rid[0]) {
      if (margin < 0.02) ++top1_soft; else ++top1_hard;
    }
    for (int i = 0; i < k; ++i) {
      const int32_t id = got[static_cast<size_t>(t)][i].first;
      const int32_t* p = std::find(rid, rid + k, id);
      if (p == rid + k) { ++set_miss; continue; }
      max_rel = std::max(max_rel, std::fabs(rval[p - rid] - got[static_cast<size_t>(t)][i].second) /
                                      (std::fabs(rval[p - rid]) + 1e-30));
    }
  }
  std::printf("[ .. ] logits: top-1 hard mismatches %d, near-tie %d of %d rows; top-%d set misses %d of %d; "
              "matched logits within %.3g relative\n",
              top1_hard, top1_soft, T, k, set_miss, T * k, max_rel);
  if (top1_hard != 0 || set_miss > T * k / 4) ok = false;
  // ---- the draft block: its rows over the prompt ---------------------------------
  if (dump.tensors.count("mtp_topk_ids") && cfg.mtp_layer() >= 0) {
    Qwen35Model mtp = make_model(cfg, dir, T, true);
    const Qwen35Model::Outputs d = mtp.mtp_forward(tokens);
    const int R = T - 1;
    const Dump::Tensor& mh = dump.tensor("mtp_final_hidden");
    require(mh.dtype == "BF16" && mh.shape.size() == 2 && mh.shape[0] == R && mh.shape[1] == H, "dump: mtp_final_hidden shape");
    require(d.final_hidden_bits.size() == static_cast<size_t>(R) * H, "mtp_forward: hidden rows");
    const Stats ms = compare_bf16(d.final_hidden_bits.data(), reinterpret_cast<const uint16_t*>(mh.data),
                                  static_cast<size_t>(R) * H, 16, 128);
    std::printf("[ .. ] draft final hidden: l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", ms.l2, ms.max_ulps, ms.soft,
                ms.hard, ms.total);
    if (ms.total > 0 && (ms.l2 > l2_budget || static_cast<double>(ms.hard) / ms.total > hard_budget)) ok = false;
    const Dump::Tensor& mid = dump.tensor("mtp_topk_ids");
    const Dump::Tensor& mlog = dump.tensor("mtp_topk_logits");
    require(mid.dtype == "I32" && mlog.dtype == "F32" && mid.shape[0] == R && mid.shape[1] == k, "dump: mtp topk shape");
    const int32_t* mref_ids = reinterpret_cast<const int32_t*>(mid.data);
    const float* mref_vals = reinterpret_cast<const float*>(mlog.data);
    const auto mgot = topk(d.logits, R, cfg.vocab_size, k);
    int mhard = 0, msoft = 0;
    for (int t = 0; t < R; ++t) {
      const int32_t r1 = mref_ids[static_cast<size_t>(t) * k];
      const float v1 = mref_vals[static_cast<size_t>(t) * k], v2 = mref_vals[static_cast<size_t>(t) * k + 1];
      const double margin = std::fabs(v1 - v2) / (std::fabs(v1) + 1e-30);
      if (mgot[static_cast<size_t>(t)][0].first != r1) {
        if (margin < 0.02) ++msoft; else ++mhard;
      }
    }
    std::printf("[ .. ] draft logits: top-1 hard mismatches %d, near-tie %d of %d rows\n", mhard, msoft, R);
    if (mhard != 0) ok = false;
  }
  std::printf("[ %s ] qwen35_forward_dump_parity (%s)\n", ok ? "OK" : "FAIL",
              relaxed ? "end-to-end, relaxed" : "teacher-forced, strict");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixture, mixed_fixture, smoke, checkpoint, dump, states;
  bool relaxed = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--write-fixture" && i + 1 < argc) fixture = argv[++i];
    else if (a == "--write-mixed-fixture" && i + 1 < argc) mixed_fixture = argv[++i];
    else if (a == "--smoke" && i + 1 < argc) smoke = argv[++i];
    else if (a == "--checkpoint-dir" && i + 1 < argc) checkpoint = argv[++i];
    else if (a == "--dump-file" && i + 1 < argc) dump = argv[++i];
    else if (a == "--engine-states" && i + 1 < argc) states = argv[++i];
    else if (a == "--relaxed") relaxed = true;
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  try {
    if (!fixture.empty()) {
      qwen35fx::write_fixture(fixture);
      std::printf("[ OK ] wrote the fixture to %s\n", fixture.c_str());
      return 0;
    }
    if (!mixed_fixture.empty()) {
      qwen35fx::write_mixed_fixture(mixed_fixture);
      std::printf("[ OK ] wrote the mixed fixture to %s\n", mixed_fixture.c_str());
      return 0;
    }
    if (!smoke.empty()) return run_smoke(smoke);
    if (!checkpoint.empty() && !dump.empty()) return run_dump_parity(checkpoint, dump, states, relaxed);
    std::fprintf(stderr,
                 "usage: --write-fixture DIR | --write-mixed-fixture DIR | --smoke DIR | "
                 "--checkpoint-dir DIR --dump-file FILE [--engine-states FILE] [--relaxed]\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
