# Qwen3.8-27B dense on the dual 5070 Ti — the NVFP4 MLP and the discrete memory plan (2026-10-03)

Status: the at-load NVFP4 MLP is in-tree and gated; the two-rank boot is
not (that is dense plan §6's P4 — family entry, rxe transport, cluster
template). This doc records the quant form, the kernel and loader changes
behind it, the discrete-GPU memory-plan split in `dgpp_serve`, and the
traps found on the way. The port itself is `docs/qwen38_27b_dense_plan.md`
(landed 2026-10-03, commit 0c97db2); the platform (sm_120a build,
Soft-RoCE bus loopback) is commit f7694bd.

The deployment target: **world 2 over the two RTX 5070 Ti on this box**
(two Spark-class discrete GPUs where the family's earlier deployments were
one GB10 node — "dual spark"). 16 GB a board, no GPUDirect, no RDMA NIC:
the bus rides Soft-RoCE loopback (dense plan §4).

## 1. Why: 16 GB boards, and the MLP is the slab

The dense form's FP8 stack (dense plan §4) plans **14.0 GB resident a
rank** — 11.9 GB layer weights (every dense projection block-FP8 at load)
+ 1.27 GB replicated embed + 0.64 GB head slice + 0.2 GB MTP — leaving
≈2.3 GB for KV (262 KB/tok/rank bf16), GDN states (75 MB/slot/rank),
graphs and scratch. That fits the parity boot but not a useful serving
shape; the plan named the follow-up: the MLPs are 17.4 GB of the rank's
23.9 GB BF16 layer weights — **the single highest-leverage quant**, and
the same modelopt NVFP4 triple the vLLM deployment on this box already
serves.

`engine.dense_weights = "nvfp4"` lands it. The ledger, a rank at world 2:

| | BF16 | FP8 | NVFP4 |
|---|---|---|---|
| 65 dense SwiGLUs (64 + draft) | 17.4 GB | 8.7 GB | **4.89 GB** |
| every other dense projection | 6.5 GB | 3.25 GB | 3.25 GB |
| resident total (with embed/head/MTP) | — | 14.0 GB | **10.25 GB** |

0.5625 B/element (e2m1 pairs + one e4m3 per 16 + the global) against
FP8's 1 + scales. The margin for KV, states, graphs and scratch roughly
doubles: 2.3 → 5.3 GB. Decode weight traffic a step drops with it
(8.7 → 4.9 GB of MLP reads a rank a step) — the form is a fit decision
first and a speed decision second; the decode GEMV's effective bandwidth
on this board is the same as the Flash-Next experts' (docs/nvfp4_plan.md).

Quality, honestly stated: same recipe as the served NVFP4 checkpoint
(modelopt triple, no calibration — amax over the rank's slice), but
**all 65 MLPs** where vLLM's hybrid leaves layers 56–63 in FP8, and
encoded at load from the BF16 original rather than an offline dynamic
quant. The gates below pin the wiring (recipe bitwise vs the reference
reimplementation) and the walk (decode audit within 2e-2, near-tie
counted).

## 2. The knob and the resident form

- `engine.dense_weights = "nvfp4"` (cluster config / fabric `WorldSettings`
  / `dgpp_serve --dense-weights`), implying the fp8 form for everything
  else: `set_dense_weights_fp8(true)` + `set_dense_mlp_nvfp4(true)`
  (`QwenLayerStream`). `fp8_head mma` now accepts either fp8 or nvfp4.
  BF16 releases only — a shipped-FP8 checkpoint (`dense_fp8_shipped`)
  keeps its shipped form; the loader branch falls to fp8.
- `QwenMlpResident.fp4[3]` (`GlmFp4Matrix`) beside the fp8 triple; the
  draft layer's MLP takes the same form.
- The resident image's format gains bit 16 — an image written under the
  other form is rebuilt, not misread.
- The loader (`WeightBuilder::encode_fp4` / `load_bf16_rows_fp4` /
  `load_bf16_cols_fp4`): the fp8 slicing recipe (gate/up by intermediate
  rows, down by intermediate columns) with the host-side encode from
  `loaders/nvfp4_quant.hpp` — the same encoder the GLM-4.7 requant path
  uses. The scale grid anchors at the slice origin: a K slice starts on a
  16-element boundary by construction (TP slice widths are multiples).
  The device global is the recipe's `1 / weight_scale_2`.
- The counting build derives `layer_bytes` from the same builders, so the
  memory plan and the resident capacity follow the form with no
  closed-form duplicate. The loader test pins
  `bytes_nvfp4 < bytes_fp8 < bytes_bf16` and the source-byte plan.

## 3. The fp4 GEMV at K = 8704: the fifth pass

The down's K at world 2 is I/2 = **8704** = 272 chunks — 16 lanes × 17 —
one over the four passes of four the core allowed (`kMaxPasses = 4`,
kMaxK 16384). The gate/up ride the existing 5120 (Flash-Next's hidden
width, already compiled).

`kMaxPasses` is 4 → **5** (kMaxK 20480). The pass count is a loop bound
only — `Geom<K>::passes` derives per K — so the existing widths'
geometry, and their bits, are untouched; the compiled set gains
`dispatch_k`/`k_compiled` 8704 (and the error string). `fp4_gemv_test`
still gates the core.

The world guard is explicit: `QwenDenseMlp` fails at construction when
any of the three Ks is outside the compiled set. World 1's down K 17408
*would* fit the fifth pass but is not compiled — the dual-spark form
targets world 2 — and world 4's 4352 is not either; those deployments
keep fp8, fail-fast at boot rather than mid-serve.

## 4. The prefill bridge

`gemm_dense_fp4` (layers.cpp) splits on the row count exactly like the
fp8 path it sits beside:

- **m ≤ 128** — `launch_fp4_gemv_bf16`, the single-matrix GEMV, bitwise
  invariant to m (the core's contract; the decode gate pins it).
- **m > 128** — dequantize once per chunk into the shared BF16 bridge and
  run the bf16 GEMM: `launch_fp4_dequant` (kernels/fp4_dequant.cu), the
  fp8 bridge's fp4 twin — one thread an element, one RNE round to bf16 at
  the end, `bf16(e2m1 × e4m3 / global)`. No fp4 GEMM exists for prefill
  shapes, so unlike fp8 there is **no scale-GEMM fallback**: the bridge
  holds the matrix or the enqueue throws.

That difference is the trap this port's review caught: the bridge is
sized by `QwenModel::dense_bridge_bytes`, which took every dense
projection *except the MLP* — correct under fp8, where the MLP prefill
falls back to the scale GEMM, and wrong here. At world 2 the MLP's slice
([8704, 5120] = 85 MiB bf16) **bounds** q_proj's [6144, 5120] (60 MiB),
so the fp4 prefill would have thrown at the first real boot with a
bridge sized 30 MiB too small. The tiny fixture cannot see it (its MLP
is far smaller than its q_proj). `dense_bridge_bytes` now takes the MLP
slice under the knob, and `qwen_decode_test --dense-mlp-fp4` opens with
a **world-2 shape gate** built from the real 27B dims (a hand-built
config, no weights): the slice Ks in the compiled set, worlds 1/4 off
it, and the bridge pinned to `mlp_elems * 2` under nvfp4 vs
`q_proj_elems * 2` under fp8. The plan item's name follows the form.

The bridge, scratch and activations aside, the fp4 form adds **no** new
per-step buffers: the GEMV reads the resident image; the bridge is the
fp8 stack's own.

## 5. The discrete memory plan in `dgpp_serve`

`check_memory_plan` assumed GB10: one unified pool, device free and host
available two views of it, 4 GiB headroom against the max. A 5070 Ti rank
is two pools with nothing shared. The check now branches on
`prop.integrated`:

- integrated: the old sum, unchanged;
- discrete: device needs (plan's device bytes + **1 GiB** headroom — the
  CUDA context ~0.25 GiB on the 610 driver, capture pools beside the
  plan's activation workspace, allocator growth) against device free;
  pinned needs (pinned + prefix arena + engine buffers) + the 4 GiB host
  headroom against host available. The binding side names itself in the
  log ("discrete: device fits/over … host fits/over …"), and the
  largest-context hint extrapolates along the binding side's slope —
  the arena and engine buffers move with the host side under the
  discrete branch.

`kDeviceHeadroomBytes` is a first measurement, to be revisited with the
resident soak's ledger on new hardware.

## 6. Validation state (all green on this box, 2026-10-03)

- `qwen_loader_dense_mlp_nvfp4_at_load_is_the_recipe` — the recipe
  reimplemented from `nvfp4_quant.hpp`'s comment (amax → ws2 → block
  scales → e2m1 pairs, nibble order, `1/ws2` global), world 1 and the
  world-2 rank-1 slices (rows/cols recipe), the draft layer's MLP, the
  decoded-back half-step bound, byte plan order, source-byte plan.
- `qwen_dense_mlp_fp4_decode` (new ctest, dense twin fixture) — the
  world-2 shape gate; the GEMV-path prefill bitwise the forward's last
  row; the 160-row bridge prefill bitwise; 12-step decodes after both,
  within the audit's 2e-2 with near ties counted.
- The fp4 GEMV core (`fp4_gemv_test`) — 8704 through the fifth pass.
- The dense bf16 13-gate and dense fp8 13-gate suites, and the
  Flash-Next suite (the shared `layers.cpp` regression): all green.
- The decode fixture's QSA fix below is covered by the dense suites
  under fp8 dense weights.

## 7. Traps found (2026-10-03, resolved in-tree)

- **The dense QSA tripped the indexer's dense path**: `attn_partial`'s
  `np == 3` branch (the AutoRound hybrid's BF16 indexer projection)
  fired for the dense form, whose QSA has no indexer at all — a null
  `index_qk_proj` through `gemm_dense` under fp8 dense weights. The
  branch now requires `has_idx_`.
- **The bridge sizing** (§4): `dense_bridge_bytes` missed the MLP take;
  caught in review before any real boot, pinned by the shape gate.
- **`fp8_head mma` vs the new form**: the check rejected nvfp4 on the
  old `dense_weights != "fp8"` clause; it now reads `== "checkpoint"`.

## 8. What remains (dense plan §6 P4/P5)

- The family entry and cluster template (world 2, both GPUs, rxe) and
  the first 2-GPU boot under this form — expect ~10.25 GB resident and
  the discrete plan check's log to name the fitted shape.
- Serving parity against the box's vLLM endpoint (the greedy
  cross-check target named in the port plan).
- The shared-memory bus lane (the rxe follow-up) if the μs-scale
  collectives bound decode at the target shape.
