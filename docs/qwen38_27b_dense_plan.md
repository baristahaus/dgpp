# Qwen3.8-27B (dense, qwen3_5) on dgpp — port plan for the 5070 Ti node (2026-10-02)

Status: planned. This is the first family port that targets this machine
(hephaestus, 2× RTX 5070 Ti 16 GB, no RDMA NIC) rather than the Spark/GB10
fleet, so the plan covers the model **and** the two platform deltas: sm_120a
kernels and a software-RDMA transport.

Checkpoint references:
- `Qwen/Qwen3.8-27B` (BF16, official) — `/nfs/models--Qwen--Qwen3.8-27B`,
  tensor census from its safetensors headers.
- `Qwen/Qwen3.8-27B` NVFP4 (Unsloth dynamic, `~/models`, served by vLLM on
  this box at :8000) — same `text_config` byte-for-byte; useful as the greedy
  cross-check target, not as a load target.
- transformers 5.15.0 `models/qwen3_5/modeling_qwen3_5.py` (in the vLLM image,
  tag `6b084be…`) — the reference implementation for the dense text stack.
  It explicitly ignores `mtp.*` (`_keys_to_ignore_on_load_unexpected`), so
  the MTP reference is vLLM's `qwen3_5_mtp.py` in the same image.

## 0. Summary

Qwen3.8-27B is the dense sibling of the engine's first Qwen family
(Flash-Next, `qwen4_exp`): the same hybrid GDN / full-attention pattern and
vocabulary, minus every routed structure. 64 layers, 5120 hidden, 48
Gated DeltaNet layers and 16 plain full-attention layers in a 3:1 pattern
(`full_attention_interval = 4`), a dense SwiGLU MLP (17 408) on every layer,
zero-centered RMSNorm pre-ops, plain residual — no Gated Residual, no MoE, no
PLE, no indexer. One MTP draft layer. A vision tower ships in the checkpoint
and is not served. 26.8 B text parameters → ≈13.4 GB per rank at FP8/world 2
before KV, GDN states, graphs and scratch.

What is shared with Flash-Next vs what is new:

| Piece | 27B | Flash-Next | Reuse |
|---|---|---|---|
| GDN mixer | same tensors, gate `silu(z)` ("swish") | `sigmoid(z)` | kernels + gate variant |
| Attention | `Qwen3_5Attention` + output gate from q's high half, no indexer | same + QSA indexer | attention core as-is, select-all path |
| MLP | dense SwiGLU 17408 | 512-expert MoE + shared expert | shared-expert FP8 path, widened |
| Residual | plain (norm → add) | 4-branch Gated Residual | new residual site, same call shape |
| Norms | zero-centered RMSNorm | same | as-is |
| MTP | fused `fc [H,2H]`, plain draft layer | fc_embedding/fc_hidden + GR mixer | split fused fc; plain draft walk |
| RoPE | partial 0.25×256, mrope interleaved [11,11,10] | same | as-is |

## 1. Architecture (reference-pinned)

Semantics verified against transformers 5.15 source, quoted where the engine
must match bit-for-bit through the FP8 path:

- **Decoder layer** (plain residual): `x = input_layernorm(r)`;
  `r += attn(x)`; `x = post_attention_layernorm(r)`; `r += mlp(x)`.
- **Norm**: zero-centered RMSNorm — `(x * w).to(dtype)` with `w` init zeros,
  computed in fp32. The engine's `qwen_norm.hpp` already implements this form.
- **GDN**: `in_proj_qkv/z/a/b`, depthwise `conv1d` (width 4, no bias, causal
  left-pad), `dt_bias`, `A_log`, `RMSNormGated(128, silu)`, `out_proj`. The
  gated norm is `norm(x) * silu(z)` with silu in fp32 (`ACT2FN["silu"]`) —
  the 27B's `output_gate_type: "swish"` names silu; Flash-Next's config
  says `sigmoid` and its kernels bake `sigmoid(z)`. This is a per-family
  gate-kind knob on the GDN gated-norm sites (chunked and recurrent paths).
- **Attention** (the 16 full layers): q_proj outputs `heads*256*2` — per head,
  the low 256 dims are the query, the high 256 the gate. `q_norm`/`k_norm`
  are per-head RMSNorm over head_dim (256). Partial RoPE applies to the
  first 64 dims (`partial_rotary_factor 0.25`), mrope interleaved with
  section [11,11,10] (sum 32 = rotary_dim/2). GQA 24:4 over full history.
  `attn_output = attn_output * sigmoid(gate)` before o_proj. Identical to
  Flash-Next's QSA attention minus the indexer: with no indexer the
  selection is *all* pools of the request (select-all), which the walk
  expresses as a full-history KV read.
- **MLP**: `down(silu(gate(x)) * up(x))`, SwiGLU, 17408 intermediate.
- **MTP** (vLLM reference): `e = pre_fc_norm_embedding(embed(next_tok))`;
  `h = pre_fc_norm_hidden(h_last)`; `x = fc(cat[e, h])` with fc weight
  `[H, 2H]` and the **embedding half first**; draft layer = a standard
  full-attention decoder layer with `residual = None` entry; output =
  `norm(x, residual)`; shared embedding and lm_head
  (`mtp_use_dedicated_embeddings: false`). No hyper-state window: the plain
  hidden is the fc input, which removes the GR state plumbing Flash-Next's
  draft needed.
- **Vision tower** (`model.visual.*`): present in the checkpoint, not served;
  the loader skips the tensors the way the GLM-5.2-Vision port does.

### 1.1 Config deltas (text_config)

The 27B's `text_config` parses with the Flash-Next schema except that the
gated-residual, indexer and MoE fields are absent and the gate kind differs:

- `hc_count`/`hc_lowrank`: **optional** — absent ⇒ plain residual. All
  `hyper_width()` uses collapse to `hidden_size` (5120).
- `indexer_n_heads`/`indexer_kv_heads`/`indexer_head_dim`/`indexer_budget`/
  `indexer_compress_ratio`: **optional** — absent ⇒ the 16 attention layers
  run select-all over full history.
- `num_experts`/`num_experts_per_tok`/`moe_intermediate_size`/
  `shared_expert_intermediate_size`: **optional** — absent ⇒ dense MLP, and
  `intermediate_size` becomes required (17 408, multiple of 8).
- `output_gate_type`: accepts `swish` in addition to `sigmoid`; the gate kind
  rides the layer plan into the GDN kernels.
- `attn_output_gate: true` — no parser change (the gate is structural: q_proj
  is 2×head_dim wide; same as Flash-Next).
- Everything else (rope_parameters with partial factor + mrope, layer_types,
  mtp block, eos list) already parses; 5120 hidden, 64 layers,
  head_dim 256, GDN 16:48 heads of 128 pass the existing checks.

Top level: `architectures[0] = "Qwen3_5ForConditionalGeneration"`,
`model_type = "qwen3_5"` — a new `ModelArchitecture::Qwen35` dispatch arm.
The 27B is text-only for serving purposes; the engine reads only
`text_config` keys (embed/lm_head/norm/mtp), so a nested VL checkpoint needs
no special case beyond the visual-tensor skip.

## 2. Loader and binding

- Dense MLP tensors per layer: `mlp.gate_proj/up_proj [17408, 5120]`,
  `mlp.down_proj [5120, 17408]`. At `dense_weights=fp8` these encode at load
  through the existing FP8 path (per-tensor + act scales, the shared-expert
  format) and shard: gate/up split by rows (8704/rank), down by columns.
  The FP8 GEMV/GEMM kernels for dense SwiGLU already exist for the
  shared expert; the port reuses them at the wider N and the walk's
  prefill/decode enqueue sites move to a `QwenDenseMlp` layer class.
- Attention layers without indexer: bind q/k/v/o (+q_norm/k_norm) only;
  the indexer tensors and their FP8 forms are simply absent from the
  expected-tensor set when the config has no indexer.
- MTP: `mtp.fc.weight [5120, 10240]` splits at load into fc_embedding
  (`[:, :5120]`) and fc_hidden (`[:, 5120:]`) halves — same two-GEMV shape
  Flash-Next binds, so the draft fc path is unchanged code. The split is
  three roundings against the reference's one fused fp32 accumulation over
  `cat[e, h]` — accepted: the draft steers speculation only, the served
  output stays the trunk's, and the P5 draft-head gate runs tolerance, not
  bitwise. `pre_fc_norm_*` bind at hidden_size (not hyper_width). No
  `mtp.hyper_connection_mixer`. The draft layer binds the dense-attention +
  dense-MLP tensor set.
- Embedding/lm_head/norm: unchanged. The census' `model.visual.*` tensors
  are skipped explicitly (release-load filter), matching the GLM vision
  precedent.

## 3. Walk and layers

- `QwenResidualSite`: the GR site's plain replacement — `mix` = pre-op norm
  of the residual, `defer_combine` = accumulate into the residual, final
  `mixer_mix` = final norm. Same call shape as `QwenGrSite` so the walk
  selects the site by config and the decode/prefill/graph walks stay shared.
  Scratch: `hyper_width()` returns 5120 in dense mode, so W-sized buffers
  (the GR combine rows, the MTP fc input) resize for free.
- `QwenDenseMlp`: enqueue_decode (FP8 GEMV + SwiGLU), enqueue_prefill (FP8
  GEMM or dequant-BF16 per the dense-weights plan), matching the shared
  expert's two paths. TP split at the GEMM level as above.
- Attention select-all: with no indexer the layer enqueues the full-history
  attention (the QSA kernel path with a "all pools" selection list — the
  selection machinery already supports explicit lists for the
  reference-check path; the port routes around scoring only).
- GDN gate kind: one template/branch on the gated-norm (`silu` vs `sigmoid`)
  at the chunked prefill kernel and the recurrent decode kernel.
- MTP draft walk: fc(norm(e), norm(h)) → draft layer (attention + MLP,
  plain residual) → final norm → head. The GDN-state and KV plumbing for the
  draft layer reuses the existing MTP graph plumbing; the GR window rows
  drop out.

## 4. Serving, memory, transport on this node

- **Family entry** (`dgpp_serve.cpp`): model_type `qwen3_5`, arch
  `Qwen3_5ForConditionalGeneration`; lat slot bytes = 5120 rows × fp32 fold
  (20 480 B) per collective class; the allreduce stream budget sizes from
  128 lat-class collectives per decode step (64 layers × 2 folds).
- **Memory plan** (16 GB/rank, FP8 dense): 11.9 GB layer weights/rank +
  1.27 GB replicated embed + 0.64 GB head slice + 0.2 GB MTP ≈ 14.0 GB
  resident, leaving ≈2.3 GB for KV (bf16, GQA 4×256×2B ≈ 4 KB/tok/layer ×
  64 ≈ 262 KB/tok/rank), GDN states (48 layers × 24 v-heads × 128×128 fp32
  ≈ 75 MB/slot/rank), graphs and scratch. Expect 2–4 slots and 16–32 K
  context; the plan arbitrates at boot, and the first bring-up logs the
  fitted shape. If FP8 does not fit the target shape, the follow-up is
  at-load NVFP4 for the MLPs (17.1 GB of the 23.9 GB layer weights — the
  single highest-leverage quant, matching what the vLLM deployment on this
  box already does); **not** part of this port.
- **Transport — Soft-RoCE**: no RDMA NIC on this node. The bus's data path
  is host-pinned memory end to end (NIC-DMA'd slabs, device reads via
  mapped pointers), which rxe serves without GPUDirect: `rdma link add
  rxe0 type rxe netdev enp10s0`, persisted as a boot unit
  (`rxe-link.service`), `DGPP_ROCE_DEVICES=rxe0`. Local QP pairs loop
  in-kernel. Expected per-collective latency is software (μs-scale, ~128
  collectives/step ⇒ ≤1 ms/step ceiling) — acceptable for bring-up and the
  parity gates; a shared-memory lane is the optimization follow-up.
- **Cluster template**: `deploy/cluster_qwen-3.8-27b_fp8_w2.json` — nodes
  localhost×2, `CUDA_VISIBLE_DEVICES` 0/1 per node, `DGPP_ROCE_DEVICES=rxe0`,
  dense_weights fp8, MTP on once the draft walk lands, memory plan defaults
  from the family entry.
- **sm_120a**: the CMake hard-sets `CMAKE_CUDA_ARCHITECTURES=121a` (GB10);
  make it overridable (`-DCMAKE_CUDA_ARCHITECTURES=120a` for this node) —
  no preset change. The FP4/NVFP4 inline asm (e2m1x2 cvt, FP4 MMA) is
  sm_120a-capable; any 121-only instruction surfaces as a compile error and
  gets adapted then. First build on this box is the gate.

## 5. Validation

1. **Reference dump + cold forward check**: `tools/qwen_reference_dump.py`
   gains the dense variant (plain residual, swish GDN gate, select-all
   attention, dense MLP; MTP per the vLLM reference above) and runs in the
   vLLM image (transformers 5.15 present) against `/nfs` BF16.
   `apps/qwen_forward_check.cpp` gates logits/hidden parity as in
   `docs/testing.md`.
2. **Bus transport**: two-endpoint loopback over rxe0 on this box
   (the fabric test suite's local mode), then the full engine on 2 GPUs.
3. **Serving parity**: greedy transcripts vs the live vLLM NVFP4 server
   (:8000) over a fixed prompt set — same tokenizer, greedy, compare
   tokens (the deployments differ in quant, so exact-match is the gate for
   short prompts and prefix-divergence stats for long ones).
4. **End-to-end**: OpenAI-API smoke on the cluster template, decode
   throughput recorded in `docs/benchmarks/` once stable.

## 6. Staging

- P0: sm_120a build + host test suite on this node; rxe link + bus loopback.
- P1: architecture arm + config (optional GR/indexer/MoE, swish gate) — the
  `/nfs` config parses and the tensor census binds.
- P2: loader/binding + dense MLP + residual site + select-all attention.
- P3: MTP variant.
- P4: family entry + cluster template + first 2-GPU boot, memory plan fitted.
- P5: validation gates (forward check, serving parity), benchmarks, notes.

### 6.1 Status (2026-10-03)

P1–P3 landed and gated on the dense twin fixture (`qwen_forward_test
--write-fixture DIR --dense`): smoke (finite, deterministic) and the full
13-gate `qwen_decode_test` pass, including graph parity, teacher-forced MTP
verification and bf12 companions; the Flash-Next suite stays green. Deviations
and traps found on the way, all resolved in-tree:

- The draft fc runs the Flash-Next two-projection + fuse chain with hc=1
  (§2's split note) — the fused single-GEMM variant was built and dropped:
  identical shape to Flash-Next code won (draft-side tolerance, not bitwise).
- `glm_embed_bcast_kernel` hardcoded GLM's four residual branches; it now
  takes a branch count (GLM call sites pass 4, the dense form 1).
- Select-all attention: `attn_partial_kernel`'s `resolve` chased the null
  `topk` list; the contiguous form reads base+index, as
  `qsa_select_all_counts` counted. The warp prefill kernel already had it.
- The plain site's first `build_layer_objects` created the objects but never
  bound the per-layer norms (the GR branch needed no first-pass bind).
- `run_rows` computed `W = hc_count * H` (0 in dense): layer captures and the
  draft's h window sizing now use `hyper_width()`.
- The constructor, `plan_memory` and `graph_prepare` built MoE machinery
  unconditionally (`GlmMoeConfig: n_experts must be positive`, null `moe_`);
  all three gate on `has_moe()`, and the plan adds the dense MLP scratch.

P4/P5 remain: family entry, rxe transport, reference-dump dense variant and
serving parity.

2026-10-03, later: the §4 "not part of this port" NVFP4 MLP follow-up
landed — `engine.dense_weights = "nvfp4"`, the at-load modelopt triple for
all 65 dense SwiGLUs, 4.89 GB a rank at world 2 against fp8's 8.7
(resident 14.0 → 10.25 GB, the KV/graph margin 2.3 → 5.3 GB). The fp4
GEMV's fifth pass (K = 8704), the prefill bridge, the discrete memory
plan and the traps: `docs/qwen38_dual_spark.md`.
