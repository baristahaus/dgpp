# Qwen3.8-27B NVFP4 mixed release on model35

The `unsloth/Qwen3.8-27B-NVFP4` checkpoint (compressed-tensors) is a second
weight form of the dense 27B family, and the one that fits this node's
16 GiB cards with room for KV. The upstream model35 family (merged in
d43d125) implements the FP8 block release only; the `Nvfp4Mixed` quant kind
is parsed (`config35.cpp`) but nothing binds, loads or runs it yet. This
plan is the implementation, stage by stage, with the invariants called out.

## The checkpoint (verified against the safetensors index)

```
layers 0-55   linear_attn (GDN) or self_attn (Full), per config35's table:
              attn projections  F8_E4M3 [n, k]  + BF16 weight_scale [n, 1]   (CHANNEL fp8)
              mlp.*             weight_packed U8 [I, H/2]                    (NVFP4)
                                weight_scale F8_E4M3 [I, H/16]
                                weight_global_scale F32 [1]
                                input_global_scale F32 [1]
layers 56-63  attn the same channel fp8; the MLP is channel fp8 too:
              weight F8_E4M3 [I, H] / [H, I] + BF16 weight_scale [rows, 1]
lm_head       F8_E4M3 [248320, 5120] + BF16 weight_scale [248320, 1]  (channel)
embed_tokens  BF16 [248320, 5120]
model_mtp     all BF16 (self_attn q/k/v/o, mlp gate/up/down, fc, norms)
visual.*      ignored (text-only scope, same as the FP8 release)
```

Channel fp8 = one e4m3 scale per weight row: `GlmQuantMatrix` with
`scale_block_rows = 1, scale_block_cols = cols`, scales `[rows]` F32.
The grid log2 form: `rs = 0, cs = ceil_log2(cols)` — `scale_cols = 1`, so
every 16-byte chunk sits inside one scale column (`cs >= 4` holds: the
widths are 5120/6144/8704/17408 → cs 13/13/14/15).

## What the release means numerically

- The fp4 MLPs are **W4A16**: `fp4_gemv` (decode rows) and
  `launch_dense_mma_fp4_*` consume bf16 activations; weights dequantize
  exactly (e2m1 x e4m3 is exact in f16, one global-scale multiply).
  `input_global_scale` is a calibrated activation scale for a W4A4 path we
  do not run: note-read and discard, like the MOE loader's `input_scale`.
- The global scale convention matches the MOE release DGPP already serves:
  `weight_global_scale` is the reciprocal-form scale (modelopt
  `weight_scale_2` renamed); the loader stores its reciprocal into the
  bump and the kernels multiply. `load_fp4_rows_mo` is the precedent; the
  dense triple only differs in tensor names (`weight_packed`, not
  `weight`).
- Bitwise pins: a site's per-row chain must be stable across batch sizes.
  - fp4 MLP: one kernel family per site (`dense_mma_fp4`) at **every** row
    count — its per-row MMA chain is m-independent, so decode, verify and
    prefill are bitwise each other (stronger than the fp8 form, which
    reorders across its GEMV/MMA/tile bounds under the near-tie rule).
  - channel fp8 sites: the grid launchers' GEMV/tile forms apply per the
    same near-tie rule as the block form today.
- `k_scale` / `v_scale` ([1] BF16, per full-attn layer) are the fp8-KV
  hint. KV stays BF16 this release: note-read and discard. The 8-bit KV
  pool is a separate follow-up.

## Stage 1 — kernels: generalize the grid bounds

- `launch_scale_gemm_grid_*` relaxes `rs/cs ∈ 5..7` to `rs ∈ 0..7`,
  `cs ∈ 4..15` (all three consumers — the tile kernel, the fp8 GEMV chunk
  core, `mma_gemv` — already index `scales[(n >> rs) * scale_cols +
  (k >> cs)]` generically; the bound was a dsv41-era range check).
  Validation: `2^rs == scale_block_rows` (power of two), `cs` from
  `ceil_log2(scale_block_cols)` with the `cs >= 4` floor.
- `launch_fp8_dequant_blocks` gains `rs`/`cs` (defaults 7/7): the flat
  elementwise kernel indexes `scales[(n >> rs) * scale_cols + (k >> cs)]`.
  Channel dequant: rs=0, cs=13..15.
- No fp4 kernel changes: `check_fp4_mma_shape` takes any k multiple of 16,
  so `dense_mma_fp4` runs k = 5120 / 8704 / 17408 today. The fp4 GEMV
  (kMaxK 16384, compiled set without 8704/17408) is **not** used — the
  down projection's 17-chunk rows need five passes and 34 KiB single-row
  staging; the MMA tile reads the matrix once at any m and is the better
  decode form anyway (the fp8 down's own 2026-10-03 note: weights once,
  per-row chain m-independent).

## Stage 2 — binding35: form-aware expectation tables

`qwen35_expected_*` take the quant kind:

- `expect_gdn35` / `expect_full35`: under Nvfp4Mixed register the channel
  pair (`weight` F8_E4M3 [n, k] + `weight_scale` BF16 [n, 1], roles
  ChannelFp8Payload/Scale); under Fp8Block the block pair (`_scale_inv`)
  as today. New roles, not Fp8Payload/Fp8Scale — the loader dispatches on
  them and the digest must distinguish the forms.
- `expect_dense_mlp35`: Nvfp4Mixed + layer < 56 → the fp4 quadruple
  (`weight_packed`, `weight_scale`, `weight_global_scale`,
  `input_global_scale`, one shared set per matrix); layers 56-63 → the
  channel pair; Fp8Block → the block pair as today.
- MTP layer under Nvfp4Mixed: BF16 self_attn q/k/v/o + BF16 mlp (the
  release's draft is unquantized) — the Fp8Block table's MTP rows were
  fp8.
- Globals: `lm_head` under Nvfp4Mixed is the channel pair; `embed_tokens`,
  the final norm, and the MTP fc/norm rows stay BF16.
- `k_scale`/`v_scale` rows (Full layers): note-read and discard; register
  them so the binding table accounts for their bytes.
- `loader_format()` 2 → 3 under Nvfp4Mixed (Fp8Block images stay valid).

## Stage 3 — loader35: the three load paths

- `load_fp8_channel_rows/cols`: e4m3 rows + per-row BF16 scale widened to
  F32, `GlmQuantMatrix{scale_block_rows=1, scale_block_cols=k}`. Row
  slicing is trivial (any bound); column slicing keeps whole rows.
  Channel `_cols` (o_proj, out_proj, down) slices columns of every row —
  the scale stays per row, so no scale surgery at all.
- `load_fp4_dense(base, rows/cols slice)`: the MOE loader's byte-copy
  packing with the release's names. Rows slice for gate/up; columns slice
  16-aligned for down (scales slice at cols/16).
- MTP under mixed: bf16 loads into the existing bf16 fields
  (`QwenFullAttnResident::q_proj` … , new bf16 mlp fields +
  `Qwen35DenseMlpResident::form`), the fp8 fields stay null.
- The lm head loads as a channel `GlmQuantMatrix` (globals side) under
  mixed; `globals_.lm_head` (bf16) stays for Fp8Block.
- Act/kv hint tensors: `input_global_scale` and `k_scale`/`v_scale`
  note-read + discard (counting pass accounts their bytes).

## Stage 4 — layers / model35: the form dispatch

- `gemm_dense` routes fp8 through the grid launchers with (rs, cs) from
  the matrix; the `prefill_fp8_gemm` opt-in stays block-128-gated (off
  for channel); the bridge dequant becomes grid-aware. One edit covers
  GDN, Full attention and every shared projection.
- GDN / Full decode multi-problems: `Fp8GemvProblem.rs/cs` per matrix
  (block: 7/7; channel: 0/13..15).
- `Qwen35Model::dense_mlp` dispatches on the resident's form:
  - Fp4: `launch_dense_mma_fp4_bf16` x3 + `qwen35_swiglu_bf16`, every
    row count, no PT, no bridge.
  - Fp8 (block or channel): today's path; grid dequant in the bridge,
    grid launchers on the streaming tail, down keeps the
    weights-once form.
  - Bf16 (MTP under mixed): `gemm_.matmul` x3 + swiglu.
- `head_gemv` under mixed: rows ≤ 128 the grid GEMV/MMA on the channel
  head (f32 out); wider rows the chunked bridge (dequant V-chunks into
  `gw_.dequant` + Lt), since there is no BF16 head to fall back to and
  the plain grid tile kernel measured behind Lt at wide m.
- PT recipe under mixed: attention slots requant with grid dequant
  (channel) or direct bf16 maxabs (MTP draft); **MLP PT slots are off**
  (fp4 layers have no recipe; the eight channel layers bridge). The
  65-slot x 3 x IH allocation disappears with them.
- `plan_memory` / the resident image digest pick up the form via
  `loader_format()` + the binding-table roles.

## Stage 5 — validation

1. `qwen_load_check` on the mixed checkpoint, world 1: binding report,
   byte accounting, resident-image save/restore round trip.
2. `qwen_forward_check` walks: greedy transcript equality against vLLM
   (`qwen3.8-27b-nvfp4`, the container already on this box) — the same
   gate the fp8 form shipped with, now on the mixed weights.
3. World 2 across the two 5070 Ti: the walks + `dflash2` spec-decode
   transcripts (the draft is bf16 under this release — its own pin).
4. `bench_qwen35.sh` cells vs the FP8 release's record and the vLLM
   baseline; record in `docs/measurements.md` and the serving notes.

## The fit (why this release)

Per card, world 2 (weights only, from the index): fp4 MLPs 0-55 ≈ 4.1 GB,
channel MLPs 56-63 ≈ 1.1 GB, channel attn ≈ 3.1 GB, embed 1.27 GB, fp8
head 0.64 GB, bf16 MTP ≈ 0.31 GB → ≈ 10.5 GB against the 16 GiB card's
~15 GiB usable after context — KV and activations fit with margin the
fp8-block release never had (its MLPs alone were 8.3 GB per card).
