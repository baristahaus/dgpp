# The mixed NVFP4 release on a 16 GiB card — 2026-10-04

Release: `/home/barista/models` (`Qwen3.8-27B-NVFP4-MoE-NVFP4-Mix`). Node:
`hephaestus`, 2× RTX 5070 Ti 16 GiB, PCIe Gen5 x8 wired, Soft-RoCE `rxe0`.
Branch `5070ti`, commits `d94093f` (the mixed forms), `50fe518` (the serve
flags), `ae760d3` (the fixture's global). Pushed.

## What the release actually is (read off the files, not assumed)

* 64 layers: NVFP4 MLPs everywhere; **channel fp8 attention** in the 8
  stragglers 0,1,6,7,22,23,38,39 (`weight` U8 e4m3 [N,K] + `weight_scale`
  BF16 [N,1]); `layer_types` is 48 `linear_attention` (GDN) + 16
  `full_attention`.
* NVFP4 triple per matrix: `weight_packed` U8 [N,K/2], `weight_scale`
  F8_E4M3 [N,K/16] row-major k-major, `weight_global_scale` F32 [1]
  (layer 0: gate/up 6400, down 2752) plus a note-read `input_global_scale`
  (the release ships W4A4; this engine runs W4A16 off the BF16 activations).
  **Dequant is `code·s/weight_global_scale` and the kernels divide**, so the
  loader stores the value DIRECT. Storing the reciprocal (what the loader
  did) squares the error: the mixed MLPs came out ~4e7× too large, the walk
  froze on one garbage residual and the lm head wrote exact zeros — the
  served token was a uniform draw. The fixture's globals were ~1.0, which
  is why the gate never saw it; the fixture now carries 495.
* Channel scales **multiply** and are stored direct in both loaders
  (`load_fp8_channel_rows` widens BF16→F32 with no inversion) — verified
  against the real tensors through the stream's dequant path, all 64 layers.
* The draft is BF16 in its own `model_mtp.safetensors` (0.31 GiB with `fc`).
  `generation_config.json` carries a `dflash_config`, so DFlash is a real
  option for this family, but `~/dflash2-zdz621` is not on this machine —
  the drafter weights were never downloaded here, so today's draft is MTP.

## What runs, and how fast

| shape | plan | runs? |
|---|---|---|
| world 1, streaming, no draft | 8.20 GiB device + 3.72 GiB pinned | **yes** — answers correctly ("Paris", "101, 103, 107"), **0.3–0.6 tok/s**: a streamed 20.3 GiB stack crosses PCIe once per token |
| world 1, resident (the draft needs residency) | 20.95 GiB of weights | no: 15.51 GiB card |
| world 2 (both cards), resident + decode graph + MTP | 14.82 GiB device, 3.06 GiB pinned | refused: 14.82 + 1.00 headroom > 15.28 free — short by ~0.5 GiB at `default_max_tokens` 512 |

The next shape to try is world 2 with `default_max_tokens` 256,
`prefix_cache_gib` 0, `kv_capacity` 8192 and the spec-snapshot rows cut:
the shortfall is half a GiB and lives in the activation scratch, the dense
bridge and the snapshot rows. The drafter would then need
`scripts/download-dflash2.sh` into `~/dflash2-zdz621` (3.7 GB, HF token).

## The serve-flag bug this campaign died on

`apps/dgpp_serve.cpp` used `graph_world` (the CUDA-graph knob, true at any
world when `decode_graph` is on) as the **fabric** flag in the memory doctor
and in `build_model`. The family keys residency and head sharding off
fabric, so a single card asked for the resident, vocab-sharded image: the
doctor refused shapes that stream fine, and the boot died in a
`weight_build` cudaMalloc during the warmup while the plan claimed it
fitted. Fabric is `world > 1` now, and the qwen3_5 family carries the draft
at world 1 too (`--mtp` still requires `--decode-graph`, the engine's rule).
The streaming branch of `plan_memory` also never carried the host staging,
so the plan read ~3.6 GiB smaller than the boot.

## Reproduce

```sh
cd ~/zmodern/dgpp
./build-release/qwen35_forward_test --smoke /home/barista/models    # real-config gate
DGPP_QWEN35_TRACE=1 DGPP_Q35_SMOKE_T=8 \
  ./build-release/qwen35_forward_test --smoke /home/barista/models  # per-layer walk
./build-release/dgpp-serve --config deploy/serve_qwen3.8-27b_nvfp4_w1.json --memory-plan
nohup ./dgpp-mixed-run.sh > ~/dgpp-mixed.log 2>&1 &                 # streaming, :18080
```
