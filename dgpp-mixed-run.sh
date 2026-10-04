#!/bin/bash
# dgpp-mixed-run.sh — launch dgpp-serve on the mixed NVFP4 Qwen3.8-27B release.
#
# Streaming residency on one RTX 5070 Ti: the release's 64 layers (20.3 GiB)
# do not fit a 16 GiB card, so one layer streams at a time and the draft
# stays off (DFlash requires the resident image; see NOTES.md).
#
#   ctx per request = engine.kv_capacity (65536 here, the shared pool)
#   launch:           nohup ~/zmodern/dgpp/dgpp-mixed-run.sh > ~/dgpp-mixed.log 2>&1 &
#   verify:           curl -s localhost:18080/v1/models
set -euo pipefail
cd "$HOME/zmodern/dgpp"
export DGPP_LOG_LEVEL="${DGPP_LOG_LEVEL:-info}"
exec ./build-release/dgpp-serve \
  --config deploy/serve_qwen3.8-27b_nvfp4_w1.json \
  --bind-host 127.0.0.1 --port 18080
