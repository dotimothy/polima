#!/usr/bin/env bash
# Compile the latest ACT checkpoint (gewb_2_final, grey eraser) and pack a bundle.
#   ./compile-gewb.sh                 export + compile + pack   (~10 min cold)
#   ./compile-gewb.sh --stop-after export     just check the ONNX  (~32 s)
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${CHECKPOINT:-$ROOT/ACT/outputs/gewb_2_final_act_20260806_195153/checkpoints/100000/pretrained_model}"
BUILD_DIR="${BUILD_DIR:-$ROOT/ACT/outputs/polima_gewb_act_100000}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate act

exec "$ROOT/polima/bin/polima" compile \
    --checkpoint "$CHECKPOINT" \
    --build-dir "$BUILD_DIR" \
    "$@"
