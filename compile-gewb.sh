#!/usr/bin/env bash
# Compile the latest ACT checkpoint (gewb_2_final, grey eraser) and pack a bundle.
#   ./compile-gewb.sh                 export + compile + pack   (~10 min cold)
#   ./compile-gewb.sh --stop-after export     just check the ONNX  (~32 s)
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
POLIMA_ROOT="$ROOT/polima"
POLIMA_OUTPUTS_ROOT="${POLIMA_OUTPUTS:-$POLIMA_ROOT/outputs}"
CHECKPOINT="${CHECKPOINT:-$ROOT/ACT/outputs/gewb_2_final_act_20260806_195153/checkpoints/100000/pretrained_model}"
BUILD_DIR="${BUILD_DIR:-$POLIMA_OUTPUTS_ROOT/build/polima_gewb_act_100000}"

exec "$POLIMA_ROOT/bin/polima" compile \
    --checkpoint "$CHECKPOINT" \
    --build-dir "$BUILD_DIR" \
    "$@"
