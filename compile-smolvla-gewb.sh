#!/usr/bin/env bash
# Build a native PoLiMa SmolVLA bundle for the GEWB checkpoint.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
POLIMA_ROOT="$ROOT/polima"
POLIMA_OUTPUTS_ROOT="${POLIMA_OUTPUTS:-$POLIMA_ROOT/outputs}"
CHECKPOINT="${CHECKPOINT:-$ROOT/SmolVLA/outputs/gewb_2_final_single_task_smolvla_20260806_194521/checkpoints/045000/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-/ml_datasets/gewb_2_final}"
BUILD_DIR="${BUILD_DIR:-$POLIMA_OUTPUTS_ROOT/build/polima_gewb_smolvla_045000}"
DATASET="${DATASET:-gewb_2_final}"
STEPS="${STEPS:-45000}"
COMPILER_PYTHON="${COMPILER_PYTHON:-/home/timothydo/sima-sdk-extensions/model-compiler/bin/python}"

[[ -f "$CHECKPOINT/model.safetensors" ]] || { echo "No checkpoint: $CHECKPOINT" >&2; exit 1; }

MODEL_COMPILER_BIN="$(dirname -- "$COMPILER_PYTHON")" exec "$POLIMA_ROOT/bin/polima" compile \
    --policy smolvla \
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --build-dir "$BUILD_DIR" \
    --dataset "$DATASET" \
    --steps "$STEPS" \
    "$@"
