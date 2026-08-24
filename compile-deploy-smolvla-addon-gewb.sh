#!/usr/bin/env bash
# Compile, bundle, and deploy the newest addon + GEWB SmolVLA checkpoint.
set -Eeuo pipefail

POLIMA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$POLIMA_ROOT/.." && pwd)"
POLIMA_OUTPUTS_ROOT="${POLIMA_OUTPUTS:-$POLIMA_ROOT/outputs}"

CHECKPOINT="${CHECKPOINT:-$WORKSPACE_ROOT/SmolVLA/outputs/addon_final_gewb_2_final_combined_smolvla_base_20260817_102630/checkpoints/035000/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-$WORKSPACE_ROOT/SmolVLA/outputs/combined_datasets/addon_final_gewb_2_final_combined_20260817_102633}"
BUILD_DIR="${BUILD_DIR:-$POLIMA_OUTPUTS_ROOT/build/polima_addon_final_gewb_combined_smolvla_035000}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$POLIMA_OUTPUTS_ROOT/bundles}"
BOARD="${BOARD:-sima@192.168.91.211}"
JOBS="${JOBS:-1}"
COMPILER_PYTHON="${COMPILER_PYTHON:-/home/timothydo/sima-sdk-extensions/model-compiler/bin/python}"

DEPLOY=true
START=true
BUILD_SERVER=true
REUSE=false

usage() {
    cat <<EOF
Compile, bundle, and deploy the newest addon + GEWB SmolVLA checkpoint.

Usage: $(basename "$0") [options]

  --checkpoint PATH     checkpoint to export
  --dataset-root PATH   LeRobot dataset used for the reference fixture
  --build-dir PATH      ONNX/compiler build tree
  --bundle-root PATH    destination for content-addressed bundles
  --board USER@HOST     Modalix board (default: $BOARD)
  -j, --jobs N          parallel graph compiles (default: $JOBS)
  --reuse               reuse graphs whose content key is unchanged
  --compile-only        compile and bundle without deploying
  --no-start            deploy and activate without starting the server
  --no-build            skip rebuilding the native server on the board
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --bundle-root) BUNDLE_ROOT="$2"; shift 2 ;;
        --board) BOARD="$2"; shift 2 ;;
        -j|--jobs) JOBS="$2"; shift 2 ;;
        --reuse) REUSE=true; shift ;;
        --compile-only) DEPLOY=false; shift ;;
        --no-start) START=false; shift ;;
        --no-build) BUILD_SERVER=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -f "$CHECKPOINT/model.safetensors" ]] || {
    echo "Missing SmolVLA checkpoint: $CHECKPOINT" >&2
    exit 1
}
[[ -f "$DATASET_ROOT/meta/info.json" ]] || {
    echo "Missing LeRobot dataset: $DATASET_ROOT" >&2
    exit 1
}
[[ -x "$COMPILER_PYTHON" ]] || {
    echo "Missing SmolVLA export/compiler Python: $COMPILER_PYTHON" >&2
    exit 1
}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
    echo "--jobs must be a positive integer" >&2
    exit 2
}

compile_args=(
    compile
    --policy smolvla
    --checkpoint "$CHECKPOINT"
    --dataset-root "$DATASET_ROOT"
    --build-dir "$BUILD_DIR"
    --output-root "$BUNDLE_ROOT"
    --dataset addon_final_gewb_2_final_combined
    --steps 35000
    --jobs "$JOBS"
)
if [[ "$REUSE" == true ]]; then
    compile_args+=(--reuse)
fi

compile_log="$(mktemp)"
trap 'rm -f -- "$compile_log"' EXIT
MODEL_COMPILER_BIN="$(dirname -- "$COMPILER_PYTHON")" "$POLIMA_ROOT/bin/polima" \
    "${compile_args[@]}" | tee "$compile_log"

BUNDLE="$(awk '$1 == "root" { path=$2 } END { print path }' "$compile_log")"
[[ -f "$BUNDLE/bundle.json" ]] || {
    echo "Compile succeeded but no bundle path was reported" >&2
    exit 1
}

if [[ "$DEPLOY" == false ]]; then
    echo "SmolVLA bundle ready: $BUNDLE"
    exit 0
fi

deploy_args=(deploy --bundle "$BUNDLE" --board "$BOARD")
if [[ "$START" == true ]]; then
    deploy_args+=(--start)
fi
if [[ "$BUILD_SERVER" == false ]]; then
    deploy_args+=(--no-build)
fi
"$POLIMA_ROOT/bin/polima" "${deploy_args[@]}"
