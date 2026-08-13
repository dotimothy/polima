#!/usr/bin/env bash
# Build a deployable SmolVLA bundle for the gewb checkpoint.
#
# This IMPORTS an existing compiled tree rather than compiling from the
# checkpoint. SmolVLA's export side (polima/policies/smolvla/graphs.py) is not
# written yet -- the spec and the runtime plan are done and verified against the
# recorded pipeline, but nothing yet turns a checkpoint into ONNX. Until then the
# ELFs come from the legacy compile:
#
#     SmolVLA/scripts/compile_deploy_smolvla_som.sh
#
# What this does give you is a real PoLiMa bundle -- content-addressed id,
# plan.json, the wire and robot descriptions -- served by the same polima_server
# that serves ACT.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/SmolVLA/outputs/modalix_gewb_smolvla_045000_v2/final_som_compile}"
DATASET="${DATASET:-gewb_2_final}"
STEPS="${STEPS:-45000}"

[[ -d "$BUILD_DIR" ]] || { echo "No such build tree: $BUILD_DIR" >&2; exit 1; }

exec "$ROOT/polima/bin/polima" compile \
    --policy smolvla \
    --import-legacy "$BUILD_DIR" \
    --dataset "$DATASET" \
    --steps "$STEPS" \
    "$@"
