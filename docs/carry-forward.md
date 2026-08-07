# Legacy fixes PoLiMa must not lose

The four legacy stacks encode operational knowledge that was learned the hard
way, often as one line in a shell script. A rewrite silently drops that unless
it is written down. Everything here is a real fix in `ACT/` or `SmolVLA/` that
the corresponding PoLiMa module has to reproduce.

Check this list when implementing each phase.

## Camera pixel format — `fourcc: MJPG`   [DONE]

Both robot launchers pass `fourcc: MJPG` in `--robot.cameras`:

    ACT/scripts/start_act_robot_client_on_som.sh
    SmolVLA/scripts/start_smolvla_robot_client_on_som.sh
    SmolVLA/inference_smolvla.sh          (CAMERA_CONFIG_REMOTE)

Two 640x480@30 USB cameras on one controller exceed the bus budget in
uncompressed YUYV, so the stream silently drops frames. Nothing errors; control
just degrades.

**In PoLiMa:** `RobotSpec.camera_fourcc` (default `MJPG`) and
`RobotSpec.camera_config()`, which is the single place that renders the draccus
blob. The legacy stacks each hand-wrote it, which is why the fix had to be
applied twice.

## `LD_LIBRARY_PATH` for TorchCodec   [Phase 1c]

    ACT/install_act_env.sh
    ACT/train_act_local.sh

    conda activate "$ENV_NAME"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

TorchCodec dlopens FFmpeg's versioned shared libraries at runtime. Installing
the conda `ffmpeg=7.1.1` package is not enough — its `lib/` must be on the
runtime linker path, or `from torchcodec.decoders import VideoDecoder` fails
with a bare import error that says nothing about FFmpeg.

Note this also means `conda run` is insufficient: the variable has to be exported
into the activated environment.

**In PoLiMa:** `polima/train/env.py::conda_run` must export it after activating.
`train_act_local.sh` also preflights the import and points at the installer,
which the TrainRunner should keep.

## SO-101 CLI registration   [Phase 1d]

    ACT/run_robot_client_with_live_view.py
    SmolVLA/run_robot_client_with_live_view.py

    import lerobot.robots.so_follower  # noqa: F401  # Register SO-100/SO-101 CLI choices.

A side-effect import. Without it, draccus does not know `--robot.type=so101_follower`
and argument parsing fails before anything runs.

**In PoLiMa:** `polima/robot/client.py` needs the same import, and it must
survive linting — it looks unused.

## `--calibrate` with a backup   [Phase 1d]

Both on-board launchers grew a `--calibrate` flag that prompts, timestamps a
backup of the existing calibration, then runs `lerobot-calibrate`:

    cp -p "$calibration" "$calibration.bak.$(date +%Y%m%d_%H%M%S)"
    lerobot-calibrate --robot.type=so101_follower --robot.port=... \
        --robot.id=so-arm101 --robot.calibration_dir=...

The backup matters: a bad calibration run otherwise overwrites a working one
with no way back, and the arm then moves incorrectly.

**In PoLiMa:** `polima robot --calibrate`, keeping the confirmation prompt and
the timestamped backup. `RobotSpec.supports_calibrate` / `calibration_id`.

## Policy-server working directory   [Phase 4]

    SmolVLA/inference_smolvla.sh   (remote-server mode)

    # The checkpoint processor references its bundled tokenizer as the
    # relative path `tokenizer`; resolve that path from the checkpoint.
    if [[ -d "$POLICY" ]]; then cd "$POLICY"; fi

SmolVLA checkpoints reference their tokenizer by a *relative* path, so the
server must run with the checkpoint as cwd. Otherwise loading fails somewhere
inside transformers with an unhelpful path error.

**In PoLiMa:** whatever launches a SmolVLA policy server sets cwd to the
checkpoint.

## Camera focus is per-rig, not per-policy

`SmolVLA/.camera_focus_config.json` moved `focus` 35 -> 30. This is a property of
the physical camera and lens, not of the policy, and both stacks keep their own
copy that can drift.

**In PoLiMa:** one `robot/camera_focus_config.json` on the board, referenced by
config rather than duplicated per policy.

## Compiled stages must be reusable across runs   [Phase 1b]

    SmolVLA/scripts/compile_deploy_smolvla_som.sh   --reuse-vision-dir

The vision tower is the most expensive stage to compile and the one that changes
least: it is the frozen SmolVLM backbone, so it is identical across every
fine-tune of the same base. Recompiling it on each iteration of the prefix /
suffix / denoise stages wastes the bulk of the wall clock, and the script notes
those stages already have to run sequentially because each needs substantial
host RAM.

The flag skips the compile and `rsync`s a previously unpacked
`models_uncompressed/vision_llima_bf16` into place instead. Note it also has to
move the *existence check* for that stage, since `$VISION_ONNX` need not exist
when the compiled output is being reused.

**In PoLiMa:** this is a general property, not a SmolVLA quirk -- ACT's vision
backbone has the same character. The compile driver should key each stage by a
content hash of (onnx, calibration input, compile flags) and skip stages whose
key is unchanged, so reuse is automatic rather than a manual flag. That is the
same content-addressing already used for bundles and for the native build skip.

## Task strings belong to the dataset

The SmolVLA task moved from "Place the red cube in the black basket." to
"Place the grey eraser in white basket." across five files. The dataset already
carries it: `/ml_datasets/gewb_2_final` reports
`place the grey eraser in white basket` via `meta/tasks.parquet`.

**In PoLiMa:** default the task from the dataset's own label rather than
hardcoding it per script, so it cannot drift out of sync with the checkpoint.
`polima data validate --json` already surfaces it as `tasks[0]`.
