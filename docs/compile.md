# The compile path

`polima compile` turns a build tree of ONNX graphs into MLA ELFs and then into a
deployable bundle. It replaces four scripts:

| Legacy | Replaced by |
|---|---|
| `ACT/scripts/sima_compile_onnx_tensors.py` (158 lines) | `polima/compile/tensor.py` |
| `SmolVLA/scripts/sima_compile_onnx_tensors.py` (235 lines) | ″ |
| `GR00T-N1.6/scripts/sima_compile_onnx_tensors.py` (125 lines) | ″ |
| `SmolVLA/scripts/unpack_sima_mpk.py` | `polima/compile/mpk.py` |

plus the per-graph loops in `compile_deploy_act_som.sh`,
`act_modalix_compile_controller.py` and `smolvla_modalix_compile_controller.py`,
which become `polima/compile/driver.py` driven by `PolicySpec.compile.graphs`.

## The three interpreters

The compiler is never imported, always subprocessed:

    training env      torch + lerobot,  no afe
    model-compiler    afe + onnx,       no torch
    board             numpy,            neither

`polima compile` itself runs in the neutral middle and shells out. That is why
`polima/compile/tensor.py` imports afe *inside* its functions -- the module has
to be importable in all three, and `polima doctor --imports` proves it is.

The compiler venv has no polima installed, so the driver puts `src/` on its
`PYTHONPATH` and runs `python -m polima.compile.tensor`.

## Reproduction proof

The three legacy copies were unified as a union, not a rewrite: where they
disagreed, the behaviour that produced the currently-deployed ELFs is the
default and the other is a flag. The check is exact rather than approximate.

Driving `polima/compile/driver.py` over the same build tree that produced the
deployed ACT bundle regenerates every ELF **byte for byte**:

| graph | sha256 (first 16) | bytes | compile |
|---|---|---|---|
| `vision_backbone` | `4a7a9a8a67f90137` | 31,027,744 | 163 s |
| `encoder_layer_00_stem` | `7184354b6ffc517b` | 17,514,096 | 93 s |
| `encoder_layer_01` | `09609dedc6746fc2` | 17,189,232 | 90 s |
| `encoder_layer_02` | `9058ad2b8b8c8e77` | 17,189,232 | 91 s |
| `encoder_layer_03` | `4b532ed0b4fc7e0c` | 17,189,232 | 91 s |
| `decoder_action_tail` | `b1eece6992dbddc6` | 13,384,840 | 40 s |

ModelSDK 2.1.0, bf16, all six on the first precision attempt. ~9.5 minutes for a
cold build of the whole policy.

Byte identity is a stronger claim than a numerical tolerance and a cheaper one
to check, so it is the standing regression test for this path. It also
establishes that afe compilation is deterministic given identical inputs, which
is what makes the content-keyed resume below sound.

### Reproduced again, from the Palette container

The same check on a *second* checkpoint (`gewb_2_final`, the grey-eraser task),
compiled inside the running Palette container rather than on the host, against an
independent reference: the ELFs the legacy build had already deployed to the
board.

| graph | sha256 (first 16) | compile |
|---|---|---|
| `vision_backbone` | `c326e846fd3d73f8` | 112 s |
| `encoder_layer_00_stem` | `885e0eff650e2197` | 61 s |
| `encoder_layer_01` | `15d2ce815420bf42` | 60 s |
| `encoder_layer_02` | `6b4b0c90538cf3dd` | 58 s |
| `encoder_layer_03` | `cba658fa833f194b` | 58 s |
| `decoder_action_tail` | `efea65bedcc31cf5` | 30 s |

All six identical to `/media/nvme/polima/models/ACT_gewb_100000/`. So byte
identity holds across checkpoints and across the host/container boundary, which
is what makes the content-keyed resume safe to trust.

### Running it in the Palette container

The container is long-lived and already mounts the workspace and the compiler,
so this is `docker exec`, not `docker run`:

```bash
docker exec -it <palette-container> bash -lc '
  export PATH=/workspace/MLSandbox/polima/bin:$PATH
  polima compile --build-dir <dir>
'
```

Note the image alone is not enough: a bare `docker run` of the same image fails
with `libLLVM.so.18.1: cannot open shared object file`, because the provisioned
container has libraries the image does not. Compiling belongs in the container
you already have, not a fresh one.

To re-run it:

```bash
REF=ACT/outputs/modalix_rcwb_f_t_act_100000_llima
mkdir -p /tmp/repro/onnx /tmp/repro/calibration
cp $REF/onnx/*.onnx /tmp/repro/onnx/
cp $REF/calibration/*.npz /tmp/repro/calibration/
polima compile --build-dir /tmp/repro --stop-after compile
for g in vision_backbone encoder_layer_00_stem encoder_layer_01 \
         encoder_layer_02 encoder_layer_03 decoder_action_tail; do
  cmp $REF/retained/$g/${g}_stage1_mla.elf \
      /tmp/repro/retained/$g/${g}_stage1_mla.elf && echo "match $g"
done
```

## What the three copies disagreed on

Everything was kept; nothing was chosen over anything else except one default.

**Calibration layout.** The curated `quantize_compile` helper transposes
calibration data NCHW→NHWC unconditionally, which is right for image models and
corrupts SmolVLA's rank-2/rank-3 action tensors. The rule all three converge on
is narrower: transpose only rank-4, and only when the model is NCHW. It is now
in one place (`calibration.to_calibration_layout`) and applied identically to
all three calibration sources, so the source cannot change what the layout flag
means.

**Shape inference.** SmolVLA runs `onnx.shape_inference` after simplification;
ACT does not. Both work for their own graphs, but inference rewrites the proto's
`value_info`, so enabling it everywhere would change the bytes handed to afe --
and ACT's ELFs are the reproduction target. It is therefore opt-in
(`--infer-shapes`), requested per graph. Re-stamping the input dims, which is
the other half of what SmolVLA added, *is* unconditional: it is a no-op whenever
onnxsim honoured `overwrite_input_shapes`, and cheap insurance against a
symbolic dim surviving into afe, where the resulting error names neither the
input nor the shape.

**Precision.** ACT and GR00T try int8 then fall back to bf16; that ordering is
per graph in `GraphSpec.precisions`. SmolVLA's split weight/activation precision
is available as `--activation-precision` / `--weight-precision`.

## Success is an ELF, not an exit code

afe can return 0 having quietly placed a graph on the APU, producing no MLA ELF.
Both the ACT and GR00T controllers check for this, and so does the driver: a
compile counts only if `locate_elf` finds one. Otherwise it tries the next
precision and, failing that, reports which log to read.

The same applies inside an mpk — `mpk.has_elf` exists because an archive can be
well-formed and empty of ELFs.

## Resume is content-keyed

The legacy `--resume` checked whether an output file existed, so re-exporting a
graph and re-running kept the stale ELF. `Driver.stage_key` hashes the ONNX, the
calibration data, the compile flags and the ModelSDK version:

* re-export a graph → its key changes → it recompiles, others are reused
* upgrade afe → every key changes → everything recompiles
* move or rename the build tree → keys unchanged (no absolute paths in the key)

This generalizes SmolVLA's `--reuse-vision-dir` flag, whose purpose was to skip
recompiling a frozen backbone. Nothing has to be declared reusable; a stage that
did not change is not rebuilt. `--force` overrides.

## Build tree layout

Deliberately identical to the legacy trees, so the Phase-1a bundle packer
consumes a fresh tree unchanged and a legacy tree stays importable:

    onnx/<name>.onnx                       input  (export stage)
    calibration/<name>.npz                 input  (export stage)
    compiled/<precision>/<name>/           afe's own output
    retained/<name>/<name>_stage1_mla.elf  the ELF the board runs
    models_uncompressed/<name>/share/      what bundle packing reads
    logs/compile_<name>_<precision>.log
    compile_state.json                     resume keys
    artifact_manifest.json
