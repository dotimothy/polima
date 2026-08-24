"""Quantize and compile one fixed-shape ONNX graph for a SiMa target.

This replaces the three copies of `sima_compile_onnx_tensors.py` in `ACT/`,
`SmolVLA/` and `GR00T-N1.6/`. Lined up side by side those copies are the same
algorithm with different subsets of the knobs exposed, so this is their union
rather than a rewrite:

  ACT      --calibration-npz, --precision, NCHW rank-4 calibration transpose
  SmolVLA  IO overrides, --calibration-input-file, gen1/gen2, split precision,
           --no-compile, --calib-method, --requant-mode, shape re-inference
  GR00T    (a strict subset of ACT; no tessellation, NCHW only)

Nothing was dropped. Where the copies disagreed, the behaviour that produced the
currently-deployed ELFs is the default, and the other is a flag -- see
`prepare()` for the one case where that distinction matters.

## Where this runs

Under the SiMa model-compiler venv (python 3.12 + afe + onnx + onnxsim, and
notably *no* torch), not under the training env. That is why `afe` is imported
inside the functions that need it: `polima doctor` and the unit tests import
this module in environments where afe does not exist, and they only need the
argument surface and the plan, not the compiler.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import threading
import time
from pathlib import Path
from typing import Sequence

from polima.compile import calibration as calib

Shape = tuple[int, ...]


# --------------------------------------------------------------------- onnx io


def detect_io(path: str | Path) -> tuple[list[str], list[Shape], list[str]]:
    """Read input names/shapes and output names straight off the graph.

    `load_external_data=False` matters for the vision backbones, whose weights
    live in a sidecar file: this only ever touches graph metadata, and loading
    the tensors to read a shape costs seconds and gigabytes for nothing.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    names: list[str] = []
    shapes: list[Shape] = []
    for value in model.graph.input:
        shape = tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(
                f"input {value.name!r} of {path} is not static: {shape}. "
                "Re-export with fixed axes; the MLA cannot take a dynamic shape."
            )
        names.append(value.name)
        shapes.append(shape)
    return names, shapes, [value.name for value in model.graph.output]


def prepare(path: str | Path, names: Sequence[str], shapes: Sequence[Shape],
            infer_shapes: bool = False) -> Path:
    """Simplify to a static graph and write `<stem>_tensor_prepared.onnx`.

    `infer_shapes` is the one place the legacy copies genuinely disagreed.
    SmolVLA runs `onnx.shape_inference` after simplification; ACT does not. Both
    produce working ELFs for their own graphs, but shape inference rewrites the
    proto's `value_info`, so turning it on unconditionally would change the bytes
    handed to afe for ACT -- and ACT's ELFs are the reproduction target. It is
    therefore opt-in, requested per graph by `GraphSpec`.

    Re-stamping the input dims is unconditional and is a no-op whenever onnxsim
    honoured `overwrite_input_shapes`. It is cheap insurance against the case
    SmolVLA hit, where a symbolic dim survives simplification and afe then fails
    much later with a message that does not mention the shape.
    """
    import onnx
    from onnxsim import simplify

    path = Path(path)
    shape_map = {name: list(shape) for name, shape in zip(names, shapes, strict=True)}
    proto, ok = simplify(str(path), overwrite_input_shapes=shape_map,
                         dynamic_input_shape=False)
    if not ok:
        raise RuntimeError(f"onnxsim validation failed for {path}")

    for tensor in proto.graph.input:
        if tensor.name not in shape_map:
            continue
        tensor.type.tensor_type.shape.ClearField("dim")
        for size in shape_map[tensor.name]:
            tensor.type.tensor_type.shape.dim.add().dim_value = size

    if infer_shapes:
        proto = onnx.shape_inference.infer_shapes(proto)

    # afe's importer rejects newer IR versions; every legacy copy clamps to 8.
    proto.ir_version = min(proto.ir_version, 8)
    prepared = path.with_name(f"{path.stem}_tensor_prepared.onnx")
    onnx.save(proto, prepared)
    return prepared


def promote_rank3_hwc(path: str | Path, layout: str = "NHWC") -> Path:
    """Wrap each rank-3 [N,W,C] boundary as a layout-correct rank-4 tensor.

    The reshapes are numerical no-ops. They only satisfy ModelSDK 2.1's
    four-dimensional external-layout packaging path, which is required for
    directly sharing an HWC MLABuffer between token stages. Rank-4 boundaries
    are left alone, which also covers SmolVLA vision's NCHW image input and
    rank-3 token output. NCHW graphs use [N,C,1,W], so the compiler sees the
    real channel count while an HWC external buffer remains token-major.
    """
    import onnx

    path = Path(path)
    model = onnx.load(str(path))
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("rank-3 HWC promotion requires one input and one output")
    graph_input, graph_output = model.graph.input[0], model.graph.output[0]
    input_shape = [dim.dim_value for dim in graph_input.type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in graph_output.type.tensor_type.shape.dim]
    if len(input_shape) not in (3, 4) or len(output_shape) not in (3, 4):
        raise ValueError(
            f"expected rank-3/rank-4 I/O, got input={input_shape}, output={output_shape}"
        )
    if len(input_shape) == 4 and len(output_shape) == 4:
        return path

    input_name, output_name = graph_input.name, graph_output.name
    rank3_input = input_name + "__rank3" if len(input_shape) == 3 else input_name
    rank3_output = output_name + "__rank3" if len(output_shape) == 3 else output_name
    for node in model.graph.node:
        node.input[:] = [rank3_input if name == input_name else
                         rank3_output if name == output_name else name for name in node.input]
        node.output[:] = [rank3_output if name == output_name else name for name in node.output]

    def rank4(shape: list[int]) -> list[int]:
        if layout.upper() == "NCHW":
            return [shape[0], shape[2], 1, shape[1]]
        return [shape[0], 1, shape[1], shape[2]]

    if len(input_shape) == 3:
        input_shape_name = input_name + "__shape3"
        model.graph.initializer.append(
            onnx.helper.make_tensor(
                input_shape_name, onnx.TensorProto.INT64, [3], input_shape
            )
        )
        model.graph.node.insert(0, onnx.helper.make_node(
            "Reshape", [input_name, input_shape_name], [rank3_input],
            name=input_name + "__unwrap_hwc",
        ))
    if len(output_shape) == 3:
        output_shape_name = output_name + "__shape4"
        model.graph.initializer.append(onnx.helper.make_tensor(
            output_shape_name, onnx.TensorProto.INT64, [4],
            rank4(output_shape),
        ))
        model.graph.node.append(onnx.helper.make_node(
            "Reshape", [rank3_output, output_shape_name], [output_name],
            name=output_name + "__wrap_hwc",
        ))
    for value, shape in ((graph_input, input_shape), (graph_output, output_shape)):
        if len(shape) != 3:
            continue
        value.type.tensor_type.shape.ClearField("dim")
        for size in rank4(shape):
            value.type.tensor_type.shape.dim.add().dim_value = size
    promoted = path.with_name(f"{path.stem}_rank4_hwc.onnx")
    onnx.save(model, promoted)
    return promoted


# -------------------------------------------------------------- tessellation


def tessellation_parameters(quantized):
    """Direct HWC in / HWC16 out for the MLA segment.

    This is what lets the board hand the MLA a plain contiguous buffer and read
    a de-tessellated one back, which is the whole reason `plan.cpp` can be a
    generic interpreter instead of per-model glue.

    It reaches into `quant_model._net.nodes["MLA_0"]`, a private attribute, and
    is the most fragile line in the compile path -- an afe upgrade that renames
    that node breaks it. SmolVLA's copy added an explicit check for exactly that,
    so it is kept here: the failure is otherwise a KeyError with no context.
    """
    from afe.apis.defines import TensorDRAMLayout, TensorTessellateParameters
    from afe.ir.node import node_is_tuple

    nodes = quantized._net.nodes
    if "MLA_0" not in nodes:
        raise RuntimeError(
            f"MLA_0 not found in the quantized net; nodes={list(nodes)[:30]}. "
            "The graph may have fallen back to the APU, or afe renamed the "
            "segment -- check the quantize log for unsupported operators."
        )
    mla_node = nodes["MLA_0"]

    params = {}
    inbound = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0), enable_mla=True, dram_layout=TensorDRAMLayout.HWC
    )
    for input_name in mla_node.input_names:
        params[input_name] = dataclasses.replace(inbound)

    output_node = mla_node.ir.nodes[mla_node.ir.output_node_name]
    output_names = (
        output_node.input_node_names if node_is_tuple(output_node) else [output_node.name]
    )
    outbound = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0), enable_mla=True, dram_layout=TensorDRAMLayout.HWC16
    )
    for output_name in output_names:
        params[f"{output_name}_output"] = dataclasses.replace(outbound)
    return params


def external_layout_parameters(quantized, layout_name: str):
    """Force the same external layout on every MLA input and output."""
    from afe.apis.defines import TensorDRAMLayout, TensorTessellateParameters
    from afe.ir.node import node_is_tuple

    nodes = quantized._net.nodes
    if "MLA_0" not in nodes:
        raise RuntimeError(
            "MLA_0 not found in the quantized net; cannot set an external DRAM layout"
        )
    mla_node = nodes["MLA_0"]
    template = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0), enable_mla=True,
        dram_layout=TensorDRAMLayout[layout_name],
    )
    params = {
        name: dataclasses.replace(template) for name in mla_node.input_names
    }
    output_node = mla_node.ir.nodes[mla_node.ir.output_node_name]
    output_names = (
        output_node.input_node_names if node_is_tuple(output_node) else [output_node.name]
    )
    params.update({
        f"{name}_output": dataclasses.replace(template) for name in output_names
    })
    return params


# --------------------------------------------------------------------- driver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polima-compile-tensor", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("modalix", "mlsoc"), default="modalix")

    parser.add_argument("--precision", choices=("int8", "bf16"), default="bf16",
                        help="sets both sides unless overridden individually")
    parser.add_argument("--activation-precision", choices=("int8", "bf16"))
    parser.add_argument("--weight-precision", choices=("int8", "bf16"))

    parser.add_argument("--calibration-npz", type=Path,
                        help="one array per input, shaped (N, *input_shape)")
    parser.add_argument("--calibration-raw-f32", type=Path,
                        help="flat float32 samples for a single-input graph")
    parser.add_argument("--calib-method", default="mse")
    parser.add_argument("--requant-mode", choices=("sima", "tflite"), default="sima")

    parser.add_argument("--model-layout", choices=("NCHW", "NHWC"), default="NCHW")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mla-tessellation", action="store_true")
    parser.add_argument(
        "--external-dram-layout", choices=("compiler", "HWC", "HWC16"),
        default="compiler", help="force one layout on every MLA input and output",
    )
    parser.add_argument(
        "--promote-rank3-hwc", action="store_true",
        help="wrap [N,W,C] graph I/O as [N,1,W,C] for explicit HWC layouts",
    )
    parser.add_argument("--retain-compile-dir", type=Path,
                        help="keep the compiler temp dir; this is where the ELF lands")
    parser.add_argument(
        "--exit-on-stable-elf", action="store_true",
        help="work around ModelSDK builds that do not return after writing a complete ELF",
    )

    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--infer-shapes", action="store_true",
                        help="run onnx shape inference after simplification (SmolVLA)")
    parser.add_argument("--no-compile", action="store_true",
                        help="quantize and save only, skip ELF generation")

    parser.add_argument("--input-names", nargs="+")
    parser.add_argument("--input-shapes", nargs="+", metavar="D0,D1,...")
    parser.add_argument("--input-types", nargs="+", choices=("float32", "int8"))
    parser.add_argument("--output-names", nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from afe.apis.defines import (
        CalibrationMethod,
        RequantizationMode,
        bfloat16_scheme,
        default_quantization,
        gen1_target,
        gen2_target,
        quantization_scheme,
    )
    from afe.apis.error_handling_variables import enable_verbose_error_messages
    from afe.apis.loaded_net import load_model
    from afe.apis.release_v1 import get_model_sdk_version
    from afe.ir.tensor_type import ScalarType
    from afe.load.importers.general_importer import ImporterParams, ModelFormat

    args = build_parser().parse_args(argv)
    enable_verbose_error_messages()

    names, shapes, outputs = detect_io(args.model_path)
    if args.input_names:
        names = args.input_names
    if args.input_shapes:
        shapes = [tuple(int(p) for p in s.split(",") if p) for s in args.input_shapes]
    if args.output_names:
        outputs = args.output_names
    type_names = args.input_types or ["float32"] * len(names)
    if len(type_names) != len(names):
        raise SystemExit("--input-types must give exactly one type per input")
    input_types = [ScalarType[name] for name in type_names]

    stem = args.model_path.stem
    output = args.build_dir / stem
    output.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Model Compiler Version: {get_model_sdk_version()}")
    print(f"[INFO] Target: {args.device}   Layout: {args.model_layout}")
    print(f"[INFO] Inputs: {list(zip(names, shapes, strict=True))}")
    print(f"[INFO] Outputs: {outputs}")
    print(f"[INFO] Output directory: {output}")

    calibration_shapes = list(shapes)
    original_shapes = dict(zip(names, shapes, strict=True))
    model_path = (
        args.model_path if args.no_simplify
        else prepare(args.model_path, names, shapes, infer_shapes=args.infer_shapes)
    )
    if args.promote_rank3_hwc:
        if args.external_dram_layout == "compiler":
            raise SystemExit("--promote-rank3-hwc requires --external-dram-layout")
        model_path = promote_rank3_hwc(model_path, args.model_layout)
        names, shapes, outputs = detect_io(model_path)

    importer = ImporterParams(
        format=ModelFormat.onnx,
        file_paths=[str(model_path)],
        input_names=list(names),
        input_shapes=[tuple(s) for s in shapes],
        input_types=input_types,
        layout=args.model_layout,
        output_names=list(outputs),
    )
    target = gen2_target if args.device == "modalix" else gen1_target
    loaded = load_model(importer, target=target)

    activation = args.activation_precision or args.precision
    weights = args.weight_precision or args.precision
    all_bf16 = activation == "bf16" and weights == "bf16"

    if args.calibration_npz and args.calibration_raw_f32:
        raise SystemExit("pass at most one of --calibration-npz / --calibration-raw-f32")

    kind, source, note = calib.plan(
        activation, weights, args.calibration_npz, args.calibration_raw_f32
    )
    if note:
        print(f"[{'WARN' if 'drift' in note else 'INFO'}] {note}")

    samples = calib.build(
        kind, source, names, calibration_shapes if args.promote_rank3_hwc else shapes,
        layout=args.model_layout,
        types=dict(zip(names, type_names, strict=True)),
    )
    if args.promote_rank3_hwc:
        for sample in samples:
            for name in names:
                if len(original_shapes[name]) == 3:
                    sample[name] = sample[name].reshape(
                        sample[name].shape[0], 1, *sample[name].shape[1:]
                    )
    print(f"[INFO] Calibration: {kind}, {len(samples)} sample(s)")
    config = (
        default_quantization
        .with_activation_quantization(
            bfloat16_scheme() if activation == "bf16" else quantization_scheme(True, False, 8)
        )
        .with_weight_quantization(
            bfloat16_scheme() if weights == "bf16" else quantization_scheme(False, True, 8)
        )
        .with_requantization_mode(
            RequantizationMode.sima if args.requant_mode == "sima" else RequantizationMode.tflite
        )
        .with_calibration(CalibrationMethod.from_str(args.calib_method))
    )

    quantized = loaded.quantize(
        calibration_data=samples,
        quantization_config=config,
        any_shape_on_mla=True,
        automatic_layout_conversion=True,
        model_name=stem,
        log_level=logging.INFO,
    )
    quantized.save(model_name=stem, output_directory=str(output))
    print("[INFO] Quantized model saved")

    if args.no_compile:
        return 0

    if args.exit_on_stable_elf:
        if not args.retain_compile_dir:
            raise SystemExit("--exit-on-stable-elf requires --retain-compile-dir")
        _start_elf_watchdog(args.retain_compile_dir / f"{stem}_stage1_mla.elf")

    if args.mla_tessellation and args.external_dram_layout != "compiler":
        raise SystemExit("--mla-tessellation and --external-dram-layout are mutually exclusive")
    compile_layout = (
        tessellation_parameters(quantized) if args.mla_tessellation else
        external_layout_parameters(quantized, args.external_dram_layout)
        if args.external_dram_layout != "compiler" else None
    )
    quantized.compile(
        output_path=str(output),
        batch_size=args.batch_size,
        log_level=logging.INFO,
        tessellate_parameters=compile_layout,
        retained_temporary_directory_name=(
            str(args.retain_compile_dir) if args.retain_compile_dir else None
        ),
    )
    print("[INFO] Compilation complete")
    return 0


def _start_elf_watchdog(path: Path, stable_seconds: float = 30.0) -> None:
    """Exit successfully if ModelSDK hangs after finishing the retained ELF.

    ModelSDK 2.1.0 can remain inside ``quantized.compile`` indefinitely for the
    large native SmolVLA vision graph even after logging code-generation
    completion and closing the 590 MiB ELF.  ELF presence is already PoLiMa's
    compile success condition.  This opt-in watchdog additionally requires an
    unchanged size and mtime for 30 seconds, so it cannot accept a file that is
    still being streamed.
    """
    def watch() -> None:
        previous: tuple[int, int] | None = None
        stable_since: float | None = None
        while True:
            try:
                stat = path.stat()
                current = (stat.st_size, stat.st_mtime_ns)
            except FileNotFoundError:
                current = None
            now = time.monotonic()
            if current is not None and current == previous and current[0] > 0:
                stable_since = stable_since or now
                if now - stable_since >= stable_seconds:
                    print(
                        f"[INFO] ELF stable for {stable_seconds:.0f}s; "
                        "leaving stuck ModelSDK post-generation cleanup",
                        flush=True,
                    )
                    os._exit(0)
            else:
                previous = current
                stable_since = now if current is not None else None
            time.sleep(2.0)

    threading.Thread(target=watch, name="polima-elf-watchdog", daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
