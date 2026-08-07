"""The policy plugin contract -- PoLiMa's central abstraction.

Adding a policy should mean writing one PolicySpec, not copying ~3,000 lines.
A spec declares, declaratively and in one place:

  * DatasetContract  -- what the on-disk LeRobotDataset must look like
  * TrainSpec        -- how to invoke training (lerobot-train vs launch_finetune)
  * CompilePlan      -- the ONNX subgraph decomposition and how each graph compiles
  * RuntimePlan      -- what runs on the MLA, in what order, with what host glue
  * WireSpec         -- the TCP protocol between the robot client and the SoM server
  * RobotSpec        -- cameras, joints, chunk size for the robot client

Everything here is a frozen dataclass and JSON round-trippable, because a spec
has to cross three interpreters: the training conda env (torch), the
model-compiler venv (afe, no torch), and the board (py3.11, no torch).
Consequently `builder`/`entrypoint`-style fields hold *dotted import paths as
strings*, not callables -- they are resolved lazily, in the interpreter that can
actually import them.

STDLIB ONLY. No numpy, no torch. See the dependency-floor note in pyproject.toml.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from math import prod
from typing import Any, Literal, Mapping, Sequence

Layout = Literal["NCHW", "NHWC"]
Precision = Literal["int8", "bf16", "mixed"]
DramLayout = Literal["plain", "hwc16"]
Compiler = Literal["afe", "llima"]
Device = Literal["modalix", "mlsoc"]


class SpecError(ValueError):
    """A PolicySpec is internally inconsistent."""


def resolve(dotted: str) -> Any:
    """Import 'pkg.module:attr' lazily.

    Specs name their torch modules and entry points this way so that importing
    polima.policies.act does not drag torch into the compiler venv or the board.
    """
    if ":" not in dotted:
        raise SpecError(f"expected 'module:attr', got {dotted!r}")
    module_name, attr = dotted.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise SpecError(f"{module_name} has no attribute {attr!r}") from exc


# --------------------------------------------------------------------- tensors


@dataclass(frozen=True)
class TensorSpec:
    """A fully static tensor at a graph or wire boundary.

    `dram_layout` describes how the MLA lays this tensor out in DRAM when it
    crosses the accelerator boundary. 'hwc16' means the 16-lane byte-planed
    tessellated form that TensorDRAMLayout.HWC16 produces and that
    smolvla_som_server.cpp currently un-packs by hand; the runtime needs
    logical_width/logical_channels to reverse it.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    dram_layout: DramLayout = "plain"
    logical_width: int | None = None
    logical_channels: int | None = None

    @property
    def elements(self) -> int:
        return prod(self.shape) if self.shape else 0

    def validate(self, where: str) -> None:
        if not self.shape or any(d <= 0 for d in self.shape):
            raise SpecError(f"{where}: {self.name} must be fully static, got {self.shape}")
        if self.dram_layout == "hwc16" and not (self.logical_width and self.logical_channels):
            raise SpecError(
                f"{where}: {self.name} is hwc16 but lacks logical_width/logical_channels"
            )


# --------------------------------------------------------------------- dataset


@dataclass(frozen=True)
class DatasetConverter:
    """A required dataset rewrite before training.

    Exists for GR00T, which needs LeRobotDataset v3.0 downgraded to v2.1 in a
    *second* conda env (groot-n1.6-convert). Expressing it here is what stops
    that from being 200 lines of interleaved bash in train_groot_local.sh.
    """

    name: str
    conda_env: str
    entrypoint: tuple[str, ...]
    target_codebase_version: str
    post_copy: tuple[tuple[str, str], ...] = ()   # (src_in_repo, dst_relative_to_dataset)
    space_factor: float = 2.2                     # convert needs ~2.2x the source size


@dataclass(frozen=True)
class DatasetContract:
    """What a dataset must look like for this policy to train on it.

    Generalizes ACT/scripts/validate_act_datasets.py, whose SO-101 assertions are
    hardcoded. Making them data means SmolVLA and GR00T get real validation for
    free instead of an inline `[[ -f meta/info.json ]]`.
    """

    state_names: tuple[str, ...]
    state_dim: int
    action_dim: int
    camera_keys: tuple[str, ...]
    camera_shape: tuple[int, int, int]            # HWC, as stored in meta/info.json
    fps: int
    #: LeRobot feature names. Standard across every policy here, but named rather
    #: than hardcoded because export and normalization both index by them, and a
    #: dataset that renamed them would otherwise fail deep inside torch.
    state_key: str = "observation.state"
    action_key: str = "action"
    codebase_version: str = "v3.0"
    single_task: bool = True                      # False => language-conditioned
    task_canonicalizer: str | None = None         # dotted path
    rename_map: Mapping[str, str] = field(default_factory=dict)
    empty_cameras: int = 0
    converter: DatasetConverter | None = None
    extra_meta_files: tuple[str, ...] = ()

    def validate(self, where: str) -> None:
        if len(self.state_names) != self.state_dim:
            raise SpecError(
                f"{where}: {len(self.state_names)} state_names but state_dim={self.state_dim}"
            )
        if not self.camera_keys:
            raise SpecError(f"{where}: at least one camera key is required")
        if len(self.camera_shape) != 3:
            raise SpecError(f"{where}: camera_shape must be (H, W, C)")


# -------------------------------------------------------------------- training


@dataclass(frozen=True)
class TrainSpec:
    backend: Literal["lerobot-train", "groot-launch-finetune"]
    conda_env: str
    entrypoint: tuple[str, ...]
    build_args: str                               # dotted path -> (cfg, spec) -> list[str]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    repo_dir_hint: str | None = None              # cwd for the invocation, e.g. "ACT/lerobot"
    augmentation_tfs: str | None = None           # the ColorJitter/RandomAffine JSON blob
    checkpoint_glob: str = "checkpoints/*/pretrained_model"
    requires_offline_env: bool = True


# --------------------------------------------------------------------- compile


@dataclass(frozen=True)
class CalibrationSource:
    kind: Literal["npz", "raw_f32", "random"] = "npz"
    producer: str | None = None
    samples: int = 8


@dataclass(frozen=True)
class GraphSpec:
    """One ONNX subgraph -> one MLA ELF.

    `name` is the single identifier used end to end: the ONNX stem, the ELF stem,
    the bundle's models/<name>/ directory, and the plan.json `graph` key. Keeping
    them equal is what lets bundle.model_elf(name) never branch.
    """

    name: str
    builder: str                                  # dotted path -> nn.Module factory
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    layout: Layout = "NCHW"
    compiler: Compiler = "afe"
    precision: Precision = "bf16"
    precision_fallback: tuple[Precision, ...] = ()
    calibration: CalibrationSource = field(default_factory=CalibrationSource)
    mla_tessellation: bool = True
    elf_from: Literal["retained", "mpk"] = "retained"
    llima_args: tuple[str, ...] = ()

    @property
    def elf_name(self) -> str:
        return f"{self.name}_stage1_mla.elf"

    @property
    def precisions(self) -> tuple[Precision, ...]:
        """Try `precision` first, then each fallback -- ACT's and GR00T's
        controllers both do `for precision in ("int8", "bf16")`."""
        return (self.precision, *self.precision_fallback)

    def validate(self, where: str) -> None:
        where = f"{where}.{self.name}"
        if not self.inputs or not self.outputs:
            raise SpecError(f"{where}: needs at least one input and one output")
        for tensor in (*self.inputs, *self.outputs):
            tensor.validate(where)
        if self.compiler == "llima" and self.mla_tessellation:
            raise SpecError(f"{where}: llima-compiled graphs do not take MLA tessellation")


@dataclass(frozen=True)
class CompilePlan:
    graphs: tuple[GraphSpec, ...]                 # ordered; also the export order
    export_entry: str
    verify_entry: str | None = None
    fixture_entry: str | None = None
    normalization_entry: str | None = None
    #: Reference tensors written by `fixture_entry` and read back by the verify
    #: step and by bundle packing. Named here so the generic export driver has no
    #: policy-specific filename in it; ACT keeps its legacy `act_fixture.npz`
    #: because existing build trees on disk use that name.
    fixture_file: str = "fixture.npz"
    verify_atol: float = 1e-4
    verify_rtol: float = 1e-3

    def graph(self, name: str) -> GraphSpec:
        for graph in self.graphs:
            if graph.name == name:
                return graph
        raise KeyError(f"no graph {name!r}; have {[g.name for g in self.graphs]}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.graphs)


# ---------------------------------------------------------- on-device runtime


#: Opcodes the native plan interpreter (native/src/plan.cpp) understands.
#: ACT needs only run_elf/pack/gather_strided -- that is the whole Phase-1
#: interpreter. The rest land with SmolVLA in Phase 4.
OPCODES = (
    "run_elf",          # Runner[graph].run(in -> out)
    "pack",             # scatter sub-spans into one zeroed buffer at fixed offsets
    "slice",            # contiguous copy-out
    "gather_strided",   # for i<count: copy(src + i*stride, take)
    "scale",            # multiply by a scalar
    "matvec",           # y[o] = b[o] + sum_i W[o*K+i] * x[i], from .f32 sidecars
    "sincos_time",      # min/max-period sin/cos time embedding
    "euler",            # loop body xN with x -= dt*v
    "normalize",        # (x - mean) / std
    "denormalize",      # x * std + mean
)


@dataclass(frozen=True)
class Step:
    op: str
    out: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, where: str, buffers: Mapping[str, int]) -> None:
        if self.op not in OPCODES:
            raise SpecError(f"{where}: unknown opcode {self.op!r}; known: {list(OPCODES)}")
        if self.out not in buffers:
            raise SpecError(f"{where}: writes undeclared buffer {self.out!r}")


@dataclass(frozen=True)
class RuntimePlan:
    """Serialized to plan.json and interpreted on the board.

    This is what replaces act_llima.cpp's hand-written predict(): the ELF paths,
    element counts and packing offsets currently baked into the binary's
    constructor become data, so one polima_server serves every bundle.
    """

    buffers: Mapping[str, int]
    steps: tuple[Step, ...]
    result: str
    sidecars: tuple[str, ...] = ()
    loops: Mapping[str, int] = field(default_factory=dict)

    def validate(self, where: str, compile_plan: CompilePlan | None = None) -> None:
        if self.result not in self.buffers:
            raise SpecError(f"{where}: result {self.result!r} is not a declared buffer")
        for index, step in enumerate(self.steps):
            step.validate(f"{where}.steps[{index}]", self.buffers)
            if step.op == "run_elf":
                graph = step.args.get("graph")
                if compile_plan is not None and graph not in compile_plan.names:
                    raise SpecError(
                        f"{where}.steps[{index}]: run_elf names unknown graph {graph!r}"
                    )
                for source in step.args.get("in", ()):
                    if source not in self.buffers:
                        raise SpecError(
                            f"{where}.steps[{index}]: reads undeclared buffer {source!r}"
                        )


# ------------------------------------------------------------------------ wire


@dataclass(frozen=True)
class WireSpec:
    """The SoM TCP protocol.

    Both legacy clients (act_som_client.py, smolvla_som_client.py) hardcode
    struct.Struct("<IIII") / ("<IIIIfI") independently. Here it is data, so
    polima.wire.protocol derives the packers and there is one client.
    """

    magic: int
    version: int = 1
    default_port: int = 8092
    request_header: str = "<IIII"                 # magic, version, request_id, flags
    response_header: str = "<IIIIfI"              # magic, version, request_id, status, ms, count
    request_tensors: tuple[TensorSpec, ...] = ()
    response_shape: tuple[int, ...] = ()
    normalization_side: Literal["client", "server"] = "client"
    stats_file: str = "normalization_stats.npz"

    @property
    def response_elements(self) -> int:
        return prod(self.response_shape) if self.response_shape else 0

    @property
    def magic_ascii(self) -> str:
        """Readable form -- 0x4D544341 is 'ACTM' little-endian."""
        try:
            return self.magic.to_bytes(4, "little").decode("ascii")
        except (UnicodeDecodeError, OverflowError):
            return hex(self.magic)


# ----------------------------------------------------------------------- robot


@dataclass(frozen=True)
class RobotSpec:
    camera_roles: tuple[tuple[str, str], ...]     # (lerobot camera key, UI label)
    joint_names: tuple[str, ...]
    actions_per_chunk: int
    default_fps: int = 30
    max_relative_target: int = 12                 # degrees; the safety clamp
    aggregate_fn: str = "weighted_average"
    task_string: str | None = None
    image_preprocessor: str | None = None         # dotted path; None => identity
    #: Camera pixel format. MJPG is not cosmetic: two 640x480@30 USB cameras on
    #: one controller exceed the bus budget in uncompressed YUYV, so the stream
    #: silently drops to a lower rate. Both legacy launchers were fixed to pass
    #: `fourcc: MJPG` in --robot.cameras; PoLiMa must emit it too or the robot
    #: client regresses in a way that only shows up as degraded control.
    camera_fourcc: str = "MJPG"
    #: Whether `polima robot` offers --calibrate (backs up the existing
    #: calibration, then runs lerobot-calibrate).
    supports_calibrate: bool = True
    calibration_id: str = "so-arm101"

    def camera_config(self, devices: Mapping[str, str], fps: int | None = None) -> str:
        """Render lerobot's --robot.cameras draccus blob.

        One place builds this string; the legacy stacks each hand-wrote it, which
        is how the MJPG fix had to be applied twice.
        """
        rate = fps or self.default_fps
        entries = ", ".join(
            f"{key}: {{type: opencv, index_or_path: '{devices[key]}', "
            f"width: 640, height: 480, fps: {rate}, fourcc: {self.camera_fourcc}}}"
            for key, _ in self.camera_roles
            if key in devices
        )
        return "{ " + entries + "}"

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.camera_roles)


# ------------------------------------------------------------------- the spec


@dataclass(frozen=True)
class PolicySpec:
    name: str
    display_name: str
    dataset: DatasetContract
    train: TrainSpec
    compile: CompilePlan
    wire: WireSpec
    robot: RobotSpec
    runtime_plan_builder: str
    checkpoint_validator: str | None = None
    bundle_format: str = "polima-bundle-v1"

    def graph(self, name: str) -> GraphSpec:
        return self.compile.graph(name)

    def build_runtime_plan(self, context: Any = None) -> RuntimePlan:
        return resolve(self.runtime_plan_builder)(self, context)

    def validate(self) -> None:
        """Cross-field consistency. Called by the registry at import time, so a
        malformed spec fails loudly at `polima doctor` rather than mid-deploy."""
        where = f"policy[{self.name}]"
        self.dataset.validate(where)

        if not self.compile.graphs:
            raise SpecError(f"{where}: no graphs declared")
        seen: set[str] = set()
        for graph in self.compile.graphs:
            if graph.name in seen:
                raise SpecError(f"{where}: duplicate graph name {graph.name!r}")
            seen.add(graph.name)
            graph.validate(f"{where}.compile")

        # The wire's action count must match what the robot client expects.
        expected = self.robot.actions_per_chunk * self.dataset.action_dim
        if self.wire.response_elements != expected:
            raise SpecError(
                f"{where}: wire.response_shape={self.wire.response_shape} is "
                f"{self.wire.response_elements} elements, but "
                f"actions_per_chunk*action_dim = {expected}"
            )

        # camera_roles may name either the short role ("overhead") or the full
        # feature key ("observation.images.overhead"); both must resolve.
        short_keys = {key.rsplit(".", 1)[-1] for key in self.dataset.camera_keys}
        unknown_cameras = set(self.robot.camera_keys) - short_keys - set(self.dataset.camera_keys)
        if unknown_cameras:
            raise SpecError(
                f"{where}: robot.camera_roles reference {sorted(unknown_cameras)}, "
                f"which are not in dataset.camera_keys {list(self.dataset.camera_keys)}"
            )
        if len(self.robot.camera_roles) != len(self.dataset.camera_keys):
            raise SpecError(
                f"{where}: {len(self.robot.camera_roles)} camera_roles but "
                f"{len(self.dataset.camera_keys)} dataset camera_keys"
            )

        plan = self.build_runtime_plan()
        plan.validate(f"{where}.plan", self.compile)

        # Every buffer a run_elf reads/writes must match the graph's element count.
        for index, step in enumerate(plan.steps):
            if step.op != "run_elf":
                continue
            graph = self.compile.graph(step.args["graph"])
            out_elements = sum(t.elements for t in graph.outputs)
            declared = plan.buffers[step.out]
            if declared != out_elements:
                raise SpecError(
                    f"{where}.plan.steps[{index}]: buffer {step.out!r} holds {declared} "
                    f"elements but graph {graph.name!r} outputs {out_elements}"
                )
            in_elements = sum(t.elements for t in graph.inputs)
            sources = step.args.get("in", ())
            supplied = sum(plan.buffers[s] for s in sources)
            if supplied != in_elements:
                raise SpecError(
                    f"{where}.plan.steps[{index}]: inputs {list(sources)} supply {supplied} "
                    f"elements but graph {graph.name!r} expects {in_elements}"
                )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "graphs": list(self.compile.names),
            "state_dim": self.dataset.state_dim,
            "action_dim": self.dataset.action_dim,
            "cameras": list(self.dataset.camera_keys),
            "fps": self.dataset.fps,
            "actions_per_chunk": self.robot.actions_per_chunk,
            "wire": {
                "magic": hex(self.wire.magic),
                "magic_ascii": self.wire.magic_ascii,
                "default_port": self.wire.default_port,
            },
            "train_backend": self.train.backend,
            "conda_env": self.train.conda_env,
        }


def sequence_names(items: Sequence[Any]) -> list[str]:
    return [getattr(i, "name", str(i)) for i in items]
