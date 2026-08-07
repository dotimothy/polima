"""SmolVLA -- vision-language-action policy, second port into PoLiMa.

Every constant is transcribed from the working legacy stack, not invented:

  SmolVLA/devkit/smolvla_som_server/smolvla_som_server.cpp   pipeline, wire, sizes
  SmolVLA/scripts/compile_deploy_smolvla_som.sh              per-graph compile flags
  SmolVLA/scripts/smolvla_som_client.py                      client-side protocol
  SmolVLA/scripts/extract_smolvla_som_constants.py           the constants/ sidecars
  SmolVLA/inference_smolvla.sh                               robot launch, task string

## How it differs from ACT, structurally

ACT is a fixed feed-forward chain: six graphs, one pass, done. SmolVLA is a
*flow-matching* policy, so the action is produced by integrating a learned
velocity field over ten Euler steps. Two of its four graphs therefore run ten
times per inference, not once -- twenty MLA calls in total against ACT's six.

That is why PoLiMa's runtime is a plan interpreter rather than a fixed pipeline:
the loop, the time embedding and the Euler update are all data in plan.json.
See `polima.policies.smolvla.runtime`.

## Language conditioning

SmolVLA is language-conditioned, so `single_task` is False -- unlike ACT, a
dataset with several task strings is legitimate. The language embedding itself is
*precomputed* into `constants/language_embedding.f32` at export: the task string
is fixed for a deployed checkpoint, so tokenizing and embedding it on every
inference would burn latency on a constant. Changing the task means re-exporting.
"""

from __future__ import annotations

from polima.policies.base import (
    CalibrationSource,
    CompilePlan,
    DatasetContract,
    GraphSpec,
    PolicySpec,
    RobotSpec,
    TensorSpec,
    TrainSpec,
    WireSpec,
)
from polima.policies.registry import register_policy
from polima.policies.smolvla.runtime import (
    ACTION_DIM,
    CACHE_ELEMENTS,
    CHUNK,
    DEFAULT_PORT,
    DENOISE_IN_ELEMENTS,
    IMAGE_ELEMENTS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NOISE_ELEMENTS,
    PREFIX_ELEMENTS,
    STATE_DIM,
    SUFFIX_IN_ELEMENTS,
    SUFFIX_OUT_ELEMENTS,
    VISION_ELEMENTS,
    WIRE_MAGIC,
)

#: SO-101 follower joints, in dataset order. Same arm as ACT.
JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

#: Cameras are captured at 480x640 and letterboxed to 512x512 by the client
#: before the wire, matching SmolVLM2's expected input geometry.
CAPTURE_HEIGHT, CAPTURE_WIDTH = 480, 640


SMOLVLA_SPEC = PolicySpec(
    name="smolvla",
    display_name="SmolVLA (SmolVLM2 + flow-matching action expert)",
    # --------------------------------------------------------------- dataset --
    dataset=DatasetContract(
        state_names=JOINT_NAMES,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        camera_keys=("observation.images.overhead", "observation.images.wrist"),
        camera_shape=(CAPTURE_HEIGHT, CAPTURE_WIDTH, 3),
        fps=30,
        codebase_version="v3.0",
        # Language-conditioned: several task strings in one dataset is fine.
        single_task=False,
        task_canonicalizer="polima.data.contract:canonical_task",
    ),
    # ------------------------------------------------- checkpoint provenance --
    train=TrainSpec(
        backend="lerobot-train",
        conda_env="act",
        repo_dir_hint="SmolVLA/lerobot",
        entrypoint=("lerobot-train",),
        build_args="polima.policies.smolvla.train:build_args",
        defaults={"steps": 35_000, "batch_size": 8, "chunk_size": CHUNK},
    ),
    # --------------------------------------------------------------- compile --
    compile=CompilePlan(
        export_entry="polima.policies.smolvla.graphs:export_all",
        verify_entry="polima.policies.smolvla.graphs:verify_chain",
        fixture_entry="polima.policies.smolvla.graphs:write_fixtures",
        normalization_entry="polima.export.normalization:from_lerobot_checkpoint",
        fixture_file="smolvla_fixture.npz",
        verify_atol=1e-3,
        verify_rtol=1e-2,
        graphs=(
            # The SmolVLM2 vision tower. The only NCHW graph, and the only one
            # compiled through LLiMa rather than straight afe -- it is a VLM
            # backbone, which is exactly what sima_lmm.host.compile_lmm is for.
            # Run twice per inference, once per camera.
            GraphSpec(
                name="vision",
                builder="polima.policies.smolvla.graphs:VisionTower",
                inputs=(TensorSpec("image", (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)),),
                outputs=(TensorSpec("image_tokens", (1, 64, 960)),),
                layout="NCHW",
                precision="bf16",
                calibration=CalibrationSource("random"),
                compiler="llima",
                mla_tessellation=False,
            ),
            # Prefix: 241 tokens of vision + language + state -> the packed KV
            # cache every denoise step reads. Runs once.
            GraphSpec(
                name="prefix",
                builder="polima.policies.smolvla.graphs:PrefixCache",
                inputs=(TensorSpec("prefix_embeddings", (1, 241, 960)),),
                outputs=(TensorSpec("cache", (CACHE_ELEMENTS,)),),
                layout="NHWC",
                precision="bf16",
                calibration=CalibrationSource("random"),
            ),
            # Action expert input projection + time conditioning. Runs 10x.
            GraphSpec(
                name="suffix",
                builder="polima.policies.smolvla.graphs:SuffixProjection",
                inputs=(TensorSpec("suffix_input", (1, 50, 752)),),
                outputs=(TensorSpec("suffix_output", (1, 50, 720)),),
                layout="NHWC",
                precision="bf16",
                calibration=CalibrationSource("random"),
            ),
            # The velocity field. Reads the cache plus this step's suffix and
            # emits dx/dt over the 50x32 action lane. Runs 10x -- the single
            # most expensive thing in the pipeline.
            GraphSpec(
                name="denoise",
                builder="polima.policies.smolvla.graphs:DenoiseExpert",
                inputs=(TensorSpec("denoise_input", (DENOISE_IN_ELEMENTS,)),),
                outputs=(TensorSpec("velocity", (1, 50, 32)),),
                layout="NHWC",
                precision="bf16",
                calibration=CalibrationSource("random"),
            ),
        ),
    ),
    # ------------------------------------------------------------------ wire --
    wire=WireSpec(
        magic=WIRE_MAGIC,                 # "SMOL" little-endian
        version=1,
        default_port=DEFAULT_PORT,
        request_header="<IIII",
        response_header="<IIIIfI",
        request_tensors=(
            # Letterboxed to 512x512 and normalized by the client, NCHW.
            TensorSpec("image0", (3, IMAGE_HEIGHT, IMAGE_WIDTH)),
            TensorSpec("image1", (3, IMAGE_HEIGHT, IMAGE_WIDTH)),
            TensorSpec("state", (STATE_DIM,)),
            # The flow-matching seed. Supplied by the client rather than drawn
            # on the board so a run is reproducible from its request alone.
            TensorSpec("noise", (CHUNK, 32)),
        ),
        response_shape=(CHUNK, ACTION_DIM),
        # Unlike ACT, normalization is server-side: the board already holds the
        # statistics as sidecars because the state projection needs them there.
        normalization_side="server",
        stats_file="normalization_stats.npz",
    ),
    # ----------------------------------------------------------------- robot --
    robot=RobotSpec(
        camera_roles=(("overhead", "Overhead"), ("wrist", "Wrist")),
        camera_hints={"overhead": "C920", "wrist": "Sonix"},
        joint_names=JOINT_NAMES,
        actions_per_chunk=CHUNK,
        default_fps=30,
        max_relative_target=12,
        aggregate_fn="weighted_average",
    ),
    runtime_plan_builder="polima.policies.smolvla.runtime:build_plan",
    checkpoint_validator="polima.policies.smolvla.graphs:validate_checkpoint",
)

register_policy(SMOLVLA_SPEC)

__all__ = ["SMOLVLA_SPEC", "JOINT_NAMES"]
