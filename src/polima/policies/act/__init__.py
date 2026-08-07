"""ACT (Action Chunking Transformer) -- the first policy ported to PoLiMa.

Every constant below is transcribed from the working legacy stack, not invented:

  ACT/scripts/export_act_modalix.py       graph decomposition, shapes, calibration
  ACT/scripts/compile_deploy_act_som.sh   per-graph layout and precision
  ACT/devkit/act_llima/act_llima.cpp      element counts, packing offsets, magic
  ACT/scripts/act_som_client.py           wire protocol, client-side normalization
  ACT/scripts/validate_act_datasets.py    the SO-101 dataset contract
  ACT/train_act_local.sh                  training defaults and augmentations

ACT has no language conditioning, which is why the dataset contract is
single-task by default (relaxed per-invocation with --allow-mixed-tasks).
"""

from __future__ import annotations

from polima.policies.act.runtime import (
    ACTION_DIM,
    CAMERA_ELEMENTS,
    CAMERA_TOKENS,
    CHUNK,
    ENCODER_TOKENS,
    HIDDEN,
    HIDDEN_ELEMENTS,
    PADDED_ACTION_DIM,
    STEM_ELEMENTS,
    STEM_TOKENS,
)
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

HEIGHT, WIDTH = 480, 640
STATE_DIM = 6

#: The six SO-101 follower joints, in dataset order.
JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

#: Verbatim from ACT/train_act_local.sh. Applied only with --augmentations.
AUGMENTATION_TFS = (
    '{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},'
    '"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},'
    '"affine":{"weight":0.5,"type":"RandomAffine",'
    '"kwargs":{"degrees":[-1.0,1.0],"translate":[0.02,0.02]}}}'
)

# The encoder hidden state, shared by four graphs. Rank 4 because the MLA path
# needs NHWC; the graph reshapes to (1, 602, 512) internally.
_HIDDEN_IN = TensorSpec("hidden", (1, 1, ENCODER_TOKENS, HIDDEN))
_HIDDEN_OUT = TensorSpec("hidden_out", (1, 1, ENCODER_TOKENS, HIDDEN))


def _encoder_layer(index: int) -> GraphSpec:
    return GraphSpec(
        name=f"encoder_layer_{index:02d}",
        builder="polima.policies.act.graphs:EncoderLayerRank4",
        inputs=(_HIDDEN_IN,),
        outputs=(_HIDDEN_OUT,),
        layout="NHWC",
        precision="bf16",
        precision_fallback=("int8",),
        calibration=CalibrationSource("npz", samples=8),
    )


ACT_SPEC = PolicySpec(
    name="act",
    display_name="ACT (Action Chunking Transformer)",
    # --------------------------------------------------------------- dataset --
    dataset=DatasetContract(
        state_names=JOINT_NAMES,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        camera_keys=("observation.images.overhead", "observation.images.wrist"),
        camera_shape=(HEIGHT, WIDTH, 3),
        fps=30,
        codebase_version="v3.0",
        single_task=True,
        task_canonicalizer="polima.data.contract:canonical_task",
    ),
    # -------------------------------------------------------------- training --
    train=TrainSpec(
        backend="lerobot-train",
        conda_env="act",
        repo_dir_hint="ACT/lerobot",
        entrypoint=("lerobot-train",),
        build_args="polima.policies.act.train:build_args",
        defaults={
            "steps": 100_000,
            "batch_size": 8,
            "num_workers": 4,
            "device": "cuda",
            "keep_last_checkpoints": 1,
            "chunk_size": CHUNK,
            "n_action_steps": CHUNK,
        },
        augmentation_tfs=AUGMENTATION_TFS,
    ),
    # --------------------------------------------------------------- compile --
    compile=CompilePlan(
        export_entry="polima.policies.act.graphs:export_all",
        verify_entry="polima.policies.act.graphs:verify_chain",
        fixture_entry="polima.policies.act.graphs:write_fixtures",
        normalization_entry="polima.export.normalization:from_lerobot_checkpoint",
        fixture_file="act_fixture.npz",
        verify_atol=1e-4,
        verify_rtol=1e-3,
        graphs=(
            # The only NCHW graph -- it takes a raw image. Everything downstream
            # is NHWC token data. compile_deploy_act_som.sh encodes the same rule.
            GraphSpec(
                name="vision_backbone",
                builder="polima.policies.act.graphs:VisionBackbone",
                inputs=(TensorSpec("image", (1, 3, HEIGHT, WIDTH)),),
                outputs=(TensorSpec("camera_tokens", (1, CAMERA_TOKENS, HIDDEN)),),
                layout="NCHW",
                precision="bf16",
                precision_fallback=("int8",),
                calibration=CalibrationSource("npz", samples=8),
            ),
            # Consumes the host-packed (1, 601, 512): token 0 = state,
            # tokens 1..300 = camera 0, tokens 301..600 = camera 1.
            GraphSpec(
                name="encoder_layer_00_stem",
                builder="polima.policies.act.graphs:EncoderStemPacked",
                inputs=(TensorSpec("stem_input", (1, 1, STEM_TOKENS, HIDDEN)),),
                outputs=(_HIDDEN_OUT,),
                layout="NHWC",
                precision="bf16",
                precision_fallback=("int8",),
                calibration=CalibrationSource("npz", samples=8),
            ),
            _encoder_layer(1),
            _encoder_layer(2),
            _encoder_layer(3),
            # Action head widened 6 -> 16 for MLA channel alignment and
            # zero-filled; the host strides past the padding.
            GraphSpec(
                name="decoder_action_tail",
                builder="polima.policies.act.graphs:DecoderActionRank4",
                inputs=(_HIDDEN_IN,),
                outputs=(
                    TensorSpec("normalized_actions", (1, 1, CHUNK, PADDED_ACTION_DIM)),
                ),
                layout="NHWC",
                precision="bf16",
                precision_fallback=("int8",),
                calibration=CalibrationSource("npz", samples=8),
            ),
        ),
    ),
    # ------------------------------------------------------------------ wire --
    wire=WireSpec(
        magic=0x4D544341,                 # "ACTM" little-endian
        version=1,
        # act_llima occupies 8082 on the live board; PoLiMa lands on 8092 so both
        # can serve during the Phase-1 parity proof.
        default_port=8092,
        request_header="<IIII",
        response_header="<IIIIfI",
        request_tensors=(
            # HWC, already normalized by the client. export_act_modalix.py writes
            # direct_inputs/vision_input_{i}.f32 with transpose(0, 2, 3, 1).
            TensorSpec("image0", (HEIGHT, WIDTH, 3)),
            TensorSpec("image1", (HEIGHT, WIDTH, 3)),
            TensorSpec("state", (STATE_DIM,)),
        ),
        response_shape=(CHUNK, ACTION_DIM),
        normalization_side="client",
        stats_file="normalization_stats.npz",
    ),
    # ----------------------------------------------------------------- robot --
    robot=RobotSpec(
        camera_roles=(("overhead", "Overhead"), ("wrist", "Wrist")),
        # From the legacy launcher's --perspective-camera / --wrist-camera
        # defaults: a Logitech C920 overhead and a Sonix wrist camera.
        camera_hints={"overhead": "C920", "wrist": "Sonix"},
        joint_names=JOINT_NAMES,
        actions_per_chunk=CHUNK,
        default_fps=30,
        max_relative_target=12,
        aggregate_fn="weighted_average",
    ),
    runtime_plan_builder="polima.policies.act.runtime:build_plan",
    checkpoint_validator="polima.policies.act.graphs:validate_checkpoint",
)

register_policy(ACT_SPEC)

__all__ = ["ACT_SPEC", "JOINT_NAMES", "AUGMENTATION_TFS"]
