"""Draw calibration samples from the dataset the checkpoint was trained on.

Shared by every policy: ACT, SmolVLA and GR00T all calibrate against real
observations passed through the checkpoint's own preprocessor, because that is
what the graphs will see at inference. Calibrating on raw dataset tensors instead
would quantize against the wrong range entirely -- the preprocessor is where
normalization happens.

The dataset is resolved from `train_config.json` inside the checkpoint rather
than being asked for, so calibration cannot silently use a different dataset than
training did. An explicit root still overrides it, because datasets get moved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def resolve_dataset(checkpoint: str | Path, override: str | Path | None = None) -> tuple[str, Path]:
    """(repo_id, root) from the checkpoint's own training config."""
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found; pass --dataset-root to name the dataset explicitly"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(override) if override else Path(config["dataset"]["root"])
    return config["dataset"]["repo_id"], root.resolve()


def load(policy, checkpoint: str | Path, observation_keys: Sequence[str],
         dataset_root: str | Path | None = None, count: int = 8,
         lerobot_src: str | Path | None = None):
    """`count` preprocessed observations spread evenly across the dataset.

    Even spacing matters more than it looks: episodes are recorded in order, so
    the first N frames of a LeRobot dataset are all the same moment of the same
    episode. Calibrating on those would cover a fraction of the activation range.
    """
    import sys

    if lerobot_src and str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors

    repo_id, root = resolve_dataset(checkpoint, dataset_root)
    # A checkpoint may rename raw dataset features before the model sees them
    # (the combined SmolVLA run maps overhead/wrist to camera1/camera2).  The
    # graph module names processed tensors; dataset lookup must use the inverse
    # map and then let the saved preprocessor perform the rename normally.
    inverse_rename: dict[str, str] = {}
    processor_manifest = Path(checkpoint) / "policy_preprocessor.json"
    if processor_manifest.is_file():
        manifest = json.loads(processor_manifest.read_text(encoding="utf-8"))
        for step in manifest.get("steps", []):
            if step.get("registry_name") == "rename_observations_processor":
                rename = step.get("config", {}).get("rename_map") or {}
                inverse_rename.update({target: source for source, target in rename.items()})
    dataset = LeRobotDataset(repo_id=repo_id, root=str(root))
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    indices = np.linspace(0, len(dataset) - 1, num=min(count, len(dataset)), dtype=int)
    samples = []
    for index in indices:
        item = dataset[int(index)]
        observation = {
            inverse_rename.get(key, key): item[inverse_rename.get(key, key)]
            for key in observation_keys
        }
        processed = preprocessor(observation)
        # Language-conditioned preprocessors replace the string ``task`` with
        # generated token and attention-mask tensors. Preserve every tensor the
        # processor emits rather than projecting back down to the raw keys;
        # ACT sees the same state/images as before, while SmolVLA also receives
        # ``observation.language.{tokens,attention_mask}``.
        samples.append({
            key: value.detach().cpu().float() if value.is_floating_point()
            else value.detach().cpu()
            for key, value in processed.items()
            if hasattr(value, "detach")
        })
    return samples, postprocessor, root
