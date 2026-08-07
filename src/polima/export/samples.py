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
    dataset = LeRobotDataset(repo_id=repo_id, root=str(root))
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )

    indices = np.linspace(0, len(dataset) - 1, num=min(count, len(dataset)), dtype=int)
    samples = []
    for index in indices:
        item = dataset[int(index)]
        observation = {key: item[key] for key in observation_keys}
        processed = preprocessor(observation)
        samples.append({
            key: processed[key].detach().cpu().float() for key in observation
        })
    return samples, postprocessor, root
