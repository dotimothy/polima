"""How a GR00T N1.7 fine-tune is invoked. PoLiMa does not run this.

`TrainSpec.build_args` exists so `polima data validate` can report what a run
expected without shelling out. The flags are those of
`gr00t/experiment/launch_finetune.py` at the `n1.7-release` tag, which kept
N1.6's command-line surface.

STDLIB ONLY -- this is imported wherever the spec is, including the board.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_args(config: Mapping[str, Any], spec) -> list[str]:
    """The `launch_finetune.py` invocation for one run."""
    defaults = dict(spec.train.defaults)
    defaults.update(config)
    args = [
        "--base-model-path", str(defaults.get("base_model", "nvidia/GR00T-N1.7-3B")),
        "--dataset-path", str(defaults["dataset_path"]),
        "--embodiment-tag", str(defaults["embodiment_tag"]),
        "--modality-config-path", str(defaults.get("modality_config", "config/so101_config.py")),
        "--num-gpus", str(defaults.get("num_gpus", 1)),
        "--output-dir", str(defaults["output_dir"]),
        "--max-steps", str(defaults["steps"]),
        "--global-batch-size", str(defaults["batch_size"]),
        "--gradient-accumulation-steps", str(defaults.get("gradient_accumulation", 1)),
        "--dataloader-num-workers", str(defaults.get("num_workers", 8)),
        "--save-steps", str(defaults.get("save_steps", 5_000)),
        "--save-total-limit", str(defaults.get("save_total_limit", 4)),
    ]
    for flag, key in (("llm", "tune_llm"), ("visual", "tune_visual"),
                      ("projector", "tune_projector"),
                      ("diffusion-model", "tune_diffusion_model")):
        args.append(f"--tune-{flag}" if defaults.get(key) else f"--no-tune-{flag}")
    if spec.train.augmentation_tfs:
        import json

        jitter = json.loads(spec.train.augmentation_tfs)
        args.append("--color-jitter-params")
        for name, value in jitter.items():
            args += [name, str(value)]
    return args
