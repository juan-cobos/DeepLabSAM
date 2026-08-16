"""
Self-contained utilities for loading SuperAnimal model configs and snapshots.

Replaces the DLC imports from:
  - deeplabcut.pose_estimation_pytorch.modelzoo.utils
  - deeplabcut.core.config
  - deeplabcut.pose_estimation_pytorch.config.utils
  - deeplabcut.pose_estimation_pytorch.config.make_pose_config

No DeepLabCut dependency required.
"""

import copy
from pathlib import Path
from typing import Optional

import yaml

CONFIGS_DIR = Path(__file__).parent / "configs"
# Authoritative SuperAnimal torch model URLs (vendored from dlclibrary's
# modelzoo_urls_pytorch.yaml). Entries are "<org>/<repo>/<filename>"; note the
# repos span different HF orgs (mwmathis/ and DeepLabCut/).
_PYTORCH_URLS_PATH = CONFIGS_DIR / "modelzoo_urls_pytorch.yaml"


def read_config_as_dict(config_path) -> dict:
    """Read a YAML config file and return as a plain Python dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def replace_default_values(
    config,
    num_bodyparts: Optional[int] = None,
    num_individuals: Optional[int] = None,
    backbone_output_channels: Optional[int] = None,
    **kwargs,
):
    """Recursively replace placeholder strings in a config dict with actual values.

    Supports arithmetic expressions like "num_bodyparts x 2", "num_bodyparts // 2",
    "num_bodyparts + 1".
    """

    def get_updated_value(variable: str):
        parts = variable.strip().split(" ")
        var_name = parts[0]
        if updated_values.get(var_name) is None:
            raise ValueError(
                f"Found '{variable}' in config but no value was provided for '{var_name}'."
            )
        base = updated_values[var_name]
        if len(parts) == 1:
            return base
        if len(parts) == 3:
            op, factor = parts[1], parts[2]
            if not factor.isdigit():
                raise ValueError(f"Factor must be an integer in: '{variable}'")
            factor = int(factor)
            if op == "+":
                return base + factor
            if op == "x":
                return base * factor
            if op == "//":
                return base // factor
            raise ValueError(f"Unknown operator '{op}' in: '{variable}'")
        raise ValueError(f"Cannot parse config variable: '{variable}'")

    updated_values = {
        "num_bodyparts": num_bodyparts,
        "num_individuals": num_individuals,
        "backbone_output_channels": backbone_output_channels,
        **kwargs,
    }

    config = copy.deepcopy(config)
    keys = list(config.keys()) if isinstance(config, dict) else range(len(config))

    for k in keys:
        v = config[k]
        if isinstance(v, (dict, list)):
            config[k] = replace_default_values(
                v, num_bodyparts, num_individuals, backbone_output_channels, **kwargs
            )
        elif isinstance(v, str) and v.strip().split(" ")[0] in updated_values:
            config[k] = get_updated_value(v)

    return config


def available_pose_models(dataset: str) -> list[str]:
    """List the torch pose model names available for a SuperAnimal dataset."""
    url_map = read_config_as_dict(_PYTORCH_URLS_PATH)
    return list(url_map.get(dataset, {}).get("pose_models", {}).keys())


def get_super_animal_snapshot_path(
    dataset: str, model_name: str, kind: str = "pose_models"
) -> Path:
    """Return the path to a SuperAnimal model snapshot, downloading if needed.

    Resolves the HF repo + filename from the bundled ``modelzoo_urls_pytorch.yaml``
    (so the differing orgs — ``mwmathis/`` vs ``DeepLabCut/`` — and non-standard
    filenames are handled correctly). Uses huggingface_hub caching.

    Args:
        dataset: SuperAnimal dataset name, e.g. "superanimal_topviewmouse".
        model_name: model name, e.g. "hrnet_w32" or "resnet_50".
        kind: "pose_models" (default) or "detectors".

    Returns:
        Path to the .pt checkpoint file.
    """
    from huggingface_hub import hf_hub_download

    url_map = read_config_as_dict(_PYTORCH_URLS_PATH)
    entry = url_map.get(dataset, {}).get(kind, {})
    path = entry.get(model_name)
    if path is None:
        raise ValueError(
            f"No torch {kind} '{model_name}' for '{dataset}'. Available: {list(entry)}"
        )
    parts = path.split("/")
    repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def load_super_animal_config(
    super_animal: str,
    model_name: str,
    detector_name: Optional[str] = None,
    max_individuals: int = 30,
    device: Optional[str] = None,
) -> dict:
    """Load a SuperAnimal model config from bundled YAML files.

    Equivalent to deeplabcut.pose_estimation_pytorch.modelzoo.utils.load_super_animal_config
    but reads from configs bundled in this submodule instead of the DLC package.

    Args:
        super_animal: SuperAnimal name, e.g. "superanimal_topviewmouse"
        model_name: pose model name, e.g. "hrnet_w32"
        detector_name: detector name, e.g. "fasterrcnn_resnet50_fpn_v2", or None for BU
        max_individuals: maximum number of individuals to detect
        device: device string ("cuda", "cpu", etc.)

    Returns:
        Model configuration dict ready to pass to PoseModel.build() / DETECTORS.build()
    """
    project_config = read_config_as_dict(CONFIGS_DIR / f"{super_animal}.yaml")
    model_config = read_config_as_dict(CONFIGS_DIR / f"{model_name}.yaml")

    # Equivalent to add_metadata()
    bodyparts = project_config.get("bodyparts", [])
    model_config = copy.deepcopy(model_config)
    model_config["metadata"] = {
        "project_path": project_config.get("project_path", ""),
        "pose_config_path": str(CONFIGS_DIR / f"{model_name}.yaml"),
        "bodyparts": bodyparts,
        "unique_bodyparts": [],
        "individuals": project_config.get("individuals", ["animal"]),
        "with_identity": project_config.get("identity", False),
    }

    # Equivalent to update_config() → replace_default_values()
    model_config = replace_default_values(
        model_config,
        num_bodyparts=len(bodyparts),
        num_individuals=max_individuals,
        backbone_output_channels=model_config["model"].get("backbone_output_channels"),
    )
    model_config["metadata"]["individuals"] = [
        f"animal{i}" for i in range(max_individuals)
    ]

    model_config["device"] = device

    if detector_name is None and super_animal != "superanimal_humanbody":
        model_config["method"] = "BU"
    else:
        model_config["method"] = "TD"
        if super_animal != "superanimal_humanbody":
            detector_cfg = read_config_as_dict(CONFIGS_DIR / f"{detector_name}.yaml")
            model_config["detector"] = detector_cfg

    if model_config.get("detector") is not None:
        model_config["detector"]["device"] = device

    return model_config
