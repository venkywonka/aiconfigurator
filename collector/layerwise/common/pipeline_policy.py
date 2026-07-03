"""Safety policy for metadata-only layerwise model bundles."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_ERROR_MESSAGES = {
    "missing_metadata_dir": "Metadata dummy mode requires a local metadata directory.",
    "invalid_metadata_dir": "Metadata dummy mode requires a valid local metadata directory.",
    "missing_config": "The metadata directory must contain a regular config.json file.",
    "forbidden_model_file": "The metadata directory contains a forbidden model entry.",
    "forbidden_override": "Metadata dummy mode forbids real-data overrides.",
    "forbidden_credential": "Metadata dummy mode forbids model-download credentials.",
}
_CREDENTIAL_VARIABLES = ("HF_TOKEN", "HF_TOKEN_FILE")
_REAL_DATA_FLAGS = ("MOE_REAL_ROUTER", "PHYSICAL_TP_REAL_WEIGHTS")
_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".msgpack",
    ".onnx",
    ".gguf",
    ".npy",
    ".npz",
)
_CHECKPOINT_INDEX_SUFFIXES = tuple(f"{suffix}.index.json" for suffix in _WEIGHT_SUFFIXES)
_WEIGHT_BASENAME_PREFIXES = (
    "model-",
    "pytorch_model",
    "tf_model",
    "flax_model",
    "consolidated",
    "adapter_model",
)


class MetadataDummyPolicyError(ValueError):
    """Report a metadata-dummy policy violation without exposing environment values."""

    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(_ERROR_MESSAGES.get(reason, "Metadata dummy policy validation failed."))


@dataclass(frozen=True)
class MetadataDummyPolicy:
    """Validated settings for offline metadata-only model loading."""

    model_dir: Path
    load_format: str = "dummy"
    shape_source: str = "synthetic"
    offline: bool = True


def _is_weight_like(name: str) -> bool:
    normalized = name.lower()
    return (
        normalized.endswith(_WEIGHT_SUFFIXES)
        or normalized.endswith(_CHECKPOINT_INDEX_SUFFIXES)
        or normalized.startswith(_WEIGHT_BASENAME_PREFIXES)
    )


def _validate_bundle_entries(model_dir: Path) -> None:
    pending = [model_dir]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            raise MetadataDummyPolicyError("forbidden_model_file")
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                            continue
                        if not entry.is_file(follow_symlinks=False) or _is_weight_like(entry.name):
                            raise MetadataDummyPolicyError("forbidden_model_file")
                    except OSError:
                        raise MetadataDummyPolicyError("forbidden_model_file") from None
        except OSError:
            raise MetadataDummyPolicyError("forbidden_model_file") from None


def validate_metadata_dummy_environment(environ: Mapping[str, str]) -> MetadataDummyPolicy | None:
    """Validate metadata-dummy inputs and return their fixed offline policy."""

    if environ.get("AIC_MODEL_MODE") != "metadata_dummy":
        return None

    if any(environ.get(variable) for variable in _CREDENTIAL_VARIABLES):
        raise MetadataDummyPolicyError("forbidden_credential")

    if any(environ.get(variable) for variable in _REAL_DATA_FLAGS):
        raise MetadataDummyPolicyError("forbidden_override")
    if environ.get("FPM_REAL_WORKLOAD_SHAPE_SOURCE") not in (None, "", "synthetic"):
        raise MetadataDummyPolicyError("forbidden_override")
    if environ.get("AIC_LOAD_FORMAT") not in (None, "", "dummy"):
        raise MetadataDummyPolicyError("forbidden_override")

    model_dir_value = environ.get("AIC_MODEL_METADATA_DIR")
    if not model_dir_value:
        raise MetadataDummyPolicyError("missing_metadata_dir")

    model_dir = Path(model_dir_value)
    if not model_dir.is_absolute() or model_dir.is_symlink() or not model_dir.is_dir():
        raise MetadataDummyPolicyError("invalid_metadata_dir")
    try:
        model_dir = model_dir.resolve(strict=True)
    except OSError:
        raise MetadataDummyPolicyError("invalid_metadata_dir") from None

    config_path = model_dir / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise MetadataDummyPolicyError("missing_config")

    _validate_bundle_entries(model_dir)
    return MetadataDummyPolicy(model_dir=model_dir)
