"""Safety policy for metadata-only layerwise model bundles."""

from __future__ import annotations

import argparse
import os
import sys
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


def validate_metadata_dummy_runtime(
    environ: Mapping[str, str],
    *,
    model: str | None = None,
    load_format: str | None = None,
) -> MetadataDummyPolicy | None:
    """Validate metadata mode plus runtime aliases and an optional model path."""

    policy = validate_metadata_dummy_environment(environ)
    if policy is None:
        return None

    for value in (environ.get("LOAD_FORMAT"), load_format):
        if value not in (None, ""):
            runtime_environ = dict(environ)
            runtime_environ["AIC_LOAD_FORMAT"] = value
            validate_metadata_dummy_environment(runtime_environ)
    for variable in ("FPM_SHAPE_SOURCE", "REAL_WORKLOAD_SHAPE_SOURCE"):
        value = environ.get(variable)
        if value not in (None, ""):
            runtime_environ = dict(environ)
            runtime_environ["FPM_REAL_WORKLOAD_SHAPE_SOURCE"] = value
            validate_metadata_dummy_environment(runtime_environ)

    if model not in (None, ""):
        candidate = Path(model)
        try:
            candidate = candidate.resolve(strict=True)
        except OSError:
            raise MetadataDummyPolicyError("forbidden_override") from None
        if not candidate.is_dir() or candidate != policy.model_dir:
            raise MetadataDummyPolicyError("forbidden_override")
    return policy


def enforce_metadata_dummy_load_format(
    args: list[str] | tuple[str, ...],
    environ: Mapping[str, str],
) -> list[str]:
    """Return CLI args with one fixed dummy load format in metadata mode."""

    policy = validate_metadata_dummy_runtime(environ)
    if policy is None:
        return list(args)

    normalized: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--load-format":
            if index + 1 >= len(args) or args[index + 1] != policy.load_format:
                raise MetadataDummyPolicyError("forbidden_override")
            index += 2
            continue
        if token.startswith("--load-format="):
            if token.partition("=")[2] != policy.load_format:
                raise MetadataDummyPolicyError("forbidden_override")
            index += 1
            continue
        normalized.append(token)
        index += 1
    normalized.append(f"--load-format={policy.load_format}")
    return normalized


def metadata_dummy_environment_overrides(policy: MetadataDummyPolicy | None) -> dict[str, str]:
    """Return the fixed environment inherited by metadata-only subprocesses."""

    if policy is None:
        return {}
    return {
        "AIC_LOAD_FORMAT": policy.load_format,
        "AIC_MODEL_METADATA_DIR": str(policy.model_dir),
        "AIC_MODEL_MODE": "metadata_dummy",
        "FPM_REAL_WORKLOAD_SHAPE_SOURCE": policy.shape_source,
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate layerwise metadata-only runtime policy.")
    parser.add_argument("--model")
    parser.add_argument("--load-format")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    extra_args = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
    try:
        policy = validate_metadata_dummy_runtime(
            os.environ,
            model=args.model,
            load_format=args.load_format,
        )
        enforce_metadata_dummy_load_format(extra_args, os.environ)
    except MetadataDummyPolicyError as error:
        print(error, file=sys.stderr)
        return 2
    if policy is not None:
        print(policy.model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
