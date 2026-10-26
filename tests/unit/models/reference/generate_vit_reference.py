#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Generate pinned timm ViT/16 Tiny, Small and Base random references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys


ARCHITECTURES = (
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
)
TIMM_COMMIT = "af97f99efef62bd9bd7eaa908fffe5b7a2d98e2c"
SOURCE_HASHES = {
    "timm/models/vision_transformer.py":
        "e3435821723dffe0d0e58d86d72e8337ce1106c1041f7b87ed37ed685ea3cd79",
    "timm/layers/attention.py":
        "91094ccb54a743b7e70f38431c7e330e750ce4f575d49ba34153b16829bb5622",
    "timm/layers/patch_embed.py":
        "535cf495fb01c6216536b3a8d1fdaa25f67600df70522fd253d034b63b0a8c7b",
    "timm/layers/mlp.py":
        "c024128a1c7e2a55ebc5e33a7c94c585ae1e960cd032b39b08445ea30f05c5a5",
    "timm/layers/pos_embed.py":
        "550a15f5dd3b47eae90e32416afec00d7635d481b594b4c41e68e3d30f5a5ac7",
    "timm/layers/norm.py":
        "afe3abf01a4379065c1c11bf6b6f947a1a743aff7f5e51849b353697b0db557d",
    "timm/version.py":
        "abcf490ed112377cc1cf5f697fca8a48aadb89c81bb7b6033cfa8af593dbdf5c",
}
VERSIONS = {
    "python": "3.11.9",
    "torch": "2.14.1+cpu",
    "timm": "1.0.31.dev0",
    "safetensors": "0.6.2",
}
FROZEN_FILES: dict[str, dict[str, int | str]] = {
    "vit_base_patch16_224-input.safetensors": {
        "size_bytes": 61520,
        "sha256": "925f5afae36eb544abae13955bdaf38043b3f9a2896bd6294c8f27b70bb27a79",
    },
    "vit_base_patch16_224-output.safetensors": {
        "size_bytes": 130176,
        "sha256": "95dacd0bcea8a5c1c32bd4dd2fa646d579d9e8e6e65ec5e6e5eda88f93ed7455",
    },
    "vit_base_patch16_224-state.safetensors": {
        "size_bytes": 346284720,
        "sha256": "2487fd0ef6843d84385519d9203ec77b2d2708fff4bcbeb41d77edf68fa717a0",
    },
    "vit_small_patch16_224-input.safetensors": {
        "size_bytes": 61520,
        "sha256": "37414e39ba673efa7e1067ba798e78e7cff6843fb5681d6d9a58f5bf5984de25",
    },
    "vit_small_patch16_224-output.safetensors": {
        "size_bytes": 67200,
        "sha256": "c30956800473b73928faff052c8ae5c3adb260a44d67c1b7308ef3dc2c29c58d",
    },
    "vit_small_patch16_224-state.safetensors": {
        "size_bytes": 88216496,
        "sha256": "74369f12ddb44da2b4d136dcdea37a1eeffe1d80ec7c482fd4892b359eaae52b",
    },
    "vit_tiny_patch16_224-input.safetensors": {
        "size_bytes": 61520,
        "sha256": "0e4e52430c8f5b41acb0613a0d62a0fd6cf95d2c592bb5f4643dd788f1aff2d3",
    },
    "vit_tiny_patch16_224-output.safetensors": {
        "size_bytes": 35712,
        "sha256": "0220d6beb10b36e144583e95c641cbc5317d3b9c96faafc98a3344d5de88ab23",
    },
    "vit_tiny_patch16_224-state.safetensors": {
        "size_bytes": 22883344,
        "sha256": "f7eb01186c3278faa69785b45f7a96c53b332b45ee093959e8116cc25515d52f",
    },
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def file_identities(directory: Path) -> dict:
    return {
        name: {
            "size_bytes": (directory / name).stat().st_size,
            "sha256": digest(directory / name),
        }
        for architecture in ARCHITECTURES
        for name in (
            f"{architecture}-state.safetensors",
            f"{architecture}-input.safetensors",
            f"{architecture}-output.safetensors",
        )
    }


def verify_files(directory: Path) -> dict:
    if len(FROZEN_FILES) != 9:
        raise ValueError("All nine reviewed ViT reference identities are required")
    observed = file_identities(directory)
    if observed != FROZEN_FILES:
        raise ValueError("Fixed ViT reference identities differ")
    return observed


def generate(directory: Path, timm_source: Path) -> dict:
    for name, expected in SOURCE_HASHES.items():
        if digest(timm_source / name) != expected:
            raise ValueError("Pinned timm source differs: " + name)
    sys.path.insert(0, str(timm_source))
    import torch
    import timm
    import safetensors
    from safetensors.torch import save_file

    if not Path(timm.__file__).resolve().is_relative_to(timm_source):
        raise ValueError("Imported timm is not the pinned source tree")
    observed_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "safetensors": safetensors.__version__,
    }
    if observed_versions != VERSIONS:
        raise ValueError(f"Reference toolchain differs: {observed_versions}")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    models = {}
    for architecture in ARCHITECTURES:
        torch.manual_seed(1729)
        model = timm.create_model(
            architecture, pretrained=False, dynamic_img_size=True
        ).eval()
        image = torch.randn((1, 3, 64, 80), dtype=torch.float32)
        with torch.inference_mode():
            tokens = model.forward_features(image)
            logits = model.forward_head(tokens)
            grid = tokens[:, 1:].transpose(1, 2).reshape(
                1, tokens.shape[-1], 4, 5
            )
        for role, tensors in (
            ("state", model.state_dict()),
            ("input", {"input": image}),
            ("output", {"output": logits, "tokens": tokens, "grid": grid}),
        ):
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in tensors.items()
                },
                str(directory / f"{architecture}-{role}.safetensors"),
            )
        models[architecture] = {
            "parameters": sum(value.numel() for value in model.parameters()),
            "state_tensors": len(model.state_dict()),
            "input_shape": list(image.shape),
            "logits_shape": list(logits.shape),
            "tokens_shape": list(tokens.shape),
            "grid_shape": list(grid.shape),
            "preprocessing_candidate": {
                key: model.default_cfg[key]
                for key in (
                    "input_size",
                    "interpolation",
                    "crop_pct",
                    "mean",
                    "std",
                )
            },
        }
        del model

    files = file_identities(directory)
    if FROZEN_FILES and files != FROZEN_FILES:
        raise ValueError("Generated bytes differ from reviewed fixed identities")
    return {
        "schema_id": "jyppx.models.vit-generation",
        "schema_version": 1,
        "generator_sha256": digest(Path(__file__)),
        "platform": platform.platform(),
        "versions": observed_versions,
        "timm_commit": TIMM_COMMIT,
        "source_sha256": SOURCE_HASHES,
        "seed": 1729,
        "reference_source": "timm.create_model(name, pretrained=False, dynamic_img_size=True)",
        "pretrained": False,
        "models": models,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--timm-source", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    directory = args.fixture_dir.resolve()
    if args.verify_only:
        print(json.dumps(verify_files(directory), indent=2, sort_keys=True))
        return 0
    if args.timm_source is None:
        parser.error("--timm-source is required when generating")
    directory.mkdir(parents=True, exist_ok=True)
    evidence = generate(directory, args.timm_source.resolve())
    (directory / "vit-generation-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
