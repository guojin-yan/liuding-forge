#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Generate a fixed timm Swin-T random architecture reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys


ARCHITECTURE = "swin_tiny_patch4_window7_224"
TIMM_COMMIT = "af97f99efef62bd9bd7eaa908fffe5b7a2d98e2c"
SOURCE_HASHES = {
    "timm/models/swin_transformer.py":
        "e15412dcc7cdc2a85203ac02d7cb0871fdcfc9b4744589fca884c494411038da",
    "timm/layers/classifier.py":
        "688e8f77964fa00e24a682c69c3a63c6522bfa1919116bcbcbc515548061e2d5",
    "timm/layers/patch_embed.py":
        "535cf495fb01c6216536b3a8d1fdaa25f67600df70522fd253d034b63b0a8c7b",
    "timm/layers/mlp.py":
        "c024128a1c7e2a55ebc5e33a7c94c585ae1e960cd032b39b08445ea30f05c5a5",
    "timm/layers/drop.py":
        "2047ce241398537d6e502716f1f3bb7a285b006554f2e27e9a68618d68376ac2",
    "timm/version.py":
        "abcf490ed112377cc1cf5f697fca8a48aadb89c81bb7b6033cfa8af593dbdf5c",
}
VERSIONS = {
    "python": "3.11.9",
    "torch": "2.14.1+cpu",
    "timm": "1.0.31.dev0",
    "safetensors": "0.6.2",
}
FROZEN_FILES = {
    "swin_tiny_patch4_window7_224-input.safetensors": {
        "size_bytes": 56528,
        "sha256": "d023736e18dca4c3a9194b1e4d7b0b6eee1eba26304ac062772dbe91fec5cf39",
    },
    "swin_tiny_patch4_window7_224-output.safetensors": {
        "size_bytes": 231712,
        "sha256": "0e7745d7c4bb2a73c9f8c1d2f12df0717851c5623bd8ec8776d745739140a632",
    },
    "swin_tiny_patch4_window7_224-state.safetensors": {
        "size_bytes": 113170824,
        "sha256": "f357952715690e031314b8f7c6fae72119ed2c2f741afd003a363938c50f4bfe",
    },
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_files(directory: Path) -> dict:
    observed = {
        name: {
            "size_bytes": (directory / name).stat().st_size,
            "sha256": digest(directory / name),
        }
        for name in FROZEN_FILES
    }
    if observed != FROZEN_FILES:
        raise ValueError("Fixed Swin-T reference identities differ")
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
    torch.manual_seed(1729)
    model = timm.create_model(
        ARCHITECTURE, pretrained=False, dynamic_img_size=True
    ).eval()
    image = torch.randn((1, 3, 56, 84), dtype=torch.float32)
    stages = {}
    handles = []
    for index in range(4):
        def capture(_module, _inputs, output, stage=index):
            stages[f"stage{stage}"] = output.permute(0, 3, 1, 2).contiguous()
        handles.append(model.layers[index].register_forward_hook(capture))
    with torch.inference_mode():
        logits = model(image)
    for handle in handles:
        handle.remove()

    files = {}
    for role, tensors in (
        ("state", model.state_dict()),
        ("input", {"input": image}),
        ("output", {"output": logits, **stages}),
    ):
        name = f"{ARCHITECTURE}-{role}.safetensors"
        path = directory / name
        save_file(
            {key: value.detach().cpu().contiguous()
             for key, value in tensors.items()},
            str(path),
        )
        files[name] = {"size_bytes": path.stat().st_size, "sha256": digest(path)}

    return {
        "schema_id": "jyppx.models.swin-generation",
        "schema_version": 1,
        "timm_commit": TIMM_COMMIT,
        "source_sha256": SOURCE_HASHES,
        "versions": observed_versions,
        "platform": platform.platform(),
        "seed": 1729,
        "pretrained": False,
        "reference_source": "timm.create_model(name, pretrained=False, dynamic_img_size=True)",
        "architecture": ARCHITECTURE,
        "parameters": sum(value.numel() for value in model.parameters()),
        "state_tensors": len(model.state_dict()),
        "depths": [len(stage.blocks) for stage in model.layers],
        "input_shape": list(image.shape),
        "logits_shape": list(logits.shape),
        "stage_shapes": {name: list(value.shape) for name, value in stages.items()},
        "preprocessing_candidate": {
            "input_size": list(model.pretrained_cfg["input_size"]),
            "interpolation": model.pretrained_cfg["interpolation"],
            "crop_pct": model.pretrained_cfg["crop_pct"],
            "mean": list(model.pretrained_cfg["mean"]),
            "std": list(model.pretrained_cfg["std"]),
        },
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
    verify_files(directory)
    (directory / "swin-generation-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
