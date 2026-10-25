#!/usr/bin/env python3
"""Generate offline random-state DINOv3 ViT references from a pinned Meta checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


UPSTREAM_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
SOURCE_SHA256 = {
    "dinov3/hub/backbones.py": "821b0f6f034fd866862d1b355036321389436fd534766829251cd4e19da2ef45",
    "dinov3/models/vision_transformer.py": "1395c5a6297861101db35af44a04613a0c60a502c947fd3b54092852ddd682b5",
    "dinov3/layers/rope_position_encoding.py": "41f233d8a34f304e16a621508ff8040d58e4a99a445a6bf9e3b08b2e742b391c",
    "dinov3/layers/attention.py": "ac8677e640f6c365c63f60d684bc01c38a4018298e3cfc2238d44e5e359364f2",
    "dinov3/layers/block.py": "1a5639a0147ffa26d9afb89315121ecf1930774a443070ea5ee43f107ef95629",
    "dinov3/layers/ffn_layers.py": "62875b278c7045b6538581f8e5c19fc2b89505c028832bf95359ce4f3d8b3741",
    "dinov3/layers/patch_embed.py": "a9ec7336332ac0b0073a60ff79040a250d05e719f3ab85fcb6759a7d97dff6be",
}
VARIANTS = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
}
INPUT_SHAPE = (2, 3, 35, 50)
SEED = 20261025


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    upstream = args.upstream_root.resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError("DINOv3 checkout differs from the pinned revision")
    for relative, expected in SOURCE_SHA256.items():
        if digest(upstream / relative) != expected:
            raise ValueError(f"DINOv3 source differs: {relative}")

    sys.path.insert(0, str(upstream))
    import numpy
    import safetensors
    import torch
    from dinov3.hub import backbones
    from safetensors.torch import save_file

    if (platform.python_version() != "3.13.12" or
            torch.__version__ != "2.9.1+cpu" or
            numpy.__version__ != "2.5.4" or
            safetensors.__version__ != "0.8.0"):
        raise ValueError("Python reference dependencies differ from the reviewed lock")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cases = {}
    for index, (name, dimension) in enumerate(VARIANTS.items()):
        torch.manual_seed(SEED + index)
        model = getattr(backbones, name)(pretrained=False).cpu().eval()
        inputs = torch.randn(INPUT_SHAPE, dtype=torch.float32)
        with torch.inference_mode():
            features = model.forward_features(inputs)
            tokens = torch.cat(
                (features["x_norm_clstoken"].unsqueeze(1),
                 features["x_storage_tokens"],
                 features["x_norm_patchtokens"]), dim=1
            )
            cls = model(inputs)
        if tokens.shape != (2, 11, dimension) or cls.shape != (2, dimension):
            raise ValueError(f"Unexpected official DINOv3 output shape: {name}")
        if not torch.isfinite(tokens).all() or not torch.isfinite(cls).all():
            raise ValueError(f"Nonfinite official DINOv3 output: {name}")

        values = {
            "state": {key: value.detach().cpu().contiguous()
                      for key, value in model.state_dict().items()},
            "input": {"input": inputs.contiguous()},
            "output": {"tokens": tokens.contiguous(), "output": cls.contiguous()},
        }
        files = {}
        for role, tensors in values.items():
            path = output / f"{name}-{role}.safetensors"
            save_file(tensors, str(path))
            files[role] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
        cases[name] = {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "state_keys": len(values["state"]),
            "input_shape": list(INPUT_SHAPE),
            "token_shape": list(tokens.shape),
            "cls_shape": list(cls.shape),
            "files": files,
        }
        del model, inputs, features, tokens, cls, values

    evidence = {
        "schema_id": "jyppx.models.dinov3-reference",
        "schema_version": 1,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_sha256": SOURCE_SHA256,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "numpy_version": numpy.__version__,
        "safetensors_version": safetensors.__version__,
        "device": "cpu",
        "intraop_threads": 2,
        "interop_threads": 1,
        "seed": SEED,
        "pretrained": False,
        "weight_license": "NOASSERTION",
        "qualification": "random-architecture-differential-only",
        "cases": cases,
    }
    (output / "dinov3-reference-evidence.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "generated", "cases": cases}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
