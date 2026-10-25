#!/usr/bin/env python3
"""Generate pinned Meta DINOv3 ConvNeXt random-state references."""

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
    "dinov3/models/convnext.py": "a8c8017839c7db5621fdee50570d664e3027a1ddcd5cad85f60c01f0139f5f9d",
}
VARIANTS = ("dinov3_convnext_tiny", "dinov3_convnext_small")
INPUT_SHAPE = (1, 3, 65, 97)
SEED = 20261026
ALIASES = {
    "norms.3.weight": "norm.weight",
    "norms.3.bias": "norm.bias",
}


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
    for index, name in enumerate(VARIANTS):
        torch.manual_seed(SEED + index)
        model = getattr(backbones, name)(pretrained=False).cpu().eval()
        official_state = model.state_dict()
        for alias, canonical in ALIASES.items():
            if (alias not in official_state or canonical not in official_state or
                    official_state[alias].data_ptr() != official_state[canonical].data_ptr() or
                    not torch.equal(official_state[alias], official_state[canonical])):
                raise ValueError(f"Unexpected official state alias: {alias}")
        state = {key: value.detach().cpu().contiguous()
                 for key, value in official_state.items() if key not in ALIASES}
        if set(official_state) != set(state) | set(ALIASES):
            raise ValueError("Unexpected official state keys")
        if set(state) != set(dict(model.named_parameters())):
            raise ValueError("Canonical state differs from unique model parameters")

        inputs = torch.randn(INPUT_SHAPE, dtype=torch.float32)
        cropped = inputs[..., :64, :96]
        with torch.inference_mode():
            stages = model.get_intermediate_layers(
                cropped, n=[0, 1, 2, 3], reshape=True, norm=False
            )
            global_feature = model(cropped)
        expected_shapes = (
            (1, 96, 16, 24), (1, 192, 8, 12),
            (1, 384, 4, 6), (1, 768, 2, 3),
        )
        if tuple(tuple(stage.shape) for stage in stages) != expected_shapes:
            raise ValueError(f"Unexpected official stage shapes: {name}")
        if global_feature.shape != (1, 768):
            raise ValueError(f"Unexpected official global shape: {name}")
        if not all(torch.isfinite(stage).all() for stage in stages):
            raise ValueError(f"Nonfinite official stage: {name}")
        if not torch.isfinite(global_feature).all():
            raise ValueError(f"Nonfinite official global feature: {name}")

        values = {
            "state": state,
            "input": {"input": inputs.contiguous()},
            "output": {
                **{f"stage{stage_index}": stage.contiguous()
                   for stage_index, stage in enumerate(stages)},
                "output": global_feature.contiguous(),
            },
        }
        files = {}
        for role, tensors in values.items():
            path = output / f"{name}-{role}.safetensors"
            save_file(tensors, str(path))
            files[role] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
        cases[name] = {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "official_state_keys": len(official_state),
            "canonical_state_keys": len(state),
            "input_shape": list(INPUT_SHAPE),
            "stage_shapes": [list(stage.shape) for stage in stages],
            "global_shape": list(global_feature.shape),
            "files": files,
        }

    evidence = {
        "schema_id": "jyppx.models.dinov3-convnext-reference",
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
        "input_contract": "floor-crop to 64x96 before official forward",
        "verified_duplicate_state_aliases": ALIASES,
        "weight_license": "NOASSERTION",
        "qualification": "random-architecture-differential-only",
        "cases": cases,
    }
    (output / "dinov3-convnext-reference-evidence.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "generated", "cases": cases}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
