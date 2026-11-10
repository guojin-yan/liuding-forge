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
SOURCE_SHA256_LF = {
    "dinov3/hub/backbones.py": "6c9a18aca0b5fcb1fce439213a94a528e072376a6023718e03dbd9fcfaa10deb",
    "dinov3/models/vision_transformer.py": "1af05405e19bc42188f80bfcb204b87ed1ec72af998e0bbd4658a85d7fae93f2",
    "dinov3/layers/rope_position_encoding.py": "af8ea35acb76d18ad4d53b844d5057bf01472a848375c51d70583a87f044efd1",
    "dinov3/layers/attention.py": "dfc21def6ba17d00cba18ea23a8726e7f1dd5a0dc8e43510852c999a173d37a8",
    "dinov3/layers/block.py": "809a3615ec93019042ab440f31c21d5242186d9e503c67ac0f42a324ceae1950",
    "dinov3/layers/ffn_layers.py": "fa2caadb52ff43a7e6888966a00ee07842fed9fd05c68305962cc1577d221fbe",
    "dinov3/layers/patch_embed.py": "c592c7262779c8e1789ed502aee81bdd55eba4607065f8d04ae95eea683d0ef1"
}
LOW_RESOURCE_VARIANTS = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
}
HIGH_RESOURCE_VARIANTS = {
    "dinov3_vitl16": 1024,
    "dinov3_vitl16plus": 1024,
}
PROFILES = {
    "low": (LOW_RESOURCE_VARIANTS, (2, 3, 35, 50), 20261025),
    "high": (HIGH_RESOURCE_VARIANTS, (1, 3, 35, 50), 20261026),
    "huge": ({"dinov3_vith16plus": 1280}, (1, 3, 35, 50), 20261110),
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
    parser.add_argument("--profile", choices=PROFILES, default="low")
    parser.add_argument("--environment", choices=("legacy", "cpu214"), default="legacy")
    parser.add_argument("--source-archive-provenance", type=Path)
    parser.add_argument("--verify-source-only", action="store_true")
    args = parser.parse_args()
    variants, input_shape, seed = PROFILES[args.profile]
    upstream = args.upstream_root.resolve(strict=True)
    if args.source_archive_provenance is None:
        revision = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip()
    else:
        provenance = json.loads(args.source_archive_provenance.read_text(encoding="utf-8"))
        revision = provenance["commit"]
        if provenance["archive_sha256"] != "94da41da596633a73619078bb3bbde64ff3be95c55e8658638fa57c542cc4fe7":
            raise ValueError("DINOv3 source archive differs from the pinned revision")
        for relative, identity in provenance["files"].items():
            source = (upstream / relative).resolve(strict=True)
            if not source.is_relative_to(upstream) or source.is_symlink():
                raise ValueError("DINOv3 source provenance leaves the pinned cache")
            if source.stat().st_size != identity["bytes"] or digest(source) != identity["sha256"]:
                raise ValueError(f"DINOv3 cached source differs: {relative}")
    if revision != UPSTREAM_COMMIT:
        raise ValueError("DINOv3 checkout differs from the pinned revision")
    source_digests = {}
    for relative, expected in SOURCE_SHA256.items():
        source_digests[relative] = digest(upstream / relative)
        if source_digests[relative] not in (expected, SOURCE_SHA256_LF[relative]):
            raise ValueError(f"DINOv3 source differs: {relative}")
    if args.verify_source_only:
        print(json.dumps({"status": "source-verified", "upstream_commit": revision,
                          "source_sha256": source_digests, "official_code_executed": False,
                          "model_executed": False}, sort_keys=True))
        return 0

    sys.path.insert(0, str(upstream))
    import numpy
    import safetensors
    import torch
    from dinov3.hub import backbones
    from safetensors.torch import save_file

    actual_environment = (platform.python_version(), torch.__version__, numpy.__version__, safetensors.__version__)
    locked_environments = {
        "legacy": ("3.13.12", "2.9.1+cpu", "2.5.4", "0.8.0"),
        "cpu214": ("3.11.18", "2.14.1+cpu", "2.3.3", "0.6.2"),
    }
    if actual_environment != locked_environments[args.environment]:
        raise ValueError("Python reference dependencies differ from the selected reviewed lock")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    threads = 1 if args.environment == "cpu214" else 2
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    cases = {}
    for index, (name, dimension) in enumerate(variants.items()):
        torch.manual_seed(seed + index)
        model = getattr(backbones, name)(pretrained=False).cpu().eval()
        inputs = torch.randn(input_shape, dtype=torch.float32)
        huge_intermediates = {}
        with torch.inference_mode():
            features = model.forward_features(inputs)
            tokens = torch.cat(
                (features["x_norm_clstoken"].unsqueeze(1),
                 features["x_storage_tokens"],
                 features["x_norm_patchtokens"]), dim=1
            )
            cls = model(inputs)
            if args.profile == "huge":
                selected = model.get_intermediate_layers(
                    inputs, n=[0, 15, 31], return_class_token=True,
                    return_extra_tokens=True, norm=True)
                for block, (patches, class_token, storage) in zip((0, 15, 31), selected):
                    huge_intermediates[f"block{block}_tokens"] = torch.cat(
                        (class_token.unsqueeze(1), storage, patches), dim=1).contiguous()

        patch_count = (input_shape[2] // 16) * (input_shape[3] // 16)
        if tokens.shape != (input_shape[0], 5 + patch_count, dimension) or \
                cls.shape != (input_shape[0], dimension):
            raise ValueError(f"Unexpected official DINOv3 output shape: {name}")
        if not torch.isfinite(tokens).all() or not torch.isfinite(cls).all():
            raise ValueError(f"Nonfinite official DINOv3 output: {name}")

        values = {
            "state": {key: value.detach().cpu().contiguous()
                      for key, value in model.state_dict().items()},
            "input": {"input": inputs.contiguous()},
            "output": {"tokens": tokens.contiguous(), "output": cls.contiguous()},
        }
        for key, value in huge_intermediates.items():
            if value.shape != tokens.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid official H+ intermediate: {key}")
        values["output"].update(huge_intermediates)
        files = {}
        for role, tensors in values.items():
            path = output / f"{name}-{role}.safetensors"
            save_file(tensors, str(path))
            files[role] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
        cases[name] = {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "state_keys": len(values["state"]),
            "parameter_tensors": len(list(model.named_parameters())),
            "buffer_tensors": len(list(model.named_buffers())),
            "intermediate_blocks": [0, 15, 31] if args.profile == "huge" else [],
            "state_inventory": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                "kind": "parameter" if key in dict(model.named_parameters()) else "buffer"}
                for key, value in values["state"].items()},

            "input_shape": list(input_shape),
            "token_shape": list(tokens.shape),
            "cls_shape": list(cls.shape),
            "files": files,
        }
        del model, inputs, features, tokens, cls, values

    evidence = {
        "schema_id": "jyppx.models.dinov3-reference",
        "schema_version": 1,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_sha256": source_digests,
        "historical_CRLF_source_sha256": SOURCE_SHA256,
        "reference_environment": args.environment,
        "profile": args.profile,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "numpy_version": numpy.__version__,
        "safetensors_version": safetensors.__version__,
        "device": "cpu",
        "intraop_threads": threads,
        "interop_threads": 1,
        "seed": seed,
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
