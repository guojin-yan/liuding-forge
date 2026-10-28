#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Create offline actual-weight references; no task-accuracy qualification.

Requires the explicitly supplied pinned files and reviewed Python packages.
Outputs are independent Python logits/tokens for the C++ external-weight gate.
The evaluation image is synthetic RGB, using torchvision tensor bicubic resize;
it does not establish PIL pixel equivalence or real-dataset accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import sys


FROZEN_WEIGHTS = {
    "resnet50.a1_in1k.safetensors": (102469840, "773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb"),
    "dinov2-small.facebook.safetensors": (88249960, "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"),
    "dinov2-small.liuding.safetensors": (88240512, "f3ef1ae01fd4af3cd15926a0d5e631fa0caea50ca68b29db8a68c8c84c9810e5"),
}
TIMM_COMMIT = "af97f99efef62bd9bd7eaa908fffe5b7a2d98e2c"
TRANSFORMERS_SOURCE_SHA256 = "81116a0fddfd3d96f9d175cc56a22ded11382eb21bc78748e79b29d34660f862"
TIMM_RESNET_SOURCE_SHA256 = "a60602278b16dd235189212426629c824ce8a6d923764295520d62a4272f2179"
REFERENCE_IDENTITY = (9884068, "4736a2ef8f98bfd773700c6d06fe2614a0e5fb8d6d8c02c36b9f0c643d6250a6")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--timm-dir", required=True, type=Path)
    p.add_argument("--reference-packages", type=Path)
    p.add_argument("--jyppx-commit", required=True)
    args = p.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.jyppx_commit) is None:
        raise ValueError("Full source revision required")
    sys.path.append(str(args.timm_dir.resolve(strict=True)))
    if args.reference_packages:
        sys.path.append(str(args.reference_packages.resolve(strict=True)))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import safetensors
    from safetensors.torch import load_file, save_file
    import timm
    import torch
    import torchvision
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    from transformers import Dinov2Config
    from transformers.models.dinov2 import modeling_dinov2

    if (torch.__version__ != "2.14.1+cpu" or torchvision.__version__ != "0.29.1+cpu" or
            safetensors.__version__ != "0.6.2" or importlib.metadata.version("transformers") != "4.56.2"):
        raise ValueError("Reference packages differ from the reviewed lock")
    hf_source = Path(inspect.getsourcefile(modeling_dinov2)).resolve()
    if digest(hf_source) != TRANSFORMERS_SOURCE_SHA256:
        raise ValueError("Transformers DINOv2 implementation differs")
    for name, (size, sha) in FROZEN_WEIGHTS.items():
        path = args.weights_dir / name
        if path.stat().st_size != size or digest(path) != sha:
            raise ValueError("Weight identity mismatch: " + name)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "pretrained-reference.json"
    if report_path.exists():
        raise ValueError("Use a new output directory to preserve previous evidence")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    # No RNG or image decoder is needed for the input; exact rational pixels.
    image = (torch.arange(3 * 311 * 457, dtype=torch.int64).remainder(251).to(torch.float32) / 250).reshape(3, 311, 457)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tensors = {"image": image}
    for prefix, shorter in [("resnet50", 235), ("dinov2", 256)]:
        resized = TF.resize(image, shorter, InterpolationMode.BICUBIC, antialias=True)
        tensors[prefix + "_input"] = TF.normalize(TF.center_crop(resized, [224, 224]), mean, std).unsqueeze(0).contiguous()
    tensors["dinov2_dynamic_input"] = TF.resize(tensors["dinov2_input"], [224, 280], InterpolationMode.BICUBIC, antialias=True)
    tensors["dinov2_native_input"] = TF.resize(tensors["dinov2_input"], [518, 518], InterpolationMode.BICUBIC, antialias=True)
    sources = {}
    with torch.inference_mode():
        model = timm.create_model("resnet50.a1_in1k", pretrained=False).cpu().eval()
        model.load_state_dict(load_file(str(args.weights_dir / "resnet50.a1_in1k.safetensors")), strict=True)
        tensors["resnet50_output"] = model(tensors["resnet50_input"]).contiguous()
        resnet_source = Path(inspect.getsourcefile(timm.models.resnet)).resolve()
        if digest(resnet_source) != TIMM_RESNET_SOURCE_SHA256:
            raise ValueError("timm ResNet implementation differs")
        sources["timm_resnet"] = {"revision": TIMM_COMMIT, "sha256": digest(resnet_source)}
        del model
        config = Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                              image_size=518, patch_size=14, layer_norm_eps=1e-6, use_swiglu_ffn=False)
        config._attn_implementation = "eager"
        model = modeling_dinov2.Dinov2Model(config).cpu().eval()
        model.load_state_dict(load_file(str(args.weights_dir / "dinov2-small.facebook.safetensors")), strict=True)
        for tag in ["dinov2", "dinov2_dynamic", "dinov2_native"]:
            result = model(pixel_values=tensors[tag + "_input"])
            tensors[tag + "_tokens"] = result.last_hidden_state.contiguous()
            tensors[tag + "_output"] = result.pooler_output.contiguous().clone()
        sources["transformers_dinov2"] = {"version": "4.56.2", "sha256": digest(hf_source), "attention": "eager"}
        del model
    for value in tensors.values():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Reference contains nonfinite or non-F32 data")
    path = output / "pretrained-reference.safetensors"
    save_file(dict(sorted(tensors.items())), str(path))
    if (path.stat().st_size, digest(path)) != REFERENCE_IDENTITY:
        raise ValueError("Generated reference differs from the reviewed complete identity")
    inventory = {name: {"shape": list(t.shape), "dtype": str(t.dtype)} for name,t in tensors.items()}
    report = {
        "schema_id": "jyppx.liuding.pretrained-reference", "schema_version": 1,
        "source_base": args.jyppx_commit, "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "safetensors": safetensors.__version__, "sources": sources,
        "weights": {n: {"size_bytes": s, "sha256": h} for n, (s,h) in FROZEN_WEIGHTS.items()},
        "file": {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest(path)},
        "tensors": inventory,
        "task_qualification": "none: deterministic synthetic RGB; actual pretrained bytes and inference only",
        "preprocessing": "torchvision tensor bicubic antialias, shortest edge then center crop, unit RGB ImageNet mean/std; no PIL pixel-equivalence claim",
    }
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file": report["file"], "sources": sources}), flush=True)


if __name__ == "__main__":
    main()
