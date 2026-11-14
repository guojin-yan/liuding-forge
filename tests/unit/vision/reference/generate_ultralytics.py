# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Generate development-only state/output/gradient fixtures from local upstream.

Requires an explicitly supplied Ultralytics checkout and the independent CPU
PyTorch test environment. It downloads neither models nor datasets. The output
records the hashes of all upstream Python/YAML sources used for this run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--task", default="detect",
                        choices=["detect", "classify", "instance_segment", "pose", "obb"])
    parser.add_argument("--variant", default="p5")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--normalization-offset", type=float, default=0)
    parser.add_argument("--objective-output", type=int, default=-1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "settings").mkdir()
    root = args.upstream_root.resolve()
    os.environ.update(YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true",
                      YOLO_CONFIG_DIR=str(args.output / "settings"),
                      MPLCONFIGDIR=str(args.output / "matplotlib"))
    sys.path.insert(0, str(root))
    import torch
    import yaml
    import ultralytics
    from safetensors.torch import save_file
    from ultralytics.nn.tasks import (ClassificationModel, DetectionModel,
                                     OBBModel, PoseModel, SegmentationModel)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    family, scale = args.architecture[:-1], args.architecture[-1]
    directory = {"yolov8": "v8", "yolo11": "11", "yolo12": "12", "yolo26": "26"}[family]
    suffix = {"detect": "", "classify": "-cls", "instance_segment": "-seg",
              "pose": "-pose", "obb": "-obb"}[args.task]
    variant = {"p5": "", "p2": "-p2", "p6": "-p6", "ghost_p5": "-ghost",
               "ghost_p2": "-ghost-p2", "ghost_p6": "-ghost-p6"}[args.variant]
    config_path = root / "ultralytics/cfg/models" / directory / (family + suffix + variant + ".yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["scale"] = scale
    factory = {"detect": DetectionModel, "classify": ClassificationModel,
               "instance_segment": SegmentationModel, "pose": PoseModel, "obb": OBBModel}[args.task]
    model = factory(config, ch=3, nc=3, verbose=False)
    if args.normalization_offset:
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.bias.copy_(torch.linspace(-args.normalization_offset,
                                                     args.normalization_offset, module.num_features))
    model.eval()
    # Only select the raw-head schema. Child BN/dropout modules remain in eval.
    head = model.model[-1]
    head.training = True
    if family == "yolo26" and args.task == "instance_segment":
        # Include training-only semantic logits while retaining eval BN/dropout
        # in every child. Native raw outputs declare that auxiliary branch.
        head.proto.training = True
    state_path = args.output / "state.safetensors"
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(state_path))
    image = torch.randn(1, 3, args.size, args.size, requires_grad=True)
    predictions = model(image)

    def raw(prediction, heads):
        parts = [prediction["boxes"], prediction["scores"]]
        for key in ["mask_coefficient", "kpts", "kpts_sigma"]:
            if key in prediction:
                parts.append(prediction[key])
        if "angle" in prediction:
            features = prediction["feats"]
            parts.append(torch.cat([heads["angle_head"][i](x).flatten(2)
                                    for i, x in enumerate(features)], 2))
        return torch.cat(parts, 1)

    if args.task == "classify":
        outputs = [predictions]
    else:
        many = predictions.get("one2many", predictions)
        outputs = [raw(many, head.one2many)]
        if "one2one" in predictions:
            outputs.append(raw(predictions["one2one"], head.one2one))
        if "proto" in many:
            proto = many["proto"]
            outputs.extend(proto if isinstance(proto, tuple) else [proto])

    def record(value):
        value = value.detach().cpu().float()
        return {"shape": list(value.shape), "values": value.flatten().tolist()}

    fixture = {"schema": "vision.independent-reference/v1", "architecture": args.architecture,
               "task": args.task, "variant": args.variant, "classes": 3,
               "upstream_version": ultralytics.__version__, "seed": 17,
               "normalization_offset": args.normalization_offset,
               "torch_version": str(torch.__version__), "state": str(state_path.resolve()),
               "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
               "parameter_elements": sum(p.numel() for p in model.parameters()),
               "input": record(image), "outputs": [record(x) for x in outputs]}
    weights = [(i + 1) if args.objective_output in [-1, i] else 0 for i in range(len(outputs))]
    fixture["objective_weights"] = weights
    sum(weight * x.square().mean() for weight, x in zip(weights, outputs)).backward()
    fixture["input_gradient"] = record(image.grad)
    available = [(k, p) for k, p in model.named_parameters() if p.grad is not None]
    # Cover the first convolution and both ends of the task head inventory.
    selected = dict([available[0], available[len(available) // 2], available[-1]])
    fixture["gradients"] = {k: record(p.grad) for k, p in selected.items()}
    torch.optim.SGD(model.parameters(), lr=.001).step()
    fixture["updated"] = {k: record(p) for k, p in selected.items()}
    fixture["source_sha256"] = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "ultralytics").rglob("*.py"))}
    fixture["source_sha256"][config_path.relative_to(root).as_posix()] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (args.output / "fixture.json").write_text(json.dumps(fixture, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Independent fixture generated:", args.architecture, args.task, args.variant,
          "parameters=", fixture["parameter_elements"], flush=True)


if __name__ == "__main__":
    main()
