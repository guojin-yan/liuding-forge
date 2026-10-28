#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Offline Fashion-MNIST linear-probe oracle for two pinned pretrained models.

Uses the first 100 samples per class in each official split, without a holdout
search. Grayscale is repeated to RGB before the independently sourced weight
preprocessing. No dataset, weight or package acquisition is performed here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import struct
import sys
import time

from generate_pretrained_reference import (
    FROZEN_WEIGHTS, TIMM_COMMIT, TIMM_RESNET_SOURCE_SHA256,
    TRANSFORMERS_SOURCE_SHA256, digest,
)

DATA_REVISION = "b2617bb6d3ffa2e429640350f613e3291e10b141"
DATA_SHA256 = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--weights-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--timm-dir", required=True, type=Path)
    p.add_argument("--reference-packages", required=True, type=Path)
    a = p.parse_args()
    sys.path.append(str(a.timm_dir.resolve(strict=True)))
    sys.path.append(str(a.reference_packages.resolve(strict=True)))
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
        raise ValueError("Reference package lock differs")
    for name, (size, sha) in FROZEN_WEIGHTS.items():
        path = a.weights_dir / name
        if path.stat().st_size != size or digest(path) != sha:
            raise ValueError("Weight identity differs: " + name)
    resnet_source = Path(inspect.getsourcefile(timm.models.resnet)).resolve()
    hf_source = Path(inspect.getsourcefile(modeling_dinov2)).resolve()
    if digest(resnet_source) != TIMM_RESNET_SOURCE_SHA256 or digest(hf_source) != TRANSFORMERS_SOURCE_SHA256:
        raise ValueError("Independent implementation lock differs")
    for name, sha in DATA_SHA256.items():
        if digest(a.data_dir / name) != sha:
            raise ValueError("Fashion-MNIST identity differs: " + name)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    tensors = {}
    for split, prefix, count in [("train", "train", 60000), ("test", "t10k", 10000)]:
        with gzip.open(a.data_dir / (prefix + "-images-idx3-ubyte.gz"), "rb") as f:
            raw = f.read(count * 784 + 17)
        with gzip.open(a.data_dir / (prefix + "-labels-idx1-ubyte.gz"), "rb") as f:
            labels = f.read(count + 9)
        if (len(raw) != count * 784 + 16 or struct.unpack(">4I", raw[:16]) != (2051, count, 28, 28) or
                len(labels) != count + 8 or struct.unpack(">2I", labels[:8]) != (2049, count)):
            raise ValueError("Malformed IDX source")
        selected, seen = [], [0] * 10
        for i, label in enumerate(labels[8:]):
            if label >= 10:
                raise ValueError("Fashion-MNIST label out of range")
            if seen[label] < 100:
                selected.append(i)
                seen[label] += 1
        if seen != [100] * 10:
            raise ValueError("Class quota unavailable")
        tensors[split + "_indices"] = torch.tensor(selected, dtype=torch.int64)
        tensors[split + "_labels"] = torch.tensor([labels[8+i] for i in selected], dtype=torch.int64)
        pixels = bytearray(b"".join(raw[16+i*784:16+(i+1)*784] for i in selected))
        tensors[split + "_images"] = torch.frombuffer(pixels, dtype=torch.uint8).clone().reshape(1000, 1, 28, 28)
        del raw, labels, pixels
    selection = hashlib.sha256(b"jyppx.liuding.fashion-mnist.first100-per-class.v1\n")
    for key in ["train_indices", "train_images", "train_labels", "test_indices", "test_images", "test_labels"]:
        selection.update(key.encode() + b"\0")
        selection.update(tensors[key].numpy().tobytes())
    results = {}
    for name, shorter in [("resnet50", 235), ("dinov2", 256)]:
        started = time.monotonic()
        if name == "resnet50":
            model = timm.create_model("resnet50.a1_in1k", pretrained=False)
            model.load_state_dict(load_file(str(a.weights_dir / "resnet50.a1_in1k.safetensors")), strict=True)
            model.reset_classifier(0)
        else:
            config = Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                                  image_size=518, patch_size=14, layer_norm_eps=1e-6, use_swiglu_ffn=False)
            config._attn_implementation = "eager"
            model = modeling_dinov2.Dinov2Model(config)
            model.load_state_dict(load_file(str(a.weights_dir / "dinov2-small.facebook.safetensors")), strict=True)
        model.eval().requires_grad_(False)
        for split in ["train", "test"]:
            features = []
            with torch.no_grad():
                for start in range(0, 1000, 16):
                    x = tensors[split + "_images"][start:start+16].to(torch.float32).div(255).expand(-1, 3, -1, -1)
                    x = TF.resize(x, shorter, InterpolationMode.BICUBIC, antialias=True)
                    x = TF.normalize(TF.center_crop(x, [224,224]), [0.485,0.456,0.406], [0.229,0.224,0.225])
                    output = model(x) if name == "resnet50" else model(pixel_values=x).pooler_output
                    features.append(output.contiguous())
            value = torch.cat(features)
            tensors[name + "_" + split + "_features"] = torch.nn.functional.normalize(value, dim=1).contiguous()
            print(name, split, "features", list(value.shape), flush=True)
        del model
        # Exactly 40 full-batch, zero-initialized, unregularized SGD steps.
        x = tensors[name + "_train_features"]
        head = torch.nn.Linear(x.shape[1], 10)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        optimizer = torch.optim.SGD(head.parameters(), lr=10.)
        losses = []
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(x), tensors["train_labels"])
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            prediction = head(tensors[name + "_test_features"])
            labels = tensors["test_labels"]
            counts = torch.bincount(labels * 10 + prediction.argmax(1), minlength=100).reshape(10,10)
            correct = counts.diag().double()
            precision = correct / counts.sum(0).clamp_min(1)
            recall = correct / counts.sum(1).clamp_min(1)
            f1 = 2*correct / (counts.sum(0)+counts.sum(1)).clamp_min(1)
            results[name] = {"accuracy": float(correct.sum()/1000), "macro_precision": float(precision.mean()),
                             "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
                             "confusion": counts.tolist(), "losses": losses,
                             "elapsed_seconds": round(time.monotonic()-started,3)}
            tensors[name + "_head_weight"] = head.weight.detach().contiguous()
            tensors[name + "_head_bias"] = head.bias.detach().contiguous()
            tensors[name + "_test_logits"] = prediction.contiguous()
        print(name, results[name]["accuracy"], results[name]["macro_f1"], flush=True)
    path = a.output_dir / "classification-reference.safetensors"
    save_file(dict(sorted(tensors.items())), str(path))
    report = {"schema_id": "jyppx.liuding.classification-reference", "schema_version": 1,
              "dataset": {"repo": "zalandoresearch/fashion-mnist", "revision": DATA_REVISION,
                          "license": "MIT", "selection_sha256": selection.hexdigest(), "files": DATA_SHA256, "selection": "first 100 per class per official split",
                          "class_order": ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]},
              "file": {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest(path)},
              "reference": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                            "safetensors": safetensors.__version__, "timm_commit": TIMM_COMMIT,
                            "resnet_source": digest(resnet_source), "transformers_source": digest(hf_source)},
              "task": "frozen L2-normalized pretrained embedding, new 10-class linear head; 40 full-batch SGD lr=10, zero initialization",
              "thresholds_frozen_before_execution": {"accuracy": 0.65, "macro_f1": 0.60},
              "limits": "balanced 1000/1000 subset only; no full fine-tuning, checkpoint, complete dataset or product performance qualification",
              "results": results}
    (a.output_dir / "classification-reference.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report["file"]),flush=True)


if __name__ == "__main__":
    main()
