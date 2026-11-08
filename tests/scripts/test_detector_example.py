# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Optional real-data C++ example and independent decode/COCO checker."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(directory):
    # These dependencies only score saved native predictions; no Python model
    # or training implementation participates in the example.
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    value = json.loads((directory / "predictions.json").read_text())
    result = json.loads((directory / "results.json").read_text())
    raw = np.fromfile(directory / "raw.f32", dtype="<f4").reshape(25, 3, 5, 8, 8)
    instances = np.fromfile(directory / "instances.i64", dtype="<i8").reshape(25, 128, 128)
    require(len(set(value["ids"])) == 25, "Duplicate real image IDs")
    anchors = ((16., 48.), (32., 80.), (64., 112.))

    def sigmoid(x):
        return 1. / (1. + math.exp(-max(-80., min(80., float(x)))))

    def iou(a, b):
        intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
        return intersection / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

    for image in range(25):
        ground_truth = []
        for identity in sorted(set(instances[image].reshape(-1)) - {0}):
            y, x = np.nonzero(instances[image] == identity)
            ground_truth.append([int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1])
        require(ground_truth == value["targets"][image], "Native instance-to-target boxes differ")
        candidates = []
        for anchor, (width, height) in enumerate(anchors):
            for y in range(8):
                for x in range(8):
                    v = raw[image, anchor, :, y, x]
                    score = sigmoid(v[4])
                    if score < .01:
                        continue
                    cx, cy = (x+sigmoid(v[0]))*16., (y+sigmoid(v[1]))*16.
                    w = width * math.exp(max(-4., min(4., float(v[2]))))
                    h = height * math.exp(max(-4., min(4., float(v[3]))))
                    box = [max(0., min(128., number)) for number in (cx-w/2, cy-h/2, cx+w/2, cy+h/2)]
                    if box[2] > box[0] and box[3] > box[1]:
                        candidates.append({"box": box, "score": score})
        # Python's stable sort independently preserves anchor/y/x tie order.
        candidates.sort(key=lambda row: -row["score"])
        selected = []
        for candidate in candidates:
            if all(iou(candidate["box"], kept["box"]) <= .5 for kept in selected):
                selected.append(candidate)
            if len(selected) == 100:
                break
        native = value["predictions"][image]
        require(len(native) == len(selected), "Native NMS count differs")
        for actual, expected in zip(native, selected):
            require(abs(actual["score"]-expected["score"]) < 1e-12, "Decoded confidence differs")
            require(max(abs(a-b) for a, b in zip(actual["box"], expected["box"])) < 1e-10, "Decoded box differs")

    annotations, detections = [], []
    for i, (ground_truth, predictions) in enumerate(zip(value["targets"], value["predictions"]), 1):
        for box in ground_truth:
            x, y, right, bottom = box
            annotations.append({"id": len(annotations)+1, "image_id": i, "category_id": 1,
                                "bbox": [x, y, right-x, bottom-y], "area": (right-x)*(bottom-y), "iscrowd": 0})
        for prediction in predictions:
            x, y, right, bottom = prediction["box"]
            detections.append({"image_id": i, "category_id": 1, "bbox": [x, y, right-x, bottom-y],
                               "score": prediction["score"]})
    summaries = []
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"info": {}, "images": [{"id": i} for i in range(1, 26)],
                      "categories": [{"id": 1, "name": "person"}], "annotations": annotations}
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = dict(gt.dataset, annotations=[])
            dt.createIndex()
        for source in (None, 0, 1):
            ids = [i+1 for i, identity in enumerate(value["sources"]) if source is None or identity == source]
            evaluator = COCOeval(gt, dt, "bbox")
            evaluator.params.imgIds = ids
            evaluator.params.iouThrs = np.asarray([.5, .55, .6, .65, .7, .75, .8, .85, .9, .95])
            evaluator.params.areaRng = [[0., 1e10]]
            evaluator.params.areaRngLbl = ["all"]
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            precision = evaluator.eval["precision"][:, :, 0, 0, 2]
            summaries.append({"ap50": float(precision[0][precision[0] >= 0].mean()),
                              "map": float(precision[precision >= 0].mean()),
                              "ground_truth": sum(1 for row in annotations if row["image_id"] in ids),
                              "predictions": sum(1 for row in detections if row["image_id"] in ids)})
    for actual, expected in zip([result, *result["by_source"]], summaries):
        for key in ("ap50", "map"):
            require(abs(actual[key]-expected[key]) < 1e-12, f"Independent COCO {key} differs")
        for key in ("ground_truth", "predictions"):
            require(actual[key] == expected[key], f"Independent COCO {key} count differs")
    require(result["partition_report_equal"] and result["resumed_report_equal"] and result["complete_no_reread"],
            "Evaluation completion/resumption contract failed")
    report = json.loads((directory / "quality.json").read_text())
    require(report["verdict"] == 1, "Quality report does not pass")
    reference_result = {"passed": True, "implementation": "pycocotools 2.0.11", "images": 25,
                        "decoded_outputs_checked": int(raw.size), "summaries": summaries,
                        "raw_sha256": digest(directory / "raw.f32")}
    (directory / "reference.json").write_text(json.dumps(reference_result, indent=2)+"\n", encoding="utf-8")
    return reference_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--data", type=Path, default=os.environ.get("JYPPX_LIUDING_DETECTOR_DATA_DIR"))
    parser.add_argument("--output", type=Path, default=os.environ.get("JYPPX_LIUDING_DETECTOR_OUTPUT_DIR"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=os.environ.get("JYPPX_LIUDING_DETECTOR_DEVICE", "cpu"))
    parser.add_argument("--reference-only", type=Path)
    args = parser.parse_args()
    if args.reference_only:
        print(json.dumps(reference(args.reference_only)))
        return 0
    if args.data is None:
        print("Pinned external Penn-Fudan data not selected; no training or download")
        return 77
    require(args.binary and args.binary.is_file(), "Build the native detector example first")
    require(args.output is not None and not args.output.exists(), "A fresh external output directory is required")
    require(digest(args.data / "manifest.json") == "38386432386c0fc3b9560c98575677e150708b3dbfb4fc11825035309f4dbd95",
            "Frozen split identity differs")
    # Resolve checker-only packages before spending the finite native workload.
    import numpy  # noqa: F401
    import pycocotools.cocoeval  # noqa: F401
    args.output.mkdir(parents=True)
    launches = []

    def run(action, name, device, *tail):
        command = [str(args.binary), action, str(args.data), str(args.output/name), device, *map(str, tail)]
        started = time.monotonic()
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900, check=False)
        (args.output/(name+".log")).write_bytes(completed.stdout)
        launches.append({"name": name, "argv": command, "returncode": completed.returncode,
                         "elapsed_seconds": time.monotonic()-started})
        (args.output/"launches.json").write_text(json.dumps(launches, indent=2)+"\n", encoding="utf-8")
        require(completed.returncode == 0, name+" failed: "+completed.stdout.decode(errors="replace")[-4000:])

    run("train", "continuous", args.device)
    baseline = json.loads((args.output/"continuous/training.json").read_text())
    require(baseline["global_step"] == baseline["applied_updates"] == 1200, "Continuous update count differs")
    run("evaluate", "evaluate-"+args.device, args.device, args.output/"continuous/model.safetensors", baseline["weights_sha256"])
    reference(args.output/("evaluate-"+args.device))
    run("train", "paused", args.device, "-", 137)
    run("train", "resumed", args.device, args.output/"paused/checkpoint.jyck")
    paused = json.loads((args.output/"paused/training.json").read_text())
    resumed = json.loads((args.output/"resumed/training.json").read_text())
    require(paused["global_step"] == paused["applied_updates"] == 137 and paused["paused"], "Pause boundary differs")
    require(resumed["initial_global_step"] == 137 and resumed["global_step"] == 1200 and resumed["applied_updates"] == 1063,
            "Fresh-process restored update count differs")
    for name in ("model.safetensors", "checkpoint.jyck"):
        require(digest(args.output/"continuous"/name) == digest(args.output/"resumed"/name), "Full restored state differs: "+name)
    if args.device == "cuda":
        run("evaluate", "evaluate-cpu", "cpu", args.output/"continuous/model.safetensors", baseline["weights_sha256"])
        reference(args.output/"evaluate-cpu")
    result = {"passed": True, "native_launches": len(launches), "native_training_launches": 3,
              "logical_updates": baseline["applied_updates"]+paused["applied_updates"]+resumed["applied_updates"],
              "exact_full_checkpoint": True, "exact_model": True, "training_device": args.device,
              "weights_sha256": baseline["weights_sha256"], "checkpoint_sha256": baseline["checkpoint_sha256"]}
    require(result["logical_updates"] == 2400, "Frozen training budget differs")
    (args.output/"summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
