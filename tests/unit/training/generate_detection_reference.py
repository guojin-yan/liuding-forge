# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Regenerate the bbox oracle with official pycocotools==2.0.10 (test only)."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import random

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert importlib.metadata.version("pycocotools") == "2.0.10"
    rng = random.Random(839)
    images = []
    for n, image_id in enumerate([7, 2, 19, 5, 11, 23, 3, 29]):
        targets, predictions = [], []
        for cls in range(4):
            for j in range(2):
                size = [8, 20, 40, 110][(n + cls + j) % 4]
                x, y = rng.randrange(80), rng.randrange(80)
                box = [x, y, x + size, y + size]
                crowd = (n + cls + j) % 7 == 0
                target = {"box": box, "class_index": cls, "crowd": crowd,
                          "area": size * size if (n + cls) % 3 else size * size / 2}
                targets.append(target)
                for shift in [0, 2, 10]:
                    predictions.append({"box": [x + shift, y, x + size + shift, y + size],
                                        "score": rng.choice([.3, .5, .7, .9]), "class_index": cls})
            predictions.append({"box": [200, 200, 210, 210], "score": .9, "class_index": cls})
        # No-GT class predictions must not silently become perfect AP.
        predictions.append({"box": [0, 0, 20, 20], "score": .95, "class_index": 4})
        images.append({"image_id": image_id, "source_index": n % 2,
                       "targets": targets, "predictions": predictions})
    images.append({"image_id": 31, "source_index": 1, "targets": [], "predictions": []})
    digest = hashlib.sha256(json.dumps(images, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    coco = COCO()
    annotations, detections = [], []
    for image in sorted(images, key=lambda x: x["image_id"]):
        for target in image["targets"]:
            x, y, x2, y2 = target["box"]
            annotations.append({"id": len(annotations) + 1, "image_id": image["image_id"],
                                "category_id": target["class_index"], "bbox": [x, y, x2-x, y2-y],
                                "area": target["area"], "iscrowd": int(target["crowd"])})
        for prediction in image["predictions"]:
            x, y, x2, y2 = prediction["box"]
            detections.append({"image_id": image["image_id"], "category_id": prediction["class_index"],
                               "bbox": [x, y, x2-x, y2-y], "score": prediction["score"]})
    coco.dataset = {"info": {}, "images": [{"id": x["image_id"]} for x in images],
                    "categories": [{"id": i, "name": str(i)} for i in range(5)], "annotations": annotations}
    cases = []
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
        dt = coco.loadRes(detections)
        for area in [[0, 1e10], [0, 32**2], [32**2, 96**2], [96**2, 1e10]]:
            for cap in [1, 10, 100]:
                summaries = []
                for source in [None, 0, 1]:
                    evaluator = COCOeval(coco, dt, "bbox")
                    evaluator.params.imgIds = sorted(x["image_id"] for x in images if source is None or x["source_index"] == source)
                    evaluator.params.iouThrs = np.array([.5, .55, .6, .65, .7, .75, .8, .85, .9, .95])
                    evaluator.params.areaRng = [area]
                    evaluator.params.areaRngLbl = ["configured"]
                    evaluator.params.maxDets = [cap]
                    evaluator.evaluate()
                    evaluator.accumulate()
                    precision = evaluator.eval["precision"][:, :, :, 0, 0]
                    recall = evaluator.eval["recall"][:, :, 0, 0]
                    per_class = []
                    for cls in range(5):
                        valid = precision[:, :, cls]
                        per_class.append({"ap_by_iou": [None if (a < 0).all() else float(a[a >= 0].mean()) for a in valid],
                                          "recall_by_iou": [None if a < 0 else float(a) for a in recall[:, cls]]})
                    summaries.append({"source": source, "mean_ap": float(precision[precision >= 0].mean()) if (precision >= 0).any() else None,
                                      "per_class": per_class})
                cases.append({"min_area": area[0], "max_area": area[1], "max_det": cap, "summaries": summaries})
    reference = {"producer": "official pycocotools==2.0.10 COCOeval bbox", "dataset_id": "sha256:" + digest,
                 "num_classes": 5, "sources": ["alpha", "beta", "empty"], "images": images, "cases": cases}
    args.output.write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(images), "cases": len(cases), "summaries": 36,
                      "dataset_id": reference["dataset_id"], "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
