#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent semantic IoU/Dice oracle using pixel sets and exact fractions.

No Torch, framework code, datasets, downloads or external packages are used.
Print JSON for C++ differential tests: the consumer supplies the same fixed
arithmetic image fixture. Background exclusion removes scoring classes, never
pixels, and the fifth class is absent in both truth and predictions.
"""
from fractions import Fraction
import json


def reference():
    truth = [-1 if i % 13 == 0 else (i * 11 + i // 3) % 4 for i in range(646)]
    prediction = [(i * 37 + i // 7) % 4 for i in range(646)]
    accepted = {i for i, label in enumerate(truth) if label != -1}
    classes = []
    for label in range(5):
        t = {i for i in accepted if truth[i] == label}
        p = {i for i in accepted if prediction[i] == label}
        classes.append((t, p))
    cases = []
    for dice in [False, True]:
        for background in [False, True]:
            for absent in ["exclude", "zero", "one"]:
                for average in ["micro", "macro", "weighted"]:
                    selected = [pair for i, pair in enumerate(classes) if background or i != 2]
                    if average == "micro":
                        numerator = sum(len(t & p) for t, p in selected) * (2 if dice else 1)
                        den = sum(len(t) + len(p) if dice else len(t | p) for t, p in selected)
                        score = Fraction(numerator, den)
                    else:
                        terms = []
                        for t, p in selected:
                            den = len(t) + len(p) if dice else len(t | p)
                            if not den and absent == "exclude":
                                continue
                            value = Fraction((2 if dice else 1) * len(t & p), den) if den else Fraction(absent == "one")
                            terms.append((value, len(t) if average == "weighted" else 1))
                        score = sum(value * weight for value, weight in terms) / sum(weight for _, weight in terms)
                    cases.append({"dice": dice, "background": background, "absent": absent,
                                  "average": average, "score": float(score), "exact": str(score)})
    return {"schema": 1, "algorithm": "sets-of-pixel-indices/exact-rational",
            "shape": [2, 17, 19], "classes": 5, "ignore": -1, "background": 2,
            "valid_pixels": len(accepted), "cases": cases,
            "counts": [{"intersection": len(t & p), "target": len(t), "predicted": len(p)} for t, p in classes]}


if __name__ == "__main__":
    print(json.dumps(reference(), indent=2, sort_keys=True))
