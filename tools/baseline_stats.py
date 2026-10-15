#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build and compare stable statistics for repeated benchmark samples."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize repeated benchmark samples and enforce stability."
    )
    parser.add_argument("input", type=Path, help="Raw repeated-sample JSON file")
    parser.add_argument("--output", required=True, type=Path, help="Statistics JSON")
    parser.add_argument("--reference", type=Path, help="Reference thresholds to compare")
    parser.add_argument(
        "--write-reference", type=Path, help="Write a reference from stable samples"
    )
    parser.add_argument("--max-cv-percent", type=float, default=10.0)
    parser.add_argument("--min-regression-percent", type=float, default=10.0)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and greater than zero")
    return result


def _summarize_metric(
    metric: dict[str, Any], min_regression_percent: float
) -> tuple[str, dict[str, Any]]:
    name = metric.get("name")
    direction = metric.get("direction")
    unit = metric.get("unit")
    samples = metric.get("samples")
    if not isinstance(name, str) or not name:
        raise ValueError("each metric requires a non-empty name")
    if direction not in ("higher", "lower"):
        raise ValueError(f"{name}: direction must be 'higher' or 'lower'")
    if not isinstance(unit, str) or not unit:
        raise ValueError(f"{name}: unit must be a non-empty string")
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError(f"{name}: at least three independent samples are required")

    values = [_finite_positive(value, f"{name} sample") for value in samples]
    mean = statistics.fmean(values)
    sample_stddev = statistics.stdev(values)
    cv_percent = sample_stddev / mean * 100.0
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    allowed_regression = max(min_regression_percent, 3.0 * cv_percent)
    return name, {
        "direction": direction,
        "unit": unit,
        "sample_count": len(values),
        "samples": values,
        "minimum": min(values),
        "maximum": max(values),
        "mean": mean,
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "sample_stddev": sample_stddev,
        "cv_percent": cv_percent,
        "allowed_regression_percent": round(allowed_regression, 2),
    }


def _build_statistics(
    report: dict[str, Any], max_cv_percent: float, min_regression_percent: float
) -> dict[str, Any]:
    if report.get("schema_version") != 1:
        raise ValueError("raw sample schema_version must be 1")
    environment_key = report.get("environment_key")
    if not isinstance(environment_key, str) or not environment_key:
        raise ValueError("raw samples require a non-empty environment_key")
    metrics = report.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("raw samples require at least one metric")

    summaries: dict[str, Any] = {}
    for raw_metric in metrics:
        if not isinstance(raw_metric, dict):
            raise ValueError("each metric must be an object")
        name, summary = _summarize_metric(raw_metric, min_regression_percent)
        if name in summaries:
            raise ValueError(f"duplicate metric: {name}")
        summaries[name] = summary

    unstable = [
        name
        for name, metric in summaries.items()
        if metric["cv_percent"] > max_cv_percent
    ]
    return {
        "schema_version": 1,
        "kind": "jyppx-repeated-benchmark-statistics",
        "environment_key": environment_key,
        "metadata": report.get("metadata", {}),
        "stability": {
            "max_cv_percent": max_cv_percent,
            "passed": not unstable,
            "unstable_metrics": unstable,
        },
        "metrics": summaries,
    }


def _reference_from_statistics(statistics_report: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        name: {
            "direction": metric["direction"],
            "unit": metric["unit"],
            "median": metric["median"],
            "cv_percent": metric["cv_percent"],
            "allowed_regression_percent": metric["allowed_regression_percent"],
        }
        for name, metric in statistics_report["metrics"].items()
    }
    return {
        "schema_version": 1,
        "kind": "jyppx-benchmark-reference",
        "environment_key": statistics_report["environment_key"],
        "source": statistics_report.get("metadata", {}),
        "metrics": metrics,
    }


def _compare_reference(
    statistics_report: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    if reference.get("schema_version") != 1:
        raise ValueError("reference schema_version must be 1")
    if reference.get("kind") != "jyppx-benchmark-reference":
        raise ValueError("reference kind must be jyppx-benchmark-reference")
    if reference.get("environment_key") != statistics_report["environment_key"]:
        raise ValueError("reference environment_key does not match the current run")
    reference_metrics = reference.get("metrics")
    if not isinstance(reference_metrics, dict) or not reference_metrics:
        raise ValueError("reference requires metrics")

    failures: list[str] = []
    comparisons: dict[str, Any] = {}
    for name, expected in reference_metrics.items():
        if name not in statistics_report["metrics"]:
            failures.append(f"{name}: missing from current run")
            continue
        if not isinstance(expected, dict):
            raise ValueError(f"{name}: reference metric must be an object")
        current = statistics_report["metrics"][name]
        if expected.get("direction") != current["direction"]:
            raise ValueError(f"{name}: direction differs from reference")
        if expected.get("unit") != current["unit"]:
            raise ValueError(f"{name}: unit differs from reference")

        reference_median = _finite_positive(expected.get("median"), f"{name} median")
        allowed = _finite_positive(
            expected.get("allowed_regression_percent"),
            f"{name} allowed_regression_percent",
        )
        current_median = current["median"]
        if current["direction"] == "higher":
            degradation = (reference_median - current_median) / reference_median * 100.0
        else:
            degradation = (current_median - reference_median) / reference_median * 100.0
        passed = degradation <= allowed
        if not passed:
            failures.append(
                f"{name}: degradation {degradation:.2f}% exceeds {allowed:.2f}%"
            )
        comparisons[name] = {
            "reference_median": reference_median,
            "current_median": current_median,
            "degradation_percent": degradation,
            "allowed_regression_percent": allowed,
            "passed": passed,
        }
    return {"passed": not failures, "failures": failures, "metrics": comparisons}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    args = _parse_arguments()
    if args.reference and args.write_reference:
        raise ValueError("--reference and --write-reference are mutually exclusive")
    max_cv = _finite_positive(args.max_cv_percent, "max_cv_percent")
    min_regression = _finite_positive(
        args.min_regression_percent, "min_regression_percent"
    )
    statistics_report = _build_statistics(_read_object(args.input), max_cv, min_regression)

    if args.reference:
        statistics_report["comparison"] = _compare_reference(
            statistics_report, _read_object(args.reference)
        )
    _write_json(args.output, statistics_report)

    if args.write_reference:
        if not statistics_report["stability"]["passed"]:
            raise ValueError("cannot write a reference from unstable samples")
        _write_json(args.write_reference, _reference_from_statistics(statistics_report))

    for name, metric in statistics_report["metrics"].items():
        print(
            f"{name}: median={metric['median']:.3f} {metric['unit']}, "
            f"p95={metric['p95']:.3f}, cv={metric['cv_percent']:.2f}%, "
            f"threshold={metric['allowed_regression_percent']:.2f}%"
        )
    if not statistics_report["stability"]["passed"]:
        print(
            "unstable metrics: "
            + ", ".join(statistics_report["stability"]["unstable_metrics"])
        )
        return 1
    comparison = statistics_report.get("comparison")
    if comparison and not comparison["passed"]:
        print("regressions: " + "; ".join(comparison["failures"]))
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"baseline statistics error: {error}")
        raise SystemExit(2) from error

