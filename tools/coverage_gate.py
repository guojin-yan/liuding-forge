#!/usr/bin/env python3
"""Enforce target-owned coverage baselines from a gcovr JSON summary."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path, PurePosixPath
from typing import Any


METRIC_NAMES = ("line", "function", "branch")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate a gcovr JSON summary by owned module and reject "
            "coverage or instrumented-denominator regressions."
        )
    )
    parser.add_argument("report", type=Path, help="gcovr --json-summary report")
    parser.add_argument("policy", type=Path, help="checked-in coverage policy")
    parser.add_argument(
        "--output", type=Path, help="optional machine-readable observed summary"
    )
    return parser.parse_args()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_percent(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number in [0, 100]")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 100.0:
        raise ValueError(f"{label} must be a finite number in [0, 100]")
    return result


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _normalize_filename(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    if "\\" in value:
        raise ValueError(f"{label} must use '/' separators: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("./"):
        raise ValueError(f"{label} must be a normalized repository-relative path")
    return value


def _read_report_files(report: dict[str, Any]) -> list[dict[str, Any]]:
    files = report.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("coverage report files must be a non-empty array")

    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_file in enumerate(files):
        label = f"coverage report files[{index}]"
        if not isinstance(raw_file, dict):
            raise ValueError(f"{label} must be an object")
        filename = _normalize_filename(raw_file.get("filename"), f"{label}.filename")
        if filename in seen:
            raise ValueError(f"coverage report repeats file: {filename}")
        seen.add(filename)

        metrics: dict[str, dict[str, int]] = {}
        for metric_name in METRIC_NAMES:
            covered = _require_int(
                raw_file.get(f"{metric_name}_covered"),
                f"{label}.{metric_name}_covered",
            )
            total = _require_int(
                raw_file.get(f"{metric_name}_total"),
                f"{label}.{metric_name}_total",
            )
            if covered > total:
                raise ValueError(
                    f"{label}.{metric_name} has covered > total: {covered}/{total}"
                )
            metrics[metric_name] = {"covered": covered, "total": total}
        parsed.append({"filename": filename, "metrics": metrics})
    return parsed


def _read_policy_targets(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if policy.get("schema_version") != 1:
        raise ValueError("coverage policy schema_version must be 1")
    if policy.get("kind") != "jyppx-target-coverage-policy":
        raise ValueError("coverage policy kind must be jyppx-target-coverage-policy")
    baseline = policy.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("coverage policy baseline must be an object")
    _require_string(baseline.get("commit"), "coverage policy baseline.commit")
    _require_string(
        baseline.get("environment"), "coverage policy baseline.environment"
    )

    raw_targets = policy.get("targets")
    if not isinstance(raw_targets, dict) or not raw_targets:
        raise ValueError("coverage policy targets must be a non-empty object")

    targets: dict[str, dict[str, Any]] = {}
    all_prefixes: dict[str, str] = {}
    for target_name, raw_target in raw_targets.items():
        label = f"coverage policy target {target_name!r}"
        if not isinstance(target_name, str) or not target_name:
            raise ValueError("coverage policy target names must be non-empty strings")
        if not isinstance(raw_target, dict):
            raise ValueError(f"{label} must be an object")

        raw_prefixes = raw_target.get("path_prefixes")
        if not isinstance(raw_prefixes, list) or not raw_prefixes:
            raise ValueError(f"{label}.path_prefixes must be a non-empty array")
        prefixes: list[str] = []
        for index, raw_prefix in enumerate(raw_prefixes):
            prefix = _normalize_filename(
                raw_prefix, f"{label}.path_prefixes[{index}]"
            )
            if not prefix.endswith("/"):
                raise ValueError(f"{label} path prefix must end with '/': {prefix}")
            owner = all_prefixes.get(prefix)
            if owner is not None:
                raise ValueError(
                    f"coverage path prefix {prefix!r} belongs to both {owner!r} "
                    f"and {target_name!r}"
                )
            all_prefixes[prefix] = target_name
            prefixes.append(prefix)

        raw_metrics = raw_target.get("metrics")
        if not isinstance(raw_metrics, dict):
            raise ValueError(f"{label}.metrics must be an object")
        metrics: dict[str, dict[str, Any]] = {}
        for metric_name in METRIC_NAMES:
            raw_metric = raw_metrics.get(metric_name)
            metric_label = f"{label}.metrics.{metric_name}"
            if not isinstance(raw_metric, dict):
                raise ValueError(f"{metric_label} must be an object")
            covered = _require_int(
                raw_metric.get("baseline_covered"),
                f"{metric_label}.baseline_covered",
            )
            total = _require_int(
                raw_metric.get("baseline_total"),
                f"{metric_label}.baseline_total",
                minimum=1,
            )
            if covered > total:
                raise ValueError(
                    f"{metric_label} has baseline_covered > baseline_total"
                )
            metrics[metric_name] = {
                "baseline_covered": covered,
                "baseline_total": total,
                "minimum_percent": _require_percent(
                    raw_metric.get("minimum_percent"),
                    f"{metric_label}.minimum_percent",
                ),
            }
        targets[target_name] = {"path_prefixes": prefixes, "metrics": metrics}
    return targets


def _percent(covered: int, total: int) -> float:
    return covered * 100.0 / total if total else 100.0


def _aggregate(
    files: list[dict[str, Any]], targets: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    observed = {
        target_name: {
            "files": [],
            "metrics": {
                metric_name: {"covered": 0, "total": 0}
                for metric_name in METRIC_NAMES
            },
        }
        for target_name in targets
    }

    for file_entry in files:
        filename = file_entry["filename"]
        owners = [
            target_name
            for target_name, target in targets.items()
            if any(filename.startswith(prefix) for prefix in target["path_prefixes"])
        ]
        if not owners:
            raise ValueError(f"coverage file has no target owner: {filename}")
        if len(owners) != 1:
            raise ValueError(
                f"coverage file has multiple target owners ({', '.join(owners)}): "
                f"{filename}"
            )
        target_observed = observed[owners[0]]
        target_observed["files"].append(filename)
        for metric_name in METRIC_NAMES:
            metric = file_entry["metrics"][metric_name]
            target_observed["metrics"][metric_name]["covered"] += metric["covered"]
            target_observed["metrics"][metric_name]["total"] += metric["total"]

    for target_name, target_observed in observed.items():
        if not target_observed["files"]:
            raise ValueError(f"coverage target has no observed files: {target_name}")
        target_observed["files"].sort()
        for metric_name in METRIC_NAMES:
            metric = target_observed["metrics"][metric_name]
            if metric["total"] < 1:
                raise ValueError(
                    f"coverage target {target_name!r} has no {metric_name} denominator"
                )
            metric["percent"] = round(
                _percent(metric["covered"], metric["total"]), 6
            )
    return observed


def _verify_report_totals(
    report: dict[str, Any], observed: dict[str, dict[str, Any]]
) -> None:
    for metric_name in METRIC_NAMES:
        covered = sum(
            target["metrics"][metric_name]["covered"] for target in observed.values()
        )
        total = sum(
            target["metrics"][metric_name]["total"] for target in observed.values()
        )
        report_covered = _require_int(
            report.get(f"{metric_name}_covered"),
            f"coverage report {metric_name}_covered",
        )
        report_total = _require_int(
            report.get(f"{metric_name}_total"),
            f"coverage report {metric_name}_total",
        )
        if (covered, total) != (report_covered, report_total):
            raise ValueError(
                f"coverage report {metric_name} totals do not equal owned files: "
                f"report={report_covered}/{report_total}, owned={covered}/{total}"
            )


def _evaluate(
    observed: dict[str, dict[str, Any]], targets: dict[str, dict[str, Any]]
) -> list[str]:
    failures: list[str] = []
    for target_name, target_policy in targets.items():
        for metric_name in METRIC_NAMES:
            actual = observed[target_name]["metrics"][metric_name]
            expected = target_policy["metrics"][metric_name]
            if actual["total"] < expected["baseline_total"]:
                failures.append(
                    f"{target_name}.{metric_name} denominator shrank: "
                    f"{actual['total']} < {expected['baseline_total']}"
                )
            if actual["covered"] < expected["baseline_covered"]:
                failures.append(
                    f"{target_name}.{metric_name} covered count regressed: "
                    f"{actual['covered']} < {expected['baseline_covered']}"
                )
            if actual["percent"] + 1e-9 < expected["minimum_percent"]:
                failures.append(
                    f"{target_name}.{metric_name} percentage regressed: "
                    f"{actual['percent']:.2f}% < "
                    f"{expected['minimum_percent']:.2f}%"
                )
    return failures


def _write_output(
    path: Path,
    report_path: Path,
    policy_path: Path,
    policy: dict[str, Any],
    observed: dict[str, dict[str, Any]],
) -> None:
    output = {
        "schema_version": 1,
        "kind": "jyppx-target-coverage-observation",
        "report": str(report_path),
        "policy": str(policy_path),
        "baseline": policy["baseline"],
        "targets": observed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    args = _parse_arguments()
    report = _read_object(args.report, "coverage report")
    policy = _read_object(args.policy, "coverage policy")
    files = _read_report_files(report)
    targets = _read_policy_targets(policy)
    observed = _aggregate(files, targets)
    _verify_report_totals(report, observed)

    if args.output is not None:
        _write_output(args.output, args.report, args.policy, policy, observed)

    failures = _evaluate(observed, targets)
    for target_name, target_observed in observed.items():
        metrics = target_observed["metrics"]
        summary = ", ".join(
            f"{name}={metrics[name]['covered']}/{metrics[name]['total']} "
            f"({metrics[name]['percent']:.2f}%)"
            for name in METRIC_NAMES
        )
        print(f"{target_name}: {summary}")
    if failures:
        raise ValueError("coverage gate failed: " + "; ".join(failures))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from None
