#!/usr/bin/env python3
"""Enforce target-owned coverage baselines from a gcovr JSON summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
    parser.add_argument("report", nargs="?", type=Path, help="gcovr --json-summary report")
    parser.add_argument("policy", type=Path, help="checked-in coverage policy")
    parser.add_argument("--source-root", type=Path, help="source tree for v2 owner verification")
    parser.add_argument("--cmake-cache", type=Path, help="actual configured cache for the v2 profile")
    parser.add_argument("--audit-owners", action="store_true", help="verify source ownership only; no coverage evaluated")
    parser.add_argument(
        "--output", type=Path, help="optional machine-readable observed summary"
    )
    return parser.parse_args()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} repeats JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=unique_keys)
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
    version = policy.get("schema_version")
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("coverage policy schema_version must be 1 or 2")
    if version == 2 and set(policy) != {
        "schema_version", "kind", "baseline", "targets", "observations",
        "non_executable_files", "profile",
    }:
        raise ValueError("v2 coverage policy has unknown or missing fields")
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
            for other, owner in all_prefixes.items():
                if prefix == other or (owner != target_name and
                                        (prefix.startswith(other) or other.startswith(prefix))):
                    raise ValueError(f"coverage path prefixes overlap for {owner!r} and {target_name!r}: {prefix}")
            all_prefixes[prefix] = target_name
            prefixes.append(prefix)

        status = raw_target.get("baseline_status", "ratchet")
        if status == "pending-observation":
            if version != 2 or set(raw_target) != {"path_prefixes", "baseline_status"}:
                raise ValueError(f"{label} pending baseline must have no metrics and requires v2")
            targets[target_name] = {"path_prefixes": prefixes, "baseline_status": status}
            continue
        if status != "ratchet":
            raise ValueError(f"{label} has invalid baseline_status")
        if version == 2 and (not {"path_prefixes", "metrics"} <= set(raw_target) or
                             set(raw_target) - {"path_prefixes", "metrics", "baseline_status", "baseline_observation"}):
            raise ValueError(f"{label} has unknown or missing fields")
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
        targets[target_name] = {"path_prefixes": prefixes, "metrics": metrics, "baseline_status": status}
    return targets


def _read_non_executable_files(policy: dict[str, Any]) -> dict[str, dict[str, str]]:
    if policy["schema_version"] == 1:
        return {}
    records = policy.get("non_executable_files")
    if not isinstance(records, list):
        raise ValueError("non_executable_files must be an array")
    parsed = {}
    for record in records:
        if not isinstance(record, dict) or set(record) not in (
            {"path", "sha256", "reason"}, {"path", "sha256", "reason", "normalization"}
        ):
            raise ValueError("invalid non-executable file descriptor")
        path = _normalize_filename(record["path"], "non-executable path")
        if path in parsed or not path.startswith("include/jyppx/") or not path.endswith(".hpp"):
            raise ValueError("non-executable exception must name one unique public header")
        if not isinstance(record["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"]):
            raise ValueError("non-executable exception requires an exact SHA-256")
        if "normalization" in record and record["normalization"] != "LF":
            raise ValueError("non-executable normalization must be LF when declared")
        _require_string(record["reason"], "non-executable reason")
        parsed[path] = record
    return parsed


def _include_only_header(raw: bytes) -> bool:
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", raw.decode("utf-8"), flags=re.S)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    guard = re.fullmatch(r"#ifndef\s+([A-Za-z_][A-Za-z0-9_]*)", lines[0])
    return bool(guard and lines[1] == "#define " + guard[1] and lines[-1] == "#endif" and
                all(re.fullmatch(r'#include\s+(?:<[^<>]+>|"[^"\n]+")', line) for line in lines[2:-1]))


def _verify_source_ownership(source_root: Path, targets: dict[str, dict[str, Any]],
                             exceptions: dict[str, dict[str, str]]) -> dict[str, Any]:
    owned = {}
    excluded = []
    counts = {name: 0 for name in targets}
    suffixes = {".h", ".hh", ".hpp", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".inc"}
    root = source_root.resolve()
    for relative in ("include/jyppx", "src/jyppx"):
        directory = root / relative
        if not directory.is_dir():
            raise ValueError(f"source ownership root is missing: {relative}")
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f"source ownership file is aliased outside the root: {path}")
            filename = path.relative_to(root).as_posix()
            owners = [name for name, target in targets.items()
                      if any(filename.startswith(prefix) for prefix in target["path_prefixes"])]
            if filename in exceptions:
                raw = path.read_bytes()
                fingerprinted = raw
                if exceptions[filename].get("normalization") == "LF":
                    fingerprinted = raw.replace(b"\r\n", b"\n")
                    if b"\r" in fingerprinted:
                        raise ValueError(f"non-executable LF normalization rejects bare CR: {filename}")
                if owners or hashlib.sha256(fingerprinted).hexdigest() != exceptions[filename]["sha256"] or not _include_only_header(raw):
                    raise ValueError(f"non-executable exception changed or emits code: {filename}")
                excluded.append(filename)
            elif len(owners) != 1:
                raise ValueError(f"source candidate must have exactly one owner: {filename} ({owners})")
            else:
                owned[filename] = owners[0]
                counts[owners[0]] += 1
    if set(excluded) != set(exceptions) or any(count == 0 for count in counts.values()):
        raise ValueError("source ownership has a missing exception or an empty owner")
    return {"kind": "jyppx-coverage-owner-audit", "file_owners": owned,
            "non_executable_files": excluded, "owner_file_counts": counts,
            "source_file_count": len(owned) + len(excluded),
            "coverage_evaluated": False, "release_qualified": False}


def _read_profile(policy: dict[str, Any]) -> dict[str, Any] | None:
    if policy["schema_version"] == 1:
        return None
    profile = policy.get("profile")
    if not isinstance(profile, dict) or set(profile) != {"id", "cmake_options"}:
        raise ValueError("v2 profile must have id and cmake_options")
    _require_string(profile["id"], "coverage profile id")
    options = profile["cmake_options"]
    if not isinstance(options, dict) or not options:
        raise ValueError("coverage profile cmake_options must not be empty")
    if any(not isinstance(key, str) or not re.fullmatch(r"JYPPX_[A-Z0-9_]+", key) or
           not isinstance(value, str) or value not in {"ON", "OFF"}
           for key, value in options.items()):
        raise ValueError("coverage profile requires explicit JYPPX boolean options")
    return profile


def _verify_profile(cache: Path | None, profile: dict[str, Any]) -> None:
    if cache is None:
        raise ValueError("v2 coverage requires the actual --cmake-cache")
    entries = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        name_type, value = line.split("=", 1)
        name, kind = name_type.split(":", 1)
        if name in profile["cmake_options"]:
            if name in entries:
                raise ValueError(f"coverage profile cache repeats key: {name}")
            entries[name] = (kind, value)
    for name, value in profile["cmake_options"].items():
        if entries.get(name) != ("BOOL", value):
            raise ValueError(f"coverage profile option missing or changed: {name}")


def _percent(covered: int, total: int) -> float:
    return covered * 100.0 / total if total else 100.0


def _aggregate(
    files: list[dict[str, Any]], targets: dict[str, dict[str, Any]],
    non_executable_files: dict[str, dict[str, str]] | None = None,
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
        if non_executable_files and filename in non_executable_files:
            if any(metric["total"] or metric["covered"] for metric in file_entry["metrics"].values()):
                raise ValueError(f"non-executable exception has coverage entities: {filename}")
            continue
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
        if target_policy.get("baseline_status") == "pending-observation":
            failures.append(f"{target_name} baseline pending observation and review")
            continue
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
    if policy["schema_version"] == 2:
        output["pending_baseline_targets"] = [name for name, target in policy["targets"].items()
                                               if target.get("baseline_status") == "pending-observation"]
        output["profile"] = policy["profile"]["id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    args = _parse_arguments()
    policy = _read_object(args.policy, "coverage policy")
    targets = _read_policy_targets(policy)
    exceptions = _read_non_executable_files(policy)
    profile = _read_profile(policy)
    inventory = None
    if policy["schema_version"] == 2 or args.audit_owners:
        if args.source_root is None:
            raise ValueError("source ownership verification requires --source-root")
        inventory = _verify_source_ownership(args.source_root, targets, exceptions)
    if args.audit_owners:
        if args.report is not None:
            raise ValueError("owner audit must not consume a coverage report")
        if profile is not None and args.cmake_cache is not None:
            _verify_profile(args.cmake_cache, profile)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Source ownership verified; coverage was not evaluated or qualified")
        return 0
    if args.report is None:
        raise ValueError("a coverage report is required unless --audit-owners is used")
    if profile is not None:
        _verify_profile(args.cmake_cache, profile)
    report = _read_object(args.report, "coverage report")
    files = _read_report_files(report)
    if inventory is not None:
        known = set(inventory["file_owners"]) | set(inventory["non_executable_files"])
        if any(file["filename"] not in known for file in files):
            raise ValueError("coverage report includes a file outside the audited source inventory")
    observed = _aggregate(files, targets, exceptions)
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
