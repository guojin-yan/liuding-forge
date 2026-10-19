#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Collect environment-bound CUDA Trainer performance samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


SUPPORTED_MODES = (
    "cuda-fp32-single-sgd",
    "cuda-fp16-multi-sgd-amp",
)
EXPECTED_OPTIMIZERS = {
    "cuda-fp32-single-sgd": 1,
    "cuda-fp16-multi-sgd-amp": 2,
}
METRICS = (
    ("samples_per_second", "higher", "samples_per_second"),
    ("mean_step_milliseconds", "lower", "milliseconds"),
    ("peak_allocated_bytes", "lower", "bytes"),
    ("peak_reserved_bytes", "lower", "bytes"),
)
BUILD_FIELDS = (
    "jyppx_version",
    "git_commit",
    "compiler",
    "cxx_standard",
    "os_arch",
    "build_type",
    "libtorch_version",
    "libtorch_build",
    "accelerator_runtime",
    "dependency_lock_digest",
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent CUDA Trainer benchmark processes and enforce "
            "stable, environment-bound statistics."
        )
    )
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        action="append",
        choices=SUPPORTED_MODES,
        help="Mode to collect; repeat to select modes (default: both)",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--max-cv-percent", type=float, default=10.0)
    parser.add_argument("--min-regression-percent", type=float, default=10.0)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--write-reference", type=Path)
    parser.add_argument(
        "--baseline-stats",
        type=Path,
        default=Path(__file__).with_name("baseline_stats.py"),
    )
    return parser.parse_args()


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and greater than zero")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def validate_sample(
    sample: dict[str, Any], expected_mode: str | None = None
) -> dict[str, Any]:
    """Validate one benchmark document and return its stable identity."""
    if sample.get("schema_version") != 1:
        raise ValueError("sample schema_version must be 1")
    if sample.get("kind") != "jyppx-cuda-training-benchmark-sample":
        raise ValueError("sample kind is not a CUDA training benchmark sample")
    mode = _nonempty_string(sample.get("mode"), "sample mode")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported sample mode: {mode}")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"expected mode {expected_mode}, received {mode}")

    workload = _object(sample.get("workload"), "workload")
    _nonempty_string(workload.get("id"), "workload.id")
    _nonempty_string(workload.get("digest"), "workload.digest")
    for field in (
        "seed",
        "batch_size",
        "steps",
        "sample_count",
        "input_features",
        "hidden_features",
        "output_features",
        "worker_count",
        "prefetch_batches",
    ):
        _positive_integer(workload.get(field), f"workload.{field}")
    if workload["sample_count"] != workload["batch_size"] * workload["steps"]:
        raise ValueError("workload.sample_count must equal batch_size * steps")

    build = _object(sample.get("build_fingerprint"), "build_fingerprint")
    for field in BUILD_FIELDS:
        _nonempty_string(build.get(field), f"build_fingerprint.{field}")
    if build["build_type"].lower() != "release":
        raise ValueError("CUDA performance samples require a Release build")
    if not build["accelerator_runtime"].startswith("cuda:available:"):
        raise ValueError("build fingerprint does not report available CUDA")

    cuda = _object(sample.get("cuda"), "cuda")
    if cuda.get("device_index") != 0:
        raise ValueError("benchmark must use visible CUDA device index zero")
    _nonempty_string(cuda.get("device_name"), "cuda.device_name")
    _nonempty_string(cuda.get("compute_capability"), "cuda.compute_capability")
    _positive_integer(
        cuda.get("compile_toolkit_version"), "cuda.compile_toolkit_version"
    )

    runtime = _object(sample.get("runtime"), "runtime")
    _positive_integer(runtime.get("intraop_threads"), "runtime.intraop_threads")
    _positive_integer(runtime.get("interop_threads"), "runtime.interop_threads")

    result = _object(sample.get("result"), "result")
    for field, _, _ in METRICS:
        _positive_number(result.get(field), f"result.{field}")
    steps = workload["steps"]
    if result.get("global_step") != steps:
        raise ValueError("result.global_step does not match workload.steps")
    optimizer_steps = result.get("optimizer_steps")
    expected_optimizers = EXPECTED_OPTIMIZERS[mode]
    if not isinstance(optimizer_steps, list) or len(optimizer_steps) != expected_optimizers:
        raise ValueError(
            f"{mode} requires exactly {expected_optimizers} optimizer step counters"
        )
    if any(value != steps for value in optimizer_steps):
        raise ValueError("every optimizer step counter must match workload.steps")

    scaler = result.get("grad_scaler")
    if mode == "cuda-fp16-multi-sgd-amp":
        scaler_object = _object(scaler, "result.grad_scaler")
        if scaler_object.get("enabled") is not True:
            raise ValueError("AMP sample requires an enabled GradScaler")
        _positive_number(scaler_object.get("scale"), "result.grad_scaler.scale")
        if scaler_object.get("checkpoint_ready") is not True:
            raise ValueError("AMP sample GradScaler is not checkpoint-ready")
    elif scaler is not None:
        raise ValueError("full-precision sample must not publish GradScaler state")

    return {
        "mode": mode,
        "workload": workload,
        "build_fingerprint": build,
        "cuda": cuda,
        "runtime": runtime,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _parse_nvidia_row(output: str, fields: tuple[str, ...]) -> dict[str, str]:
    rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(fields):
        raise ValueError("nvidia-smi returned an unexpected row shape")
    values = [value.strip() for value in rows[0]]
    if any(not value for value in values):
        raise ValueError("nvidia-smi returned an empty identity field")
    return dict(zip(fields, values, strict=True))


def _query_gpu_identity(device_index: int) -> dict[str, str]:
    fields = (
        "index",
        "name",
        "uuid",
        "driver_version",
        "power.limit",
        "persistence_mode",
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "nvidia-smi identity query failed: " + completed.stderr.strip()
        )
    return _parse_nvidia_row(completed.stdout, fields)


def _query_gpu_noise(device_index: int) -> dict[str, str]:
    fields = (
        "timestamp",
        "temperature.gpu",
        "pstate",
        "clocks.current.graphics",
        "clocks.current.memory",
        "power.draw",
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return {"status": "unavailable", "reason": completed.stderr.strip()}
    return {"status": "recorded", **_parse_nvidia_row(completed.stdout, fields)}


def _run_process(
    benchmark: Path, mode: str, output_dir: Path, label: str
) -> dict[str, Any]:
    completed = subprocess.run(
        [str(benchmark), "--mode", mode],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=os.environ.copy(),
    )
    (output_dir / f"{label}.stdout.json").write_text(
        completed.stdout, encoding="utf-8", newline="\n"
    )
    (output_dir / f"{label}.stderr.log").write_text(
        completed.stderr, encoding="utf-8", newline="\n"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark {label} failed with exit code {completed.returncode}; "
            f"see {label}.stderr.log"
        )
    try:
        sample = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"benchmark {label} did not emit one JSON document") from error
    if not isinstance(sample, dict):
        raise ValueError(f"benchmark {label} output must be a JSON object")
    validate_sample(sample, mode)
    return sample


def build_raw_report(
    measured: dict[str, list[dict[str, Any]]],
    warmups: dict[str, list[dict[str, Any]]],
    gpu_identity: dict[str, str],
    noise: list[dict[str, str]],
) -> dict[str, Any]:
    """Build a baseline_stats-compatible report after identity validation."""
    if not measured:
        raise ValueError("at least one measured mode is required")
    identities: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    for mode, samples in measured.items():
        if mode not in SUPPORTED_MODES or not samples:
            raise ValueError(f"invalid or empty measured mode: {mode}")
        all_samples = list(warmups.get(mode, [])) + samples
        expected_identity = validate_sample(all_samples[0], mode)
        expected_canonical = _canonical_json(expected_identity)
        for sample in all_samples[1:]:
            identity = validate_sample(sample, mode)
            if _canonical_json(identity) != expected_canonical:
                raise ValueError(f"{mode} changed identity between processes")
        if expected_identity["cuda"]["device_name"] != gpu_identity.get("name"):
            raise ValueError(
                f"{mode} CUDA device name differs from nvidia-smi identity"
            )
        identities[mode] = expected_identity
        for field, direction, unit in METRICS:
            metric_rows.append(
                {
                    "name": f"training.{mode}.{field}",
                    "direction": direction,
                    "unit": unit,
                    "samples": [sample["result"][field] for sample in samples],
                }
            )

    environment = {
        "runner_schema_version": 1,
        "system": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "thread_environment": {
            name: os.environ.get(name, "unset")
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "gpu": gpu_identity,
        "mode_identities": identities,
    }
    environment_digest = hashlib.sha256(
        _canonical_json(environment).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "environment_key": f"jyppx-cuda-training-v1:{environment_digest}",
        "metadata": {
            "protocol": {
                "independent_processes": True,
                "warmup_processes_per_mode": min(
                    len(warmups.get(mode, [])) for mode in measured
                ),
                "sample_processes_per_mode": min(
                    len(samples) for samples in measured.values()
                ),
                "cuda_synchronize_timing_boundary": True,
            },
            "environment": environment,
            "noise_snapshots": noise,
        },
        "metrics": metric_rows,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _prepare_output_directory(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(f"output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _run_statistics(args: argparse.Namespace, samples_path: Path) -> None:
    statistics_path = args.output_dir / "statistics.json"
    command = [
        sys.executable,
        str(args.baseline_stats),
        str(samples_path),
        "--output",
        str(statistics_path),
        "--max-cv-percent",
        str(args.max_cv_percent),
        "--min-regression-percent",
        str(args.min_regression_percent),
    ]
    if args.reference is not None:
        command.extend(("--reference", str(args.reference.resolve())))
    if args.write_reference is not None:
        command.extend(("--write-reference", str(args.write_reference.resolve())))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    (args.output_dir / "statistics.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8", newline="\n"
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"baseline statistics failed with exit code {completed.returncode}"
        )


def main() -> int:
    args = _parse_arguments()
    if args.samples < 5:
        raise ValueError("--samples must be at least 5 for the CUDA gate")
    if args.warmup < 1:
        raise ValueError("--warmup must be at least 1")
    if args.device_index < 0:
        raise ValueError("--device-index must be non-negative")
    if args.reference is not None and args.write_reference is not None:
        raise ValueError("--reference and --write-reference are mutually exclusive")
    benchmark = args.benchmark.resolve()
    if not benchmark.is_file():
        raise ValueError(f"benchmark executable does not exist: {benchmark}")
    baseline_stats = args.baseline_stats.resolve()
    if not baseline_stats.is_file():
        raise ValueError(f"baseline statistics tool does not exist: {baseline_stats}")
    args.baseline_stats = baseline_stats
    args.output_dir = _prepare_output_directory(args.output_dir)
    modes = tuple(dict.fromkeys(args.mode or SUPPORTED_MODES))

    identity_before = _query_gpu_identity(args.device_index)
    noise = [_query_gpu_noise(args.device_index)]
    warmups: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    measured: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    for mode in modes:
        for index in range(1, args.warmup + 1):
            warmups[mode].append(
                _run_process(
                    benchmark,
                    mode,
                    args.output_dir,
                    f"warmup-{mode}-{index}",
                )
            )
        for index in range(1, args.samples + 1):
            measured[mode].append(
                _run_process(
                    benchmark,
                    mode,
                    args.output_dir,
                    f"sample-{mode}-{index}",
                )
            )
            noise.append(_query_gpu_noise(args.device_index))

    identity_after = _query_gpu_identity(args.device_index)
    if identity_after != identity_before:
        raise ValueError("nvidia-smi identity changed during the benchmark run")
    report = build_raw_report(measured, warmups, identity_before, noise)
    samples_path = args.output_dir / "samples.json"
    _write_json(samples_path, report)
    _run_statistics(args, samples_path)
    print(f"raw samples: {samples_path}")
    print(f"statistics: {args.output_dir / 'statistics.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"CUDA baseline error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
