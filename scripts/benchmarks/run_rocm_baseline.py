#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Collect development-only ROCm samples with native HIP device identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

MODES = {
    "training": ("rocm-fp32-single-sgd", "rocm-fp16-multi-sgd-amp"),
    "optimizer": tuple(
        f"rocm-fp32-optimizer-{name}"
        for name in ("sgd", "adam", "adamw", "rmsprop", "adagrad", "nadam", "adadelta", "lamb")
    ),
}
METRICS = {
    "training": (("samples_per_second", "higher", "samples_per_second"),),
    "optimizer": (("parameter_elements_per_second", "higher", "elements_per_second"),),
}
COMMON_METRICS = (
    ("mean_step_milliseconds", "lower", "milliseconds"),
    ("peak_allocated_bytes", "lower", "bytes"),
    ("peak_reserved_bytes", "lower", "bytes"),
)
BUILD_FIELDS = (
    "jyppx_version", "git_commit", "compiler", "cxx_standard", "os_arch",
    "build_type", "libtorch_version", "libtorch_build", "accelerator_runtime",
    "dependency_lock_digest",
)
COMPARISON_BLOCKED_REASON = "No qualified ROCm power/clock control observer"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def positive(value: Any, label: str, *, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        raise ValueError(f"{label} has the wrong numeric type")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


def text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError(f"{label} must be bounded nonempty text")


def validate_identity(document: dict[str, Any]) -> dict[str, Any]:
    """Validate actual HIP identity; a device name alone is insufficient."""
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("unsupported ROCm schema")
    build = document["build_fingerprint"]
    for field in BUILD_FIELDS:
        text(build[field], f"build.{field}")
    if build["build_type"].lower() != "release":
        raise ValueError("ROCm measurements require Release")
    if not re.fullmatch(r"rocm:available:[1-9][0-9]*", build["accelerator_runtime"]):
        raise ValueError("binary does not observe an available ROCm runtime")
    if not build["libtorch_build"].startswith("rocm-"):
        raise ValueError("binary does not report ROCm LibTorch")
    device = document["rocm"]
    if type(device["device_index"]) is not int or device["device_index"] != 0:
        raise ValueError("benchmark must use visible HIP device zero")
    for field in ("device_name", "architecture", "pci_bus_id", "hip_uuid"):
        text(device[field], f"rocm.{field}")
    if not re.fullmatch(r"[0-9a-f]{32}", device["hip_uuid"]) or int(device["hip_uuid"], 16) == 0:
        raise ValueError("invalid native HIP UUID")
    if not re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", device["pci_bus_id"]):
        raise ValueError("invalid native HIP PCI identity")
    if not re.fullmatch(r"gfx[0-9a-f]+(?::[A-Za-z0-9_+:-]+)?", device["architecture"]):
        raise ValueError("invalid HIP architecture")
    for field in ("hip_driver_api_version", "hip_runtime_version", "compile_hip_version", "total_memory_bytes"):
        positive(device[field], f"rocm.{field}", integer=True)
    for field in ("intraop_threads", "interop_threads"):
        positive(document["runtime"][field], f"runtime.{field}", integer=True)
    return {key: document[key] for key in ("build_fingerprint", "rocm", "runtime")}


def validate_sample(sample: dict[str, Any], suite: str, mode: str) -> dict[str, Any]:
    if mode not in MODES[suite] or sample["mode"] != mode:
        raise ValueError("unexpected ROCm mode")
    if sample["kind"] != f"jyppx-rocm-{suite}-benchmark-sample":
        raise ValueError("unexpected ROCm sample kind")
    identity = validate_identity(sample)
    workload = sample["workload"]
    expected_id = f"jyppx.rocm.{suite}.{'mlp' if suite == 'training' else 'dense'}/v1"
    if workload["id"] != expected_id or not re.fullmatch(r"fnv1a64:[0-9a-f]{16}", workload["digest"]):
        raise ValueError("unexpected ROCm workload identity")
    fields = ("seed", "steps", "batch_size", "sample_count", "input_features", "hidden_features", "output_features", "worker_count", "prefetch_batches") if suite == "training" else ("seed", "steps", "parameter_count", "parameter_elements")
    for field in fields:
        positive(workload[field], f"workload.{field}", integer=True)
    result = sample["result"]
    for field, _, _ in METRICS[suite] + COMMON_METRICS:
        positive(result[field], f"result.{field}", integer=field.endswith("_bytes"))
    if not result["peak_allocated_bytes"] <= result["peak_reserved_bytes"] <= identity["rocm"]["total_memory_bytes"]:
        raise ValueError("allocator peaks exceed reserved or physical memory")
    steps = workload["steps"]
    counter = result["global_step" if suite == "training" else "completed_steps"]
    if type(counter) is not int or counter != steps:
        raise ValueError("uncommitted benchmark steps")
    if suite == "training":
        if workload["sample_count"] != workload["steps"] * workload["batch_size"]:
            raise ValueError("inconsistent training sample count")
        expected_count = 2 if mode == "rocm-fp16-multi-sgd-amp" else 1
        counters = result["optimizer_steps"]
        if not isinstance(counters, list) or len(counters) != expected_count or any(type(x) is not int or x != steps for x in counters):
            raise ValueError("optimizer step counters do not match the mode")
        scaler = result["grad_scaler"]
        if expected_count == 2:
            if scaler["enabled"] is not True or scaler["checkpoint_ready"] is not True or scaler["scale"] != 128.0:
                raise ValueError("GradScaler did not commit the expected finite steps")
        elif scaler is not None:
            raise ValueError("FP32 mode must not report a scaler")
    else:
        if result["parameters_finite"] is not True or result["parameters_changed"] is not True:
            raise ValueError("optimizer numerical postconditions failed")
        for field in ("dtype", "gradient_policy", "option_profile"):
            text(workload[field], f"workload.{field}")
    return {**identity, "suite": suite, "mode": mode, "workload": workload}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_native(binary: Path, arguments: list[str], output: Path, label: str, timeout: float) -> dict[str, Any]:
    try:
        completed = subprocess.run([str(binary), *arguments], capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        (output / f"{label}.stdout.json").write_bytes(error.stdout or b"")
        (output / f"{label}.stderr.log").write_bytes(error.stderr or b"")
        write_json(output / f"{label}.process.json", {"status": "timeout", "timeout_seconds": timeout})
        raise RuntimeError(f"{label} exceeded its process time budget") from error
    (output / f"{label}.stdout.json").write_bytes(completed.stdout)
    (output / f"{label}.stderr.log").write_bytes(completed.stderr)
    write_json(output / f"{label}.process.json", {"status": "exited", "exit_code": completed.returncode})
    if completed.returncode:
        raise RuntimeError(f"{label} failed: exit {completed.returncode}; raw output preserved")
    if len(completed.stdout) > 1024 * 1024:
        raise ValueError("native report exceeded the document bound")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("native report must be a JSON object")
    return value


def observe_host() -> dict[str, Any]:
    observation = {"system": platform.system(), "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "logical_cpus": os.cpu_count()}
    if platform.system() == "Windows":
        # This is an independent list of Windows adapters, not a HIP UUID mapping.
        script = "[pscustomobject]@{cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors);memory_bytes=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory;display_adapters=@(Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,DriverVersion)} | ConvertTo-Json -Depth 5 -Compress"
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError("Windows host/display driver observation failed")
        observation["windows_cim"] = json.loads(result.stdout)
    else:
        observation["display_driver"] = {"status": "unavailable", "reason": "No OS display driver observer for this platform"}
    return observation


def build_report(measured: dict[str, list[dict[str, Any]]], warmups: dict[str, list[dict[str, Any]]], suite: str, native_identity: dict[str, Any], host: dict[str, Any], binary_sha256: str) -> dict[str, Any]:
    device_identity = validate_identity(native_identity)
    mode_identities = {}
    rows = []
    if not measured:
        raise ValueError("no measured modes")
    for mode, samples in measured.items():
        if len(samples) < 5 or not warmups.get(mode):
            raise ValueError("each mode requires a warmup and at least five independent samples")
        expected = None
        for sample in warmups[mode] + samples:
            current = validate_sample(sample, suite, mode)
            if any(current[k] != v for k, v in device_identity.items()):
                raise ValueError("sample differs from the native HIP/build/thread preflight")
            if expected is not None and current != expected:
                raise ValueError("sample workload changed between independent processes")
            expected = current
        mode_identities[mode] = expected
        for field, direction, unit in METRICS[suite] + COMMON_METRICS:
            rows.append({"name": f"{suite}.{mode}.{field}", "direction": direction, "unit": unit, "samples": [x["result"][field] for x in samples]})
    environment = {
        "host": host, "binary_sha256": binary_sha256,
        "mode_identities": mode_identities,
        "process_environment": {key: os.environ.get(key, "unset") for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
    }
    digest = hashlib.sha256(canonical(environment).encode()).hexdigest()
    return {
        "schema_version": 1, "environment_key": f"jyppx-rocm-{suite}-v1:{digest}",
        "comparison_eligibility": {"passed": False, "blocked_reasons": [COMPARISON_BLOCKED_REASON]},
        "metadata": {"environment": environment, "protocol": {"independent_processes": True, "warmup_processes_per_mode": min(map(len, warmups.values())), "sample_processes_per_mode": min(map(len, measured.values())), "hip_synchronize_timing_boundary": True}, "limitations": ["HIP driver API version is not the Windows display driver version", "Windows adapter observations do not establish a HIP UUID mapping", "No power/clock/temperature/noise telemetry; no cross-device performance claim", "Allocator peaks exclude driver/context and earlier model preparation"]},
        "metrics": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite", choices=tuple(MODES), default="training")
    parser.add_argument("--mode", choices=sum(MODES.values(), ()), action="append")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--max-cv-percent", type=float, default=10)
    parser.add_argument("--min-regression-percent", type=float, default=10)
    references = parser.add_mutually_exclusive_group()
    references.add_argument("--reference", type=Path)
    references.add_argument("--write-reference", type=Path)
    args = parser.parse_args()
    if args.samples < 5 or args.warmup < 1:
        raise ValueError("require at least five samples and one warmup")
    for name in ("timeout_seconds", "max_cv_percent", "min_regression_percent"):
        positive(getattr(args, name), name)
    modes = tuple(dict.fromkeys(args.mode or MODES[args.suite]))
    if any(mode not in MODES[args.suite] for mode in modes):
        raise ValueError("selected mode is outside the suite")
    if args.reference or args.write_reference:
        raise ValueError(COMPARISON_BLOCKED_REASON + "; collect standalone samples without reference options")
    binary = args.benchmark.resolve(strict=True)
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    before = run_native(binary, ["--identity"], output, "identity-before", args.timeout_seconds)
    if before.get("kind") != "jyppx-rocm-benchmark-identity":
        raise ValueError("preflight did not observe a ROCm benchmark")
    validate_identity(before)
    host_before = observe_host()
    write_json(output / "host-before.json", host_before)
    measured, warmups = {}, {}
    for mode in modes:
        measured[mode], warmups[mode] = [], []
        for label, count, destination in (("warmup", args.warmup, warmups), ("sample", args.samples, measured)):
            for index in range(1, count + 1):
                sample = run_native(binary, ["--mode", mode], output, f"{label}-{mode}-{index}", args.timeout_seconds)
                validate_sample(sample, args.suite, mode)
                destination[mode].append(sample)
                print(f"{label} {mode} {index}/{count} passed", flush=True)
    after = run_native(binary, ["--identity"], output, "identity-after", args.timeout_seconds)
    host_after = observe_host()
    write_json(output / "host-after.json", host_after)
    if after != before or host_after != host_before or hashlib.sha256(binary.read_bytes()).hexdigest() != binary_sha:
        raise ValueError("native device/build, host/display driver, or executable changed")
    report = build_report(measured, warmups, args.suite, before, host_before, binary_sha)
    write_json(output / "samples.json", report)
    command = [sys.executable, str(Path(__file__).with_name("baseline_stats.py")), str(output / "samples.json"), "--output", str(output / "statistics.json"), "--max-cv-percent", str(args.max_cv_percent), "--min-regression-percent", str(args.min_regression_percent)]
    statistics = subprocess.run(command, capture_output=True, timeout=120)
    (output / "statistics.log").write_bytes(statistics.stdout + statistics.stderr)
    print(statistics.stdout.decode("utf-8", errors="replace"))
    return statistics.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ROCm baseline error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
