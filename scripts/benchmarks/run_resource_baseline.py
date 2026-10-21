#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0.
"""Sample real resource profiles in bounded independent processes (stdlib only)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

MODELS = ("resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large", "transformer_encoder")
PROFILES = ("storage", "checkpoint", "telemetry", *MODELS, "quantization", "serving")
BUILD_FIELDS = ("jyppx_version", "git_commit", "compiler", "cxx_standard", "os_arch", "build_type",
                "libtorch_version", "libtorch_build", "accelerator_runtime", "dependency_lock_digest")
ENV_BUILD_FIELDS = tuple(key for key in BUILD_FIELDS if key not in ("git_commit", "jyppx_version"))
ENV_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                 "CUDA_DEVICE_ORDER", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
                 "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                 "OMP_PROC_BIND", "OMP_PLACES", "OMP_DYNAMIC", "OMP_WAIT_POLICY", "MKL_DYNAMIC",
                 "KMP_AFFINITY", "KMP_BLOCKTIME")
COMMON_METRICS = {"peak_rss_bytes", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "output_bytes"}
PROFILE_METRICS = {
    "storage": {"load_seconds", "publish_seconds", "throughput_bytes_per_second", "reopen_read_seconds", "artifact_payload_bytes"},
    "checkpoint": {"checkpoint_pause_seconds", "restore_seconds", "throughput_bytes_per_second"},
    "telemetry": {"serialization_seconds", "load_seconds", "emit_flush_seconds", "throughput_events_per_second", "reopen_replay_seconds", "canonical_bytes"},
    "quantization": {"calibration_conversion_seconds", "serialization_seconds", "load_seconds", "warmup_seconds", "forward_seconds", "throughput_items_per_second"},
    "serving": {"publish_seconds", "load_seconds", "warmup_seconds", "forward_seconds", "batch_seconds", "throughput_requests_per_second"},
}
for _model in MODELS:
    PROFILE_METRICS[_model] = {"load_seconds", "warmup_seconds", "forward_seconds", "throughput_items_per_second", "parameter_elements"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > (16 << 20):
        raise ValueError(f"JSON report exceeds 16 MiB: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    return object_value(value, str(path))


def object_value(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def workload(profile: str) -> tuple[dict[str, Any], int, tuple[int, int]]:
    if profile == "storage":
        return {"objects": 8, "bytes_per_object": 262144, "generator": "(i*37+i/251+object)%256",
                "durability": "SQLite FULL plus CAS publication"}, 8, (2097152, 16 << 20)
    if profile == "checkpoint":
        return {"checkpoints": 4, "payload_bytes": 1048576, "generator": "(i*37+i/251+checkpoint)%256",
                "synchronize": True}, 4, (4194304, 12 << 20)
    if profile == "telemetry":
        return {"events": 256, "encode_rounds": 16, "message": "fixed finite local resource event",
                "sink": "JSONL observer; stream flush, no fsync"}, 256, (25600, 4 << 20)
    if profile in MODELS:
        transformer = profile == "transformer_encoder"
        steps = 256 if transformer else 32
        size = 4096 if transformer else 56
        return {"model": profile, "input_shape": [2, 16, 32] if transformer else [2, 3, 32, 32],
                "seed": 20261021, "input": "arange%31 /31 -0.5 FP32", "classes": 7,
                "warmup_forwards": 1, "forwards": steps,
                "transformer": {"dimension": 32, "heads": 4, "feedforward": 64, "layers": 2,
                                "dropout": 0.0, "gelu": True, "final_norm": True}}, steps, (size, size)
    if profile == "quantization":
        return {"in_features": 64, "out_features": 32, "batch": 16, "forwards": 512,
                "codec_rounds": 128, "weights": "((i*13)%3)-1", "input": "(((i*7)%17)-8)/8",
                "bias": "(o%7)/8", "warmup_forwards": 1, "calibration_samples": 2}, 512, (2048, 65536)
    if profile == "serving":
        return {"fixture": "examples/serving/linear_bundle.hpp", "operation": "y[n,c]=x[n,c]*[2,-3][c]",
                "direct_requests": 128, "batch_rounds": 32, "batch_width_limit": 8, "max_wait_ms": 5,
                "request_shape": [1, 2], "request_value": 1.0, "warmup_predictions": 1,
                "transport": "in-process public API; no HTTP"}, 384, (32, 4 << 20)
    raise ValueError(f"unknown profile: {profile}")


def effective_device(profile: str, selected: str) -> str:
    return selected if profile in (*MODELS, "serving") else "cpu"


def validate_sample(sample: dict[str, Any], profile: str, device: str, commit: str) -> None:
    if sample.get("schema_version") != 1 or sample.get("kind") != "jyppx-resource-benchmark-sample":
        raise ValueError("unexpected resource sample protocol")
    if sample.get("profile") != profile:
        raise ValueError("sample profile mismatch")
    expected, operations, size_bounds = workload(profile)
    frozen = {"id": profile + "/v1", "descriptor": expected,
              "digest": "sha256:" + hashlib.sha256(canonical(expected).encode()).hexdigest()}
    if canonical(sample.get("workload")) != canonical(frozen):
        raise ValueError("fixed workload descriptor or digest mismatch")
    build = object_value(sample.get("build_fingerprint"), "build fingerprint")
    for key in BUILD_FIELDS:
        nonempty(build.get(key), "build." + key)
    if build["git_commit"] != commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("binary/source commit mismatch; reconfigure and incrementally rebuild from a clean commit")
    if build["build_type"].lower() != "release":
        raise ValueError("resource performance evidence requires Release")
    execution = object_value(sample.get("execution"), "execution")
    if execution.get("device") != device:
        raise ValueError("actual execution device mismatch")
    nonempty(execution.get("path"), "execution.path")
    nonempty(execution.get("timing"), "execution.timing")
    if profile in (*MODELS, "serving"):
        if execution.get("intraop_threads") != 1 or execution.get("interop_threads") != 1:
            raise ValueError("model/serving workload requires one intraop and interop thread")
    if device == "cuda:0":
        cuda = object_value(execution.get("cuda"), "execution.cuda")
        if cuda.get("device_index") != 0 or "synchronize" not in execution["timing"]:
            raise ValueError("CUDA timing/device protocol mismatch")
        for key in ("device_name", "device_uuid", "compute_capability"):
            nonempty(cuda.get(key), "cuda." + key)
        if not re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", cuda["device_uuid"]):
            raise ValueError("CUDA UUID is not an actual device identity")
        for key in ("compile_toolkit_version", "runtime_version", "driver_api_version"):
            positive(cuda.get(key), "cuda." + key)
        if not build["accelerator_runtime"].startswith("cuda:available:"):
            raise ValueError("CUDA runtime is not reported available")
    validation = object_value(sample.get("validation"), "validation")
    if (validation.get("expected_operations") != operations or
            validation.get("completed_operations") != operations or
            validation.get("finite_output") is not True or validation.get("content_verified") is not True):
        raise ValueError("profile did not verify and complete its exact workload")
    if profile in (*MODELS, "serving", "quantization"):
        output_sum = validation.get("output_sum")
        if isinstance(output_sum, bool) or not isinstance(output_sum, (int, float)) or not math.isfinite(output_sum):
            raise ValueError("missing finite output evidence")
    if profile == "serving":
        if (validation.get("batch_rows") != 256 or
                not 1 < validation.get("largest_actual_batch", 0) <= 8 or
                not 0 < validation.get("batch_forward_calls", 0) < 256):
            raise ValueError("serving profile did not execute actual batching")
    if profile == "quantization":
        if (execution.get("path") != "storage-quantized-dequantize-to-fp32-linear" or
                execution.get("compute_quantized") is not False or execution.get("qat") is not False or
                execution.get("fallback_modules") != 1):
            raise ValueError("quantization profile must declare storage-only FP32 fallback")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", validation.get("artifact_digest", "")):
            raise ValueError("quantization artifact digest missing")
    metrics = object_value(sample.get("metrics"), "metrics")
    if metrics.keys() != COMMON_METRICS | PROFILE_METRICS[profile]:
        raise ValueError("required resource metric set mismatch")
    for name, raw in metrics.items():
        metric = object_value(raw, name)
        expected_unit = ("seconds" if name.endswith("_seconds") else "elements" if name == "parameter_elements"
                         else name.removeprefix("throughput_").replace("_per_", "/") if name.startswith("throughput_")
                         else "bytes")
        expected_direction = "higher" if name.startswith("throughput_") else "lower"
        diagnostic = (name in ("load_seconds", "warmup_seconds", "calibration_conversion_seconds") or
                      (profile == "serving" and name == "publish_seconds"))
        if metric.get("unit") != expected_unit or metric.get("direction") != expected_direction:
            raise ValueError(f"{name}: metric unit or direction differs from the fixed contract")
        if type(metric.get("gate")) is not bool:
            raise ValueError(f"{name}: invalid comparison contract")
        if metric.get("status") == "available":
            positive(metric.get("value"), name)
            if metric["gate"] != (not diagnostic):
                raise ValueError(f"{name}: required stability gate was changed")
            if name.startswith("peak_cuda") and device == "cpu":
                raise ValueError("CPU profile cannot report measured CUDA allocation")
        elif metric.get("status") == "unavailable":
            nonempty(metric.get("reason"), name + ".reason")
            if "value" in metric or metric["gate"]:
                raise ValueError(f"{name}: unavailable metric must have no fabricated value or pass gate")
            if name not in ("peak_rss_bytes", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
                raise ValueError(f"{name}: required capability cannot be unavailable")
            if name.startswith("peak_cuda") and device == "cuda:0":
                raise ValueError("actual CUDA execution requires allocator peak evidence")
        else:
            raise ValueError(f"{name}: metric requires explicit available/unavailable status")
    output_bytes = metrics["output_bytes"]["value"]
    if not size_bounds[0] <= output_bytes <= size_bounds[1]:
        raise ValueError("profile output size exceeds frozen bounds")


def run_process(command: list[str], directory: Path, *, timeout: float,
                max_output_bytes: int, environment: dict[str, str] | None = None) -> dict[str, Any]:
    """Drain both pipes with bounded storage; retain logs and metadata on failure."""
    directory.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, Any] = {"command": command, "timeout_seconds": timeout,
                               "started_at_unix_ns": time.time_ns(),
                               "working_directory": str(directory.resolve()),
                               "max_output_bytes_per_stream": max_output_bytes}
    started = time.monotonic()
    process = None
    failures: list[str] = []
    lengths = {"stdout": 0, "stderr": 0}
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   stdin=subprocess.DEVNULL, cwd=directory, env=environment, shell=False)

        def drain(stream: Any, name: str) -> None:
            try:
                with (directory / (name + ".log")).open("wb") as log:
                    while chunk := stream.read(16384):
                        remaining = max(0, max_output_bytes - lengths[name])
                        log.write(chunk[:remaining])
                        lengths[name] += len(chunk)
                        if lengths[name] > max_output_bytes:
                            failures.append(name + " exceeded output bound")
                            process.kill()
                            break
            except (OSError, ValueError) as error:
                failures.append(name + " pipe failure: " + str(error))
                if process.poll() is None:
                    process.kill()
            finally:
                stream.close()

        workers = [threading.Thread(target=drain, args=(process.stdout, "stdout")),
                   threading.Thread(target=drain, args=(process.stderr, "stderr"))]
        for worker in workers:
            worker.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            failures.append("process timeout")
            process.kill()
            process.wait(timeout=10)
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                raise RuntimeError("benchmark pipe did not terminate after process exit")
        metadata["returncode"] = process.returncode
        if process.returncode != 0:
            failures.append("nonzero process exit: " + str(process.returncode))
        if failures:
            raise ValueError("; ".join(failures))
        if lengths["stdout"] == 0:
            raise ValueError("benchmark emitted no JSON sample")
        sample = read_json(directory / "stdout.log")
        write_json(directory / "sample.json", sample)
        metadata["status"] = "captured"
        return sample
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        metadata.update(elapsed_seconds=time.monotonic() - started, output_bytes=lengths)
        write_json(directory / "process.json", metadata)


def source_identity(root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(["git", "-C", str(root), *arguments], check=True,
                                capture_output=True, encoding="utf-8", timeout=15)
        return result.stdout.strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != root.resolve():
        raise ValueError("--source-root must be the repository root")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("resource evidence requires a clean source checkout")
    return {"git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"), "clean": True}


def cache_identity(build: Path, source: Path) -> dict[str, str]:
    cache = build / "CMakeCache.txt"
    entries = {}
    for line in cache.read_text(encoding="utf-8-sig").splitlines():
        if line and not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            entries[key.split(":", 1)[0]] = value
    if Path(entries.get("CMAKE_HOME_DIRECTORY", "")).resolve() != source.resolve():
        raise ValueError("build cache belongs to a different source directory")
    selected = {key: value for key, value in entries.items()
                if (key.startswith(("JYPPX_BUILD_", "JYPPX_ENABLE_")) or
                    re.fullmatch(r"CMAKE_(?:CXX_FLAGS.*|EXE_LINKER_FLAGS.*|INTERPROCEDURAL_OPTIMIZATION.*|MSVC_RUNTIME_LIBRARY|GENERATOR|GENERATOR_PLATFORM|GENERATOR_TOOLSET)", key))}
    if selected.get("JYPPX_BUILD_BENCHMARKS") != "ON":
        raise ValueError("cache must enable JYPPX_BUILD_BENCHMARKS")
    return selected


def cpu_identity() -> dict[str, Any]:
    model = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    elif sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    nonempty(model, "actual CPU model")
    return {"host": platform.node(), "os": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "processor": model, "logical_cpus": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


def gpu_identity(uuid: str) -> dict[str, str]:
    fields = ("uuid", "name", "driver_version", "memory.total", "pci.bus_id")
    result = subprocess.run(["nvidia-smi", "--id=" + uuid, "--query-gpu=" + ",".join(fields),
                             "--format=csv,noheader,nounits"], check=True, capture_output=True,
                            encoding="utf-8", timeout=15)
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(fields) or rows[0][0].strip().lower() != uuid.lower():
        raise ValueError("nvidia-smi did not identify the actual visible CUDA UUID")
    return {name: nonempty(value.strip(), "GPU " + name) for name, value in zip(fields, rows[0])}


def environment_identity(host: dict[str, Any], cache: dict[str, str], samples: dict[str, dict[str, Any]],
                         environment: dict[str, str], gpu: dict[str, str] | None) -> dict[str, Any]:
    """Commit/version/binary hashes belong to provenance, never the environment key."""
    return {"host": host, "cmake_configuration": cache, "runtime_environment": environment,
            "gpu_driver": gpu,
            "profiles": {name: {"workload": sample["workload"], "execution": sample["execution"],
                "build_environment": {field: sample["build_fingerprint"][field] for field in ENV_BUILD_FIELDS},
                "metric_contract": {key: {field: value.get(field) for field in
                    ("status", "unit", "direction", "gate", "scope", "reason")}
                    for key, value in sample["metrics"].items()}}
                for name, sample in samples.items()}}


def summarize(samples: dict[str, list[dict[str, Any]]], max_cv: float) -> dict[str, Any]:
    summaries, unstable, unavailable = {}, [], []
    for profile, rows in samples.items():
        if len(rows) < 5:
            raise ValueError("at least five independent sample processes are required")
        for name, contract in rows[0]["metrics"].items():
            label = profile + "." + name
            if contract["status"] == "unavailable":
                summaries[label] = dict(contract)
                unavailable.append(label)
                continue
            values = [positive(row["metrics"][name]["value"], label) for row in rows]
            mean = statistics.fmean(values)
            cv = statistics.stdev(values) / mean * 100
            summaries[label] = {"status": "available", "unit": contract["unit"],
                "direction": contract["direction"], "gate": contract["gate"], "samples": values,
                "median": statistics.median(values), "mean": mean,
                "p95": sorted(values)[math.ceil(len(values) * 0.95) - 1], "cv_percent": cv,
                "allowed_regression_percent": max(10.0, 3 * cv)}
            if contract["gate"] and cv > max_cv:
                unstable.append(label)
    return {"metrics": summaries, "stability": {"passed": not unstable,
            "max_cv_percent": max_cv, "unstable_metrics": unstable,
            "unavailable_metrics": unavailable}}


def make_reference(report: dict[str, Any]) -> dict[str, Any]:
    if not report["stability"]["passed"]:
        raise ValueError("cannot publish a reference from unstable samples")
    return {"schema_version": 1, "kind": "jyppx-resource-benchmark-reference",
            "environment_key": report["environment_key"], "environment": report["environment"],
            "source_provenance": report["source_provenance"],
            "metrics": {name: {key: value[key] for key in
                ("status", "unit", "direction", "gate", "median", "cv_percent", "allowed_regression_percent")
                if key in value} for name, value in report["metrics"].items()}}


def compare_reference(report: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    if reference.get("schema_version") != 1 or reference.get("kind") != "jyppx-resource-benchmark-reference":
        raise ValueError("unsupported resource reference protocol")
    if reference.get("environment_key") != report["environment_key"] or reference.get("environment") != report["environment"]:
        raise ValueError("reference environment/workload differs; no cross-environment regression claim")
    if reference.get("metrics", {}).keys() != report["metrics"].keys():
        raise ValueError("reference resource metric set differs")
    provenance = object_value(reference.get("source_provenance"), "reference source provenance")
    if not re.fullmatch(r"[0-9a-f]{40}", provenance.get("git_commit", "")):
        raise ValueError("reference must preserve its actual source commit")
    failures, comparisons = [], {}
    for name, current in report["metrics"].items():
        previous = reference["metrics"][name]
        if any(previous.get(key) != current.get(key) for key in ("status", "unit", "direction", "gate")):
            raise ValueError(name + ": reference comparison contract differs")
        if current["status"] == "unavailable" or not current["gate"]:
            continue
        old = positive(previous.get("median"), name + " reference median")
        allowance = positive(previous.get("allowed_regression_percent"), name + " allowance")
        cv = previous.get("cv_percent")
        if isinstance(cv, bool) or not isinstance(cv, (int, float)) or not math.isfinite(cv) or not 0 <= cv <= 10:
            raise ValueError(name + ": reference was not stable at the required 10% CV gate")
        if allowance != max(10.0, 3 * cv):
            raise ValueError(name + ": reference regression allowance was modified")
        degradation = (current["median"] - old) / old * 100
        if current["direction"] == "higher":
            degradation = -degradation
        passed = degradation <= allowance
        comparisons[name] = {"reference_median": old, "current_median": current["median"],
                             "degradation_percent": degradation, "allowed_regression_percent": allowance,
                             "passed": passed}
        if not passed:
            failures.append(name)
    return {"passed": not failures, "failed_metrics": failures, "metrics": comparisons,
            "reference_source_provenance": provenance,
            "reference_commit_differs": provenance["git_commit"] != report["source_provenance"]["git_commit"]}


def executable(build: Path, bin_dir: Path | None, profile: str) -> Path:
    owner = "models" if profile in MODELS else profile if profile in ("quantization", "serving") else "stable"
    filename = "jyppx_bench_resource_" + owner + (".exe" if sys.platform == "win32" else "")
    candidates = [bin_dir / filename] if bin_dir else [build / "tests" / "benchmarks" / filename,
        build / "tests" / "benchmarks" / "Release" / filename]
    matches = [path.resolve() for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"profile {profile}: expected exactly one built executable, found {matches}")
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory; never overwritten")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu",
                        help="Models/serving device; storage/checkpoint/telemetry/quantization remain CPU")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-output-bytes", type=int, default=1 << 20)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--reference", type=Path)
    comparison.add_argument("--write-reference", type=Path, help="New reference only; existing evidence is immutable")
    args = parser.parse_args(argv)
    if not 1 <= args.warmup <= 10 or not 5 <= args.samples <= 100:
        raise ValueError("require 1..10 warmup processes and 5..100 measured processes")
    if not 1 <= positive(args.timeout, "timeout") <= 3600 or not 4096 <= args.max_output_bytes <= 16 << 20:
        raise ValueError("timeout/output bounds are outside allowed limits")
    if len(set(args.profiles)) != len(args.profiles):
        raise ValueError("duplicate resource profiles")
    if args.write_reference and args.write_reference.exists():
        raise ValueError("existing references must not be overwritten")
    source, build, output = args.source_root.resolve(), args.build_dir.resolve(), args.output.resolve()
    provenance = source_identity(source)
    cache = cache_identity(build, source)
    cache_hash = sha256_file(build / "CMakeCache.txt")
    host = cpu_identity()
    binaries = {name: executable(build, args.bin_dir, name) for name in args.profiles}
    binary_hashes = {name: sha256_file(path) for name, path in binaries.items()}
    provenance.update(binaries={name: {"sha256": binary_hashes[name], "path": str(path)} for name, path in binaries.items()},
                      runner_sha256=sha256_file(Path(__file__)), cmake_cache_sha256=cache_hash)
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    relevant_environment = {name: environment.get(name, "<unset>") for name in ENV_VARIABLES}
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"schema_version": 1, "kind": "jyppx-resource-benchmark-run",
        "status": "running", "started_at_unix_ns": time.time_ns(),
        "source_provenance": provenance, "profiles": args.profiles,
        "requested_device": args.device, "warmup_processes": args.warmup,
        "sample_processes": args.samples, "host": host, "runtime_environment": relevant_environment}
    write_json(output / "run.json", manifest)
    try:
        measured: dict[str, list[dict[str, Any]]] = {}
        identities: dict[str, dict[str, Any]] = {}
        gpu = None
        for profile in args.profiles:
            measured[profile] = []
            first_identity = None
            device = effective_device(profile, args.device)
            for index in range(args.warmup + args.samples):
                phase = "warmup" if index < args.warmup else "sample"
                number = index if phase == "warmup" else index - args.warmup
                directory = output / profile / (phase + f"-{number:02d}")
                command = [str(binaries[profile]), "--profile", profile, "--device", device,
                           "--work-dir", str(directory / "workload")]
                sample = run_process(command, directory, timeout=args.timeout,
                                     max_output_bytes=args.max_output_bytes, environment=environment)
                try:
                    validate_sample(sample, profile, device, provenance["git_commit"])
                except (ValueError, TypeError) as error:
                    write_json(directory / "validation.json", {"passed": False, "phase": phase,
                        "profile": profile, "error": str(error)})
                    raise
                if device == "cuda:0":
                    actual_gpu = gpu_identity(sample["execution"]["cuda"]["device_uuid"])
                    if gpu is not None and gpu != actual_gpu:
                        raise ValueError("GPU/driver identity changed during sampling")
                    gpu = actual_gpu
                current_identity = environment_identity(host, cache, {profile: sample}, relevant_environment, gpu)
                if first_identity is not None and current_identity != first_identity:
                    raise ValueError(profile + ": environment/workload/metric identity changed between processes")
                if profile in identities and sample["build_fingerprint"] != identities[profile]["build_fingerprint"]:
                    raise ValueError(profile + ": full build fingerprint changed between processes")
                first_identity = current_identity
                identities[profile] = sample
                if phase == "sample":
                    measured[profile].append(sample)
                write_json(directory / "validation.json", {"passed": True, "phase": phase,
                    "profile": profile, "build_fingerprint": sample["build_fingerprint"]})
                print(profile + " " + phase + f" {number + 1}: validated", flush=True)
        if source_identity(source) != {key: provenance[key] for key in ("git_commit", "git_tree", "clean")}:
            raise ValueError("source provenance changed during sampling")
        if sha256_file(build / "CMakeCache.txt") != cache_hash or any(
                sha256_file(path) != binary_hashes[name] for name, path in binaries.items()):
            raise ValueError("build cache or benchmark binary changed during sampling")
        env_identity = environment_identity(host, cache, identities, relevant_environment, gpu)
        provenance["build_fingerprints"] = {name: sample["build_fingerprint"] for name, sample in identities.items()}
        report = {"schema_version": 1, "kind": "jyppx-resource-benchmark-statistics",
            "environment_key": "sha256:" + hashlib.sha256(canonical(env_identity).encode()).hexdigest(),
            "environment": env_identity, "source_provenance": provenance,
            **summarize(measured, 10.0)}
        write_json(output / "statistics.json", report)
        if args.reference:
            report["comparison"] = compare_reference(report, read_json(args.reference))
            write_json(output / "statistics.json", report)
        passed = report["stability"]["passed"] and report.get("comparison", {}).get("passed", True)
        if args.write_reference and passed:
            args.write_reference.parent.mkdir(parents=True, exist_ok=True)
            with args.write_reference.open("x", encoding="utf-8") as stream:
                json.dump(make_reference(report), stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
        manifest.update(status="passed" if passed else "failed", statistics="statistics.json",
                        environment_key=report["environment_key"])
        print("resource evidence: " + str(output / "statistics.json"))
        return 0 if passed else 1
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "run.json", manifest)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print("resource baseline failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
