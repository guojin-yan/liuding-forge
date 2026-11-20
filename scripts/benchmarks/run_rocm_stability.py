#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run bounded native HIP training, pressure and checkpoint recovery processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import run_rocm_baseline as baseline


def validate_report(sample: dict, precision: str, seconds: int, pressure_mib: int, identity: dict) -> None:
    if sample.get("kind") != "jyppx-rocm-stability-sample" or type(sample.get("schema_version")) is not int or sample["schema_version"] != 1:
        raise ValueError("unexpected stability report schema")
    if sample["identity"] != identity or sample["precision"] != precision:
        raise ValueError("stability process changed precision or native identity")
    baseline.validate_identity(sample["identity"])
    if sample["requested_seconds"] != seconds or sample["elapsed_seconds"] < seconds:
        raise ValueError("native workload did not reach the requested duration")
    baseline.positive(sample["elapsed_seconds"], "elapsed_seconds")
    for field in ("committed_steps", "checkpoint_restores", "exact_continuation_steps", "pressure_bytes"):
        baseline.positive(sample[field], field, integer=True)
    if sample["pressure_bytes"] != pressure_mib * 1024 * 1024:
        raise ValueError("native pressure differs from the requested payload")
    if sample["exact_continuation_steps"] != sample["checkpoint_restores"] * 3:
        raise ValueError("not all restored objects passed three exact continuation steps")
    if sample["committed_steps"] <= sample["exact_continuation_steps"]:
        raise ValueError("invalid committed step count")
    if sample["allocator_quota_oom_recovered"] is not True or sample["shape_rejection_recovered"] is not True:
        raise ValueError("native error recovery did not complete")
    if sample["allocator_quota_bytes"] != 64 * 1024 * 1024 or sample["allocator_rejected_bytes"] != 128 * 1024 * 1024:
        raise ValueError("native allocator quota recovery used a different budget")
    skips = sample["injected_overflow_skips"]
    if type(skips) is not int or skips != (1 if precision == "fp16" else 0):
        raise ValueError("unexpected overflow recovery count")
    loss = sample["final_loss"]
    if isinstance(loss, bool) or not isinstance(loss, (float, int)) or not math.isfinite(loss) or loss < 0:
        raise ValueError("final loss must be finite and nonnegative")
    snapshots = sample["snapshots"]
    if not isinstance(snapshots, list) or len(snapshots) != sample["checkpoint_restores"]:
        raise ValueError("missing checkpoint resource observations")
    previous_time, previous_step = 0, 0
    for snapshot in snapshots:
        if snapshot["elapsed_seconds"] <= previous_time or snapshot["committed_steps"] <= previous_step:
            raise ValueError("resource observations are not in increasing execution order")
        previous_time, previous_step = snapshot["elapsed_seconds"], snapshot["committed_steps"]
        baseline.positive(snapshot["checkpoint_bytes"], "checkpoint_bytes", integer=True)
        if snapshot["checkpoint_bytes"] > 64 * 1024 * 1024:
            raise ValueError("checkpoint exceeded its artifact budget")
    total = identity["rocm"]["total_memory_bytes"]
    for memory in [sample["initial_memory"], sample["final_memory"], sample["released_memory"], *snapshots]:
        for field in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "device_free_bytes", "device_total_bytes", "rss_bytes", "peak_rss_bytes"):
            baseline.positive(memory[field], f"memory.{field}", integer=True)
        if memory["device_total_bytes"] != total:
            raise ValueError("memory observation belongs to a different physical device")
        if not memory["allocated_bytes"] <= memory["reserved_bytes"] <= memory["peak_reserved_bytes"] <= total:
            raise ValueError("invalid allocator resource observation")
        if not memory["allocated_bytes"] <= memory["peak_allocated_bytes"] <= memory["peak_reserved_bytes"]:
            raise ValueError("invalid allocator peak observation")
        if memory["device_free_bytes"] > total or memory["rss_bytes"] > memory["peak_rss_bytes"]:
            raise ValueError("invalid free-memory or RSS observation")
    if sample["final_memory"]["allocated_bytes"] - sample["released_memory"]["allocated_bytes"] < sample["pressure_bytes"]:
        raise ValueError("pressure payload was not released")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), action="append")
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--checkpoint-seconds", type=int, default=15)
    parser.add_argument("--pressure-mib", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=1000000)
    args = parser.parse_args()
    if not 2 <= args.seconds <= 1800 or not 1 <= args.checkpoint_seconds < args.seconds:
        raise ValueError("duration/checkpoint interval outside native limits")
    if not 1 <= args.pressure_mib <= 4096 or not 1 <= args.max_steps <= 2000000:
        raise ValueError("pressure/step budget outside native limits")
    binary = args.benchmark.resolve(strict=True)
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    before = baseline.run_native(binary, ["--identity"], output, "identity-before", 120)
    if before.get("kind") != "jyppx-rocm-benchmark-identity":
        raise ValueError("native executable is not a ROCm benchmark")
    baseline.validate_identity(before)
    host_before = baseline.observe_host()
    baseline.write_json(output / "host-before.json", host_before)
    summaries = {}
    for precision in dict.fromkeys(args.precision or ("fp32", "fp16", "bf16")):
        sample = baseline.run_native(binary, ["--precision", precision, "--seconds", str(args.seconds), "--checkpoint-seconds", str(args.checkpoint_seconds), "--pressure-mib", str(args.pressure_mib), "--max-steps", str(args.max_steps), "--work-dir", str(output / precision)], output, precision, args.seconds + 120)
        validate_report(sample, precision, args.seconds, args.pressure_mib, before)
        progress = [json.loads(line) for line in (output / precision / "progress.ndjson").read_text().splitlines()]
        if progress != sample["snapshots"]:
            raise ValueError("on-disk progress evidence differs from the final report")
        summaries[precision] = {key: sample[key] for key in ("elapsed_seconds", "committed_steps", "checkpoint_restores", "exact_continuation_steps", "pressure_bytes", "injected_overflow_skips", "initial_memory", "final_memory", "released_memory")}
        print(f"{precision}: {sample['elapsed_seconds']:.2f}s, {sample['committed_steps']} committed steps, {sample['checkpoint_restores']} exact file restores passed", flush=True)
    after = baseline.run_native(binary, ["--identity"], output, "identity-after", 120)
    host_after = baseline.observe_host()
    baseline.write_json(output / "host-after.json", host_after)
    if after != before or host_after != host_before or hashlib.sha256(binary.read_bytes()).hexdigest() != binary_sha:
        raise ValueError("native device/build, host/display driver or executable changed")
    baseline.write_json(output / "validation.json", {
        "schema_version": 1, "kind": "jyppx-rocm-stability-validation",
        "identity": before, "host": host_before, "binary_sha256": binary_sha,
        "protocol": {"independent_process_per_precision": True, "seconds_per_precision": args.seconds, "process_timeout_seconds": args.seconds + 120, "checkpoint_seconds": args.checkpoint_seconds, "max_steps": args.max_steps, "pressure_mib": args.pressure_mib},
        "results": summaries,
        "limitations": ["Synthetic Linear/AdamW/StepLR workload with FP32 master weights; not all models or optimizers", "Process allocator quota OOM and synchronous shape rejection; not physical VRAM exhaustion, illegal memory access or driver reset recovery", "Repeated live-payload release and bounded allocator growth; RSS is observed without a leak threshold", "Pressure and checkpoint diagnostics are included in wall duration; this is not performance sampling or an hours-long soak test"],
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ROCm stability error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
