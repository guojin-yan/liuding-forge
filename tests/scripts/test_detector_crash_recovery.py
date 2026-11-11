# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Kill a native detector after a committed checkpoint; reuse a frozen baseline."""
from __future__ import annotations

import argparse
import copy
import filecmp
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import tempfile
import time

from test_detector_example import reference, require


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arguments(binary):
    invalid = ("0", "-1", "+1", " 1", "1 ", "1200", "18446744073709551616", "137x", "")
    with tempfile.TemporaryDirectory(prefix="liuding-detector-arguments-") as root:
        root = Path(root)
        for index, value in enumerate(invalid):
            out = root / str(index)
            process = subprocess.run([str(binary), "train", str(root / "missing-data"), str(out),
                                      "cpu", "--checkpoint-every", value],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            require(process.returncode == 1 and b"Checkpoint interval must be a step" in process.stdout,
                    f"Invalid checkpoint interval accepted: {value!r}")
            require(not out.exists(), "Invalid interval created an output directory")
        for index, tail in enumerate((("--checkpoint-every",), ("--unknown", "137"),
                                      ("-", "0"), ("-", "+137"),
                                      ("--checkpoint-every", "137", "--checkpoint-every", "137")), len(invalid)):
            out = root / str(index)
            process = subprocess.run([str(binary), "train", str(root / "missing-data"), str(out), "cpu", *tail],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            require(process.returncode == 1 and not out.exists(), "Invalid option syntax reached data loading")
    print(json.dumps({"passed": True, "argument_rejections": len(invalid) + 5, "training_updates": 0}))


def checkpoint_events(log):
    result = []
    for line in log.read_bytes().splitlines():
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(value, dict) and value.get("event") == "checkpoint":
            result.append(value)
    return result


def compare_checkpoint(actual, baseline, expected_build_commit, expected_dependency_lock_digest=None):
    before, after = baseline.read_bytes(), actual.read_bytes()
    require(before[:12] == after[:12] == b"JYPPXCK1\x01\x00\x00\x00", "Unexpected checkpoint format")
    require(hashlib.sha256(before[:-32]).digest() == before[-32:] and
            hashlib.sha256(after[:-32]).digest() == after[-32:], "Checkpoint container checksum differs")
    length = struct.unpack_from("<Q", before, 12)[0]
    require(0 < length <= 1 << 20 and 20 + length + 32 <= len(before), "Checkpoint metadata boundary differs")
    metadata = before[20:20 + length]
    manifest = json.loads(metadata)

    def encode(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")

    require(encode(manifest) == metadata, "Independent manifest encoding does not reproduce the original bytes")
    expected = copy.deepcopy(manifest)
    if expected_build_commit is not None:
        expected["build"]["git_commit"] = expected_build_commit
    if expected_dependency_lock_digest is not None:
        expected["build"]["dependency_lock_digest"] = expected_dependency_lock_digest
    encoded = encode(expected)
    payload = before[:12] + struct.pack("<Q", len(encoded)) + encoded + before[20 + length:-32]
    expected_archive = payload + hashlib.sha256(payload).digest()
    require(after == expected_archive, "Full checkpoint differs from the frozen baseline and explicitly selected build provenance")
    return {"exact_expected_archive": True, "all_participant_payload_bytes_equal": True,
            "original_baseline_bytes_equal": before == after,
            "baseline_build_commit": manifest["build"]["git_commit"],
            "expected_build_commit": expected["build"]["git_commit"],
            "manifest_changes": ["build." + key for key in ("git_commit", "dependency_lock_digest")
                                 if manifest["build"][key] != expected["build"][key]],
            "expected_dependency_lock_digest": expected["build"]["dependency_lock_digest"],
            "expected_archive_sha256": hashlib.sha256(expected_archive).hexdigest(),
            "baseline_archive_sha256": hashlib.sha256(before).hexdigest()}


def finish(args, launches):
    require(len(launches) == 2 and launches[0]["name"] == "killed" and launches[1]["name"] == "resumed" and
            launches[0]["forced_termination"] and launches[1]["returncode"] == 0, "Saved launch contract differs")
    code = launches[0]["returncode"]
    require(code == -signal.SIGKILL if os.name == "posix" else code != 0, "Saved forced exit differs")
    killed_dir, resumed = args.output / "killed", args.output / "resumed"
    events = checkpoint_events(args.output / "killed.log")
    require(len(events) == 1 and events[0]["global_step"] == launches[0]["committed_step"] == 137,
            "Saved checkpoint boundary differs")
    committed_sha = digest(killed_dir / "checkpoint.jyck")
    require(committed_sha == events[0]["checkpoint_sha256"] == launches[0]["committed_checkpoint_sha256"],
            "Saved source checkpoint differs")
    require(not (killed_dir / "training.json").exists() and not (killed_dir / "model.safetensors").exists(),
            "Killed process published a completed result")
    training = json.loads((resumed / "training.json").read_text(encoding="utf-8"))
    require(training["initial_global_step"] == 137 and training["global_step"] == 1200 and
            training["applied_updates"] == 1063 and training["completed_epochs"] == 80 and
            training["participant_count"] == 7, "Fresh-process training restore differs")
    require(training["checkpoint_every"] == 137 and training["checkpoints_written"] == 8,
            "Periodic checkpoint accounting differs")
    resumed_events = checkpoint_events(args.output / "resumed.log")
    require([x["global_step"] for x in resumed_events] == [274, 411, 548, 685, 822, 959, 1096, 1200],
            "Resumed checkpoints did not preserve absolute-step alignment and the final tail")
    require(digest(args.baseline / "model.safetensors") == args.model_sha256 and
            digest(args.baseline / "checkpoint.jyck") == args.checkpoint_sha256, "Baseline identity changed")
    require(digest(resumed / "model.safetensors") == args.model_sha256 and
            filecmp.cmp(resumed / "model.safetensors", args.baseline / "model.safetensors", shallow=False),
            "Final model baseline bytes differ")
    checkpoint = compare_checkpoint(resumed / "checkpoint.jyck", args.baseline / "checkpoint.jyck",
                                    args.expected_build_commit, args.expected_dependency_lock_digest)
    require(checkpoint["expected_archive_sha256"] == training["checkpoint_sha256"] == resumed_events[-1]["checkpoint_sha256"],
            "Native final checkpoint identity differs")
    command = [str(args.binary), "evaluate", str(args.data), str(args.output / "evaluation"), args.device,
               str(resumed / "model.safetensors"), args.model_sha256]
    started = time.monotonic()
    with (args.output / "evaluation.log").open("xb") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    launches.append({"name": "evaluation", "argv": command, "returncode": completed.returncode,
                     "elapsed_seconds": time.monotonic() - started})
    (args.output / "launches-complete.json").write_text(json.dumps(launches, indent=2) + "\n", encoding="utf-8")
    require(completed.returncode == 0, "Evaluation failed; retain the native log")
    ref = reference(args.output / "evaluation")
    result = {"passed": True, "training_device": args.device, "baseline_retrained": False,
              "native_launches": len(launches), "forced_training_process_exit": code,
              "committed_checkpoint_step": 137, "resumed_applied_updates": 1063,
              "applied_updates_lower_bound": 1200, "applied_updates_upper_bound": 1337,
              "uncommitted_killed_updates_unknown": True,
              "exact_model": True, "exact_full_checkpoint": True, "checkpoint_comparison": checkpoint,
              "periodic_resume_steps": [x["global_step"] for x in resumed_events],
              "model_sha256": args.model_sha256, "checkpoint_sha256": checkpoint["expected_archive_sha256"],
              "committed_checkpoint_sha256": committed_sha, "independent_reference": ref}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


def recover(args):
    require(args.data.is_dir() and digest(args.data / "manifest.json") ==
            "38386432386c0fc3b9560c98575677e150708b3dbfb4fc11825035309f4dbd95", "Frozen dataset differs")
    require(args.baseline.is_dir(), "Select an existing immutable complete-task baseline")
    require(args.model_sha256 and args.checkpoint_sha256, "Explicit baseline SHA-256 values are required")
    require(digest(args.baseline / "model.safetensors") == args.model_sha256 and
            digest(args.baseline / "checkpoint.jyck") == args.checkpoint_sha256, "Baseline artifact identity differs")
    require(args.output is not None and not args.output.exists(), "A fresh output directory is required")
    # Resolve reference-only dependencies before starting the bounded native work.
    import numpy  # noqa: F401
    import pycocotools.cocoeval  # noqa: F401

    args.output.mkdir(parents=True)
    launches = []
    command = [str(args.binary), "train", str(args.data), str(args.output / "killed"), args.device,
               "--checkpoint-every", "137"]
    log = args.output / "killed.log"
    started = time.monotonic()
    with log.open("xb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        killed = False
        try:
            while True:
                require(process.poll() is None, "Training exited before the forced interruption")
                events = checkpoint_events(log)
                if events:
                    require(len(events) == 1 and events[0]["global_step"] == 137,
                            "The supervisor missed the frozen first-checkpoint boundary")
                    process.kill()
                    killed = True
                    break
                require(time.monotonic() - started < 90, "Checkpoint readiness timeout")
                require(log.stat().st_size < 1 << 20, "Training log cap")
                time.sleep(.005)
            code = process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    require(killed and code != 0, "A live native process was not forcibly terminated")
    if os.name == "posix":
        require(code == -signal.SIGKILL, "Expected a real SIGKILL exit")
    killed_dir = args.output / "killed"
    require(not (killed_dir / "model.safetensors").exists() and not (killed_dir / "training.json").exists(),
            "The interrupted process had already published a completed training result")
    events = checkpoint_events(log)
    require(len(events) == 1 and events[0]["global_step"] == 137, "Unexpected later checkpoint")
    committed_sha = digest(killed_dir / "checkpoint.jyck")
    require(committed_sha == events[0]["checkpoint_sha256"], "Killed process lost or replaced its committed checkpoint")
    launches.append({"name": "killed", "argv": command, "returncode": code,
                     "elapsed_seconds": time.monotonic() - started, "forced_termination": True,
                     "committed_step": 137, "committed_checkpoint_sha256": committed_sha,
                     "post_checkpoint_applied_updates": "unknown, at most 137 before the next checkpoint announcement"})

    def run(action, name, *tail):
        command = [str(args.binary), action, str(args.data), str(args.output / name), args.device, *map(str, tail)]
        started = time.monotonic()
        with (args.output / (name + ".log")).open("xb") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        launches.append({"name": name, "argv": command, "returncode": completed.returncode,
                         "elapsed_seconds": time.monotonic() - started})
        (args.output / "launches.json").write_text(json.dumps(launches, indent=2) + "\n", encoding="utf-8")
        require(completed.returncode == 0, name + " failed; retain its native log")

    run("train", "resumed", killed_dir / "checkpoint.jyck", "--checkpoint-every", 137)
    finish(args, launches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--arguments-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Verify the two already recorded native launches and run only their missing evaluation")
    parser.add_argument("--data", type=Path, default=os.environ.get("JYPPX_LIUDING_DETECTOR_DATA_DIR"))
    parser.add_argument("--output", type=Path, default=os.environ.get("JYPPX_LIUDING_DETECTOR_OUTPUT_DIR"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=os.environ.get("JYPPX_LIUDING_DETECTOR_DEVICE", "cpu"))
    parser.add_argument("--baseline", type=Path, default=os.environ.get("JYPPX_LIUDING_DETECTOR_BASELINE_DIR"))
    parser.add_argument("--model-sha256", default=os.environ.get("JYPPX_LIUDING_DETECTOR_BASELINE_MODEL_SHA256"))
    parser.add_argument("--checkpoint-sha256", default=os.environ.get("JYPPX_LIUDING_DETECTOR_BASELINE_CHECKPOINT_SHA256"))
    parser.add_argument("--expected-build-commit", default=os.environ.get("JYPPX_LIUDING_DETECTOR_EXPECTED_BUILD_COMMIT"))
    parser.add_argument("--expected-dependency-lock-digest",
                        default=os.environ.get("JYPPX_LIUDING_DETECTOR_EXPECTED_DEPENDENCY_LOCK_DIGEST"))
    args = parser.parse_args()
    require(args.binary.is_file(), "Build the native detector first")
    if args.arguments_only:
        arguments(args.binary)
    elif args.data is None or args.baseline is None:
        print("Pinned external data/baseline not selected; no training or download")
        return 77
    else:
        if args.verify_existing:
            require(args.output is not None and args.output.is_dir(), "Select the original interrupted validation output")
            launches = json.loads((args.output / "launches.json").read_text(encoding="utf-8"))
            finish(args, launches)
        else:
            recover(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
