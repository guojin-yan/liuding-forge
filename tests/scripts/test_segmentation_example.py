# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Qualify a frozen native segmentation task and fresh-process recovery."""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def events(path):
    result = []
    for line in path.read_bytes().splitlines():
        try:
            item = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(item, dict) and item.get("event") == "checkpoint":
            result.append(item)
    return result


def arguments(binary):
    with tempfile.TemporaryDirectory(prefix="liuding-segmentation-arguments-") as root:
        root = Path(root)
        tails = [("--checkpoint-every", x) for x in ("0", "-1", "+1", "1200", "137x", "18446744073709551616", "")]
        tails += [("--checkpoint-every",), ("--unknown", "137"), ("-", "0"),
                  ("--checkpoint-every", "137", "--checkpoint-every", "137")]
        for index, tail in enumerate(tails):
            output = root / str(index)
            result = subprocess.run([str(binary), "train", str(root / "missing"), str(output), "cpu", *tail],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            require(result.returncode == 1 and not output.exists(), "Invalid syntax reached output creation")
        bad_sha = subprocess.run([str(binary), "evaluate", str(root / "missing"), str(root / "bad-sha"),
                                  "cpu", str(root / "weights"), "not-a-sha"],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        require(bad_sha.returncode == 1 and not (root / "bad-sha").exists(), "Invalid weight identity reached output creation")
    print(json.dumps({"passed": True, "argument_rejections": len(tails) + 1, "training_updates": 0}))


def verify(args, launches):
    root = args.output
    expected_names = (["continuous"] if not args.recovery_only else []) + ["paused", "resumed", "killed", "recovered"]
    require([x["name"] for x in launches if x["name"] != "evaluation"] == expected_names, "Recorded native launch sequence differs")
    training = {name: read(root / name / "training.json") for name in expected_names if name != "killed"}
    for name, value in training.items():
        initial, final = (0, 137) if name == "paused" else (137, 1200) if name in ("resumed", "recovered") else (0, 1200)
        require(value["initial_global_step"] == initial and value["global_step"] == final and
                value["applied_updates"] == final - initial and value["participant_count"] == 7,
                name + " training state differs")
        require(value["completed_epochs"] == (9 if name == "paused" else 80), "Epoch boundary differs")
    reference = "resumed" if args.recovery_only else "continuous"
    model_sha = digest(root / reference / "model.safetensors")
    if args.model_sha256 is not None:
        require(model_sha == args.model_sha256, "Original frozen model SHA differs")
    for name in ("resumed", "recovered"):
        for artifact in ("model.safetensors", "checkpoint.jyck"):
            require(filecmp.cmp(root / reference / artifact, root / name / artifact, shallow=False),
                    name + " full " + artifact + " bytes differ")
        require([x["global_step"] for x in events(root / (name + ".log"))] ==
                [274, 411, 548, 685, 822, 959, 1096, 1200], "Absolute periodic save alignment differs")
        require(training[name]["checkpoints_written"] == 8 and training[name]["checkpoint_every"] == 137,
                "Periodic save accounting differs")
        require(digest(root / name / "checkpoint.jyck") == training[name]["checkpoint_sha256"] ==
                events(root / (name + ".log"))[-1]["checkpoint_sha256"], "Native saved checkpoint identity differs")
    killed = next(x for x in launches if x["name"] == "killed")
    require(killed["forced_termination"] and (killed["returncode"] == -signal.SIGKILL if os.name == "posix" else killed["returncode"] != 0),
            "A real forced native termination was not recorded")
    require(not (root / "killed/training.json").exists() and not (root / "killed/model.safetensors").exists(),
            "Killed process published completed training")
    committed = events(root / "killed.log")
    require(len(committed) == 1 and committed[0]["global_step"] == 137 and
            committed[0]["checkpoint_sha256"] == digest(root / "killed/checkpoint.jyck") == killed["committed_checkpoint_sha256"],
            "Killed process lost or replaced its committed boundary")
    evaluation = root / "evaluation"
    if not evaluation.exists():
        run(args, launches, "evaluate", "evaluation", root / reference / "model.safetensors", model_sha)
    else:
        require(any(x["name"] == "evaluation" and x["returncode"] == 0 for x in launches), "Evaluation has no successful native launch")
    reference_path = evaluation / "independent-reference.json"
    if not reference_path.exists():
        command = [sys.executable, str(Path(__file__).with_name("check_segmentation_reference.py")), str(args.data), str(evaluation)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (root / "independent-reference.log").write_bytes(result.stdout)
        require(result.returncode == 0, "Independent preprocessing/counts/COCO reference failed")
    score, independent = read(evaluation / "results.json"), read(reference_path)
    require(independent["validation"]["all_data_sha256_verified"] == 170 and
            score["iou"]["overall"] >= .60 and score["iou"]["per_class"][1] >= .35 and
            score["detection"]["ap50"] >= .10, "Frozen held-out quality threshold failed")
    summary = {"passed": True, "device": args.device, "continuous_training_route_executed": not args.recovery_only,
               "previously_qualified_continuous_route_retrained": False,
               "original_model_sha256_required": args.model_sha256, "model_sha256": model_sha,
               "checkpoint_sha256": digest(root / reference / "checkpoint.jyck"),
               "exact_model": True, "exact_full_checkpoint": True,
               "full_checkpoint_reference_route": reference, "training": training,
               "forced_training_process_exit": killed["returncode"], "committed_step": 137,
               "applied_updates_lower_bound": 2400 if args.recovery_only else 3600,
               "applied_updates_upper_bound": 2537 if args.recovery_only else 3737,
               "post_checkpoint_killed_updates_unknown": True, "native_launches": len(launches),
               "evaluation": score, "independent_reference": independent}
    write(root / "summary.json", summary)
    print(json.dumps(summary))


def run(args, launches, action, name, *tail):
    command = [str(args.binary), action, str(args.data), str(args.output / name), args.device, *map(str, tail)]
    started = time.monotonic()
    with (args.output / (name + ".log")).open("xb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
    launches.append({"name": name, "argv": command, "returncode": result.returncode, "elapsed_seconds": time.monotonic() - started})
    write(args.output / "launches.json", launches)
    require(result.returncode == 0, name + " native task failed; retain the log")


def recover(args):
    require(not args.output.exists(), "Select a fresh output directory")
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    import pycocotools.cocoeval  # noqa: F401

    args.output.mkdir(parents=True)
    launches = []
    if not args.recovery_only:
        run(args, launches, "train", "continuous")
    run(args, launches, "train", "paused", "-", 137)
    run(args, launches, "train", "resumed", args.output / "paused/checkpoint.jyck", "--checkpoint-every", 137)
    command = [str(args.binary), "train", str(args.data), str(args.output / "killed"), args.device, "--checkpoint-every", "137"]
    log = args.output / "killed.log"
    started = time.monotonic()
    with log.open("xb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while True:
                require(process.poll() is None, "Native training exited before the frozen kill boundary")
                saved = events(log)
                if saved:
                    require(len(saved) == 1 and saved[0]["global_step"] == 137, "Supervisor missed first committed checkpoint")
                    process.kill()
                    break
                require(time.monotonic() - started < 120 and log.stat().st_size < 1 << 20, "First-checkpoint observation limit")
                time.sleep(.005)
            code = process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    launches.append({"name": "killed", "argv": command, "returncode": code,
                     "elapsed_seconds": time.monotonic() - started, "forced_termination": True,
                     "committed_checkpoint_sha256": digest(args.output / "killed/checkpoint.jyck"),
                     "post_checkpoint_applied_updates": "unknown; at most 137 before next checkpoint"})
    write(args.output / "launches.json", launches)
    run(args, launches, "train", "recovered", args.output / "killed/checkpoint.jyck", "--checkpoint-every", 137)
    verify(args, launches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--arguments-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--data", type=Path, default=os.environ.get("JYPPX_LIUDING_SEGMENTATION_DATA_DIR"))
    parser.add_argument("--output", type=Path, default=os.environ.get("JYPPX_LIUDING_SEGMENTATION_OUTPUT_DIR"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=os.environ.get("JYPPX_LIUDING_SEGMENTATION_DEVICE", "cpu"))
    parser.add_argument("--recovery-only", action="store_true", default=os.environ.get("JYPPX_LIUDING_SEGMENTATION_RECOVERY_ONLY") == "1")
    parser.add_argument("--model-sha256", default=os.environ.get("JYPPX_LIUDING_SEGMENTATION_EXPECTED_MODEL_SHA256"))
    args = parser.parse_args()
    require(args.binary.is_file(), "Build the native segmentation example")
    if args.arguments_only:
        arguments(args.binary)
    elif args.data is None:
        print("External frozen task not selected; no training or download")
        return 77
    else:
        require(args.output is not None, "Select a fresh output directory")
        require(args.data.is_dir() and digest(args.data / "manifest.json") ==
                "38386432386c0fc3b9560c98575677e150708b3dbfb4fc11825035309f4dbd95", "Frozen dataset differs")
        require(not args.recovery_only or args.model_sha256 is not None, "Recovery-only requires the original frozen model SHA")
        if args.verify_existing:
            verify(args, read(args.output / "launches.json"))
        else:
            recover(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
