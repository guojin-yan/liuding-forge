#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Rehearse two installed SDKs using native training and exact-resume fences.

This is an opt-in maintenance checker, not an installer or migration API.
It never downloads, builds, signs, changes a prefix, or edits saved lineage.
The caller supplies trusted CLIs and their required runtime environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT_FIELDS = (
    "global_step", "completed_epochs", "optimizer_steps", "scheduler_steps",
    "model_sha256", "training_metrics", "validation_metrics", "accuracy",
    "mse", "confusion", "validation_samples", "next_rng_sha256",
    "next_cpu_rng_sha256", "numerics",
)
PARTICIPANTS = {
    "callbacks.dispatcher", "model.main", "recipe.lr", "recipe.sgd",
    "recipe.trainer", "recipe.training", "recipe.validation", "rng.global",
    "strategy.single", "training.diagnostics",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_state(root: Path) -> dict[str, Any]:
    """Bind file membership, content and mtimes, including SQLite and CAS."""
    result: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "run tree contains a link")
        if path.is_file():
            stat = path.stat()
            result[path.relative_to(root).as_posix()] = {
                "size": stat.st_size, "sha256": sha(path), "mtime_ns": stat.st_mtime_ns,
            }
    return result


def write_json(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


class Rehearsal:
    def __init__(self, work: Path, previous: Path, candidate: Path, timeout: int, *, saved: bool = False):
        self.work, self.previous, self.candidate = work, previous, candidate
        self.timeout = timeout
        self.saved = saved
        self.calls: list[dict[str, Any]] = []

    def invoke(self, cli: Path, name: str, *arguments: object, expected: int = 0) -> dict:
        command = [str(cli), *(str(value) for value in arguments), "--format", "json"]
        output, error = self.work / f"{name}.stdout.json", self.work / f"{name}.stderr.log"
        if self.saved:
            call = json.loads((self.work / f"{name}.call.json").read_text(encoding="utf-8"))
            require(call["name"] == name and call["command"] == command and call["returncode"] == expected,
                    name + ": saved invocation identity differs")
            require(output.stat().st_size <= 1 << 20 and error.stat().st_size <= 1 << 20 and
                    sha(output) == call["stdout_sha256"] and sha(error) == call["stderr_sha256"],
                    name + ": saved native output differs")
            self.calls.append(call)
            envelope = json.loads(output.read_text(encoding="utf-8"))
            require(envelope["schema_version"] == 1 and envelope["status"] == ("ok" if expected == 0 else "error"),
                    name + ": saved CLI envelope differs")
            return envelope
        started = time.monotonic()
        # File-backed output avoids retaining an unbounded pipe in the checker.
        with output.open("xb") as stdout, error.open("xb") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr,
                                    env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"),
                                    timeout=self.timeout, check=False)
        call = {"name": name, "command": command, "returncode": result.returncode,
                "seconds": time.monotonic() - started,
                "stdout_sha256": sha(output), "stderr_sha256": sha(error)}
        self.calls.append(call)
        write_json(self.work / f"{name}.call.json", call)
        require(result.returncode == expected, f"{name}: unexpected native exit {result.returncode}")
        require(output.stat().st_size <= 1 << 20 and error.stat().st_size <= 1 << 20,
                f"{name}: diagnostic output exceeds 1 MiB")
        envelope = json.loads(output.read_text(encoding="utf-8"))
        require(envelope["schema_version"] == 1, f"{name}: unexpected CLI envelope")
        require(envelope["status"] == ("ok" if expected == 0 else "error"),
                f"{name}: unexpected CLI status")
        return envelope

    def unchanged_refusal(self, cli: Path, name: str, config: Path, root: Path) -> None:
        proof_path = self.work / f"{name}.preservation.json"
        if self.saved:
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
            require(proof["passed"] is True and proof["content_and_mtimes_unchanged"] is True and
                    bool(proof["files"]), name + ": original preservation receipt is absent")
        else:
            before = tree_state(root)
        refused = self.invoke(cli, name, "recipe", "resume", "--file", config,
                              "--root", root, expected=2)
        error = refused["error"]
        require(error["name"] == "invalid_argument" and
                "build" in error["message"] and "resume rejected" in error["message"],
                f"{name}: refusal did not identify the exact-resume build fence")
        if not self.saved:
            require(tree_state(root) == before, f"{name}: incompatible SDK modified saved state")
            write_json(proof_path, {"passed": True, "files": before, "content_and_mtimes_unchanged": True})

    def final_state(self, cli: Path, name: str, root: Path, result: dict) -> dict:
        require(result["status"] == "completed" and result["global_step"] == 1280 and
                result["optimizer_steps"] == 1280 and result["scheduler_steps"] == 40 and
                result["completed_epochs"] == 40 and result["validation_samples"] == 256,
                f"{name}: full frozen task did not complete")
        require(result["quality_passed"] is True and result["accuracy"] >= .95 and
                result["mse"] <= .005, f"{name}: original task quality threshold failed")
        self.invoke(cli, name + "-checkpoint-verify", "checkpoint", "verify", "--path", root / "final.jyck")
        manifest = self.invoke(cli, name + "-checkpoint-inspect", "checkpoint", "inspect",
                               "--path", root / "final.jyck")["data"]["manifest"]
        participants = {entry["state"]["participant_id"]: entry for entry in manifest["participants"]}
        require(set(participants) == PARTICIPANTS, f"{name}: complete participant set differs")
        self.invoke(cli, name + "-run-verify", "run", "verify", "--root", root / "run-store")
        before = {file: sha(root / file) for file in ("final.jyck", "model.safetensors", "result.json")}
        replay = self.invoke(cli, name + "-completed-replay", "recipe", "resume",
                             "--file", self.work / "config.json", "--root", root)["data"]
        require(replay == result and all(sha(root / file) == digest for file, digest in before.items()),
                f"{name}: completed replay changed results or training state")
        return {"participants": participants, "artifacts": before, "manifest": manifest}

    def run(self, previous_commit: str, candidate_commit: str) -> dict:
        versions = {}
        for label, cli, expected in (("previous", self.previous, previous_commit),
                                     ("candidate", self.candidate, candidate_commit)):
            version = self.invoke(cli, label + "-version", "--version")["data"]
            require(version["build"]["git_commit"] == expected, label + ": binary source identity differs")
            versions[label] = version
        require(versions["previous"]["build"] != versions["candidate"]["build"],
                "two distinct SDK build fingerprints are required")
        for field in ("os_arch", "libtorch_build", "libtorch_version"):
            require(versions["previous"]["build"][field] == versions["candidate"]["build"][field],
                    "this rehearsal requires the same platform and LibTorch runtime")
        prepared = self.invoke(self.previous, "prepare-data", "data", "prepare", "--name",
                               "paired-affine", "--data-root", self.work / "datasets")["data"]
        data = Path(prepared["destination"])
        config = json.loads((ROOT / "recipes/multi-io/paired-affine.json").read_text(encoding="utf-8"))
        config.update(train_file=str(data / "train.jsonl"), validation_file=str(data / "validation.jsonl"), device="cpu")
        config_file = self.work / "config.json"
        if self.saved:
            require(json.loads(config_file.read_text(encoding="utf-8")) == config,
                    "saved full-task configuration differs")
        else:
            write_json(config_file, config)
        checked = self.invoke(self.candidate, "preflight", "recipe", "check", "--file", config_file)["data"]
        require(checked["total_steps"] == 1280 and checked["dataset"]["train"]["samples"] == 1024 and
                checked["dataset"]["validation"]["samples"] == 256, "frozen full task differs")
        baseline_root, rollback_root, candidate_root = (self.work / name for name in ("baseline", "rollback", "candidate"))
        baseline = self.invoke(self.previous, "previous-reference", "recipe", "run", "--file", config_file,
                               "--root", baseline_root)["data"]
        states = {"baseline": self.final_state(self.previous, "baseline", baseline_root, baseline)}
        for label, cli, other, root in (("rollback", self.previous, self.candidate, rollback_root),
                                        ("candidate", self.candidate, self.previous, candidate_root)):
            paused = self.invoke(cli, label + "-pause", "recipe", "run", "--file", config_file,
                                 "--root", root, "--pause-after", 259)["data"]
            require(paused["status"] == "paused" and paused["global_step"] == 259,
                    label + ": precise pause boundary differs")
            checkpoint_before = sha(root / "latest.jyck")
            self.unchanged_refusal(other, label + "-incompatible-resume", config_file, root)
            require(sha(root / "latest.jyck") == checkpoint_before, "paused checkpoint changed")
            resumed = self.invoke(cli, label + "-resume", "recipe", "resume", "--file", config_file,
                                  "--root", root)["data"]
            for field in RESULT_FIELDS:
                require(resumed[field] == baseline[field], f"{label}: recovered result differs at {field}")
            states[label] = self.final_state(cli, label, root, resumed)
            require(states[label]["participants"] == states["baseline"]["participants"],
                    label + ": full participant state differs")
            expected_build = versions["previous" if label == "rollback" else "candidate"]["build"]
            manifest = states[label]["manifest"]
            baseline_manifest = states["baseline"]["manifest"]
            require(manifest["build"] == expected_build and
                    baseline_manifest["build"] == versions["previous"]["build"],
                    label + ": checkpoint does not preserve its actual build identity")
            require({key: value for key, value in manifest.items() if key != "build"} ==
                    {key: value for key, value in baseline_manifest.items() if key != "build"},
                    label + ": checkpoint differs beyond its explicit build identity")
            require(states[label]["artifacts"]["model.safetensors"] == states["baseline"]["artifacts"]["model.safetensors"],
                    label + ": model bytes differ")
            if label == "rollback":
                require(states[label]["artifacts"]["final.jyck"] == states["baseline"]["artifacts"]["final.jyck"],
                        "rollback: whole checkpoint bytes differ under the original SDK")
        return {"passed": True, "kind": "jyppx-sdk-transition-rehearsal", "schema_version": 1,
                "SDKs": versions, "device": "cpu", "full_task_updates": 1280,
                "total_applied_updates": 3840, "pause_update": 259,
                "incompatible_resume_refusals": 2, "refusals_preserve_bytes_and_mtimes": True,
                "complete_participants": sorted(PARTICIPANTS), "states": states,
                "accuracy": baseline["accuracy"], "mse": baseline["mse"],
                "same_build_whole_checkpoint_exact": True,
                "cross_build_manifest_difference": ["build"],
                "verification_kind": "saved-native-receipts" if self.saved else "native-execution",
                "native_calls_executed_in_verification": 0 if self.saved else len(self.calls),
                "same_version_development_rehearsal": versions["previous"]["version"] == versions["candidate"]["version"],
                "release_qualification_granted": False, "calls": self.calls}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-cli", required=True, type=Path)
    parser.add_argument("--candidate-cli", required=True, type=Path)
    parser.add_argument("--expected-previous-commit", required=True)
    parser.add_argument("--expected-candidate-commit", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--verify-existing", action="store_true",
                        help="read saved native receipts without executing or overwriting them")
    args = parser.parse_args()
    require(1 <= args.timeout <= 180, "per-process timeout must be 1..180 seconds")
    for commit in (args.expected_previous_commit, args.expected_candidate_commit):
        require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is not None, "explicit full commit required")
    require(args.expected_previous_commit != args.expected_candidate_commit, "distinct source commits are required")
    previous, candidate, work = args.previous_cli.resolve(strict=True), args.candidate_cli.resolve(strict=True), args.work_dir.absolute()
    require(previous != candidate, "two distinct installed CLIs are required")
    require(not work.is_symlink() and (work.is_dir() if args.verify_existing else not work.exists()),
            "work directory must be existing for read-only verification, otherwise new")
    for cli in (previous, candidate):
        require(cli.is_file() and work not in cli.parents and
                cli.parent not in work.parents and cli.parent.parent not in work.parents,
                "work directory must be outside both SDK prefixes")
    binaries = {str(cli): {"sha256": sha(cli), "mtime_ns": cli.stat().st_mtime_ns} for cli in (previous, candidate)}
    if not args.verify_existing:
        work.mkdir()
    report_path = work / ("transition-verification.json" if args.verify_existing else "transition.json")
    require(not report_path.exists(), "refuse to overwrite an earlier verification result")
    rehearsal = Rehearsal(work, previous, candidate, args.timeout, saved=args.verify_existing)
    try:
        report = rehearsal.run(args.expected_previous_commit, args.expected_candidate_commit)
        require(binaries == {str(cli): {"sha256": sha(cli), "mtime_ns": cli.stat().st_mtime_ns} for cli in (previous, candidate)},
                "installed binary changed during rehearsal")
    except Exception as error:
        write_json(report_path, {"passed": False, "error": str(error), "calls": rehearsal.calls})
        raise
    write_json(report_path, report)
    print(json.dumps({key: report[key] for key in ("passed", "total_applied_updates", "accuracy", "mse",
                     "incompatible_resume_refusals", "release_qualification_granted")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
