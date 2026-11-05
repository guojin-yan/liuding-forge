# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

"""Native full-task recovery after losing the recipe directory.

Python prepares fixtures, injects failures and compares independent bytes;
all backup, restore, training and checkpoint operations use the C++ CLI.
Each invocation performs one baseline and one split task, without retraining.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time
import unittest

from test_image_recipe import png
from test_local_search import seed_reference


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class RecipeWorkspaceRecovery(unittest.TestCase):
    cli: Path
    source: Path
    root: Path
    evidence: Path
    recipe: str
    saved_prefix: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        if cls.saved_prefix is None:
            cls.root.mkdir(parents=True, exist_ok=False)
        elif not cls.root.is_dir():
            raise AssertionError("Saved-prefix validation needs the retained task directory")
        cls.evidence.mkdir(parents=True, exist_ok=False)
        cls.commands = []
        cls.store = cls.root / "store"
        cls.work = cls.root / "resumed"
        cls.config_path = cls.root / "config.json"
        if cls.saved_prefix is not None:
            cls.steps, cls.pause = (1280, 259) if cls.recipe == "paired" else (48, 19)
            return
        if cls.recipe == "paired":
            prepared = cls.invoke("data", "prepare", "--name", "paired-affine",
                                  "--data-root", cls.root / "datasets")
            data = Path(prepared["destination"])
            config = json.loads((cls.source / "recipes/multi-io/paired-affine.json").read_bytes())
            config.update(train_file=str(data / "train.jsonl"),
                          validation_file=str(data / "validation.jsonl"), device="cpu")
            cls.steps, cls.pause = 1280, 259
        else:
            for split, count in (("train", 8), ("validation", 4)):
                for label, name in enumerate(("green", "red")):
                    folder = cls.root / "images" / split / name
                    folder.mkdir(parents=True)
                    for number in range(count):
                        (folder / f"{number}.png").write_bytes(
                            png(label, number + (20 if split == "validation" else 0)))
            config = {"schema_version": 1, "recipe": "jyppx.liuding.imagefolder.resnet18/v1",
                "train_root": "images/train", "validation_root": "images/validation",
                "data_license": "CC0-1.0; self-authored RGB fixture", "device": "cpu",
                "seed": 20261028, "epochs": 12, "batch_size": 4, "image_size": 32,
                "workers": 2, "prefetch_batches": 2, "learning_rate": 0.01,
                "momentum": 0.9, "weight_decay": 0.0001, "lr_gamma": 0.9,
                "checkpoint_every": 16, "minimum_accuracy": 0.875,
                "max_image_bytes": 1048576, "max_data_bytes": 10485760,
                "max_samples": 1024, "image_budget_bytes": 67108864}
            cls.steps, cls.pause = 48, 19
        cls.config_path.write_text(json.dumps(config), encoding="utf-8")

    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        argv = [str(cls.cli), *map(str, arguments), "--format", "json"]
        process = subprocess.run(argv, capture_output=True, timeout=420,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                     CUBLAS_WORKSPACE_CONFIG=":4096:8"))
        number = len(cls.commands)
        for kind, raw in (("stdout", process.stdout), ("stderr", process.stderr)):
            (cls.evidence / f"command-{number:03d}.{kind}.log").write_bytes(raw)
        cls.commands.append({"argv": argv, "exit_code": process.returncode,
                             "expected_exit": expected})
        (cls.evidence / "commands.json").write_text(json.dumps(cls.commands, indent=2), encoding="utf-8")
        if process.returncode != expected:
            raise AssertionError(f"{arguments!r}\nexit={process.returncode}, expected={expected}"
                                 f"\n{process.stdout.decode(errors='replace')}"
                                 f"\n{process.stderr.decode(errors='replace')}")
        envelope = json.loads(process.stdout)
        return envelope.get("data", envelope)

    @classmethod
    def references(cls, run: str, store: Path | None = None) -> dict:
        return {row["reference_id"]: row for row in cls.invoke(
            "run", "show", "--root", store or cls.store, "--id", run)["references"]}

    @classmethod
    def blob(cls, reference: dict, store: Path | None = None) -> bytes:
        item = reference["artifact"]
        path = (store or cls.store) / "artifacts/objects" / item["sha256"][:2] / item["sha256"][2:]
        raw = path.read_bytes()
        if len(raw) != item["size_bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise AssertionError("Independent CAS check failed")
        return raw

    def prefix(self, spec_path: Path) -> dict:
        process = subprocess.Popen([str(self.cli), "local", "execute", "--root", str(self.store),
            "--id", "workspace-loss", "--spec", str(spec_path), "--format", "json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        try:
            until = time.monotonic() + 180
            while time.monotonic() < until:
                if process.poll() is not None:
                    raise AssertionError("Dispatcher finished before the confirmed retry boundary")
                if (self.store / "runs.sqlite3").is_file():
                    with closing(sqlite3.connect(self.store / "runs.sqlite3")) as db:
                        row = db.execute("SELECT r.run_id FROM jyppx_runs r JOIN jyppx_run_artifacts a "
                            "ON a.run_id=r.run_id WHERE r.parent_run_id='workspace-loss' "
                            "AND r.status=4 AND a.reference_id='local.result'").fetchone()
                        if row:
                            break
                time.sleep(.02)
            else:
                raise AssertionError("The first native recipe result was not committed")
        finally:
            if process.poll() is None:
                process.kill()  # Dispatcher only; confirmed worker/native already finished.
            stdout, stderr = process.communicate(timeout=15)
            (self.evidence / "dispatcher.stdout.log").write_bytes(stdout)
            (self.evidence / "dispatcher.stderr.log").write_bytes(stderr)
        until = time.monotonic() + 15
        while time.monotonic() < until:
            state = self.invoke("local", "show", "--root", self.store, "--id", "workspace-loss")
            if not state["owner_active"]:
                break
            time.sleep(.02)
        self.assertEqual(len(state["attempts"]), 1)
        result = state["attempts"][0]["result"]
        self.assertEqual((result["status"], result["exit_code"]), ("exited", 75))
        self.assertTrue(result["process_tree_cleanup_confirmed"])
        self.assertFalse(state["owner_active"])
        return state

    def test_full_task_recovers_lost_workspace_without_losing_history(self) -> None:
        baseline_root = self.root / "continuous"
        retained = self.root / "retained-original"
        job_id = "workspace-loss"
        if self.saved_prefix is None:
            baseline = self.invoke("recipe", "run", "--file", self.config_path, "--root", baseline_root)
            spec_path = self.root / "local.json"
            plan = self.invoke("recipe", "local-spec", "--file", self.config_path, "--root", self.work,
                "--store", self.store, "--spec", spec_path, "--executable", self.cli,
                "--pause-after", self.pause, "--timeout-ms", 360000)
            spec = plan["specification"]
            spec["retry_delay_ms"] = 4000
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            prefix = self.prefix(spec_path)
            (self.evidence / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
            (self.evidence / "prefix.json").write_text(json.dumps(prefix, indent=2) + "\n", encoding="utf-8")
            self.assertEqual(self.work.parent.resolve(), self.root.resolve())
            self.work.rename(retained)
        else:
            # An explicit new plan may consume the confirmed original prefix;
            # the failed job, partial reconstruction and baseline are retained.
            baseline = json.loads((self.saved_prefix / "baseline.json").read_bytes())
            prefix = json.loads((self.saved_prefix / "prefix.json").read_bytes())
            failed = self.invoke("local", "show", "--root", self.store, "--id", job_id)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(len(failed["attempts"]), 2)
            self.assertEqual(failed["attempts"][0], prefix["attempts"][0])
            self.assertTrue(failed["attempts"][1]["result"]["process_tree_cleanup_confirmed"])
            self.assertFalse((self.work / "latest.jyck").exists())
            self.invoke("run", "backup", "--root", self.store,
                        "--destination", self.root / "retained-failed-outer-backup")
            self.assertEqual(self.work.parent.resolve(), self.root.resolve())
            self.work.rename(self.root / "retained-failed-reconstruction")
            spec = json.loads((self.root / "local.json").read_bytes())
            spec["executable_sha256"] = sha(self.cli)
            spec["max_attempts"], spec["retry_exit_codes"], spec["retry_delay_ms"] = 1, [], 0
            del spec["training"]["retry_resume"]
            source = prefix["attempts"][0]["attempt_id"]
            checkpoint = self.references(source)["checkpoint.last"]["artifact"]
            spec["training"]["resume"] = {"checkpoint": {"run_id": source, "reference_id": "checkpoint.last",
                "sha256": checkpoint["sha256"], "size_bytes": checkpoint["size_bytes"]}, "restore_policy": "exact_only"}
            pause_index = spec["arguments"].index("--pause-after")
            del spec["arguments"][pause_index:pause_index + 2]
            spec_path = self.root / "reconstruction.json"
            with spec_path.open("x", encoding="utf-8") as stream:
                json.dump(spec, stream)
            job_id = "workspace-reconstruction"
        source = prefix["attempts"][0]["attempt_id"]
        refs = self.references(source)
        checkpoint_before = sha(retained / "latest.jyck")
        self.assertFalse(self.work.exists())
        action = "recover" if self.saved_prefix is None else "execute"
        extra = () if action == "recover" else ("--spec", spec_path)
        recovered = self.invoke("local", action, "--root", self.store, "--id", job_id, *extra)
        (self.evidence / "recovered.json").write_text(json.dumps(recovered, indent=2) + "\n", encoding="utf-8")
        self.assertEqual(recovered["status"], "completed")
        self.assertEqual(len(recovered["attempts"]), 2 if self.saved_prefix is None else 1)
        if self.saved_prefix is None:
            self.assertEqual(recovered["attempts"][0], prefix["attempts"][0])
        self.assertTrue(recovered["attempts"][-1]["result"]["training_outputs_confirmed"])
        result = json.loads((self.work / "result.json").read_bytes())
        fields = ("global_step", "optimizer_steps", "completed_epochs", "scheduler_steps", "model_sha256",
                  "training_metrics", "validation_metrics", "accuracy", "confusion",
                  "next_rng_sha256", "next_cpu_rng_sha256", "numerics")
        if self.recipe == "paired":
            fields += ("mse",)
        for field in fields:
            self.assertEqual(baseline[field], result[field], field)
        self.assertEqual(result["global_step"], self.steps)
        self.assertTrue(result["quality_passed"])
        self.assertEqual(sha(self.work / "final.jyck"), sha(baseline_root / "final.jyck"))
        self.assertEqual(sha(self.work / "model.safetensors"), sha(baseline_root / "model.safetensors"))
        self.assertEqual(sha(retained / "latest.jyck"), checkpoint_before)
        manifest = json.loads(self.blob(refs["recipe.workspace.snapshot"]))
        self.assertEqual(manifest["checkpoint_sha256"], checkpoint_before)
        self.assertEqual(manifest["source_attempt_id"], source)
        for item in manifest["files"]:
            raw = self.blob(refs[item["reference_id"]])
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (item["size_bytes"], item["sha256"]))
        self.assertTrue(self.invoke("run", "verify", "--root", self.work / "run-store")["valid"])
        with closing(sqlite3.connect(retained / "run-store/runs.sqlite3")) as old, \
                closing(sqlite3.connect(self.work / "run-store/runs.sqlite3")) as new:
            old_run = old.execute("SELECT run_id,project_id,experiment_id,created_at_ms FROM jyppx_runs").fetchall()
            self.assertEqual(old_run, new.execute("SELECT run_id,project_id,experiment_id,created_at_ms FROM jyppx_runs").fetchall())
            old_refs = old.execute("SELECT run_id,reference_id,kind,role,digest,attached_at_ms FROM jyppx_run_artifacts").fetchall()
            for row in old_refs:
                self.assertIn(row, new.execute("SELECT run_id,reference_id,kind,role,digest,attached_at_ms FROM jyppx_run_artifacts").fetchall())
        negative = self.rejections(source, refs, manifest, spec) if self.recipe == "paired" else []
        replay_before = {str(p.relative_to(self.work)): (sha(p), p.stat().st_mtime_ns)
                         for p in self.work.rglob("*") if p.is_file()}
        for action in ("recover", "execute"):
            args = ("--spec", spec_path) if action == "execute" else ()
            self.assertEqual(self.invoke("local", action, "--root", self.store, "--id", job_id, *args), recovered)
        self.assertEqual(replay_before, {str(p.relative_to(self.work)): (sha(p), p.stat().st_mtime_ns)
                                        for p in self.work.rglob("*") if p.is_file()})
        if self.saved_prefix is not None:
            self.assertEqual(self.invoke("local", "show", "--root", self.store, "--id", "workspace-loss"), failed)
        values = {"baseline": baseline, "prefix": prefix, "recovered": recovered, "result": result,
                  "workspace-manifest": manifest, "summary": {"passed": True, "recipe": self.recipe,
                    "native_update_budget": 2 * self.steps, "pause_resume": [self.pause, self.steps - self.pause],
                    "baseline_updates": self.steps, "split_updates": self.steps,
                    "saved_prefix_reused": self.saved_prefix is not None,
                    "native_updates_this_invocation": self.steps - self.pause if self.saved_prefix else 2 * self.steps,
                    "whole_checkpoint_exact": True, "prior_run_identity_and_references_retained": len(old_refs),
                    "negative_cases": negative, "automatic_retraining": False}}
        for name, value in values.items():
            (self.evidence / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def rejections(self, source: str, refs: dict, manifest: dict, original: dict) -> list[str]:
        backup = self.root / "outer-backup"
        self.invoke("run", "backup", "--root", self.store, "--destination", backup)
        cases = ("missing", "source", "checkpoint", "build", "traversal", "file-binding", "blob", "existing")
        for case in cases:
            store, work = self.root / ("store-" + case), self.root / ("work-" + case)
            self.invoke("run", "restore", "--backup", backup, "--root", store)
            changed = copy.deepcopy(manifest)
            if case == "missing":
                with closing(sqlite3.connect(store / "runs.sqlite3")) as db, db:
                    db.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id=?",
                               (source, "recipe.workspace.snapshot"))
            elif case == "file-binding":
                with closing(sqlite3.connect(store / "runs.sqlite3")) as db, db:
                    db.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id=?",
                               (source, changed["files"][0]["reference_id"]))
            elif case == "blob":
                identity = next(item["sha256"] for item in changed["files"]
                                if item["path"] == "backup/runs.sqlite3")
                path = store / "artifacts/objects" / identity[:2] / identity[2:]
                raw = path.read_bytes()
                path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            elif case == "existing":
                work.mkdir()
                (work / "sentinel").write_bytes(b"owned by another workspace")
            else:
                if case == "source": changed["source_attempt_id"] = "another-source"
                if case == "checkpoint": changed["checkpoint_sha256"] = "0" * 64
                if case == "build": changed["build"] = {"wrong": True}
                if case == "traversal": changed["files"][0]["path"] = "../escape"
                with closing(sqlite3.connect(store / "runs.sqlite3")) as db, db:
                    db.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id=?",
                               (source, "recipe.workspace.snapshot"))
                seed_reference(store, source, "recipe.workspace.snapshot", 1,
                               "recipe.workspace.snapshot", json.dumps(changed).encode())
            spec = copy.deepcopy(original)
            spec["max_attempts"] = 1
            spec["retry_exit_codes"] = []
            spec["retry_delay_ms"] = 0
            spec["training"].pop("retry_resume", None)
            checkpoint = refs["checkpoint.last"]["artifact"]
            spec["training"]["resume"] = {"checkpoint": {"run_id": source, "reference_id": "checkpoint.last",
                "sha256": checkpoint["sha256"], "size_bytes": checkpoint["size_bytes"]}, "restore_policy": "exact_only"}
            spec["arguments"][5] = str(work)
            spec_path = self.root / (case + ".json")
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            rejected = self.invoke("local", "execute", "--root", store, "--id", "reject-" + case,
                                   "--spec", spec_path, expected=1)
            self.assertEqual(rejected["status"], "failed")
            self.assertEqual(len(rejected["attempts"]), 1)
            self.assertIn(rejected["attempts"][0]["result"]["exit_code"],
                          (2, 3) if case == "blob" else (2,))
            self.assertTrue(rejected["attempts"][0]["result"]["process_tree_cleanup_confirmed"])
            if case == "existing":
                self.assertEqual((work / "sentinel").read_bytes(), b"owned by another workspace")
                self.assertEqual({p.name for p in work.iterdir()}, {"sentinel", ".recipe.lock"})
            else:
                self.assertFalse(work.exists())
            self.assertFalse((self.root / "escape").exists())
        return list(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--recipe", choices=("paired", "image"), required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--saved-prefix", type=Path,
                        help="Reuse retained baseline/prefix evidence and continue through an explicit new plan")
    args, rest = parser.parse_known_args()
    RecipeWorkspaceRecovery.cli = args.cli.resolve()
    RecipeWorkspaceRecovery.source = args.source_root.resolve()
    RecipeWorkspaceRecovery.recipe = args.recipe
    RecipeWorkspaceRecovery.saved_prefix = args.saved_prefix.resolve() if args.saved_prefix else None
    with tempfile.TemporaryDirectory(prefix="liuding-workspace-recovery-") as temporary:
        RecipeWorkspaceRecovery.root = (args.work_root or Path(temporary) / "work").resolve()
        RecipeWorkspaceRecovery.evidence = (args.evidence_dir or Path(temporary) / "evidence").resolve()
        unittest.main(argv=[__file__, *rest])
