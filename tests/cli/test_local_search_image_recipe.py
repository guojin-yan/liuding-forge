# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Licensed under the Apache License, Version 2.0.

"""Image native recipe -> one original 48-update task -> durable search evidence.

Contracts never train. Real runs four methods with one CPU ResNet18 trial:
16/8 authored images, seed 20261028, 12 epochs / 48 updates split at 19,
accuracy >= .875 and the existing train_cumulative_once BatchNorm calibration.
No baseline, export/restore or copies of model/checkpoint workspaces are added.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True

DEFINITION = {
    "schema_version": 1, "metric_id": "classification.accuracy", "version": 1,
    "unit": 5, "aggregation": 3, "direction": 3,
    "parameters": {"classes": "2", "ignore_index": "none", "zero_division": "zero",
                   "ties": "lowest_class_index", "macro_classes": "all"},
    "context_keys": ["slice", "source"],
}
CONTEXT = {"slice": "overall", "source": "all"}
PARTICIPANTS = {
    "callbacks.dispatcher", "model.main", "recipe.lr", "recipe.sgd",
    "recipe.trainer", "recipe.training", "recipe.validation", "rng.global",
    "strategy.single", "training.diagnostics",
}
EVENT_TABLES = (
    "jyppx_event_batch_items", "jyppx_metric_points", "jyppx_events",
    "jyppx_event_batches", "jyppx_event_streams", "jyppx_run_event_state",
)


def extended(path: Path) -> str:
    absolute = os.path.abspath(os.fspath(path))
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(extended(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with open(extended(path), "rb") as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    with open(extended(path), "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


@contextlib.contextmanager
def database(path: Path):
    connection = sqlite3.connect(extended(path))
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        with connection:
            yield connection
    finally:
        connection.close()


def snapshot(source: Path, destination: Path) -> None:
    with database(source) as reader, database(destination) as writer:
        reader.backup(writer)


class ImageSearchRecipe(unittest.TestCase):
    cli: Path
    source_root: Path
    device: str
    evidence_dir: Path | None = None
    temp_parent: Path | None = None

    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        cls.command_count += 1
        command = [str(cls.cli), *map(str, arguments), "--format", "json"]
        prefix = cls.evidence / f"command-{cls.command_count:03}"
        write_json(prefix.with_suffix(".request.json"), {"argv": command, "expected": expected})
        started = time.monotonic()
        try:
            process = subprocess.run(command, capture_output=True, timeout=420, check=False,
                                     env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                                              OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        except subprocess.TimeoutExpired as error:
            prefix.with_suffix(".stdout.txt").write_bytes(error.stdout or b"")
            prefix.with_suffix(".stderr.txt").write_bytes(error.stderr or b"")
            write_json(prefix.with_suffix(".receipt.json"), {
                "returncode": None, "timeout_seconds": 420,
                "elapsed_seconds": time.monotonic() - started,
                "workspace_retained": str(cls.base), "error": repr(error),
            })
            raise AssertionError("Native call timed out; preserve the owned workspace and do not retrain") from error
        prefix.with_suffix(".stdout.txt").write_bytes(process.stdout)
        prefix.with_suffix(".stderr.txt").write_bytes(process.stderr)
        write_json(prefix.with_suffix(".receipt.json"), {
            "returncode": process.returncode, "elapsed_seconds": time.monotonic() - started,
            "stdout_sha256": hashlib.sha256(process.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(process.stderr).hexdigest(),
        })
        if process.returncode != expected:
            raise AssertionError(f"{command!r}: exit={process.returncode}, expected={expected}\n"
                                 f"{process.stdout.decode('utf-8', 'replace')}\n"
                                 f"{process.stderr.decode('utf-8', 'replace')}")
        envelope = json.loads(process.stdout)
        return envelope.get("data", envelope)

    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path(tempfile.mkdtemp(prefix="is406-", dir=cls.temp_parent)).resolve()
        cls.evidence = cls.evidence_dir or cls.base / "evidence"
        cls.evidence.mkdir(parents=True, exist_ok=False)
        cls.command_count = 0
        cls.attempted = False
        cls.addClassCleanup(cls.inventory)
        write_json(cls.evidence / "retained-workspace.json", {
            "workspace": str(cls.base), "retained_on_success_and_failure": True,
            "native_update_budget": 48, "native_trial_budget": 1,
            "native_attempt_budget": 2, "new_baseline_updates": 0,
            "disk_budget_bytes": 805306368, "disk_reserve_bytes": 1073741824,
            "no_export_restore_or_workspace_copy": True,
            "evidence_policy": "retain original short TEMP; small receipts and SHA inventory only",
        })
        fixture_spec = importlib.util.spec_from_file_location(
            "image_search_authored_fixture", cls.source_root / "tests/cli/test_image_recipe.py")
        fixture = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixture)
        cls.png = staticmethod(fixture.png)
        cls.data = cls.base / "images"
        for split, count in (("train", 8), ("validation", 4)):
            for label, name in enumerate(("green", "red")):
                folder = cls.data / split / name
                folder.mkdir(parents=True)
                for number in range(count):
                    (folder / f"{number}.png").write_bytes(
                        fixture.png(label, number + (20 if split == "validation" else 0)))
        cls.config = {
            "schema_version": 1, "recipe": "jyppx.liuding.imagefolder.resnet18/v1",
            "train_root": str(cls.data / "train"), "validation_root": str(cls.data / "validation"),
            "data_license": "CC0-1.0; self-authored RGB fixture", "device": cls.device,
            "seed": 20261028, "epochs": 12, "batch_size": 4, "image_size": 32,
            "workers": 2, "prefetch_batches": 2, "learning_rate": .01,
            "momentum": .9, "weight_decay": .0001, "lr_gamma": 1.0,
            "lr_schedule": "warmup_linear", "warmup_updates": 5,
            "warmup_start_factor": .5, "lr_end_factor": .1,
            "checkpoint_every": 16, "minimum_accuracy": .875,
            "max_image_bytes": 1048576, "max_data_bytes": 10485760,
            "max_samples": 1024, "image_budget_bytes": 67108864,
            "batch_norm_calibration": "train_cumulative_once",
            "calibration_max_snapshot_bytes": 67108864,
        }
        cls.config_path = cls.base / "config.json"
        write_json(cls.config_path, cls.config)
        cls.preflight = cls.invoke("recipe", "check", "--file", cls.config_path)
        cls.config = cls.preflight["config"]
        write_json(cls.config_path, cls.config)
        cls.store = cls.base / "s"
        local = cls.invoke("recipe", "local-spec", "--file", cls.config_path,
                           "--root", cls.base / "unused", "--spec", cls.base / "local.json",
                           "--store", cls.store, "--executable", cls.cli,
                           "--pause-after", 19, "--timeout-ms", 360000)
        execution = copy.deepcopy(local["specification"])
        execution["arguments"] = ["recipe", "local-run", "--file", "${config}",
                                  "--root", "${recipe_directory}", "--format", "json",
                                  "--pause-after", "19"]
        execution["working_directory"] = "${trial_directory}"
        execution["training"]["configuration"] = "${trial_configuration}"
        for output, maximum in zip(execution["training"]["outputs"], (134217728, 67108864, 1048576)):
            output["max_size_bytes"] = maximum
        outputs = [{key: row[key] for key in ("reference_id", "kind", "role")}
                   for row in execution["training"]["outputs"]]
        cls.plan = {
            "format": "jyppx.liuding.local_search_spec", "schema_version": 1,
            "project_id": "liuding-forge", "experiment_id": "image-classification",
            "strategy": "grid", "seed": 8191, "max_trials": 1,
            "max_duration_ms": 600000, "max_failures": 1,
            "space": [{"name": "learning_rate", "config_pointer": "/learning_rate", "values": [.01]}],
            "configuration": cls.config, "execution": execution,
            "objective": {"definition": DEFINITION, "context": CONTEXT,
                          "step_kind": "sample", "selection": "last"},
            "inputs": [], "outputs": outputs,
            "recipe_bridge": {"format": "jyppx.liuding.local_search_recipe_bridge",
                              "schema_version": 1, "recipe": "jyppx.liuding.imagefolder.resnet18/v1",
                              "seed_policy": "configuration"},
        }
        cls.spec_path = cls.base / "search.json"
        write_json(cls.spec_path, cls.plan)
        specification = importlib.util.spec_from_file_location(
            "paired_search_independent_telemetry", cls.source_root / "tests/cli/test_run_telemetry.py")
        cls.telemetry = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.telemetry)

    @classmethod
    def bridge_name(cls, trial: str) -> str:
        return "search.recipe.bridge." + hashlib.sha256(trial.encode("utf-8")).hexdigest()

    @classmethod
    def inventory(cls) -> None:
        files = []
        try:
            for directory, _, names in os.walk(extended(cls.base)):
                for name in names:
                    path = Path(directory) / name
                    if cls.evidence == cls.base / "evidence" and str(path).startswith(extended(cls.evidence)):
                        continue
                    stat = path.stat()
                    files.append({"path": str(path), "size_bytes": stat.st_size,
                                  "sha256": sha(path)})
            write_json(cls.evidence / "workspace-inventory.json", {
                "workspace": str(cls.base), "files": files,
                "file_count": len(files), "total_bytes": sum(row["size_bytes"] for row in files),
                "native_attempted_once": cls.attempted,
            })
        except Exception as error:
            write_json(cls.evidence / "inventory-failure.json", {"error": repr(error)})
            raise

    @classmethod
    def reference(cls, run: str, name: str) -> tuple[bytes, str]:
        with database(cls.store / "runs.sqlite3") as connection:
            row = connection.execute(
                "SELECT a.digest,b.size_bytes FROM jyppx_run_artifacts a JOIN jyppx_blobs b USING(digest) "
                "WHERE a.run_id=? AND a.reference_id=?", (run, name)).fetchone()
        if row is None:
            raise AssertionError((run, name))
        identity, size = row
        with open(extended(cls.store / "artifacts/objects" / identity[:2] / identity[2:]), "rb") as stream:
            content = stream.read()
        if len(content) != size or hashlib.sha256(content).hexdigest() != identity:
            raise AssertionError("Independent CAS reader found a changed artifact")
        return content, identity

    @classmethod
    def json_reference(cls, run: str, name: str) -> dict:
        return json.loads(cls.reference(run, name)[0])

    @classmethod
    def replace_reference_json(cls, run: str, name: str, value: dict) -> None:
        """Controlled corruption of one durable boundary, never product resolution."""
        content = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        identity = hashlib.sha256(content).hexdigest()
        path = cls.store / "artifacts/objects" / identity[:2] / identity[2:]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        with database(cls.store / "runs.sqlite3") as connection:
            connection.execute("INSERT OR IGNORE INTO jyppx_blobs(digest,algorithm,size_bytes) VALUES(?,1,?)",
                               (identity, len(content)))
            connection.execute("UPDATE jyppx_run_artifacts SET digest=? WHERE run_id=? AND reference_id=?",
                               (identity, run, name))

    @classmethod
    def train(cls) -> None:
        if hasattr(cls, "completed"):
            return
        if cls.attempted:
            raise AssertionError("The only native trial failed; automatic retraining is forbidden")
        free = shutil.disk_usage(cls.base).free
        write_json(cls.evidence / "disk-admission.json", {
            "free_bytes": free, "required_growth_bytes": 805306368,
            "minimum_remaining_bytes": 1073741824, "new_native_updates": 48})
        if free < 805306368 + 1073741824:
            raise AssertionError("One preserved image trial needs 768 MiB growth plus the 1 GiB stop-line reserve")
        cls.attempted = True
        cls.completed = cls.invoke("search", "execute", "--root", cls.store,
                                   "--id", "image-search", "--spec", cls.spec_path)
        cls.row = cls.completed["trials"][0]
        cls.trial = cls.row["trial_id"]
        cls.training = cls.row["training_run_id"]
        cls.local = cls.invoke("local", "show", "--root", cls.store, "--id", cls.trial)
        cls.first, cls.last = cls.local["attempts"]
        cls.admission = cls.json_reference("image-search", "search.admission.1")
        cls.recipe = Path(cls.admission["execution"]["arguments"][5])
        cls.result = read_json(cls.recipe / "result.json")
        cls.bridge_reference = cls.bridge_name(cls.trial)
        cls.bridge = cls.json_reference("image-search", cls.bridge_reference)
        cls.frozen_database = cls.base / "complete.sqlite3"
        snapshot(cls.store / "runs.sqlite3", cls.frozen_database)
        cls.native_database = cls.recipe / "run-store/runs.sqlite3"
        cls.frozen_native = cls.base / "native.sqlite3"
        snapshot(cls.native_database, cls.frozen_native)
        cls.immutable_files = {
            name: (sha(cls.recipe / name), (cls.recipe / name).stat().st_mtime_ns)
            for name in ("result.json", "model.safetensors", "final.jyck")
        }
        for name, value in (("search", cls.completed), ("local", cls.local),
                            ("native-result", cls.result), ("bridge", cls.bridge)):
            write_json(cls.evidence / f"{name}.json", value)

    @classmethod
    def reset_databases(cls) -> None:
        snapshot(cls.frozen_database, cls.store / "runs.sqlite3")
        snapshot(cls.frozen_native, cls.native_database)

    @classmethod
    def pending(cls, phase: str, *, keep_bridge: bool = True) -> None:
        """Recreate an exact publication interruption from a native-complete snapshot."""
        with database(cls.store / "runs.sqlite3") as connection:
            connection.execute("DELETE FROM jyppx_run_artifacts WHERE run_id='image-search' AND "
                               "reference_id IN ('search.result','search.best','search.result.1')")
            connection.execute("UPDATE jyppx_runs SET status=2 WHERE run_id='image-search'")
            if not keep_bridge:
                connection.execute("DELETE FROM jyppx_run_artifacts WHERE run_id='image-search' AND reference_id=?",
                                   (cls.bridge_reference,))
            if phase != "after_metric":
                for table in EVENT_TABLES:
                    connection.execute(f"DELETE FROM {table} WHERE run_id=?", (cls.training,))
            if phase in ("before_start", "after_start"):
                connection.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=?", (cls.training,))
            if phase == "before_start":
                connection.execute("DELETE FROM jyppx_runs WHERE run_id=?", (cls.training,))
            else:
                connection.execute("UPDATE jyppx_runs SET status=2 WHERE run_id=?", (cls.training,))

    def assert_native_unchanged(self) -> None:
        self.assertEqual(self.invoke("local", "show", "--root", self.store,
                                     "--id", self.trial)["attempts"], self.local["attempts"])
        self.assertEqual({name: (sha(self.recipe / name), (self.recipe / name).stat().st_mtime_ns)
                          for name in self.immutable_files}, self.immutable_files)

    def test_01_closed_bridge_template_and_input_identity(self) -> None:
        mutations = []
        for key, value in (("schema_version", 2), ("recipe", "jyppx.liuding.paired-affine/v1"),
                           ("seed_policy", "implicit"), ("surprise", True)):
            plan = copy.deepcopy(self.plan)
            plan["recipe_bridge"][key] = value
            mutations.append(plan)
        for key, value in (("max_attempts", 3), ("retry_exit_codes", [23])):
            plan = copy.deepcopy(self.plan)
            plan["execution"][key] = value
            mutations.append(plan)
        for position, value in ((3, "prefix/${config}"), (5, "${trial_directory}"),
                                (5, "${recipe_directory}/child")):
            plan = copy.deepcopy(self.plan)
            plan["execution"]["arguments"][position] = value
            mutations.append(plan)
        for key, value in (("model_id", "fixture.model"), ("outputs", []),
                           ("retry_resume", {"checkpoint_reference_id": "checkpoint.last", "restore_policy": "compatible"})):
            plan = copy.deepcopy(self.plan)
            plan["execution"]["training"][key] = value
            mutations.append(plan)
        plan = copy.deepcopy(self.plan)
        del plan["recipe_bridge"]
        mutations.append(plan)
        plan = copy.deepcopy(self.plan)
        plan["objective"]["definition"]["parameters"]["ties"] = "different"
        mutations.append(plan)
        for count in ("02", "0", "1025"):
            plan = copy.deepcopy(self.plan)
            plan["objective"]["definition"]["parameters"]["classes"] = count
            mutations.append(plan)
        for index, plan in enumerate(mutations):
            with self.subTest(case=index):
                path = self.base / "bad.json"
                write_json(path, plan)
                self.invoke("search", "execute", "--root", self.store,
                            "--id", f"invalid-{index}", "--spec", path, expected=2)
                with database(self.store / "runs.sqlite3") as connection:
                    self.assertEqual(connection.execute("SELECT COUNT(*) FROM jyppx_runs WHERE run_id=?",
                                                        (f"invalid-{index}",)).fetchone()[0], 0)
        # Valid paths can still have a different unresolved identity. The real
        # resolver must reject that mismatch before admitting any native job.
        plan = copy.deepcopy(self.plan)
        plan["configuration"]["train_root"] = str(self.data / "train" / ".." / "train")
        path = self.base / "unresolved.json"
        write_json(path, plan)
        self.invoke("search", "execute", "--root", self.store,
                    "--id", "unresolved", "--spec", path, expected=2)
        with database(self.store / "runs.sqlite3") as connection:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM jyppx_runs WHERE run_id='unresolved'").fetchone()[0], 1)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM jyppx_runs WHERE parent_run_id='unresolved'").fetchone()[0], 0)

        # Valid-but-incompatible class, model/data and BN declarations are
        # rejected before any child can consume a training update.
        image_cases = []
        plan = copy.deepcopy(self.plan)
        plan["objective"]["definition"]["parameters"]["classes"] = "3"
        image_cases.append(("wrong-classes", plan))
        for name, changes in (("cuda-calibration", {"device": "cuda"}),
                              ("off-with-budget", {"batch_norm_calibration": "off"}),
                              ("non-resnet", {"model": "resnet50"})):
            plan = copy.deepcopy(self.plan)
            plan["configuration"].update(changes)
            image_cases.append((name, plan))
        plan = copy.deepcopy(self.plan)
        del plan["configuration"]["calibration_max_snapshot_bytes"]
        image_cases.append(("missing-calibration-budget", plan))
        empty = self.base / "invalid-initial.safetensors"
        empty.write_bytes(b"")
        plan = copy.deepcopy(self.plan)
        plan["configuration"].update(initial_state_file=str(empty),
                                      initial_state_sha256=hashlib.sha256(b"").hexdigest())
        image_cases.append(("invalid-complete-initial-state", plan))
        for name, plan in image_cases:
            with self.subTest(image_preflight=name):
                path = self.base / f"{name}.json"
                write_json(path, plan)
                self.invoke("search", "execute", "--root", self.store,
                            "--id", name, "--spec", path, expected=2)
                with database(self.store / "runs.sqlite3") as connection:
                    self.assertEqual(connection.execute(
                        "SELECT COUNT(*) FROM jyppx_runs WHERE parent_run_id=?", (name,)).fetchone()[0], 0)
        original = self.data / "train/green/0.png"
        original_bytes = original.read_bytes()
        try:
            original.write_bytes(self.png(0, 77))
            self.invoke("search", "execute", "--root", self.store,
                        "--id", "changed-image-bytes", "--spec", self.spec_path, expected=2)
            with database(self.store / "runs.sqlite3") as connection:
                self.assertEqual(connection.execute(
                    "SELECT COUNT(*) FROM jyppx_runs WHERE parent_run_id='changed-image-bytes'").fetchone()[0], 0)
        finally:
            original.write_bytes(original_bytes)
        # A real three-class bounded manifest reaches the executable preflight,
        # so class count is not accidentally fixed to the two-class fixture.
        third = []
        try:
            for split, count in (("train", 4), ("validation", 2)):
                folder = self.data / split / "violet"
                folder.mkdir()
                third.append(folder)
                for number in range(count):
                    (folder / f"{number}.png").write_bytes(self.png(2, number))
            checked = self.invoke("recipe", "check", "--file", self.config_path)
            self.assertEqual(len(checked["dataset"]["classes"]), 3)
            prepared = self.invoke("recipe", "local-spec", "--file", self.config_path,
                                  "--root", self.base / "unused-three", "--spec", self.base / "three-local.json",
                                  "--store", self.store, "--executable", self.cli, "--pause-after", 19)
            plan = copy.deepcopy(self.plan)
            plan["execution"]["training"]["data_manifest"] = prepared["specification"]["training"]["data_manifest"]
            plan["objective"]["definition"]["parameters"]["classes"] = "3"
            plan["execution"]["executable_sha256"] = "0" * 64
            path = self.base / "three-class.json"
            write_json(path, plan)
            error = self.invoke("search", "execute", "--root", self.store,
                                "--id", "three-class-preflight", "--spec", path, expected=2)
            self.assertIn("Local executable SHA-256 differs", json.dumps(error))
            with database(self.store / "runs.sqlite3") as connection:
                self.assertEqual(connection.execute(
                    "SELECT COUNT(*) FROM jyppx_runs WHERE parent_run_id='three-class-preflight'").fetchone()[0], 0)
        finally:
            for folder in third:
                self.assertTrue(folder.resolve().is_relative_to(self.data.resolve()))
                for file in folder.iterdir():
                    self.assertTrue(file.is_file())
                    file.unlink()
                folder.rmdir()

    def test_02_full_image_exact_retry_and_committed_objective(self) -> None:
        self.train()
        self.assertEqual(self.preflight["total_steps"], 48)
        self.assertEqual((self.preflight["dataset"]["train"]["samples"],
                          self.preflight["dataset"]["validation"]["samples"]), (16, 8))
        self.assertEqual(self.completed["status"], "completed")
        self.assertEqual(self.completed["confirmed_trials"], 1)
        self.assertEqual(self.completed["failure_count"], 0)
        self.assertEqual(self.completed["best"], self.row)
        self.assertEqual(self.completed["parallel_trials"], 1)
        self.assertEqual([part["result"]["exit_code"] for part in self.local["attempts"]], [75, 0])
        self.assertTrue(all(part["result"]["process_tree_cleanup_confirmed"] for part in self.local["attempts"]))
        paused = self.json_reference(self.first["attempt_id"], "local.stdout")["data"]
        self.assertEqual((paused["status"], paused["global_step"]), ("paused", 19))
        self.assertEqual((self.result["global_step"], self.result["optimizer_steps"]), (48, 48))
        self.assertEqual((self.result["completed_epochs"], self.result["scheduler_steps"]), (12, 48))
        self.assertEqual(self.result["validation_samples"], 8)
        self.assertGreaterEqual(self.result["accuracy"], .875)
        self.assertTrue(self.result["quality_passed"])
        self.assertEqual(self.config["minimum_accuracy"], .875)
        calibration = self.result["calibration"]
        self.assertEqual((calibration["phase"], calibration["optimizer_updates"],
                          calibration["samples"], calibration["batches"]), ("completed", 48, 16, 4))
        self.assertEqual(calibration["calibrated_model_sha256"], self.result["model_sha256"])
        self.assertNotEqual(calibration["source_model_sha256"], calibration["calibrated_model_sha256"])
        self.assertEqual(self.result["optimization"]["scheduler"]["kind"], "warmup_linear")
        self.assertEqual(self.result["optimization"]["scheduler"]["completed_steps"], 48)
        self.assertEqual(read_json(self.recipe / "model-definition.json"), {
            "format": "jyppx.liuding.onnx_model_definition", "schema_version": 1,
            "participant_id": "model.main", "factory": {"id": "jyppx.liuding.models",
            "name": "resnet18", "num_classes": 2, "input_channels": 3}})
        self.assertTrue(self.row["eligible"])
        self.assertEqual(self.row["metric"]["value"], self.result["accuracy"])
        self.assertEqual(self.row["metric"]["definition"], DEFINITION)
        self.assertEqual((self.row["metric"]["step"], self.row["metric"]["sample_count"]), (8, 8))
        self.assertEqual(self.row["recipe_bridge_sha256"], self.reference("image-search", self.bridge_reference)[1])
        self.assertEqual(self.bridge["format"], "jyppx.liuding.local_search_image_bridge")
        with database(self.store / "runs.sqlite3") as connection:
            self.assertEqual(connection.execute(
                "SELECT role FROM jyppx_run_artifacts WHERE run_id='image-search' AND reference_id=?",
                (self.bridge_reference,)).fetchone(), (self.bridge_reference,))
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM jyppx_run_artifacts WHERE run_id=? AND reference_id='search.recipe.bridge'",
                (self.trial,)).fetchone()[0], 0)
        self.assertEqual(self.bridge["seed_policy"], "configuration")
        self.assertEqual(self.bridge["training_seed"], self.config["seed"])
        self.assertEqual(self.bridge["training_seed"], 20261028)
        self.assertEqual(self.bridge["trial_seed"], self.admission["trial_seed"])
        self.assertNotEqual(self.bridge["trial_seed"], self.bridge["training_seed"])
        effective = self.json_reference(self.last["attempt_id"], "local.training.inputs")
        self.assertEqual(self.row["effective_training_inputs"], effective["training"])
        self.assertIsNone(self.row["training_inputs"]["resume"])
        self.assertEqual(effective["training"]["resume"]["restore_policy"], "exact_only")
        self.assertEqual(effective["training"]["resume"]["checkpoint"]["run_id"], self.first["attempt_id"])
        self.assertEqual(self.reference(self.training, "search.training.resume")[1],
                         self.reference(self.first["attempt_id"], "checkpoint.last")[1])
        receipt = self.json_reference(self.last["attempt_id"], "local.training.outputs")
        self.assertTrue(receipt["confirmed"])
        self.assertEqual(len(receipt["outputs"]), 3)
        for row in receipt["outputs"]:
            self.assertTrue(row["confirmed"])
            self.assertEqual(self.reference(self.training, row["reference_id"])[1], row["sha256"])
        with database(self.native_database) as connection:
            canonical = connection.execute("SELECT canonical FROM jyppx_events WHERE run_id='run.image-classification'").fetchall()
            self.assertEqual(connection.execute(
                "SELECT metric_id FROM jyppx_metric_points WHERE run_id='run.image-classification'").fetchall(),
                [("classification.accuracy",)])
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0][0].hex(), self.bridge["native_metric_canonical_hex"])
        with database(self.store / "runs.sqlite3") as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM jyppx_metric_points WHERE run_id=?",
                                                (self.training,)).fetchone()[0], 1)
        manifest = self.invoke("checkpoint", "inspect", "--path", self.recipe / "final.jyck")["manifest"]
        self.assertEqual({row["state"]["participant_id"] for row in manifest["participants"]}, PARTICIPANTS | {"recipe.batch-norm-calibration"})
        self.invoke("run", "verify", "--root", self.store)

    def test_03_partial_publication_and_uncertain_evidence(self) -> None:
        self.train()
        try:
            for phase in ("before_start", "after_start", "after_refs", "after_metric"):
                with self.subTest(replay=phase):
                    self.reset_databases()
                    self.pending(phase)
                    recovered = self.invoke("search", "recover", "--root", self.store, "--id", "image-search")
                    self.assertEqual(recovered["best"], self.row)
                    self.assert_native_unchanged()
            for conflict in ("extra_objective", "extra_reference", "partial_references",
                             "changed_seed", "missing_native_metric", "native_not_completed", "missing_owner",
                             "cleanup_unknown"):
                with self.subTest(reject=conflict):
                    self.reset_databases()
                    self.pending("after_metric" if conflict in ("extra_objective", "extra_reference", "partial_references")
                                 else "before_start",
                                 keep_bridge=conflict not in ("missing_native_metric", "native_not_completed",
                                                             "missing_owner", "cleanup_unknown"))
                    expected = 2
                    if conflict == "extra_objective":
                        fixture = self.telemetry
                        fixture.RUN_ID = self.training
                        fixture.STREAM_ID = "search.recipe.image-classification." + hashlib.sha256(self.training.encode()).hexdigest()
                        point = {"definition": DEFINITION, "step": {"kind": 4, "value": 8},
                                 "status": 1, "value": 1.0, "reason": "", "context": CONTEXT, "sample_count": 8}
                        event = fixture.envelope(2, {"kind": 2, "points": [point]})
                        event["correlation"].update(project_id="liuding-forge", run_id=self.training)
                        fixture.seed_database(self.store, [event])
                    elif conflict in ("extra_reference", "partial_references"):
                        with database(self.store / "runs.sqlite3") as connection:
                            if conflict == "partial_references":
                                connection.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id='model.weights'",
                                                   (self.training,))
                            else:
                                connection.execute("INSERT INTO jyppx_run_artifacts SELECT run_id,'unexpected',kind,'unexpected',digest,"
                                                   "logical_name,media_type,producer,attached_at_ms FROM jyppx_run_artifacts "
                                                   "WHERE run_id=? AND reference_id='model.weights'", (self.training,))
                    elif conflict == "changed_seed":
                        manifest = copy.deepcopy(self.bridge)
                        manifest["training_seed"] += 1
                        self.replace_reference_json("image-search", self.bridge_reference, manifest)
                    elif conflict == "missing_native_metric":
                        with database(self.native_database) as connection:
                            for table in EVENT_TABLES:
                                connection.execute(f"DELETE FROM {table} WHERE run_id='run.image-classification'")
                    elif conflict == "native_not_completed":
                        with database(self.native_database) as connection:
                            connection.execute("UPDATE jyppx_runs SET status=2 WHERE run_id='run.image-classification'")
                    elif conflict == "missing_owner":
                        with database(self.store / "runs.sqlite3") as connection:
                            connection.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id='local.owner'",
                                               (self.last["attempt_id"],))
                    else:
                        outcome = self.json_reference(self.last["attempt_id"], "local.result")
                        outcome["process_tree_cleanup_confirmed"] = False
                        self.replace_reference_json(self.last["attempt_id"], "local.result", outcome)
                        expected = 7
                    self.invoke("search", "recover", "--root", self.store, "--id", "image-search", expected=expected)
                    with database(self.store / "runs.sqlite3") as connection:
                        self.assertEqual(connection.execute("SELECT COUNT(*) FROM jyppx_run_artifacts WHERE run_id='image-search' "
                                                            "AND reference_id='search.result.1'").fetchone()[0], 0)
                    self.assertEqual({name: (sha(self.recipe / name), (self.recipe / name).stat().st_mtime_ns)
                                      for name in self.immutable_files}, self.immutable_files)
        finally:
            self.reset_databases()
        self.assert_native_unchanged()

    def test_04_terminal_without_native_workspace_or_data(self) -> None:
        self.train()
        self.reset_databases()
        self.assert_native_unchanged()
        retained = self.recipe.with_name("retained-native-recipe")
        retained_data = self.data.with_name("retained-images")
        for source, destination in ((self.recipe, retained), (self.data, retained_data)):
            self.assertTrue(source.resolve().is_relative_to(self.base.resolve()))
            self.assertTrue(destination.resolve().is_relative_to(self.base.resolve()))
            self.assertFalse(destination.exists())
        os.rename(extended(self.recipe), extended(retained))
        try:
            os.rename(extended(self.data), extended(retained_data))
            try:
                for action in ("show", "recover", "recover"):
                    self.assertEqual(self.invoke("search", action, "--root", self.store,
                                                 "--id", "image-search")["best"], self.row)
            finally:
                self.assertFalse(self.data.exists())
                os.rename(extended(retained_data), extended(self.data))
        finally:
            self.assertFalse(self.recipe.exists())
            os.rename(extended(retained), extended(self.recipe))
        self.assert_native_unchanged()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--task", choices=("contracts", "real"), default="contracts")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--temp-parent", type=Path)
    options, remaining = parser.parse_known_args()
    ImageSearchRecipe.cli = options.cli.resolve()
    ImageSearchRecipe.source_root = options.source_root.resolve()
    ImageSearchRecipe.device = options.device
    ImageSearchRecipe.evidence_dir = options.evidence_dir.resolve() if options.evidence_dir else None
    ImageSearchRecipe.temp_parent = options.temp_parent.resolve() if options.temp_parent else None
    names = remaining or (["ImageSearchRecipe.test_01_closed_bridge_template_and_input_identity"]
                          if options.task == "contracts" else ["ImageSearchRecipe"])
    unittest.main(argv=[sys.argv[0], *names], verbosity=2)
