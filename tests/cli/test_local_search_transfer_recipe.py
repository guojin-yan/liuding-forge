# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Licensed under the Apache License, Version 2.0.
"""Frozen ResNet50 transfer search: byte-only contracts or one four-update task.

The fixture retains the original four train/four validation images, seed,
two epochs, accuracy >= .5 and 1+3 exact continuation. No baseline is trained.
Original workspaces and small receipts remain available after either outcome.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True

from test_image_recipe import png
from test_local_search_image_recipe import (
    CONTEXT, DEFINITION, EVENT_TABLES, PARTICIPANTS,
    database, extended, read_json, sha, snapshot, write_json,
)

SEARCH = "transfer-search"
NATIVE = "run.image-classification"
GROWTH_BUDGET = 1 << 30
DISK_RESERVE = 1 << 30


class TransferSearchRecipe(unittest.TestCase):
    cli: Path
    source_root: Path
    task: str
    weights_file: Path | None = None
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
            result = subprocess.run(command, capture_output=True, timeout=420, check=False,
                env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                         OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        except subprocess.TimeoutExpired as error:
            prefix.with_suffix(".stdout.txt").write_bytes(error.stdout or b"")
            prefix.with_suffix(".stderr.txt").write_bytes(error.stderr or b"")
            write_json(prefix.with_suffix(".receipt.json"), {
                "timeout_seconds": 420, "elapsed_seconds": time.monotonic() - started,
                "workspace_retained": str(cls.base), "error": repr(error)})
            raise AssertionError("Native timeout; retain evidence and do not retrain") from error
        prefix.with_suffix(".stdout.txt").write_bytes(result.stdout)
        prefix.with_suffix(".stderr.txt").write_bytes(result.stderr)
        write_json(prefix.with_suffix(".receipt.json"), {
            "returncode": result.returncode, "elapsed_seconds": time.monotonic() - started,
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()})
        if result.returncode != expected:
            raise AssertionError(f"{command!r}: exit={result.returncode}, expected={expected}\n"
                f"{result.stdout.decode('utf-8', 'replace')}\n{result.stderr.decode('utf-8', 'replace')}")
        envelope = json.loads(result.stdout)
        return envelope.get("data", envelope)

    @classmethod
    def setUpClass(cls) -> None:
        cls.base = Path(tempfile.mkdtemp(prefix="ts409-", dir=cls.temp_parent)).resolve()
        cls.evidence = cls.evidence_dir or cls.base / "evidence"
        cls.evidence.mkdir(parents=True, exist_ok=False)
        cls.command_count, cls.attempted = 0, False
        cls.addClassCleanup(cls.inventory)
        write_json(cls.evidence / "retained-workspace.json", {
            "workspace": str(cls.base), "retained_on_success_and_failure": True,
            "native_update_budget": 4 if cls.task == "fixture" else 0,
            "native_trial_budget": 1 if cls.task == "fixture" else 0,
            "native_attempt_budget": 2 if cls.task == "fixture" else 0,
            "new_baseline_updates": 0, "disk_growth_budget_bytes": GROWTH_BUDGET,
            "disk_reserve_bytes": DISK_RESERVE,
            "large_workspace_copied": False, "full_export_restore": False,
            "weight_view": "owned hard link; original external weight is never moved or copied"})
        cls.descriptor = read_json(cls.source_root / "recipes/transfer-learning/resnet50-a1.weights.json")
        if cls.task == "contracts":
            cls.weights = cls.base / "not-model-weights.safetensors"
            cls.weights.write_bytes(b"transfer-search-byte-identity-only")
            cls.descriptor.update(sha256=sha(cls.weights), size_bytes=cls.weights.stat().st_size)
        else:
            cls.approved_weights = cls.weights_file.resolve()
            if not cls.approved_weights.is_file() or cls.approved_weights.is_relative_to(cls.base):
                raise AssertionError("An existing external approved weight file is required")
            if (cls.approved_weights.stat().st_size != cls.descriptor["size_bytes"] or
                    sha(cls.approved_weights) != cls.descriptor["sha256"]):
                raise AssertionError("Approved full pretrained weight identity differs")
            # The terminal test hides only this owned input path. A hard link
            # adds no weight copy and never renames or writes the approved file.
            cls.weights = cls.base / "input-weights.safetensors"
            try:
                os.link(extended(cls.approved_weights), extended(cls.weights))
            except OSError as error:
                raise AssertionError(
                    "The fixture requires --temp-parent on the approved weights' filesystem "
                    "for a private hard link; no weight-copy fallback is permitted") from error
            if not cls.weights.samefile(cls.approved_weights):
                raise AssertionError("The owned weight view must refer to the approved file")
            original = cls.approved_weights.stat()
            cls.original_weight_stat = (original.st_size, original.st_mtime_ns)
            write_json(cls.evidence / "weight-view.json", {
                "original": str(cls.approved_weights), "owned_path": str(cls.weights),
                "size_bytes": original.st_size, "sha256": cls.descriptor["sha256"],
                "same_file_verified": True, "new_weight_data_bytes": 0,
                "original_not_moved_or_written": True})
        cls.metadata = cls.base / "weights.json"
        write_json(cls.metadata, cls.descriptor)
        cls.data = cls.base / "images"
        for split in ("train", "validation"):
            for label, name in enumerate(("green", "red")):
                folder = cls.data / split / name
                folder.mkdir(parents=True)
                for number in range(2):
                    (folder / f"{number}.png").write_bytes(
                        png(label, number + (20 if split == "validation" else 0)))
        cls.config = {
            "schema_version": 1, "recipe": "jyppx.liuding.imagefolder.transfer/v1",
            "device": "cpu", "seed": 20261029, "batch_size": 2, "image_size": 224,
            "workers": 1, "prefetch_batches": 1, "learning_rate": .01, "momentum": .9,
            "weight_decay": .0001, "lr_gamma": .9, "checkpoint_every": 2,
            "max_image_bytes": 1048576, "max_data_bytes": 10485760,
            "max_samples": 1024, "image_budget_bytes": 134217728,
            "model": "resnet50", "weights_file": str(cls.weights),
            "weights_metadata": str(cls.metadata), "backbone_mode": "frozen", "batch_norm": "eval",
            "train_root": str(cls.data / "train"), "validation_root": str(cls.data / "validation"),
            "data_license": "CC0-1.0; self-authored RGB fixture", "epochs": 2, "minimum_accuracy": .5,
        }
        cls.config_path = cls.base / "config.json"
        write_json(cls.config_path, cls.config)
        cls.preflight = cls.invoke("recipe", "check", "--file", cls.config_path)
        if cls.preflight["total_steps"] != 4:
            raise AssertionError("The original four-update transfer task must remain intact")
        cls.config = cls.preflight["config"]
        write_json(cls.config_path, cls.config)
        cls.store = cls.base / "s"
        local = cls.invoke("recipe", "local-spec", "--file", cls.config_path,
            "--root", cls.base / "unused", "--spec", cls.base / "local.json",
            "--store", cls.store, "--executable", cls.cli, "--pause-after", 1, "--timeout-ms", 360000)
        execution = copy.deepcopy(local["specification"])
        execution["arguments"] = ["recipe", "local-run", "--file", "${config}",
            "--root", "${recipe_directory}", "--format", "json", "--pause-after", "1"]
        execution["working_directory"] = "${trial_directory}"
        execution["training"]["configuration"] = "${trial_configuration}"
        for output, maximum in zip(execution["training"]["outputs"], (134217728, 134217728, 1048576)):
            output["max_size_bytes"] = maximum
        cls.plan = {
            "format": "jyppx.liuding.local_search_spec", "schema_version": 1,
            "project_id": "liuding-forge", "experiment_id": "transfer-learning",
            "strategy": "grid", "seed": 8191, "max_trials": 1,
            "max_duration_ms": 600000, "max_failures": 1,
            "space": [{"name": "learning_rate", "config_pointer": "/learning_rate", "values": [.01]}],
            "configuration": cls.config, "execution": execution,
            "objective": {"definition": DEFINITION, "context": CONTEXT,
                          "step_kind": "sample", "selection": "last"},
            "inputs": [], "outputs": [{key: row[key] for key in ("reference_id", "kind", "role")}
                for row in execution["training"]["outputs"]],
            "recipe_bridge": {"format": "jyppx.liuding.local_search_recipe_bridge",
                "schema_version": 1, "recipe": "jyppx.liuding.imagefolder.transfer/v1",
                "seed_policy": "configuration", "transfer": cls.preflight["transfer"]},
        }
        cls.spec_path = cls.base / "search.json"
        write_json(cls.spec_path, cls.plan)

    @classmethod
    def inventory(cls) -> None:
        rows = []
        for directory, _, names in os.walk(extended(cls.base)):
            for name in names:
                path = Path(directory) / name
                if cls.evidence == cls.base / "evidence" and str(path).startswith(extended(cls.evidence)):
                    continue
                rows.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha(path)})
        write_json(cls.evidence / "workspace-inventory.json", {
            "workspace": str(cls.base), "files": rows, "file_count": len(rows),
            "total_bytes": sum(row["size_bytes"] for row in rows), "native_attempted_once": cls.attempted})

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
            raise AssertionError("Independent CAS reader found changed content")
        return content, identity

    @classmethod
    def json_reference(cls, run: str, name: str) -> dict:
        return json.loads(cls.reference(run, name)[0])

    @classmethod
    def reset_databases(cls) -> None:
        snapshot(cls.frozen_database, cls.store / "runs.sqlite3")
        snapshot(cls.frozen_native, cls.native_database)

    @classmethod
    def pending(cls, phase: str, *, keep_bridge: bool = True) -> None:
        with database(cls.store / "runs.sqlite3") as connection:
            connection.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND "
                "reference_id IN ('search.result','search.best','search.result.1')", (SEARCH,))
            connection.execute("UPDATE jyppx_runs SET status=2 WHERE run_id=?", (SEARCH,))
            if not keep_bridge:
                connection.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id=?",
                                   (SEARCH, cls.bridge_reference))
            if phase != "after_metric":
                for table in EVENT_TABLES:
                    connection.execute(f"DELETE FROM {table} WHERE run_id=?", (cls.training,))
            if phase in ("before_start", "after_start"):
                connection.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=?", (cls.training,))
            if phase == "before_start":
                connection.execute("DELETE FROM jyppx_runs WHERE run_id=?", (cls.training,))
            else:
                connection.execute("UPDATE jyppx_runs SET status=2 WHERE run_id=?", (cls.training,))

    @classmethod
    def train(cls) -> None:
        if cls.task != "fixture":
            raise AssertionError("The byte-only contracts cannot train")
        if hasattr(cls, "completed"):
            return
        if cls.attempted:
            raise AssertionError("The one native trial failed; automatic retraining is forbidden")
        free = shutil.disk_usage(cls.base).free
        write_json(cls.evidence / "disk-admission.json", {
            "free_bytes": free, "growth_budget_bytes": GROWTH_BUDGET,
            "minimum_remaining_bytes": DISK_RESERVE, "new_native_updates": 4})
        if free < GROWTH_BUDGET + DISK_RESERVE:
            raise AssertionError("Preserve a 1 GiB new-growth budget plus the 1 GiB stop reserve")
        cls.attempted = True
        cls.completed = cls.invoke("search", "execute", "--root", cls.store,
                                  "--id", SEARCH, "--spec", cls.spec_path)
        cls.row = cls.completed["trials"][0]
        cls.trial, cls.training = cls.row["trial_id"], cls.row["training_run_id"]
        cls.local = cls.invoke("local", "show", "--root", cls.store, "--id", cls.trial)
        cls.first, cls.last = cls.local["attempts"]
        cls.admission = cls.json_reference(SEARCH, "search.admission.1")
        cls.recipe = Path(cls.admission["execution"]["arguments"][5])
        cls.result = read_json(cls.recipe / "result.json")
        cls.bridge_reference = "search.recipe.bridge." + hashlib.sha256(cls.trial.encode()).hexdigest()
        cls.bridge = cls.json_reference(SEARCH, cls.bridge_reference)
        cls.native_database = cls.recipe / "run-store/runs.sqlite3"
        cls.frozen_database, cls.frozen_native = cls.base / "complete.sqlite3", cls.base / "native.sqlite3"
        snapshot(cls.store / "runs.sqlite3", cls.frozen_database)
        snapshot(cls.native_database, cls.frozen_native)
        cls.immutable_files = {name: (sha(cls.recipe / name), (cls.recipe / name).stat().st_mtime_ns)
            for name in ("result.json", "model.safetensors", "final.jyck")}
        for name, value in (("search", cls.completed), ("local", cls.local),
                            ("native-result", cls.result), ("bridge", cls.bridge)):
            write_json(cls.evidence / f"{name}.json", value)

    def assert_native_unchanged(self) -> None:
        self.assertEqual(self.invoke("local", "show", "--root", self.store,
                                     "--id", self.trial)["attempts"], self.local["attempts"])
        self.assertEqual({name: (sha(self.recipe / name), (self.recipe / name).stat().st_mtime_ns)
            for name in self.immutable_files}, self.immutable_files)

    def test_01_closed_bridge_and_frozen_descriptor(self) -> None:
        mutations = []
        for key, value in (("schema_version", 2), ("recipe", "jyppx.liuding.paired-affine/v1"),
                           ("seed_policy", "implicit"), ("surprise", True)):
            plan = copy.deepcopy(self.plan)
            plan["recipe_bridge"][key] = value
            mutations.append(plan)
        plan = copy.deepcopy(self.plan)
        del plan["recipe_bridge"]["transfer"]
        mutations.append(plan)
        for key, value in (("model_id", "vit_small_patch14_dinov2"), ("outputs", []),
                           ("retry_resume", {"checkpoint_reference_id": "checkpoint.last", "restore_policy": "compatible"})):
            plan = copy.deepcopy(self.plan)
            plan["execution"]["training"][key] = value
            mutations.append(plan)
        for index, plan in enumerate(mutations):
            with self.subTest(schema=index):
                path = self.base / "invalid.json"
                write_json(path, plan)
                self.invoke("search", "execute", "--root", self.store,
                    "--id", f"invalid-{index}", "--spec", path, expected=2)
                with database(self.store / "runs.sqlite3") as connection:
                    self.assertEqual(connection.execute("SELECT COUNT(*) FROM jyppx_runs WHERE run_id=?",
                        (f"invalid-{index}",)).fetchone()[0], 0)
        cases = []
        for name, changes in (("finetune", {"backbone_mode": "finetune"}),
                              ("bn-train", {"batch_norm": "train"}), ("cuda", {"device": "cuda"})):
            plan = copy.deepcopy(self.plan)
            plan["configuration"].update(changes)
            cases.append((name, plan))
        plan = copy.deepcopy(self.plan)
        plan["objective"]["definition"]["parameters"]["classes"] = "3"
        cases.append(("class-count", plan))
        plan = copy.deepcopy(self.plan)
        plan["recipe_bridge"]["transfer"]["weight_input_sha256"] = "0" * 64
        cases.append(("weight-identity", plan))
        plan = copy.deepcopy(self.plan)
        plan["recipe_bridge"]["transfer"]["descriptor"]["image"]["mean"][0] += .01
        cases.append(("descriptor-binding", plan))
        for name, plan in cases:
            with self.subTest(preflight=name):
                path = self.base / (name + ".json")
                write_json(path, plan)
                self.invoke("search", "execute", "--root", self.store, "--id", name, "--spec", path, expected=2)
                with database(self.store / "runs.sqlite3") as connection:
                    self.assertEqual(connection.execute(
                        "SELECT COUNT(*) FROM jyppx_runs WHERE parent_run_id=?", (name,)).fetchone()[0], 0)
        original = self.metadata.read_bytes()
        try:
            changed = copy.deepcopy(self.descriptor)
            changed["image"]["mean"][0] += .01
            write_json(self.metadata, changed)
            # Otherwise-valid drift must be rejected by the frozen binding.
            self.invoke("recipe", "check", "--file", self.config_path)
            self.invoke("search", "execute", "--root", self.store,
                "--id", "descriptor-drift", "--spec", self.spec_path, expected=2)
            with database(self.store / "runs.sqlite3") as connection:
                self.assertEqual(connection.execute(
                    "SELECT COUNT(*) FROM jyppx_runs WHERE parent_run_id='descriptor-drift'").fetchone()[0], 0)
        finally:
            self.metadata.write_bytes(original)

    def test_02_four_updates_exact_retry_and_native_accuracy(self) -> None:
        self.train()
        self.assertEqual((self.preflight["dataset"]["train"]["samples"],
                          self.preflight["dataset"]["validation"]["samples"]), (4, 4))
        self.assertEqual((self.completed["status"], self.completed["confirmed_trials"],
                          self.completed["failure_count"]), ("completed", 1, 0))
        self.assertEqual(self.completed["best"], self.row)
        self.assertEqual([row["result"]["exit_code"] for row in self.local["attempts"]], [75, 0])
        self.assertTrue(all(row["result"]["process_tree_cleanup_confirmed"] for row in self.local["attempts"]))
        paused = self.json_reference(self.first["attempt_id"], "local.stdout")["data"]
        self.assertEqual((paused["status"], paused["global_step"]), ("paused", 1))
        self.assertEqual((self.result["global_step"], self.result["optimizer_steps"],
                          self.result["completed_epochs"], self.result["scheduler_steps"]), (4, 4, 2, 2))
        self.assertEqual(self.result["validation_samples"], 4)
        self.assertGreaterEqual(self.result["accuracy"], .5)
        self.assertTrue(self.result["quality_passed"])
        transfer = self.result["transfer"]
        self.assertEqual((transfer["backbone_mode"], transfer["batch_norm"]), ("frozen", "eval"))
        self.assertEqual(transfer["initial"]["backbone_sha256"], transfer["final"]["backbone_sha256"])
        self.assertNotEqual(transfer["initial"]["head_sha256"], transfer["final"]["head_sha256"])
        self.assertEqual(transfer["initial"]["trainable_parameters"], 2049 * 2)
        self.assertEqual(transfer["weight_input_sha256"], self.preflight["transfer"]["weight_input_sha256"])
        self.assertEqual(self.json_reference(self.trial + ".inputs", "configuration"), {
            "format": "jyppx.liuding.local_transfer_configuration", "schema_version": 1,
            "configuration": self.config, "transfer": self.preflight["transfer"]})
        self.assertTrue(self.row["eligible"])
        self.assertEqual(self.row["metric"]["definition"], DEFINITION)
        self.assertEqual(self.row["metric"]["value"], self.result["accuracy"])
        self.assertEqual((self.row["metric"]["step"], self.row["metric"]["sample_count"]), (4, 4))
        self.assertEqual(self.bridge["format"], "jyppx.liuding.local_search_transfer_bridge")
        self.assertEqual(self.bridge["training_seed"], 20261029)
        self.assertEqual(self.bridge["trial_seed"], self.admission["trial_seed"])
        effective = self.json_reference(self.last["attempt_id"], "local.training.inputs")["training"]
        self.assertEqual(effective["resume"]["restore_policy"], "exact_only")
        self.assertEqual(effective["resume"]["checkpoint"]["run_id"], self.first["attempt_id"])
        self.assertEqual(self.row["effective_training_inputs"], effective)
        self.assertEqual(self.reference(self.training, "search.training.resume")[1],
                         self.reference(self.first["attempt_id"], "checkpoint.last")[1])
        outputs = self.json_reference(self.last["attempt_id"], "local.training.outputs")
        self.assertTrue(outputs["confirmed"])
        self.assertEqual(len(outputs["outputs"]), 3)
        for row in outputs["outputs"]:
            self.assertEqual(self.reference(self.training, row["reference_id"])[1], row["sha256"])
        with database(self.native_database) as connection:
            canonical = connection.execute("SELECT canonical FROM jyppx_events WHERE run_id=?", (NATIVE,)).fetchall()
            self.assertEqual(connection.execute("SELECT metric_id FROM jyppx_metric_points WHERE run_id=?",
                (NATIVE,)).fetchall(), [("classification.accuracy",)])
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0][0].hex(), self.bridge["native_metric_canonical_hex"])
        manifest = self.invoke("checkpoint", "inspect", "--path", self.recipe / "final.jyck")["manifest"]
        self.assertEqual({row["state"]["participant_id"] for row in manifest["participants"]}, PARTICIPANTS)
        self.invoke("run", "verify", "--root", self.store)

    def test_03_publication_recovery_and_missing_native_evidence(self) -> None:
        self.train()
        try:
            for phase in ("before_start", "after_start", "after_refs", "after_metric"):
                with self.subTest(replay=phase):
                    self.reset_databases()
                    self.pending(phase)
                    recovered = self.invoke("search", "recover", "--root", self.store, "--id", SEARCH)
                    self.assertEqual(recovered["best"], self.row)
                    self.assert_native_unchanged()
            for conflict in ("missing_native_metric", "incompatible_weight_role"):
                with self.subTest(reject=conflict):
                    self.reset_databases()
                    self.pending("before_start", keep_bridge=False)
                    with database(self.native_database) as connection:
                        if conflict == "missing_native_metric":
                            for table in EVENT_TABLES:
                                connection.execute(f"DELETE FROM {table} WHERE run_id=?", (NATIVE,))
                        else:
                            connection.execute("UPDATE jyppx_run_artifacts SET role='weight-input' "
                                "WHERE run_id=? AND reference_id='weight-input'", (NATIVE,))
                    self.invoke("search", "recover", "--root", self.store, "--id", SEARCH, expected=2)
                    with database(self.store / "runs.sqlite3") as connection:
                        self.assertEqual(connection.execute("SELECT COUNT(*) FROM jyppx_run_artifacts WHERE run_id=? "
                            "AND reference_id='search.result.1'", (SEARCH,)).fetchone()[0], 0)
        finally:
            self.reset_databases()
        self.assert_native_unchanged()

    def test_04_terminal_without_native_inputs(self) -> None:
        self.train()
        self.reset_databases()
        retained_recipe = self.recipe.with_name("retained-native-recipe")
        retained_data = self.data.with_name("retained-images")
        retained_metadata = self.metadata.with_name("retained-weights.json")
        retained_weights = self.weights.with_name("retained-input-weights.safetensors")
        for source, target in ((self.recipe, retained_recipe), (self.data, retained_data),
                               (self.metadata, retained_metadata), (self.weights, retained_weights)):
            self.assertTrue(source.resolve().is_relative_to(self.base.resolve()))
            self.assertTrue(target.resolve().is_relative_to(self.base.resolve()))
            self.assertFalse(target.exists())
        moved = []
        try:
            for source, target in ((self.recipe, retained_recipe), (self.data, retained_data),
                                   (self.metadata, retained_metadata), (self.weights, retained_weights)):
                os.rename(extended(source), extended(target))
                moved.append((source, target))
            for path in (self.recipe, self.data, self.metadata, self.weights):
                self.assertFalse(path.exists(), "The frozen native input path must be unavailable")
            for action in ("show", "recover", "recover"):
                self.assertEqual(self.invoke("search", action, "--root", self.store,
                    "--id", SEARCH)["best"], self.row)
        finally:
            for source, target in reversed(moved):
                self.assertFalse(source.exists())
                os.rename(extended(target), extended(source))
        original = self.approved_weights.stat()
        self.assertEqual((original.st_size, original.st_mtime_ns), self.original_weight_stat)
        self.assertTrue(self.weights.samefile(self.approved_weights))
        self.assert_native_unchanged()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--task", choices=("contracts", "fixture"), default="contracts")
    parser.add_argument("--weights-file", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--temp-parent", type=Path)
    options, remaining = parser.parse_known_args()
    if options.task == "fixture" and options.weights_file is None:
        parser.error("fixture requires an explicit existing approved --weights-file")
    TransferSearchRecipe.cli = options.cli.resolve()
    TransferSearchRecipe.source_root = options.source_root.resolve()
    TransferSearchRecipe.task = options.task
    TransferSearchRecipe.weights_file = options.weights_file
    TransferSearchRecipe.evidence_dir = options.evidence_dir.resolve() if options.evidence_dir else None
    TransferSearchRecipe.temp_parent = options.temp_parent.resolve() if options.temp_parent else None
    names = remaining or (["TransferSearchRecipe.test_01_closed_bridge_and_frozen_descriptor"]
        if options.task == "contracts" else ["TransferSearchRecipe"])
    unittest.main(argv=[sys.argv[0], *names], verbosity=2)
