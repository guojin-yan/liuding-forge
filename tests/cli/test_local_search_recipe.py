# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Licensed under the Apache License, Version 2.0.

"""Paired native recipe -> one bounded search trial -> exact durable evidence.

Contracts do not train. Real runs all four methods and trains exactly once:
the existing 1024/256, seed 20261028, 40 epoch / 1280 update task, split at
259. All recovery/failure checks reuse small SQLite snapshots of that result.
Python supplies no model, optimizer, training or objective implementation.
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


@contextlib.contextmanager
def readonly_database(path: Path):
    if not path.is_file():
        raise AssertionError("Read-only evidence database is missing")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        yield connection
    finally:
        connection.close()


def snapshot(source: Path, destination: Path) -> None:
    with database(source) as reader, database(destination) as writer:
        reader.backup(writer)


class PairedSearchRecipe(unittest.TestCase):
    cli: Path
    source_root: Path
    device: str
    evidence_dir: Path | None = None
    temp_parent: Path | None = None
    reuse_workspace: Path | None = None
    prior_evidence_dir: Path | None = None

    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        if cls.reuse_workspace is not None:
            allowed = {("search", "recover"), ("search", "show"), ("search", "export"),
                       ("local", "show"), ("checkpoint", "inspect"),
                       ("run", "verify"), ("run", "restore")}
            if tuple(map(str, arguments[:2])) not in allowed:
                raise AssertionError("Reuse mode forbids all native training dispatch, including search execute")
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
        if cls.reuse_workspace is not None:
            cls.prepare_reuse()
            return
        cls.base = Path(tempfile.mkdtemp(prefix="ps405-", dir=cls.temp_parent)).resolve()
        cls.evidence = cls.evidence_dir or cls.base / "evidence"
        cls.evidence.mkdir(parents=True, exist_ok=False)
        cls.command_count = 0
        cls.attempted = False
        cls.addClassCleanup(cls.inventory)
        write_json(cls.evidence / "retained-workspace.json", {
            "workspace": str(cls.base), "retained_on_success_and_failure": True,
            "native_update_budget": 1280, "native_trial_budget": 1,
            "native_attempt_budget": 2, "new_baseline_updates": 0,
            "evidence_policy": "retain original short TEMP; small receipts and SHA inventory only",
        })
        prepared = cls.invoke("data", "prepare", "--name", "paired-affine",
                              "--data-root", cls.base / "datasets")
        cls.data = Path(prepared["destination"])
        cls.config = read_json(cls.source_root / "recipes/multi-io/paired-affine.json")
        cls.config.update(train_file=str(cls.data / "train.jsonl"),
                          validation_file=str(cls.data / "validation.jsonl"), device=cls.device)
        cls.config_path = cls.base / "config.json"
        write_json(cls.config_path, cls.config)
        cls.preflight = cls.invoke("recipe", "check", "--file", cls.config_path)
        cls.config = cls.preflight["config"]
        write_json(cls.config_path, cls.config)
        cls.store = cls.base / "s"
        local = cls.invoke("recipe", "local-spec", "--file", cls.config_path,
                           "--root", cls.base / "unused", "--spec", cls.base / "local.json",
                           "--store", cls.store, "--executable", cls.cli,
                           "--pause-after", 259, "--timeout-ms", 360000)
        execution = copy.deepcopy(local["specification"])
        execution["arguments"] = ["recipe", "local-run", "--file", "${config}",
                                  "--root", "${recipe_directory}", "--format", "json",
                                  "--pause-after", "259"]
        execution["working_directory"] = "${trial_directory}"
        execution["training"]["configuration"] = "${trial_configuration}"
        outputs = [{key: row[key] for key in ("reference_id", "kind", "role")}
                   for row in execution["training"]["outputs"]]
        cls.plan = {
            "format": "jyppx.liuding.local_search_spec", "schema_version": 1,
            "project_id": "liuding-forge", "experiment_id": "paired-affine",
            "strategy": "grid", "seed": 8191, "max_trials": 1,
            "max_duration_ms": 600000, "max_failures": 1,
            "space": [{"name": "learning_rate", "config_pointer": "/learning_rate", "values": [.1]}],
            "configuration": cls.config, "execution": execution,
            "objective": {"definition": DEFINITION, "context": CONTEXT,
                          "step_kind": "sample", "selection": "last"},
            "inputs": [], "outputs": outputs,
            "recipe_bridge": {"format": "jyppx.liuding.local_search_recipe_bridge",
                              "schema_version": 1, "recipe": "jyppx.liuding.paired-affine/v1",
                              "seed_policy": "configuration"},
        }
        cls.spec_path = cls.base / "search.json"
        write_json(cls.spec_path, cls.plan)
        cls.load_telemetry()

    @classmethod
    def load_telemetry(cls) -> None:
        specification = importlib.util.spec_from_file_location(
            "paired_search_independent_telemetry", cls.source_root / "tests/cli/test_run_telemetry.py")
        cls.telemetry = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.telemetry)

    @classmethod
    def bridge_name(cls, trial: str) -> str:
        return "search.recipe.bridge." + hashlib.sha256(trial.encode("utf-8")).hexdigest()

    @classmethod
    def reuse_identity(cls) -> dict:
        # Read only durable, already completed native evidence. Search receipts
        # and outer-training publication are intentionally outside this identity.
        admission = cls.json_reference("paired-search", "search.admission.1")
        trial = admission["trial_id"]
        recipe = Path(admission["execution"]["arguments"][5]).resolve()
        if not recipe.is_relative_to(cls.base) or not recipe.is_dir():
            raise AssertionError("Retained recipe must exist inside the exact reused workspace")
        with readonly_database(cls.store / "runs.sqlite3") as connection:
            job = connection.execute("SELECT status,parent_run_id FROM jyppx_runs WHERE run_id=?", (trial,)).fetchone()
            attempts = connection.execute(
                "SELECT r.run_id,r.status FROM jyppx_runs r JOIN jyppx_run_artifacts a ON a.run_id=r.run_id "
                "WHERE r.parent_run_id=? AND a.reference_id='local.result' ORDER BY r.run_id", (trial,)).fetchall()
        if job != (3, "paired-search") or len(attempts) != 2 or [row[1] for row in attempts] != [4, 3]:
            raise AssertionError(("Reuse requires its completed local job and exactly the original terminal failed/completed attempts", job, attempts))
        outcomes = [cls.json_reference(row[0], "local.result") for row in attempts]
        outcomes.sort(key=lambda row: row["index"])
        if [row["exit_code"] for row in outcomes] != [75, 0] or any(
                row["status"] != "exited" or not row["process_tree_cleanup_confirmed"] for row in outcomes):
            raise AssertionError("Reuse requires the original confirmed 75/0 native cleanup chain")
        final = outcomes[-1]
        if not final["training_outputs_confirmed"]:
            raise AssertionError("Reuse requires the original three confirmed typed outputs")
        output_receipt = cls.json_reference(final["attempt_id"], "local.training.outputs")
        if not output_receipt["confirmed"] or [row["reference_id"] for row in output_receipt["outputs"]] != ["checkpoint.last", "model.weights", "training.result"]:
            raise AssertionError("Reuse output receipt differs")
        files = {}
        for output, filename in zip(output_receipt["outputs"], ("final.jyck", "model.safetensors", "result.json")):
            _, identity = cls.reference(final["attempt_id"], output["reference_id"])
            path = recipe / filename
            if not output["confirmed"] or identity != output["sha256"] or sha(path) != identity:
                raise AssertionError("Reuse native output bytes differ from their existing frozen typed receipt")
            stat = path.stat()
            files[filename] = {"sha256": identity, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        with readonly_database(recipe / "run-store/runs.sqlite3") as connection:
            native = connection.execute("SELECT status FROM jyppx_runs WHERE run_id='run.paired-affine'").fetchone()
        result = read_json(recipe / "result.json")
        data = read_json(recipe / "dataset.json")
        config = read_json(recipe / "config.json")
        if native != (3,) or result["status"] != "completed" or result["global_step"] != 1280 or result["optimizer_steps"] != 1280:
            raise AssertionError("Reuse requires the original completed 1280-update native result")
        if (data["train"]["samples"], data["validation"]["samples"], config["seed"], config["epochs"]) != (1024, 256, 20261028, 40):
            raise AssertionError("Reuse cannot alter the frozen original task")
        return {"trial_id": trial, "attempt_count": 2, "attempt_records": attempts, "attempts": outcomes,
                "typed_outputs": output_receipt, "native_files": files,
                "native_status": 3, "global_step": 1280, "configuration": config, "dataset": data,
                "frozen_execution": admission["execution"]}

    @classmethod
    def prepare_reuse(cls) -> None:
        cls.base = cls.reuse_workspace.resolve()
        if not cls.base.is_dir() or not (cls.base / "s/runs.sqlite3").is_file():
            raise AssertionError("Reuse requires an existing exact workspace and run store")
        cls.evidence = cls.evidence_dir or cls.base / "recovery-evidence"
        cls.evidence.mkdir(parents=True, exist_ok=False)
        cls.command_count = 0
        cls.attempted = False
        cls.store = cls.base / "s"
        cls.config_path = cls.base / "config.json"
        cls.spec_path = cls.base / "search.json"
        cls.config = read_json(cls.config_path)
        cls.plan = read_json(cls.spec_path)
        cls.reuse_before = cls.reuse_identity()
        if cls.config != cls.reuse_before["configuration"] or cls.plan["configuration"] != cls.config:
            raise AssertionError("Reused source configuration differs from the original frozen native task")
        cls.preflight = {"dataset": cls.reuse_before["dataset"], "total_steps": 1280,
                         "identity_source": "retained completed native dataset/configuration/result; no new check or training"}
        prior = []
        if cls.prior_evidence_dir is not None:
            if not cls.prior_evidence_dir.is_dir():
                raise AssertionError("The original failed evidence directory must remain available")
            for directory, _, names in os.walk(extended(cls.prior_evidence_dir)):
                for name in names:
                    path = Path(directory) / name
                    prior.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha(path)})
        cls.prior_evidence_inventory = prior
        write_json(cls.evidence / "reuse-before.json", {"identity": cls.reuse_before,
                   "prior_evidence_dir": str(cls.prior_evidence_dir) if cls.prior_evidence_dir else None,
                   "original_evidence_inventory": prior, "new_native_updates": 0,
                   "frozen_execution_unchanged": True, "only_existing_completed_child_recovery": True})
        write_json(cls.evidence / "retained-workspace.json", {
            "workspace": str(cls.base), "retained_on_success_and_failure": True,
            "native_update_budget": 0, "native_trial_budget": 0, "native_attempt_budget": 0,
            "original_consumed_native_updates": 1280, "original_attempt_count": 2,
            "evidence_policy": "new recovery session; original failed receipts and all native files remain in place"})
        cls.addClassCleanup(cls.inventory)
        cls.load_telemetry()

    @classmethod
    def inventory(cls) -> None:
        if cls.reuse_workspace is not None:
            after = cls.reuse_identity()
            prior_unchanged = all(Path(row["path"]).stat().st_size == row["size_bytes"] and
                                  sha(Path(row["path"])) == row["sha256"]
                                  for row in cls.prior_evidence_inventory)
            write_json(cls.evidence / "reuse-after.json", {"identity": after,
                       "unchanged": after == cls.reuse_before,
                       "prior_evidence_unchanged": prior_unchanged, "new_native_updates": 0})
            if after != cls.reuse_before or not prior_unchanged:
                raise AssertionError("The original native attempts, frozen execution or artifact bytes changed during recovery")
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
        with readonly_database(cls.store / "runs.sqlite3") as connection:
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
        cls.attempted = True
        if cls.reuse_workspace is not None:
            if cls.reuse_identity() != cls.reuse_before:
                raise AssertionError("Original native evidence changed before its only recovery")
            cls.completed = cls.invoke("search", "recover", "--root", cls.store, "--id", "paired-search")
            if cls.reuse_identity() != cls.reuse_before:
                raise AssertionError("Recovery changed original native evidence or dispatched another attempt")
        else:
            cls.completed = cls.invoke("search", "execute", "--root", cls.store,
                                       "--id", "paired-search", "--spec", cls.spec_path)
        cls.row = cls.completed["trials"][0]
        cls.trial = cls.row["trial_id"]
        cls.training = cls.row["training_run_id"]
        cls.local = cls.invoke("local", "show", "--root", cls.store, "--id", cls.trial)
        cls.first, cls.last = cls.local["attempts"]
        cls.admission = cls.json_reference("paired-search", "search.admission.1")
        cls.recipe = Path(cls.admission["execution"]["arguments"][5])
        cls.result = read_json(cls.recipe / "result.json")
        cls.bridge_reference = cls.bridge_name(cls.trial)
        cls.bridge = cls.json_reference("paired-search", cls.bridge_reference)
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
            connection.execute("DELETE FROM jyppx_run_artifacts WHERE run_id='paired-search' AND "
                               "reference_id IN ('search.result','search.best','search.result.1')")
            connection.execute("UPDATE jyppx_runs SET status=2 WHERE run_id='paired-search'")
            if not keep_bridge:
                connection.execute("DELETE FROM jyppx_run_artifacts WHERE run_id='paired-search' AND reference_id=?",
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
        if self.reuse_workspace is not None:
            self.assertEqual(self.reuse_identity(), self.reuse_before)
        self.assertEqual(self.invoke("local", "show", "--root", self.store,
                                     "--id", self.trial)["attempts"], self.local["attempts"])
        self.assertEqual({name: (sha(self.recipe / name), (self.recipe / name).stat().st_mtime_ns)
                          for name in self.immutable_files}, self.immutable_files)

    def test_01_closed_bridge_template_and_input_identity(self) -> None:
        if self.reuse_workspace is not None:
            self.fail("Reuse mode permits only tests 02/03/04; run zero-training contracts in a separate fresh session")
        mutations = []
        for key, value in (("schema_version", 2), ("recipe", "jyppx.liuding.imagefolder.resnet18/v1"),
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
        plan["configuration"]["train_file"] = str(self.data / ".." / self.data.name / "train.jsonl")
        path = self.base / "unresolved.json"
        write_json(path, plan)
        self.invoke("search", "execute", "--root", self.store,
                    "--id", "unresolved", "--spec", path, expected=2)
        with database(self.store / "runs.sqlite3") as connection:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM jyppx_runs WHERE run_id='unresolved'").fetchone()[0], 1)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM jyppx_runs WHERE parent_run_id='unresolved'").fetchone()[0], 0)

    def test_02_full_paired_exact_retry_and_committed_objective(self) -> None:
        self.train()
        self.assertEqual(self.preflight["total_steps"], 1280)
        self.assertEqual((self.preflight["dataset"]["train"]["samples"],
                          self.preflight["dataset"]["validation"]["samples"]), (1024, 256))
        self.assertEqual(self.completed["status"], "completed")
        self.assertEqual(self.completed["confirmed_trials"], 1)
        self.assertEqual(self.completed["failure_count"], 0)
        self.assertEqual(self.completed["best"], self.row)
        self.assertEqual(self.completed["parallel_trials"], 1)
        self.assertEqual([part["result"]["exit_code"] for part in self.local["attempts"]], [75, 0])
        self.assertTrue(all(part["result"]["process_tree_cleanup_confirmed"] for part in self.local["attempts"]))
        paused = self.json_reference(self.first["attempt_id"], "local.stdout")["data"]
        self.assertEqual((paused["status"], paused["global_step"]), ("paused", 259))
        self.assertEqual((self.result["global_step"], self.result["optimizer_steps"]), (1280, 1280))
        self.assertEqual((self.result["completed_epochs"], self.result["scheduler_steps"]), (40, 40))
        self.assertEqual(self.result["validation_samples"], 256)
        self.assertGreaterEqual(self.result["accuracy"], .95)
        self.assertLessEqual(self.result["mse"], .005)
        self.assertTrue(self.result["quality_passed"])
        self.assertTrue(self.row["eligible"])
        self.assertEqual(self.row["metric"]["value"], self.result["accuracy"])
        self.assertEqual(self.row["metric"]["definition"], DEFINITION)
        self.assertEqual((self.row["metric"]["step"], self.row["metric"]["sample_count"]), (256, 256))
        self.assertEqual(self.row["recipe_bridge_sha256"], self.reference("paired-search", self.bridge_reference)[1])
        with readonly_database(self.store / "runs.sqlite3") as connection:
            self.assertEqual(connection.execute(
                "SELECT role FROM jyppx_run_artifacts WHERE run_id='paired-search' AND reference_id=?",
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
            canonical = connection.execute("SELECT canonical FROM jyppx_events WHERE run_id='run.paired-affine'").fetchall()
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0][0].hex(), self.bridge["native_metric_canonical_hex"])
        with database(self.store / "runs.sqlite3") as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM jyppx_metric_points WHERE run_id=?",
                                                (self.training,)).fetchone()[0], 1)
        manifest = self.invoke("checkpoint", "inspect", "--path", self.recipe / "final.jyck")["manifest"]
        self.assertEqual({row["state"]["participant_id"] for row in manifest["participants"]}, PARTICIPANTS)
        self.invoke("run", "verify", "--root", self.store)

    def test_03_partial_publication_and_uncertain_evidence(self) -> None:
        self.train()
        try:
            for phase in ("before_start", "after_start", "after_refs", "after_metric"):
                with self.subTest(replay=phase):
                    self.reset_databases()
                    self.pending(phase)
                    recovered = self.invoke("search", "recover", "--root", self.store, "--id", "paired-search")
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
                        fixture.STREAM_ID = "search.recipe.paired-affine." + hashlib.sha256(self.training.encode()).hexdigest()
                        point = {"definition": DEFINITION, "step": {"kind": 4, "value": 256},
                                 "status": 1, "value": 1.0, "reason": "", "context": CONTEXT, "sample_count": 256}
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
                        self.replace_reference_json("paired-search", self.bridge_reference, manifest)
                    elif conflict == "missing_native_metric":
                        with database(self.native_database) as connection:
                            for table in EVENT_TABLES:
                                connection.execute(f"DELETE FROM {table} WHERE run_id='run.paired-affine'")
                    elif conflict == "native_not_completed":
                        with database(self.native_database) as connection:
                            connection.execute("UPDATE jyppx_runs SET status=2 WHERE run_id='run.paired-affine'")
                    elif conflict == "missing_owner":
                        with database(self.store / "runs.sqlite3") as connection:
                            connection.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id='local.owner'",
                                               (self.last["attempt_id"],))
                    else:
                        outcome = self.json_reference(self.last["attempt_id"], "local.result")
                        outcome["process_tree_cleanup_confirmed"] = False
                        self.replace_reference_json(self.last["attempt_id"], "local.result", outcome)
                        expected = 7
                    self.invoke("search", "recover", "--root", self.store, "--id", "paired-search", expected=expected)
                    with database(self.store / "runs.sqlite3") as connection:
                        self.assertEqual(connection.execute("SELECT COUNT(*) FROM jyppx_run_artifacts WHERE run_id='paired-search' "
                                                            "AND reference_id='search.result.1'").fetchone()[0], 0)
                    self.assertEqual({name: (sha(self.recipe / name), (self.recipe / name).stat().st_mtime_ns)
                                      for name in self.immutable_files}, self.immutable_files)
        finally:
            self.reset_databases()
        self.assert_native_unchanged()

    def test_04_terminal_export_without_native_workspace(self) -> None:
        self.train()
        self.reset_databases()
        for action in ("show", "recover", "recover"):
            self.assertEqual(self.invoke("search", action, "--root", self.store,
                                         "--id", "paired-search")["best"], self.row)
        self.assert_native_unchanged()
        backup = self.base / "export"
        exported = self.invoke("search", "export", "--root", self.store,
                               "--id", "paired-search", "--output", backup)
        self.assertFalse(exported["deployable_model_bundle"])
        self.assertTrue(exported["contains_all_trial_history"])
        retained = self.recipe.with_name("retained-native-recipe")
        self.assertTrue(self.recipe.resolve().is_relative_to(self.base.resolve()))
        self.assertTrue(retained.resolve().is_relative_to(self.base.resolve()))
        self.assertFalse(retained.exists())
        os.rename(extended(self.recipe), extended(retained))
        try:
            restored = self.base / "restored"
            self.invoke("run", "restore", "--root", restored, "--backup", backup)
            self.invoke("run", "verify", "--root", restored)
            for root in (self.store, restored):
                for action in ("show", "recover"):
                    self.assertEqual(self.invoke("search", action, "--root", root,
                                                 "--id", "paired-search")["best"], self.row)
            self.invoke("search", "export", "--root", restored, "--id", "paired-search",
                        "--output", self.base / "portable-export")
        finally:
            self.assertTrue(retained.resolve().is_relative_to(self.base.resolve()))
            self.assertTrue(self.recipe.resolve().is_relative_to(self.base.resolve()))
            self.assertFalse(self.recipe.exists())
            os.rename(extended(retained), extended(self.recipe))
        self.assert_native_unchanged()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--task", choices=("contracts", "real"), default="contracts")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--temp-parent", type=Path)
    parser.add_argument("--reuse-workspace", type=Path)
    parser.add_argument("--prior-evidence-dir", type=Path)
    options, remaining = parser.parse_known_args()
    PairedSearchRecipe.cli = options.cli.resolve()
    PairedSearchRecipe.source_root = options.source_root.resolve()
    PairedSearchRecipe.device = options.device
    PairedSearchRecipe.evidence_dir = options.evidence_dir.resolve() if options.evidence_dir else None
    PairedSearchRecipe.temp_parent = options.temp_parent.resolve() if options.temp_parent else None
    PairedSearchRecipe.reuse_workspace = options.reuse_workspace.resolve() if options.reuse_workspace else None
    PairedSearchRecipe.prior_evidence_dir = options.prior_evidence_dir.resolve() if options.prior_evidence_dir else None
    if options.reuse_workspace and options.task != "real":
        parser.error("--reuse-workspace requires --task real; it never executes a new trial")
    if options.reuse_workspace and not options.prior_evidence_dir:
        parser.error("--reuse-workspace requires --prior-evidence-dir to retain and bind the original failed receipts")
    if options.reuse_workspace and not remaining:
        remaining = ["PairedSearchRecipe.test_02_full_paired_exact_retry_and_committed_objective",
                     "PairedSearchRecipe.test_03_partial_publication_and_uncertain_evidence",
                     "PairedSearchRecipe.test_04_terminal_export_without_native_workspace"]
    if options.reuse_workspace and set(remaining) - {
            "PairedSearchRecipe.test_02_full_paired_exact_retry_and_committed_objective",
            "PairedSearchRecipe.test_03_partial_publication_and_uncertain_evidence",
            "PairedSearchRecipe.test_04_terminal_export_without_native_workspace"}:
        parser.error("--reuse-workspace accepts only the three recovery methods 02/03/04")
    names = remaining or (["PairedSearchRecipe.test_01_closed_bridge_template_and_input_identity"]
                          if options.task == "contracts" else ["PairedSearchRecipe"])
    unittest.main(argv=[sys.argv[0], *names], verbosity=2)
