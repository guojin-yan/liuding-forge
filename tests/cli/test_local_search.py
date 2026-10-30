# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Independent search contracts; Python workers and canonical events are fixtures.

The shipped training example supplies the actual native optimization acceptance.
These controlled workers exercise failure, telemetry and crash boundaries which
cannot reliably be induced in a successful training example.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True


def helper(name: str):
    specification = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


TELEMETRY = helper("test_run_telemetry")
ATTEMPTS = helper("test_local_attempt")
PROJECT, EXPERIMENT = "search.contract", "bounded"
DEFINITION = {"schema_version": 1, "metric_id": "validation.loss", "version": 1,
    "unit": 1, "aggregation": 3, "direction": 2, "parameters": {"loss": "squared"},
    "context_keys": ["dataset", "split"]}
CONTEXT = {"dataset": "independent-fixture-v1", "split": "validation"}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def cli(binary: Path, *arguments: object, expected: int = 0) -> dict:
    process = subprocess.run([str(binary), *map(str, arguments), "--format", "json"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45,
        env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
    if process.returncode != expected:
        raise AssertionError(f"exit {process.returncode}, expected {expected}: {process.stdout}\n{process.stderr}")
    return json.loads(process.stdout)


def seed_reference(root: Path, run: str, reference: str, kind: int, role: str, content: bytes) -> str:
    """Insert independently encoded CAS/reference fixtures in a native-created store."""
    identity = digest(content)
    path = root / "artifacts" / "objects" / identity[:2] / identity[2:]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    with closing(sqlite3.connect(root / "runs.sqlite3")) as database, database:
        database.execute("PRAGMA foreign_keys=ON")
        created = database.execute("SELECT created_at_ms FROM jyppx_runs WHERE run_id=?", (run,)).fetchone()[0]
        database.execute("INSERT OR IGNORE INTO jyppx_blobs(digest,algorithm,size_bytes) VALUES(?,1,?)",
                         (identity, len(content)))
        database.execute("INSERT INTO jyppx_run_artifacts(run_id,reference_id,kind,role,digest,"
            "logical_name,media_type,producer,attached_at_ms) VALUES(?,?,?,?,?,?,?,?,?)",
            (run, reference, kind, role, identity, "fixture.bin", "application/octet-stream", "independent-test", created))
    return identity


def child(directory: Path) -> None:
    ATTEMPTS.fixture("child", directory, [])


def fixture(arguments: argparse.Namespace) -> None:
    directory, root = arguments.fixture, arguments.store
    content = arguments.config.read_bytes()
    if digest(content) != arguments.config_sha256:
        raise AssertionError("Native search supplied a different config digest")
    configuration = json.loads(content)
    mode = configuration.get("case", "ok")
    launch = {"trial": arguments.trial, "run": arguments.run, "seed": arguments.seed,
        "configuration": configuration, "config_sha256": arguments.config_sha256,
        "literal": arguments.literal, "environment": os.environ.get("SEARCH_PARAMETER"), "cwd": str(Path.cwd())}
    descriptor = os.open(directory / "launches.jsonl", os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, (json.dumps(launch, ensure_ascii=True) + "\n").encode("ascii"))
    finally:
        os.close(descriptor)
    if mode == "hold":
        subprocess.Popen([sys.executable, "-I", "-u", str(Path(__file__).resolve()), "--child", str(directory)],
                         stdin=subprocess.DEVNULL)
        if not ATTEMPTS.wait_for(lambda: (directory / "child.pid").is_file()):
            raise AssertionError("Descendant did not start")
        pending = directory / "ready.pending"
        pending.write_text(json.dumps({"leader": os.getpid(), "child": int((directory / "child.pid").read_text())}),
                           encoding="ascii")
        pending.replace(directory / "ready.json")
        while True:
            time.sleep(0.01)
    # These tempting stdout values must never become the objective.
    print(json.dumps({"validation.loss": -1e30, "status": "completed"}), flush=True)
    if mode == "exit_fail":
        os._exit(23)
    if mode == "stdout_only":
        return
    metadata = {"schema_version": 1, "run_id": arguments.run, "project_id": PROJECT,
        "experiment_id": EXPERIMENT, "display_name": "Independent search fixture", "status": "running",
        "created_at_unix_ms": int(time.time() * 1000), "lineage": {
            "parent_run_id": "wrong-parent" if mode == "lineage" else arguments.trial,
            "config_digest": arguments.config_sha256, "source_digest": "fixture-source-v1",
            "build_fingerprint": "independent-test-fixture-v1"}, "tags": {"kind": "test-fixture"}}
    if mode == "missing_source":
        del metadata["lineage"]["source_digest"]
    metadata_path = arguments.config.parent / "run.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    cli(arguments.cli, "run", "start", "--root", root, "--metadata", metadata_path)
    if mode != "missing_output":
        seed_reference(root, arguments.run, "checkpoint.best", 1 if mode == "output_kind" else 2,
                       "wrong-role" if mode == "output_role" else "best", b"fixture-model:" + content)
    bound = os.environ.get("SEARCH_INPUT")
    if bound:
        with closing(sqlite3.connect(root / "runs.sqlite3")) as database:
            row = database.execute("SELECT run_id,reference_id FROM jyppx_run_artifacts WHERE digest=? LIMIT 1", (bound,)).fetchone()
        assert row is not None
        if mode == "input_mismatch":
            seed_reference(root, arguments.run, "search.input.data", 1, "search.input.data", b"wrong dataset")
        else:
            object_path = root / "artifacts" / "objects" / bound[:2] / bound[2:]
            seed_reference(root, arguments.run, "search.input.data", 1, "search.input.data", object_path.read_bytes())
    if mode != "missing_metric":
        TELEMETRY.RUN_ID = arguments.run
        TELEMETRY.STREAM_ID = "fixture." + digest(arguments.run.encode())
        number = 7.0 if mode == "tie" else (float(configuration["learning_rate"]) - 2.0) ** 2 + 0.125
        point = {"definition": copy.deepcopy(DEFINITION), "step": {"kind": 2, "value": 10},
            "status": 1, "value": number, "reason": "", "context": copy.deepcopy(CONTEXT), "sample_count": 8}
        point["definition"]["direction"] = configuration.get("objective_direction", 2)
        if mode == "wrong_semantics":
            point["definition"]["unit"] = 6
        if mode == "wrong_context":
            point["context"]["dataset"] = "another-dataset"
        if mode == "wrong_step":
            point["step"]["kind"] = 3
        points = [point]
        if mode in ("nan_last", "missing_last", "last_commit"):
            last = copy.deepcopy(point)
            last["step"]["value"] = 1  # Last commit, not greatest step counter.
            last["status"] = {"nan_last": 2, "missing_last": 5, "last_commit": 1}[mode]
            last["value"] = 23.0 if mode == "last_commit" else None
            last["reason"] = "not observed" if mode == "missing_last" else ""
            points.append(last)
        events = []
        for sequence, observation in enumerate(points, 1):
            event = TELEMETRY.envelope(sequence, {"kind": 2, "points": [observation]})
            event["correlation"].update(project_id=PROJECT, run_id=arguments.run, attempt_id=arguments.trial)
            events.append(event)
        TELEMETRY.seed_database(root, events)
    if mode != "incomplete":
        cli(arguments.cli, "run", "transition", "--root", root, "--id", arguments.run, "--status", "completed")


class LocalSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="liuding-local-search-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "搜索 fixtures"
        self.directory.mkdir()
        self.root = self.directory / "store"
        self.spec_path = self.directory / "spec.json"
        self.search_id = "bounded-search"
        executable = Path(sys.executable).resolve()
        self.spec = {"format": "jyppx.liuding.local_search_spec", "schema_version": 1,
            "project_id": PROJECT, "experiment_id": EXPERIMENT, "strategy": "grid", "seed": 71,
            "max_trials": 3, "max_duration_ms": 120000, "max_failures": 3,
            "configuration": {"learning_rate": 1.0},
            "space": [{"name": "learning_rate", "config_pointer": "/learning_rate", "values": [1.0, 2.0, 3.0]}],
            "execution": {"format": "jyppx.liuding.local_run_spec", "schema_version": 1,
                "project_id": PROJECT, "experiment_id": EXPERIMENT, "executable": str(executable),
                "executable_sha256": digest(executable.read_bytes()),
                "arguments": ["-I", "-u", str(Path(__file__).resolve()), "--fixture", str(self.directory),
                    "--cli", str(self.cli), "--store", "${store}", "--trial", "${trial_id}",
                    "--run", "${training_run_id}", "--config", "${config}", "--config-sha256", "${config_sha256}",
                    "--seed", "${trial_seed}", "--literal", "${param.learning_rate}"],
                "working_directory": "${trial_directory}", "environment": {"SEARCH_PARAMETER": "${param.learning_rate}"},
                "inherit_environment": True, "timeout_ms": 15000, "max_attempts": 1, "retry_exit_codes": [],
                "max_stdout_bytes": 65536, "max_stderr_bytes": 65536},
            "objective": {"definition": copy.deepcopy(DEFINITION), "context": copy.deepcopy(CONTEXT),
                "step_kind": "global_step", "selection": "last"},
            "inputs": [], "outputs": [{"reference_id": "checkpoint.best", "kind": "checkpoint", "role": "best"}]}

    def invoke(self, *arguments: object, expected: int = 0) -> dict:
        return cli(self.cli, *arguments, expected=expected)

    def write(self) -> None:
        self.spec_path.write_text(json.dumps(self.spec, ensure_ascii=False), encoding="utf-8")

    def search(self, action: str, *, expected: int = 0, **options) -> dict:
        if action == "execute":
            self.write()
            options["spec"] = self.spec_path
        arguments = [word for key, value in options.items() for word in ("--" + key, value)]
        return self.invoke("search", action, "--root", self.root, "--id", self.search_id,
                           *arguments, expected=expected)["data"]

    def launches(self) -> list[dict]:
        path = self.directory / "launches.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def single(self, mode: str) -> None:
        self.spec["configuration"]["case"] = mode
        self.spec["space"][0]["values"] = [2.0]
        self.spec["max_trials"] = 1

    def blob(self, root: Path, reference: dict) -> bytes:
        identity = reference["artifact"]["sha256"]
        content = (root / "artifacts" / "objects" / identity[:2] / identity[2:]).read_bytes()
        self.assertEqual(digest(content), identity)
        self.assertEqual(len(content), reference["artifact"]["size_bytes"])
        return content

    def test_grid_is_serial_uses_committed_metrics_and_replays_without_launch(self) -> None:
        result = self.search("execute")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stop_reason"], "max_trials")
        self.assertEqual(result["confirmed_trials"], 3)
        self.assertEqual(result["parallel_trials"], 1)
        self.assertEqual(result["best"]["index"], 2)
        self.assertEqual(result["best"]["metric"]["value"], 0.125)
        launches = self.launches()
        self.assertEqual([row["configuration"]["learning_rate"] for row in launches], [1, 2, 3])
        for index, row in enumerate(result["trials"], 1):
            self.assertEqual(row["index"], index)
            self.assertEqual(row["config_sha256"], launches[index - 1]["config_sha256"])
            self.assertEqual(str(row["trial_seed"]), launches[index - 1]["seed"])
            self.assertEqual(launches[index - 1]["environment"], launches[index - 1]["literal"])
            self.assertEqual(Path(launches[index - 1]["cwd"]).name, "trial-" + str(index))
            metadata = self.invoke("run", "show", "--root", self.root, "--id", row["trial_id"])["data"]["run"]
            self.assertEqual(metadata["lineage"]["parent_run_id"], self.search_id)
        self.assertEqual(self.search("execute")["trials"], result["trials"])
        self.assertEqual(self.search("recover")["best"], result["best"])
        self.assertEqual(len(self.launches()), 3)
        self.spec["seed"] += 1
        self.write()
        error = self.invoke("search", "execute", "--root", self.root, "--id", self.search_id,
                            "--spec", self.spec_path, expected=2)
        self.assertEqual(error["error"]["name"], "invalid_argument")
        self.assertEqual(len(self.launches()), 3)

    def test_random_is_seeded_without_replacement_and_exhausts_finite_space(self) -> None:
        self.spec["strategy"] = "random"
        self.spec["max_trials"] = 8
        self.spec["space"][0]["values"] = [float(index) for index in range(6)]
        first = self.search("execute")
        self.assertEqual(first["stop_reason"], "space_exhausted")
        order = [row["parameters"]["learning_rate"] for row in first["trials"]]
        self.assertEqual(sorted(order), list(range(6)))
        first_seeds = [row["trial_seed"] for row in first["trials"]]
        self.assertEqual(len(set(first_seeds)), 6)
        self.root = self.directory / "second-store"
        second = self.search("execute")
        self.assertEqual(order, [row["parameters"]["learning_rate"] for row in second["trials"]])
        self.assertEqual(first_seeds, [row["trial_seed"] for row in second["trials"]])

    def test_ties_and_maximize_use_declared_direction(self) -> None:
        self.spec["configuration"]["case"] = "tie"
        tied = self.search("execute")
        self.assertEqual(tied["best"]["index"], 1)
        self.assertEqual(tied["tie_policy"], "earliest_trial_index")
        self.root = self.directory / "maximize"
        self.spec["configuration"].update(case="ok", objective_direction=3)
        self.spec["objective"]["definition"]["direction"] = 3
        self.spec["space"][0]["values"] = [1.0, 2.0, 4.0]
        maximized = self.search("execute")
        self.assertEqual(maximized["best"]["index"], 3)
        self.assertEqual(maximized["best"]["metric"]["value"], 4.125)

    def test_last_committed_point_wins_over_larger_step(self) -> None:
        self.single("last_commit")
        result = self.search("execute")
        self.assertEqual(result["best"]["metric"]["step"], 1)
        self.assertEqual(result["best"]["metric"]["value"], 23.0)

    def test_completed_training_and_exact_metric_semantics_are_mandatory(self) -> None:
        cases = {"stdout_only": "training_not_completed", "incomplete": "training_not_completed",
            "lineage": "training_lineage_mismatch", "missing_source": "training_lineage_mismatch",
            "missing_metric": "objective_missing_or_incompatible", "wrong_context": "objective_missing_or_incompatible",
            "wrong_semantics": "objective_missing_or_incompatible", "wrong_step": "objective_missing_or_incompatible",
            "nan_last": "objective_nonfinite_or_missing", "missing_last": "objective_nonfinite_or_missing",
            "missing_output": "training_output_mismatch", "output_kind": "training_output_mismatch",
            "output_role": "training_output_mismatch"}
        for mode, reason in cases.items():
            with self.subTest(case=mode):
                self.root = self.directory / mode
                self.single(mode)
                result = self.search("execute", expected=1)
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["best"])
                self.assertFalse(result["trials"][0]["eligible"])
                self.assertEqual(result["trials"][0]["reason"], reason)

    def test_failure_budget_stops_before_next_candidate_and_never_retries(self) -> None:
        self.spec["configuration"]["case"] = "ok"
        self.spec["space"] = [{"name": "case", "config_pointer": "/case", "values": ["exit_fail", "ok"]}]
        self.spec["execution"]["environment"] = {}
        self.spec["execution"]["arguments"][-1] = "literal"
        self.spec["max_failures"] = 1
        result = self.search("execute", expected=1)
        self.assertEqual(result["stop_reason"], "failure_budget")
        self.assertEqual(result["failure_count"], 1)
        self.assertEqual(len(result["trials"]), 1)
        self.search("recover", expected=1)
        self.assertEqual(len(self.launches()), 1)
        self.root = self.directory / "valid-before-budget"
        self.spec["space"][0]["values"] = ["ok", "exit_fail", "tie"]
        partial = self.search("execute")
        self.assertEqual(partial["status"], "completed")
        self.assertEqual(partial["stop_reason"], "failure_budget")
        self.assertEqual(partial["best"]["index"], 1)
        self.assertEqual(len(partial["trials"]), 2)
        self.assertEqual(len(self.launches()), 3)

    def test_committed_child_repairs_missing_search_rows_and_final_lifecycle(self) -> None:
        self.single("ok")
        original = self.search("execute")
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND "
                "reference_id IN ('search.result.1','search.result','search.best')", (self.search_id,))
            database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id=?", (self.search_id,))
        repaired = self.search("recover")
        self.assertEqual(repaired["trials"], original["trials"])
        self.assertEqual(len(self.launches()), 1)
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id='search.best'", (self.search_id,))
            database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id=?", (self.search_id,))
        self.assertEqual(self.search("recover")["best"], original["best"])
        self.assertEqual(len(self.launches()), 1)

    def test_unknown_native_attempt_stays_fenced_even_after_cancellation(self) -> None:
        self.single("ok")
        original = self.search("execute")
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND "
                "reference_id IN ('search.result.1','search.result','search.best')", (self.search_id,))
            database.execute("DELETE FROM jyppx_run_artifacts WHERE reference_id IN ('local.result','local.stdout','local.stderr')")
            database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id=? OR run_id IN "
                "(SELECT run_id FROM jyppx_run_tags WHERE name='liuding.local.kind')", (self.search_id,))
        uncertain = self.search("recover", expected=7)
        self.assertEqual(uncertain["status"], "recovery_required")
        self.assertFalse(uncertain["owner_active"])
        self.assertEqual(uncertain["trials"], [])
        self.search("cancel")
        self.search("recover", expected=7)
        self.assertEqual(len(self.launches()), 1)
        self.assertEqual(original["confirmed_trials"], 1)

    def test_final_trial_commit_does_not_hide_persisted_cancel_on_recovery(self) -> None:
        self.single("ok")
        original = self.search("execute")
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND "
                "reference_id IN ('search.result','search.best')", (self.search_id,))
            database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id=?", (self.search_id,))
        self.search("cancel")
        recovered = self.search("recover", expected=130)
        self.assertEqual(recovered["status"], "cancelled")
        self.assertEqual(recovered["stop_reason"], "cancelled")
        self.assertEqual(recovered["trials"], original["trials"])
        self.assertEqual(len(self.launches()), 1)

    def test_frozen_admission_and_result_prefix_are_required(self) -> None:
        self.search("execute")
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id='search.result.1'", (self.search_id,))
        self.invoke("search", "show", "--root", self.root, "--id", self.search_id, expected=2)
        self.assertEqual(len(self.launches()), 3)

    def test_existing_child_without_prior_admission_cannot_be_adopted(self) -> None:
        self.single("ok")
        self.search("execute")
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND "
                "reference_id IN ('search.admission.1','search.result.1','search.result','search.best')", (self.search_id,))
            database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id=?", (self.search_id,))
        self.invoke("search", "recover", "--root", self.root, "--id", self.search_id, expected=2)
        self.assertEqual(len(self.launches()), 1)

    def test_best_export_rechecks_source_artifact_bytes(self) -> None:
        self.single("ok")
        original = self.search("execute")
        identity = original["best"]["outputs"][0]["sha256"]
        artifact = self.root / "artifacts" / "objects" / identity[:2] / identity[2:]
        content = artifact.read_bytes()
        artifact.write_bytes(bytes([content[0] ^ 1]) + content[1:])
        self.invoke("search", "export", "--root", self.root, "--id", self.search_id,
                    "--output", self.directory / "corrupt-export", expected=2)
        self.assertFalse((self.directory / "corrupt-export").exists())
        self.assertEqual(len(self.launches()), 1)

    def test_declared_input_digest_output_and_backup_keep_complete_lineage(self) -> None:
        metadata = {"schema_version": 1, "run_id": "dataset", "project_id": PROJECT, "experiment_id": EXPERIMENT,
            "display_name": "Independent input fixture", "created_at_unix_ms": int(time.time() * 1000)}
        path = self.directory / "input.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        self.invoke("run", "start", "--root", self.root, "--metadata", path)
        identity = seed_reference(self.root, "dataset", "data", 1, "dataset", b"immutable source data")
        self.spec["inputs"] = [{"name": "data", "run_id": "dataset", "reference_id": "data", "sha256": identity}]
        self.spec["execution"]["environment"]["SEARCH_INPUT"] = "${input.data}"
        original = self.search("execute")
        backup = self.directory / "backup"
        exported = self.search("export", output=backup)
        self.assertTrue(exported["contains_all_trial_history"])
        self.assertFalse(exported["deployable_model_bundle"])
        self.assertEqual(exported["best"], original["best"])
        restored = self.directory / "restored"
        self.invoke("run", "restore", "--root", restored, "--backup", backup)
        shown = self.invoke("search", "show", "--root", restored, "--id", self.search_id)["data"]
        self.assertEqual(shown["trials"], original["trials"])
        self.assertEqual(shown["best"], original["best"])
        self.invoke("run", "verify", "--root", restored)
        for row in original["trials"]:
            references = self.invoke("run", "show", "--root", restored, "--id", row["training_run_id"])["data"]["references"]
            by_id = {item["reference_id"]: item for item in references}
            self.assertEqual(by_id["search.input.data"]["artifact"]["sha256"], identity)
            for reference in references:
                self.blob(restored, reference)

    def test_mismatched_input_fails_before_launch_and_runtime_binding_is_checked(self) -> None:
        metadata = {"schema_version": 1, "run_id": "dataset", "project_id": PROJECT, "experiment_id": EXPERIMENT,
            "display_name": "Input fixture", "created_at_unix_ms": int(time.time() * 1000)}
        path = self.directory / "input.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        self.invoke("run", "start", "--root", self.root, "--metadata", path)
        identity = seed_reference(self.root, "dataset", "data", 1, "dataset", b"correct data")
        self.spec["inputs"] = [{"name": "data", "run_id": "dataset", "reference_id": "data", "sha256": "0" * 64}]
        self.write()
        self.invoke("search", "execute", "--root", self.root, "--id", self.search_id, "--spec", self.spec_path, expected=2)
        self.assertEqual(self.launches(), [])
        self.spec["inputs"][0]["sha256"] = identity
        self.spec["execution"]["environment"]["SEARCH_INPUT"] = "${input.data}"
        self.single("input_mismatch")
        result = self.search("execute", expected=1)
        self.assertEqual(result["trials"][0]["reason"], "training_input_mismatch")

    def start(self) -> subprocess.Popen:
        self.write()
        process = subprocess.Popen([str(self.cli), "search", "execute", "--root", str(self.root),
            "--id", self.search_id, "--spec", str(self.spec_path), "--format", "json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
        self.addCleanup(lambda: self.cleanup_process(process))
        return process

    def cleanup_process(self, process: subprocess.Popen) -> None:
        if process.poll() is None:
            try:
                self.search("cancel")
                process.communicate(timeout=10)
            except (AssertionError, subprocess.TimeoutExpired):
                process.kill()
                process.communicate(timeout=8)
        ready = self.directory / "ready.json"
        if ready.is_file():
            pids = json.loads(ready.read_text())
            if ATTEMPTS.alive(pids["leader"]):
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(pids["leader"]), "/T", "/F"], capture_output=True, timeout=10)
                else:
                    try:
                        os.killpg(pids["leader"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_cancel_and_concurrent_recover_terminate_owned_descendants(self) -> None:
        self.single("hold")
        process = self.start()
        self.assertTrue(ATTEMPTS.wait_for(lambda: (self.directory / "ready.json").is_file()))
        pids = json.loads((self.directory / "ready.json").read_text())
        active = self.search("recover", expected=7)
        self.assertTrue(active["owner_active"])
        self.search("cancel")
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 130, stdout + stderr)
        result = json.loads(stdout)["data"]
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["stop_reason"], "cancelled")
        for pid in pids.values():
            self.assertTrue(ATTEMPTS.wait_for(lambda: not ATTEMPTS.alive(pid)), f"Fixture PID {pid} survived")
        self.assertEqual(len(self.launches()), 1)

    def test_global_wall_clock_budget_stops_owned_tree_and_next_trial(self) -> None:
        self.spec["configuration"]["case"] = "hold"
        self.spec["max_duration_ms"] = 5000
        process = self.start()
        self.assertTrue(ATTEMPTS.wait_for(lambda: (self.directory / "ready.json").is_file()))
        pids = json.loads((self.directory / "ready.json").read_text())
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 1, stdout + stderr)
        result = json.loads(stdout)["data"]
        self.assertEqual(result["stop_reason"], "wall_time_budget")
        self.assertEqual(len(result["trials"]), 1)
        for pid in pids.values():
            self.assertTrue(ATTEMPTS.wait_for(lambda: not ATTEMPTS.alive(pid)))
        self.search("recover", expected=1)
        self.assertEqual(len(self.launches()), 1)

    def test_invalid_frozen_specs_fail_before_storage_or_launch(self) -> None:
        mutations = [lambda value: value.update(strategy="bayesian"),
            lambda value: value.update(max_trials=0), lambda value: value.update(max_duration_ms=0),
            lambda value: value["execution"].update(max_attempts=2),
            lambda value: value["execution"].update(retry_exit_codes=[23]),
            lambda value: value["objective"].update(selection="best"),
            lambda value: value["objective"]["context"].update(split="x" * 4097),
            lambda value: value["execution"].update(arguments=["--unknown=${param.learning_rate}"]),
            lambda value: value["space"][0].update(values=[1.0, 1.0]),
            lambda value: value["execution"].update(executable_sha256="0" * 64),
            lambda value: value["execution"].update(training={})]
        original = copy.deepcopy(self.spec)
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                self.spec = copy.deepcopy(original)
                self.root = self.directory / ("invalid-" + str(index))
                mutate(self.spec)
                self.write()
                rejected = self.invoke("search", "execute", "--root", self.root, "--id", self.search_id,
                    "--spec", self.spec_path, expected=3 if index == 6 else 2)
                self.assertEqual(rejected["status"], "error")
                self.assertFalse((self.root / "runs.sqlite3").exists())
        self.assertEqual(self.launches(), [])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(Path(sys.argv[2]))
    elif "--fixture" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--fixture", type=Path, required=True)
        parser.add_argument("--cli", type=Path, required=True)
        parser.add_argument("--store", type=Path, required=True)
        parser.add_argument("--trial", required=True)
        parser.add_argument("--run", required=True)
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--config-sha256", required=True)
        parser.add_argument("--seed", required=True)
        parser.add_argument("--literal", required=True)
        fixture(parser.parse_args())
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--cli", type=Path, required=True)
        arguments, rest = parser.parse_known_args()
        LocalSearchTests.cli = arguments.cli.resolve()
        unittest.main(argv=[__file__, *rest])
