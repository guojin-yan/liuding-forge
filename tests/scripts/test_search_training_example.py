#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Audit real native search training, objective selection, checkpoints and backup."""
from __future__ import annotations

import argparse
import copy
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import tempfile
import time
import unittest


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class SearchTrainingExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="liuding-search-training-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "训练 search with spaces"
        self.directory.mkdir()
        self.root = self.directory / "store"
        self.spec_path = self.directory / "search.json"
        self.search_id = "linear-regression-example"

    def invoke(self, binary: Path, *arguments: object, timeout: int = 45) -> subprocess.CompletedProcess:
        result = subprocess.run([str(binary), *map(str, arguments)], cwd=self.directory,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def cli(self, *arguments: object, timeout: int = 45) -> dict:
        return json.loads(self.invoke(CLI, *arguments, "--format", "json", timeout=timeout).stdout)["data"]

    def show_run(self, run_id: str, root: Path | None = None) -> dict:
        return self.cli("run", "show", "--root", root or self.root, "--id", run_id)

    def blob(self, reference: dict, root: Path | None = None) -> bytes:
        artifact = reference["artifact"]
        identity = artifact["sha256"]
        content = ((root or self.root) / "artifacts" / "objects" / identity[:2] / identity[2:]).read_bytes()
        self.assertEqual(digest(content), identity)
        self.assertEqual(len(content), artifact["size_bytes"])
        return content

    def parameters(self, checkpoint: bytes) -> dict[str, float]:
        """Read the two actual FP32 scalars without using Python model tooling."""
        self.assertLessEqual(len(checkpoint), 65536)
        self.assertEqual(checkpoint[:8], b"JYPPXCK1")
        self.assertEqual(checkpoint[-32:], hashlib.sha256(checkpoint[:-32]).digest())
        version, metadata_size = struct.unpack_from("<IQ", checkpoint, 8)
        self.assertEqual(version, 1)
        offset = 20 + metadata_size
        manifest = json.loads(checkpoint[20:offset])
        self.assertEqual(len(manifest["participants"]), 1)
        participant = manifest["participants"][0]
        self.assertEqual(participant["state"]["restore_level"], "exact")
        fields = {}
        for blob in participant["blobs"]:
            end = offset + blob["size_bytes"]
            self.assertLessEqual(end, len(checkpoint) - 32)
            payload = checkpoint[offset:end]
            self.assertEqual(blob["digest"]["algorithm"], "sha256")
            self.assertEqual(blob["digest"]["hex"], digest(payload))
            self.assertNotIn(blob["field_name"], fields)
            fields[blob["field_name"]] = payload
            offset = end
        self.assertEqual(offset, len(checkpoint) - 32)
        self.assertEqual(set(fields), {"topology", "tensor.0000000000", "tensor.0000000001"})

        topology = fields["topology"]
        self.assertEqual(topology[:8], b"JYPPXMDL")
        self.assertEqual(struct.unpack_from("<II", topology, 8), (1, 2))
        offset = 16
        parameters = {}
        for index in range(2):
            role, dtype, requires_grad, reserved, key_size, rank, _group, storage_offset, byte_count = (
                struct.unpack_from("<BBBBIIIqQ", topology, offset))
            offset += struct.calcsize("<BBBBIIIqQ")
            self.assertEqual((role, dtype, requires_grad, reserved), (1, 7, 1, 0))
            self.assertEqual(storage_offset, 0)
            self.assertEqual(byte_count, 4)
            self.assertLessEqual(rank, 2)
            name = topology[offset:offset + key_size].decode("utf-8")
            offset += key_size
            sizes = struct.unpack_from(f"<{rank}q", topology, offset)
            offset += 8 * rank
            strides = struct.unpack_from(f"<{rank}q", topology, offset)
            offset += 8 * rank
            self.assertEqual(sizes, (1, 1) if name == "weight" else (1,))
            self.assertEqual(strides, (1,) * rank)
            self.assertNotIn(name, parameters)
            value, = struct.unpack("<f", fields[f"tensor.{index:010d}"])
            self.assertTrue(math.isfinite(value))
            parameters[name] = value
        self.assertEqual(offset, len(topology))
        self.assertEqual(set(parameters), {"weight", "bias"})
        return parameters

    def audit_sqlite(self, root: Path, rows: list[dict]) -> set[tuple[str, int]]:
        with closing(sqlite3.connect((root / "runs.sqlite3").as_uri() + "?mode=ro", uri=True)) as database:
            self.assertEqual(database.execute("PRAGMA integrity_check").fetchall(), [("ok",)])
            self.assertEqual(database.execute("PRAGMA foreign_key_check").fetchall(), [])
            for row in rows:
                run_id, metric = row["training_run_id"], row["metric"]
                run = database.execute("SELECT parent_run_id, config_digest FROM jyppx_runs WHERE run_id=?",
                                       (run_id,)).fetchone()
                self.assertEqual(run, (row["trial_id"], row["config_sha256"]))
                points = database.execute("SELECT run_sequence, point_index, metric_id, definition_version, "
                    "step_kind, step_key, value_status FROM jyppx_metric_points WHERE run_id=?", (run_id,)).fetchall()
                self.assertEqual(len(points), 1)
                sequence, index, name, version, kind, step, status = points[0]
                self.assertEqual((sequence, index), (metric["run_sequence"], metric["point_index"]))
                self.assertEqual((name, version, kind, int.from_bytes(step, "big"), status),
                                 ("validation.mean_squared_error", 1, 2, 60, 1))
                events = database.execute("SELECT stream_id, event_id, content_digest, canonical "
                    "FROM jyppx_events WHERE run_id=?", (run_id,)).fetchall()
                self.assertEqual(len(events), 1)
                stream, event, identity, canonical = events[0]
                self.assertEqual(stream, run_id + ".training")
                self.assertEqual(event, metric["event_id"])
                self.assertEqual(digest(canonical), identity)
                definition = database.execute("SELECT definition_digest FROM jyppx_metric_definitions "
                    "WHERE metric_id=? AND definition_version=1", (name,)).fetchone()
                self.assertEqual(definition, (metric["definition_sha256"],))
            return set(database.execute("SELECT digest,size_bytes FROM jyppx_blobs WHERE digest IN "
                "(SELECT digest FROM jyppx_run_artifacts)").fetchall())

    def check_search(self, strategy: str) -> None:
        self.invoke(WORKER, "--write-spec", self.spec_path, "--strategy", strategy)
        spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        self.assertEqual(spec["strategy"], strategy)
        self.assertEqual(Path(spec["execution"]["executable"]), WORKER)
        self.assertEqual(spec["execution"]["executable_sha256"], digest(WORKER.read_bytes()))
        self.assertEqual(Path(spec["execution"]["working_directory"]), self.directory)

        result = self.cli("search", "execute", "--root", self.root, "--id", self.search_id,
                          "--spec", self.spec_path, timeout=150)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["strategy"], strategy)
        self.assertEqual(result["stop_reason"], "max_trials")
        self.assertFalse(result["owner_active"])
        self.assertFalse(result["cancel_requested"])
        self.assertEqual((result["space_cardinality"], result["planned_trials"], result["parallel_trials"]), (3, 3, 1))
        self.assertEqual(result["tie_policy"], "earliest_trial_index")
        rows = result["trials"]
        self.assertEqual(len(rows), 3)
        rates = [row["parameters"]["learning_rate"] for row in rows]
        self.assertEqual(set(rates), {0.005, 0.05, 0.2})
        if strategy == "grid":
            self.assertEqual(rates, [0.005, 0.05, 0.2])
        search_refs = {item["reference_id"]: item for item in self.show_run(self.search_id)["references"]}
        frozen_spec = self.blob(search_refs["search.spec"])
        self.assertEqual(json.loads(frozen_spec), spec)
        self.assertEqual(digest(frozen_spec), result["search_spec_sha256"])

        for index, row in enumerate(rows, 1):
            with self.subTest(strategy=strategy, trial=index):
                self.assertEqual(row["index"], index)
                self.assertTrue(row["eligible"], row["reason"])
                self.assertEqual(row["reason"], "")
                self.assertEqual(json.loads(self.blob(search_refs[f"search.result.{index}"])), row)
                configuration = self.blob(search_refs[f"search.config.{index}"])
                self.assertEqual(digest(configuration), row["config_sha256"])
                self.assertEqual(json.loads(configuration), {"learning_rate": rates[index - 1], "steps": 60})
                trial_directory = (self.root / "search-work" / digest(self.search_id.encode("utf-8"))
                                   / f"trial-{index}")
                self.assertEqual((trial_directory / "config.json").read_bytes(), configuration)
                training = self.show_run(row["training_run_id"])
                metadata = training["run"]
                self.assertEqual(metadata["status"], "completed")
                self.assertEqual(metadata["lineage"]["parent_run_id"], row["trial_id"])
                self.assertEqual(metadata["lineage"]["config_digest"], row["config_sha256"])
                self.assertEqual(metadata["tags"]["example"], "local-search")
                self.assertEqual(metadata["tags"]["seed"], str(row["trial_seed"]))
                refs = {item["reference_id"]: item for item in training["references"]}
                checkpoint = refs["checkpoint.best"]
                self.assertEqual((checkpoint["kind"], checkpoint["role"]), ("checkpoint", "best"))
                self.assertEqual(row["outputs"], [{"reference_id": "checkpoint.best", "kind": "checkpoint",
                    "role": "best", "sha256": checkpoint["artifact"]["sha256"],
                    "size_bytes": checkpoint["artifact"]["size_bytes"]}])
                checkpoint_bytes = self.blob(checkpoint)
                self.assertEqual((trial_directory / "model-best.jyck").read_bytes(), checkpoint_bytes)
                parameters = self.parameters(checkpoint_bytes)
                rate = row["parameters"]["learning_rate"]
                # Closed-form full-batch SGD is independent of the native optimizer.
                expected_weight = 3 * (1 - (1 - 2 * rate * 33 / 93) ** 60)
                expected_bias = 0.5 * (1 - (1 - 2 * rate) ** 60)
                self.assertAlmostEqual(parameters["weight"], expected_weight, delta=5e-5)
                self.assertAlmostEqual(parameters["bias"], expected_bias, delta=5e-6)
                oracle = sum(((parameters["weight"] - 3) * (-0.9 + 0.2 * item)
                              + parameters["bias"] - 0.5) ** 2 for item in range(10)) / 10
                metric = row["metric"]
                self.assertEqual(metric["definition"], spec["objective"]["definition"])
                self.assertEqual(metric["context"], spec["objective"]["context"])
                self.assertEqual((metric["step_kind"], metric["step"], metric["sample_count"], metric["status"]),
                                 ("global_step", 60, 10, 1))
                self.assertTrue(math.isfinite(metric["value"]))
                self.assertAlmostEqual(metric["value"], oracle, delta=max(1e-9, oracle * 1e-5))
                observed = self.cli("run", "metrics", "--root", self.root, "--id", row["training_run_id"],
                                    "--metric", "validation.mean_squared_error")
                self.assertEqual(observed["source"], "sqlite-canonical")
                self.assertEqual(len(observed["points"]), 1)
                point = observed["points"][0]
                self.assertEqual(point["status_name"], "finite")
                self.assertEqual(point["event_identity"]["event_id"], metric["event_id"])
                self.assertEqual(point["point"]["value"], metric["value"])
                self.assertEqual(point["point"]["definition"]["definition_digest"], metric["definition_sha256"])
                self.assertEqual(point["point"]["context"], metric["context"])

        winner = min(rows, key=lambda row: (row["metric"]["value"], row["index"]))
        self.assertEqual(result["best"], winner)
        self.assertEqual(winner["parameters"]["learning_rate"], 0.2)
        self.assertEqual(json.loads(self.blob(search_refs["search.best"])), winner)
        summary = json.loads(self.blob(search_refs["search.result"]))
        self.assertEqual(summary["confirmed_trials"], 3)
        self.assertEqual(summary["best"], winner)
        self.assertEqual(self.cli("search", "show", "--root", self.root, "--id", self.search_id), result)
        live_blobs = self.audit_sqlite(self.root, rows)
        self.assertTrue(self.cli("run", "verify", "--root", self.root)["valid"])

        backup = self.directory / "experiment-backup"
        exported = self.cli("search", "export", "--root", self.root, "--id", self.search_id, "--output", backup)
        self.assertEqual(exported["format"], "jyppx.run-store-backup/v1")
        self.assertEqual(exported["best"], winner)
        self.assertTrue(exported["contains_all_trial_history"])
        self.assertFalse(exported["deployable_model_bundle"])
        self.assertEqual(exported["artifact_count"], len(live_blobs))
        self.assertEqual(exported["artifact_bytes"], sum(size for _, size in live_blobs))
        self.assertEqual(exported["database_bytes"], (backup / "runs.sqlite3").stat().st_size)
        manifest = (backup / "manifest.jyppx-backup").read_bytes()
        self.assertEqual(manifest[-65:-1].decode("ascii"), digest(manifest[:-65]))
        lines = manifest[:-65].decode("ascii").splitlines()
        self.assertEqual(lines[:3], ["JYPPX-RUN-BACKUP", "1", "2"])
        self.assertEqual(lines[3], f"{exported['database_bytes']} {digest((backup / 'runs.sqlite3').read_bytes())}")
        self.assertEqual(int(lines[4]), len(live_blobs))
        self.assertEqual(set(lines[5:]), {f"{size} {identity}" for identity, size in live_blobs})
        self.assertEqual(self.audit_sqlite(backup, rows), live_blobs)
        for identity, size in live_blobs:
            content = (backup / "artifacts" / "objects" / identity[:2] / identity[2:]).read_bytes()
            self.assertEqual((digest(content), len(content)), (identity, size))
        backup_refs = {item["reference_id"]: item for item in self.show_run(self.search_id, backup)["references"]}
        self.assertEqual(self.blob(backup_refs["search.best"], backup), self.blob(search_refs["search.best"]))

    def test_grid_real_training_and_best_model_backup(self) -> None:
        self.check_search("grid")

    def test_random_real_training_and_best_model_backup(self) -> None:
        self.check_search("random")


class LocalTrainingExampleTests(unittest.TestCase):
    """Native training is the workload; Python only audits bytes and state."""
    setUp = SearchTrainingExampleTests.setUp
    invoke = SearchTrainingExampleTests.invoke
    cli = SearchTrainingExampleTests.cli
    show_run = SearchTrainingExampleTests.show_run
    blob = SearchTrainingExampleTests.blob

    def prepare(self, name: str, until: int, resume: str | None = None) -> tuple[Path, dict]:
        path = self.directory / (name + ".json")
        options = ["--resume-run", resume] if resume else []
        self.invoke(WORKER, "--write-local-spec", path, "--store", self.root,
                    "--until-step", until, *options)
        return path, json.loads(path.read_text(encoding="utf-8"))

    def execute(self, name: str, path: Path) -> tuple[dict, dict, dict]:
        state = self.cli("local", "execute", "--root", self.root, "--id", name, "--spec", path)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(state["attempts"]), 1)
        attempt = state["attempts"][0]["attempt_id"]
        refs = {ref["reference_id"]: ref for ref in self.show_run(attempt)["references"]}
        result = json.loads(self.blob(refs["training.result"]))
        self.assertEqual(result["checkpoint_sha256"], refs["checkpoint.last"]["artifact"]["sha256"])
        return state, refs, result

    def archive(self, content: bytes) -> tuple[dict, dict[tuple[str, str], bytes]]:
        self.assertEqual(content[:8], b"JYPPXCK1")
        self.assertLessEqual(len(content), 65536)
        self.assertEqual(content[-32:], hashlib.sha256(content[:-32]).digest())
        version, length = struct.unpack_from("<IQ", content, 8)
        self.assertEqual(version, 1)
        offset = 20 + length
        manifest = json.loads(content[20:offset])
        fields = {}
        for participant in manifest["participants"]:
            state = participant["state"]
            self.assertEqual(state["restore_level"], "exact")
            for blob in participant["blobs"]:
                end = offset + blob["size_bytes"]
                value = content[offset:end]
                self.assertLessEqual(end, len(content) - 32)
                self.assertEqual(blob["digest"], {"algorithm": "sha256", "hex": digest(value)})
                key = state["participant_id"], blob["field_name"]
                self.assertNotIn(key, fields)
                fields[key] = value
                offset = end
        self.assertEqual(offset, len(content) - 32)
        return manifest, fields

    def rejected(self, name: str, spec: dict, *, admitted: bool = False) -> dict:
        path = self.directory / (name + ".json")
        path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        process = subprocess.run([str(CLI), "local", "execute", "--root", str(self.root),
            "--id", name, "--spec", str(path), "--format", "json"], cwd=self.directory,
            capture_output=True, text=True, encoding="utf-8", timeout=45,
            env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        reply = json.loads(process.stdout)
        if admitted:
            self.assertEqual(reply["data"]["status"], "failed")
            self.assertEqual(len(reply["data"]["attempts"]), 1)
            attempt = reply["data"]["attempts"][0]["attempt_id"]
            refs = {ref["reference_id"] for ref in self.show_run(attempt)["references"]}
            self.assertNotIn("checkpoint.last", refs)
            self.assertNotIn("training.result", refs)
        else:
            self.assertEqual(reply["status"], "error")
            with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database:
                self.assertEqual(database.execute(
                    "SELECT count(*) FROM jyppx_runs WHERE parent_run_id=?", (name,)).fetchone(), (0,))
        return reply

    def seed_checkpoint(self, source: str, name: str, content: bytes) -> dict:
        """Independent negative fixture in an otherwise native-created store."""
        identity = digest(content)
        path = self.root / "artifacts" / "objects" / identity[:2] / identity[2:]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("PRAGMA foreign_keys=ON")
            stamp = database.execute("SELECT created_at_ms FROM jyppx_runs WHERE run_id=?", (source,)).fetchone()[0]
            database.execute("INSERT OR IGNORE INTO jyppx_blobs(digest,algorithm,size_bytes) VALUES(?,1,?)",
                             (identity, len(content)))
            database.execute("INSERT INTO jyppx_run_artifacts(run_id,reference_id,kind,role,digest,"
                "logical_name,media_type,producer,attached_at_ms) VALUES(?,?,2,'last',?,?,'application/octet-stream',"
                "'independent-negative-fixture',?)", (source, name, identity, name + ".jyck", stamp))
        return {"run_id": source, "reference_id": name, "sha256": identity, "size_bytes": len(content)}

    def test_explicit_twenty_plus_forty_updates_equal_uninterrupted_native_sgd(self) -> None:
        path, first_spec = self.prepare("partial", 20)
        first, first_refs, first_result = self.execute("partial-job", path)
        source = first["attempts"][0]["attempt_id"]
        self.assertEqual((first_result["initial_step"], first_result["completed_steps"], first_result["resumed"]),
                         (0, 20, False))
        path, resumed_spec = self.prepare("resume", 60, source)
        resumed, resumed_refs, result = self.execute("resumed-job", path)
        self.assertEqual((result["initial_step"], result["completed_steps"], result["resumed"]), (20, 60, True))
        path, _ = self.prepare("continuous", 60)
        _, continuous_refs, continuous_result = self.execute("continuous-job", path)
        _, actual_fields = self.archive(self.blob(resumed_refs["checkpoint.last"]))
        _, continuous_fields = self.archive(self.blob(continuous_refs["checkpoint.last"]))
        self.assertEqual({owner for owner, _ in actual_fields}, {"model.main", "optimizer.main", "run.inputs"})
        self.assertEqual(actual_fields, continuous_fields)  # Includes real SGD momentum/control blobs.
        self.assertEqual(result["weight"], continuous_result["weight"])
        self.assertEqual(result["bias"], continuous_result["bias"])
        weight, bias, vw, vb = 0.0, 0.0, 0.0, 0.0
        for _ in range(60):
            vw = 0.9 * vw + 2 * (weight - 3) * 11 / 31
            vb = 0.9 * vb + 2 * (bias - 0.5)
            weight -= 0.05 * vw
            bias -= 0.05 * vb
        self.assertAlmostEqual(result["weight"], weight, delta=3e-5)
        self.assertAlmostEqual(result["bias"], bias, delta=3e-5)
        job_refs = {ref["reference_id"]: ref for ref in self.show_run("resumed-job")["references"]}
        for key, pinned in (("configuration", "local.training.configuration"),
                            ("data_manifest", "local.training.data_manifest")):
            self.assertEqual(job_refs[pinned]["artifact"]["sha256"], resumed_spec["training"][key]["sha256"])
        self.assertEqual(job_refs["local.training.resume"]["artifact"]["sha256"],
                         first_refs["checkpoint.last"]["artifact"]["sha256"])
        frozen = json.loads(self.blob(job_refs["local.spec"]))
        self.assertEqual(frozen, resumed_spec)
        replay = self.cli("local", "recover", "--root", self.root, "--id", "resumed-job")
        self.assertEqual(replay["attempts"], resumed["attempts"])
        self.assertTrue(self.cli("run", "verify", "--root", self.root)["valid"])
        self.assertIsNone(first_spec["training"]["resume"])

    def test_typed_unknown_fields_digest_size_and_source_kind_are_rejected_before_admission(self) -> None:
        _, spec = self.prepare("inputs", 20)
        mutations = []
        bad = copy.deepcopy(spec); bad["training"]["unknown"] = True; mutations.append(bad)
        bad = copy.deepcopy(spec); bad["training"]["model_id"] = ""; mutations.append(bad)
        bad = copy.deepcopy(spec); bad["training"]["configuration"]["sha256"] = "0" * 64; mutations.append(bad)
        bad = copy.deepcopy(spec); bad["training"]["data_manifest"]["size_bytes"] += 1; mutations.append(bad)
        bad = copy.deepcopy(spec); bad["training"]["configuration"]["size_bytes"] = 1048577; mutations.append(bad)
        bad = copy.deepcopy(spec); bad["training"]["resume"] = {
            "checkpoint": copy.deepcopy(bad["training"]["configuration"]), "restore_policy": "allow_best_effort"}
        mutations.append(bad)
        bad = copy.deepcopy(spec); bad["training"]["resume"] = {
            "checkpoint": copy.deepcopy(bad["training"]["configuration"]), "restore_policy": "exact_only"}
        mutations.append(bad)  # An artifact relationship is not a checkpoint.
        for index, bad in enumerate(mutations):
            with self.subTest(case=index):
                self.rejected("invalid-input-" + str(index), bad)
        source = spec["training"]["configuration"]["run_id"]
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id=?", (source,))
        self.rejected("nonterminal-source", spec)
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("UPDATE jyppx_runs SET status=6 WHERE run_id=?", (source,))
        self.rejected("interrupted-source", spec)
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("UPDATE jyppx_runs SET status=3,project_id='other-project' WHERE run_id=?", (source,))
        self.rejected("foreign-project", spec)

    def test_corrupt_cas_is_rejected_before_admission(self) -> None:
        _, spec = self.prepare("inputs", 20)
        identity = spec["training"]["configuration"]["sha256"]
        path = self.root / "artifacts" / "objects" / identity[:2] / identity[2:]
        original = path.read_bytes()
        path.write_bytes(b"x" * len(original))
        try:
            self.rejected("corrupt-source", spec)
        finally:
            path.write_bytes(original)

    def test_model_only_checkpoint_is_not_accepted_for_training_resume(self) -> None:
        path, _ = self.prepare("partial", 20)
        state, refs, _ = self.execute("partial-job", path)
        source = state["attempts"][0]["attempt_id"]
        manifest, fields = self.archive(self.blob(refs["checkpoint.last"]))
        manifest["participants"] = [p for p in manifest["participants"]
                                    if p["state"]["participant_id"] == "model.main"]
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        payload = b"JYPPXCK1" + struct.pack("<IQ", 1, len(encoded)) + encoded
        for participant in manifest["participants"]:
            for blob in participant["blobs"]:
                payload += fields[participant["state"]["participant_id"], blob["field_name"]]
        checkpoint = self.seed_checkpoint(source, "model-only", payload + hashlib.sha256(payload).digest())
        _, spec = self.prepare("resume", 60, source)
        spec["training"]["resume"]["checkpoint"] = checkpoint
        self.rejected("model-only-resume", spec, admitted=True)

    def test_unconfirmed_attempt_stays_fenced_despite_a_valid_checkpoint(self) -> None:
        path, _ = self.prepare("partial", 20)
        state, refs, _ = self.execute("partial-job", path)
        source = state["attempts"][0]["attempt_id"]
        checkpoint = self.blob(refs["checkpoint.last"])
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id IN "
                             "('local.stdout','local.stderr','local.result')", (source,))
            database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id IN (?,?)", (source, "partial-job"))
        process = subprocess.run([str(CLI), "local", "recover", "--root", str(self.root),
            "--id", "partial-job", "--format", "json"], capture_output=True, text=True,
            encoding="utf-8", timeout=45, env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
        self.assertEqual(process.returncode, 7, process.stdout + process.stderr)
        data = json.loads(process.stdout)["data"]
        self.assertEqual(data["status"], "recovery_required")
        self.assertEqual(len(data["attempts"]), 1)
        current = {ref["reference_id"]: ref for ref in self.show_run(source)["references"]}
        self.assertEqual(self.blob(current["checkpoint.last"]), checkpoint)
        # Merely relabeling the attempt failed does not establish cleanup.
        # A new job still requires the source's immutable confirmed result.
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("UPDATE jyppx_runs SET status=4 WHERE run_id=?", (source,))
        _, spec = self.prepare("unsafe-resume", 60, source)
        self.rejected("unconfirmed-source-resume", spec)


class RetryCheckpointTrainingExampleTests(unittest.TestCase):
    """Audit real C++ continuation and independent durable-boundary fixtures."""
    setUp = SearchTrainingExampleTests.setUp
    invoke = SearchTrainingExampleTests.invoke
    cli = SearchTrainingExampleTests.cli
    show_run = SearchTrainingExampleTests.show_run
    blob = SearchTrainingExampleTests.blob
    prepare = LocalTrainingExampleTests.prepare
    execute = LocalTrainingExampleTests.execute
    archive = LocalTrainingExampleTests.archive
    seed_checkpoint = LocalTrainingExampleTests.seed_checkpoint

    def auto_spec(self, name: str = "retry") -> tuple[Path, dict]:
        path = self.directory / (name + ".json")
        self.invoke(WORKER, "--write-local-spec", path, "--store", self.root,
                    "--until-step", 60, "--checkpoint-retry-after", 20)
        return path, json.loads(path.read_text(encoding="utf-8"))

    def raw_local(self, action: str, name: str, path: Path | None = None) -> subprocess.CompletedProcess:
        options = ["--spec", str(path)] if path else []
        return subprocess.run([str(CLI), "local", action, "--root", str(self.root),
            "--id", name, *options, "--format", "json"], cwd=self.directory,
            capture_output=True, text=True, encoding="utf-8", timeout=60,
            env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))

    def refs(self, run: str) -> dict:
        return {item["reference_id"]: item for item in self.show_run(run)["references"]}

    def seed_json(self, run: str, reference: str, value: dict, *, replace: bool = False) -> str:
        content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        identity = digest(content)
        target = self.root / "artifacts" / "objects" / identity[:2] / identity[2:]
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(content)
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("PRAGMA foreign_keys=ON")
            database.execute("INSERT OR IGNORE INTO jyppx_blobs(digest,algorithm,size_bytes) VALUES(?,1,?)",
                             (identity, len(content)))
            if replace:
                self.assertEqual(database.execute("UPDATE jyppx_run_artifacts SET digest=? "
                    "WHERE run_id=? AND reference_id=?", (identity, run, reference)).rowcount, 1)
            else:
                stamp = database.execute("SELECT created_at_ms FROM jyppx_runs WHERE run_id=?", (run,)).fetchone()[0]
                database.execute("INSERT INTO jyppx_run_artifacts(run_id,reference_id,kind,role,digest,"
                    "logical_name,media_type,producer,attached_at_ms) VALUES(?,?,1,?,?,?,'application/json',"
                    "'independent-crash-fixture',?)", (run, reference, reference, identity, reference + ".json", stamp))
        return identity

    def alias_checkpoint(self, source: str, *, kind: int = 2) -> None:
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("INSERT INTO jyppx_run_artifacts(run_id,reference_id,kind,role,digest,"
                "logical_name,media_type,producer,attached_at_ms) SELECT run_id,'checkpoint.retry',?,'last',"
                "digest,logical_name,media_type,producer,attached_at_ms FROM jyppx_run_artifacts "
                "WHERE run_id=? AND reference_id='checkpoint.last'", (kind, source))

    def failed_boundary(self) -> tuple[dict, str, dict]:
        # An intentionally absent selected output leaves the real first C++
        # attempt confirmed and failed, before any second attempt is admitted.
        path, spec = self.auto_spec()
        spec["training"]["retry_resume"]["checkpoint_reference_id"] = "checkpoint.retry"
        path.write_text(json.dumps(spec), encoding="utf-8")
        process = self.raw_local("execute", "retry-job", path)
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        state = self.cli("local", "show", "--root", self.root, "--id", "retry-job")
        self.assertEqual(len(state["attempts"]), 1)
        attempt = state["attempts"][0]
        self.assertEqual(attempt["result"]["exit_code"], 23)
        self.assertTrue(attempt["result"]["process_tree_cleanup_confirmed"])
        source = attempt["attempt_id"]
        refs = self.refs(source)
        summary = json.loads(self.blob(refs["training.result"]))
        self.assertEqual((summary["initial_step"], summary["completed_steps"]), (0, 20))
        self.assertNotIn("local.training.inputs.2", self.refs("retry-job"))
        return spec, source, refs

    def assert_no_second_attempt(self) -> None:
        state = self.cli("local", "show", "--root", self.root, "--id", "retry-job")
        self.assertEqual(len(state["attempts"]), 1)
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database:
            self.assertEqual(database.execute("SELECT count(*) FROM jyppx_runs WHERE parent_run_id=?",
                                             ("retry-job",)).fetchone(), (1,))

    def second_manifest(self, spec: dict, source: str) -> dict:
        job_refs = self.refs("retry-job")
        first = json.loads(self.blob(job_refs["local.training.inputs.1"]))
        selected = self.refs(source)["checkpoint.retry"]["artifact"]
        training = copy.deepcopy(spec["training"])
        del training["retry_resume"]
        training["resume"] = {"restore_policy": "exact_only", "checkpoint": {
            "run_id": source, "reference_id": "checkpoint.retry",
            "sha256": selected["sha256"], "size_bytes": selected["size_bytes"]}}
        return {"format": "jyppx.liuding.local_training_inputs", "schema_version": 1,
            "job_id": "retry-job", "attempt_id": source.rsplit("-", 1)[0] + "-2", "index": 2,
            "spec_sha256": first["spec_sha256"],
            "previous_result_sha256": self.refs(source)["local.result"]["artifact"]["sha256"],
            "training": training}

    def test_automatic_twenty_plus_forty_restores_all_native_state_and_resource_receipts(self) -> None:
        path, spec = self.auto_spec()
        spec["resources"] = {"capacity": {"cpu_slots": 2, "memory_bytes": 4096, "gpus": {}},
                             "request": {"cpu_slots": 1, "memory_bytes": 1024, "gpu": None}}
        path.write_text(json.dumps(spec), encoding="utf-8")
        state = self.cli("local", "execute", "--root", self.root, "--id", "retry-job", "--spec", path)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(state["attempts"]), 2)
        first, second = state["attempts"]
        self.assertEqual((first["result"]["exit_code"], second["result"]["exit_code"]), (23, 0))
        self.assertEqual((first["run_status"], second["run_status"]), ("failed", "completed"))
        self.assertTrue(all(item["resource_reserved"] and item["resource_released"] for item in state["attempts"]))
        first_refs, second_refs = self.refs(first["attempt_id"]), self.refs(second["attempt_id"])
        a = json.loads(self.blob(first_refs["training.result"]))
        b = json.loads(self.blob(second_refs["training.result"]))
        self.assertEqual((a["initial_step"], a["completed_steps"], a["resumed"]), (0, 20, False))
        self.assertEqual((b["initial_step"], b["completed_steps"], b["resumed"]), (20, 60, True))
        job_refs = self.refs("retry-job")
        self.assertEqual(json.loads(self.blob(job_refs["local.spec"])), spec)
        self.assertIsNone(spec["training"]["resume"])
        frozen = json.loads(self.blob(job_refs["local.training.inputs.2"]))
        self.assertEqual(frozen["previous_result_sha256"], first_refs["local.result"]["artifact"]["sha256"])
        selected = frozen["training"]["resume"]["checkpoint"]
        self.assertEqual((selected["run_id"], selected["reference_id"], selected["sha256"]),
                         (first["attempt_id"], "checkpoint.last", first_refs["checkpoint.last"]["artifact"]["sha256"]))
        for item, refs in ((first, first_refs), (second, second_refs)):
            expected = job_refs["local.training.inputs." + str(item["index"])]["artifact"]["sha256"]
            self.assertEqual(refs["local.training.inputs"]["artifact"]["sha256"], expected)
            self.assertEqual(item["result"]["training_sha256"], expected)
            self.assertEqual(json.loads(self.blob(refs["local.intent"]))["training_sha256"], expected)
        for name in ("model_id", "configuration", "data_manifest"):
            self.assertEqual(frozen["training"][name], spec["training"][name])
        continuous_path, _ = self.prepare("continuous", 60)
        # The resource pool now requires explicit requests, including the
        # independent control job. The declared bytes are logical accounting.
        continuous_spec = json.loads(continuous_path.read_text(encoding="utf-8"))
        continuous_spec["resources"] = spec["resources"]
        continuous_path.write_text(json.dumps(continuous_spec), encoding="utf-8")
        _, control_refs, control = self.execute("continuous-job", continuous_path)
        _, actual_fields = self.archive(self.blob(second_refs["checkpoint.last"]))
        _, expected_fields = self.archive(self.blob(control_refs["checkpoint.last"]))
        self.assertEqual(actual_fields, expected_fields)
        self.assertEqual({owner for owner, _ in actual_fields}, {"model.main", "optimizer.main", "run.inputs"})
        self.assertEqual((b["weight"], b["bias"]), (control["weight"], control["bias"]))
        replay = self.cli("local", "recover", "--root", self.root, "--id", "retry-job")
        self.assertEqual(replay["attempts"], state["attempts"])
        self.assertTrue(self.cli("run", "verify", "--root", self.root)["valid"])

    def test_missing_selected_checkpoint_never_falls_back_to_restart(self) -> None:
        self.failed_boundary()
        process = self.raw_local("recover", "retry-job")
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assert_no_second_attempt()
        self.assertNotIn("local.training.inputs.2", self.refs("retry-job"))

    def test_checkpoint_kind_and_metadata_quota_reject_before_next_admission(self) -> None:
        _, source, refs = self.failed_boundary()
        self.alias_checkpoint(source, kind=1)
        process = self.raw_local("recover", "retry-job")
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assert_no_second_attempt()
        identity = refs["checkpoint.last"]["artifact"]["sha256"]
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("UPDATE jyppx_run_artifacts SET kind=2 WHERE run_id=? AND reference_id='checkpoint.retry'", (source,))
            database.execute("UPDATE jyppx_blobs SET size_bytes=? WHERE digest=?", ((1 << 30) + 1, identity))
        process = self.raw_local("recover", "retry-job")
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assert_no_second_attempt()
        self.assertNotIn("local.training.inputs.2", self.refs("retry-job"))

    def test_checkpoint_label_with_fake_bytes_fails_native_exact_restore(self) -> None:
        _, source, _ = self.failed_boundary()
        self.seed_checkpoint(source, "checkpoint.retry", b"not-a-native-checkpoint")
        process = self.raw_local("recover", "retry-job")
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        state = json.loads(process.stdout)["data"]
        self.assertEqual((state["status"], len(state["attempts"])), ("failed", 2))
        second = self.refs(state["attempts"][1]["attempt_id"])
        self.assertNotIn("checkpoint.last", second)
        self.assertNotIn("training.result", second)
        self.assertNotEqual(state["attempts"][1]["result"]["exit_code"], 0)

    def test_unconfirmed_cleanup_or_missing_result_never_starts_next_attempt(self) -> None:
        _, source, refs = self.failed_boundary()
        self.alias_checkpoint(source)
        result = json.loads(self.blob(refs["local.result"]))
        result["process_tree_cleanup_confirmed"] = False
        self.seed_json(source, "local.result", result, replace=True)
        process = self.raw_local("recover", "retry-job")
        self.assertEqual(process.returncode, 7, process.stdout + process.stderr)
        self.assertEqual(json.loads(process.stdout)["data"]["status"], "resource_cleanup_unconfirmed")
        self.assert_no_second_attempt()
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id IN "
                             "('local.result','local.stdout','local.stderr')", (source,))
            database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id=?", (source,))
        process = self.raw_local("recover", "retry-job")
        self.assertEqual(process.returncode, 7, process.stdout + process.stderr)
        self.assertEqual(json.loads(process.stdout)["data"]["status"], "recovery_required")
        self.assert_no_second_attempt()

    def test_source_ownership_and_foreign_frozen_checkpoint_are_rejected(self) -> None:
        spec, source, _ = self.failed_boundary()
        self.alias_checkpoint(source)
        original = self.second_manifest(spec, source)
        mutations = []
        bad = copy.deepcopy(original); bad["training"]["resume"]["checkpoint"]["run_id"] = "foreign-attempt"; mutations.append(bad)
        bad = copy.deepcopy(original); bad["training"]["model_id"] = "foreign-model"; mutations.append(bad)
        bad = copy.deepcopy(original); bad["training"]["configuration"]["sha256"] = "0" * 64; mutations.append(bad)
        bad = copy.deepcopy(original); bad["training"]["data_manifest"]["sha256"] = "0" * 64; mutations.append(bad)
        bad = copy.deepcopy(original); bad["previous_result_sha256"] = "0" * 64; mutations.append(bad)
        for index, frozen in enumerate(mutations):
            with self.subTest(identity=index):
                self.seed_json("retry-job", "local.training.inputs.2", frozen, replace=index != 0)
                process = self.raw_local("recover", "retry-job")
                self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assert_no_second_attempt()
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id='retry-job' AND reference_id='local.training.inputs.2'")
            database.execute("UPDATE jyppx_runs SET project_id='foreign-project' WHERE run_id=?", (source,))
        process = self.raw_local("recover", "retry-job")
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assert_no_second_attempt()

    def test_manifest_before_atomic_pins_recovers_same_identity_and_actual_checkpoint(self) -> None:
        spec, source, _ = self.failed_boundary()
        self.alias_checkpoint(source)
        frozen = self.second_manifest(spec, source)
        identity = self.seed_json("retry-job", "local.training.inputs.2", frozen)
        state = self.cli("local", "recover", "--root", self.root, "--id", "retry-job")
        self.assertEqual((state["status"], len(state["attempts"])), ("completed", 2))
        self.assertEqual(self.refs("retry-job")["local.training.inputs.2"]["artifact"]["sha256"], identity)
        second = self.refs(state["attempts"][1]["attempt_id"])
        summary = json.loads(self.blob(second["training.result"]))
        self.assertEqual((summary["initial_step"], summary["completed_steps"], summary["resumed"]), (20, 60, True))

    def test_atomic_pins_before_attempt_creation_are_reused_without_reselection(self) -> None:
        spec, source, _ = self.failed_boundary()
        self.alias_checkpoint(source)
        frozen = self.second_manifest(spec, source)
        identity = self.seed_json("retry-job", "local.training.inputs.2", frozen)
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            stamp = database.execute("SELECT created_at_ms FROM jyppx_runs WHERE run_id='retry-job'").fetchone()[0]
            for origin_run, origin_ref, target in (
                ("retry-job", "local.training.configuration", "local.training.configuration.2"),
                ("retry-job", "local.training.data_manifest", "local.training.data_manifest.2"),
                (source, "checkpoint.retry", "local.training.resume.2")):
                database.execute("INSERT INTO jyppx_run_artifacts(run_id,reference_id,kind,role,digest,logical_name,"
                    "media_type,producer,attached_at_ms) SELECT 'retry-job',?,kind,?,digest,logical_name,media_type,producer,? "
                    "FROM jyppx_run_artifacts WHERE run_id=? AND reference_id=?", (target, target, stamp, origin_run, origin_ref))
        state = self.cli("local", "recover", "--root", self.root, "--id", "retry-job")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(self.refs("retry-job")["local.training.inputs.2"]["artifact"]["sha256"], identity)

    def test_existing_next_attempt_without_result_is_fenced_after_input_commit(self) -> None:
        path, _ = self.auto_spec()
        state = self.cli("local", "execute", "--root", self.root, "--id", "retry-job", "--spec", path)
        second = state["attempts"][1]["attempt_id"]
        original_refs = self.refs(second)
        original_checkpoint = self.blob(original_refs["checkpoint.last"])
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id IN "
                             "('local.result','local.stdout','local.stderr')", (second,))
            database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id IN (?, 'retry-job')", (second,))
        process = self.raw_local("recover", "retry-job")
        self.assertEqual(process.returncode, 7, process.stdout + process.stderr)
        current = json.loads(process.stdout)["data"]
        self.assertEqual((current["status"], len(current["attempts"])), ("recovery_required", 2))
        self.assertEqual(self.blob(self.refs(second)["checkpoint.last"]), original_checkpoint)
        self.assertNotIn("local.result", self.refs(second))

    def test_retry_policy_requires_explicit_exact_bounded_configuration(self) -> None:
        _, spec = self.auto_spec()
        for name, mutate in (
            ("one-attempt", lambda value: value.update(max_attempts=1)),
            ("no-retry-codes", lambda value: value.update(retry_exit_codes=[])),
            ("best-effort", lambda value: value["training"]["retry_resume"].update(restore_policy="allow_best_effort")),
            ("unknown-field", lambda value: value["training"]["retry_resume"].update(latest=True)),
        ):
            with self.subTest(case=name):
                invalid = copy.deepcopy(spec)
                mutate(invalid)
                path = self.directory / (name + ".json")
                path.write_text(json.dumps(invalid), encoding="utf-8")
                process = self.raw_local("execute", name, path)
                self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
                with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database:
                    self.assertEqual(database.execute("SELECT count(*) FROM jyppx_runs WHERE run_id=? OR parent_run_id=?",
                                                     (name, name)).fetchone(), (0,))


    def test_terminal_recover_rejects_missing_or_same_size_corrupt_effective_manifest(self) -> None:
        for failure in ("missing", "same-size-corrupt"):
            with self.subTest(failure=failure):
                self.root = self.directory / ("terminal-manifest-" + failure)
                path, _ = self.auto_spec("terminal-manifest-" + failure)
                state = self.cli("local", "execute", "--root", self.root,
                                 "--id", "retry-job", "--spec", path)
                self.assertEqual((state["status"], len(state["attempts"])), ("completed", 2))
                reference = self.refs("retry-job")["local.training.inputs.2"]
                identity = reference["artifact"]["sha256"]
                target = self.root / "artifacts" / "objects" / identity[:2] / identity[2:]
                original = target.read_bytes()
                with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database:
                    before = database.execute("SELECT run_id,status,updated_at_ms FROM jyppx_runs ORDER BY run_id").fetchall()
                if failure == "missing":
                    target.unlink()
                else:
                    target.write_bytes(b"x" * len(original))
                    self.assertEqual(target.stat().st_size, len(original))
                try:
                    process = self.raw_local("recover", "retry-job")
                    self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
                    diagnostic = "Artifact does not exist" if failure == "missing" else "Artifact failed content-address validation"
                    self.assertIn(diagnostic, process.stdout + process.stderr)
                    with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database:
                        self.assertEqual(database.execute("SELECT run_id,status,updated_at_ms FROM jyppx_runs ORDER BY run_id").fetchall(), before)
                finally:
                    target.write_bytes(original)

    def test_cross_job_cleanup_rejects_missing_or_same_size_corrupt_effective_manifest(self) -> None:
        resources = {"capacity": {"cpu_slots": 2, "memory_bytes": 4096, "gpus": {}},
                     "request": {"cpu_slots": 1, "memory_bytes": 1024, "gpu": None}}
        for failure in ("missing", "same-size-corrupt"):
            with self.subTest(failure=failure):
                self.root = self.directory / ("cleanup-manifest-" + failure)
                path, spec = self.auto_spec("cleanup-manifest-" + failure)
                spec["resources"] = resources
                path.write_text(json.dumps(spec), encoding="utf-8")
                state = self.cli("local", "execute", "--root", self.root,
                                 "--id", "retry-job", "--spec", path)
                self.assertEqual((state["status"], len(state["attempts"])), ("completed", 2))
                second = state["attempts"][1]["attempt_id"]
                admitted_path, admitted_spec = self.prepare("new-admission-" + failure, 60)
                admitted_spec["resources"] = resources
                admitted_path.write_text(json.dumps(admitted_spec), encoding="utf-8")
                reference = self.refs("retry-job")["local.training.inputs.2"]
                identity = reference["artifact"]["sha256"]
                target = self.root / "artifacts" / "objects" / identity[:2] / identity[2:]
                original = target.read_bytes()
                # Reconstruct the reachable crash boundary after attempt 2's
                # atomic result, before its release receipt and lifecycle.
                with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
                    database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id='retry-job' AND reference_id='local.resource.release.2'")
                    database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id IN ('retry-job',?)", (second,))
                    confirmed = database.execute("SELECT reference_id,digest FROM jyppx_run_artifacts WHERE run_id=? ORDER BY reference_id", (second,)).fetchall()
                if failure == "missing":
                    target.unlink()
                else:
                    target.write_bytes(b"x" * len(original))
                    self.assertEqual(target.stat().st_size, len(original))
                try:
                    process = self.raw_local("execute", "new-admission", admitted_path)
                    self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
                    diagnostic = "Artifact does not exist" if failure == "missing" else "Artifact failed content-address validation"
                    self.assertIn(diagnostic, process.stdout + process.stderr)
                    with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database:
                        self.assertEqual(database.execute("SELECT count(*) FROM jyppx_run_artifacts WHERE run_id='retry-job' AND reference_id='local.resource.release.2'").fetchone(), (0,))
                        self.assertEqual(database.execute("SELECT count(*) FROM jyppx_runs WHERE parent_run_id='new-admission'").fetchone(), (0,))
                        self.assertEqual(database.execute("SELECT count(*) FROM jyppx_run_artifacts WHERE run_id='new-admission' AND reference_id='local.resource.lease.1'").fetchone(), (0,))
                        self.assertEqual(database.execute("SELECT reference_id,digest FROM jyppx_run_artifacts WHERE run_id=? ORDER BY reference_id", (second,)).fetchall(), confirmed)
                finally:
                    target.write_bytes(original)


class TypedSearchTrainingExampleTests(unittest.TestCase):
    """Native typed search and explicit continuation, audited without torch."""
    setUp = SearchTrainingExampleTests.setUp
    invoke = SearchTrainingExampleTests.invoke
    cli = SearchTrainingExampleTests.cli
    show_run = SearchTrainingExampleTests.show_run
    blob = SearchTrainingExampleTests.blob
    archive = LocalTrainingExampleTests.archive
    prepare = LocalTrainingExampleTests.prepare
    execute = LocalTrainingExampleTests.execute

    refs = RetryCheckpointTrainingExampleTests.refs
    seed_json = RetryCheckpointTrainingExampleTests.seed_json

    def typed(self, name: str, strategy: str = "grid", resume: str | None = None,
              retry_after: int = 0) -> tuple[Path, dict]:
        path = self.directory / (name + ".json")
        options = ["--resume-run", resume] if resume else []
        if retry_after:
            options += ["--checkpoint-retry-after", str(retry_after)]
        self.invoke(WORKER, "--write-typed-search-spec", path, "--store", self.root,
                    "--strategy", strategy, *options)
        return path, json.loads(path.read_text(encoding="utf-8"))

    def raw_search(self, action: str, name: str, path: Path | None = None) -> subprocess.CompletedProcess:
        options = ["--spec", str(path)] if path else []
        return subprocess.run([str(CLI), "search", action, "--root", str(self.root),
            "--id", name, *options, "--format", "json"], cwd=self.directory,
            capture_output=True, text=True, encoding="utf-8", timeout=45,
            env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))

    def test_automatic_search_retry_matches_continuous_training_and_final_attempt(self) -> None:
        path, _ = self.typed("control")
        control = self.cli("search", "execute", "--root", self.root, "--id", "control", "--spec", path)
        expected = {}
        for row in control["trials"]:
            refs = self.refs(row["training_run_id"])
            expected[row["parameters"]["learning_rate"]] = (
                self.archive(self.blob(refs["checkpoint.last"]))[1], row["metric"]["value"])
            self.assertNotIn("effective_training_inputs", row)
            self.assertNotIn("confirmed_training_attempt", row)
        for strategy in ("grid", "random"):
            with self.subTest(strategy=strategy):
                path, spec = self.typed("retry-" + strategy, strategy, retry_after=20)
                state = self.cli("search", "execute", "--root", self.root,
                                 "--id", strategy, "--spec", path)
                self.assertEqual((state["status"], state["confirmed_trials"], state["parallel_trials"]),
                                 ("completed", 3, 1))
                for row in state["trials"]:
                    self.assertTrue(row["eligible"])
                    self.assertIsNone(row["training_inputs"]["resume"])
                    local = self.cli("local", "show", "--root", self.root, "--id", row["trial_id"])
                    self.assertEqual(len(local["attempts"]), 2)
                    first, final = local["attempts"]
                    self.assertEqual([a["result"]["exit_code"] for a in (first, final)], [23, 0])
                    self.assertTrue(all(a["result"]["process_tree_cleanup_confirmed"] for a in (first, final)))
                    self.assertTrue(final["result"]["training_outputs_confirmed"])
                    first_refs, final_refs = self.refs(first["attempt_id"]), self.refs(final["attempt_id"])
                    before = json.loads(self.blob(first_refs["training.result"]))
                    after = json.loads(self.blob(final_refs["training.result"]))
                    self.assertEqual((before["initial_step"], before["completed_steps"]), (0, 20))
                    self.assertEqual((after["initial_step"], after["completed_steps"], after["resumed"]),
                                     (20, 60, True))
                    evidence = row["confirmed_training_attempt"]
                    self.assertEqual((evidence["attempt_id"], evidence["attempt_index"]), (final["attempt_id"], 2))
                    self.assertEqual(evidence["outputs"], row["outputs"])
                    frozen = json.loads(self.blob(final_refs["local.training.inputs"]))
                    effective = row["effective_training_inputs"]
                    self.assertEqual(effective, frozen["training"])
                    self.assertEqual(evidence["effective_inputs_sha256"], final["result"]["training_sha256"])
                    self.assertEqual(evidence["training_outputs_sha256"],
                                     final_refs["local.training.outputs"]["artifact"]["sha256"])
                    self.assertEqual(frozen["previous_result_sha256"], first_refs["local.result"]["artifact"]["sha256"])
                    self.assertEqual(effective["resume"]["checkpoint"]["run_id"], first["attempt_id"])
                    self.assertEqual(effective["resume"]["checkpoint"]["sha256"],
                                     first_refs["checkpoint.last"]["artifact"]["sha256"])
                    training = self.show_run(row["training_run_id"])
                    self.assertEqual(training["run"]["tags"]["liuding.training.owner_attempt"], final["attempt_id"])
                    refs = {r["reference_id"]: r for r in training["references"]}
                    self.assertEqual(refs["search.training.resume"]["artifact"], first_refs["checkpoint.last"]["artifact"])
                    fields = self.archive(self.blob(refs["checkpoint.last"]))[1]
                    expected_fields, metric = expected[row["parameters"]["learning_rate"]]
                    self.assertEqual(fields, expected_fields)
                    self.assertEqual(row["metric"]["value"], metric)
                    self.assertEqual(row["metric"]["step"], 60)
                    self.assertEqual(json.loads(self.blob(self.refs(row["trial_id"])["local.spec"]))["training"],
                                     row["training_inputs"])
                self.assertEqual(state["best"], min(state["trials"], key=lambda r: r["metric"]["value"]))
                self.assertEqual(self.cli("search", "recover", "--root", self.root, "--id", strategy)["trials"],
                                 state["trials"])
                self.assertTrue(self.cli("run", "verify", "--root", self.root)["valid"])

    def test_search_retry_preserves_explicit_initial_checkpoint(self) -> None:
        path, _ = self.prepare("initial", 20)
        partial, _, _ = self.execute("initial-job", path)
        source = partial["attempts"][0]["attempt_id"]
        path, _ = self.typed("resume-retry", resume=source, retry_after=40)
        state = self.cli("search", "execute", "--root", self.root, "--id", "resume-retry", "--spec", path)
        row = state["best"]
        self.assertIsNotNone(row)
        self.assertEqual(row["training_inputs"]["resume"]["checkpoint"]["run_id"], source)
        local = self.cli("local", "show", "--root", self.root, "--id", row["trial_id"])
        summaries = [json.loads(self.blob(self.refs(a["attempt_id"])["training.result"])) for a in local["attempts"]]
        self.assertEqual([(s["initial_step"], s["completed_steps"]) for s in summaries], [(20, 40), (40, 60)])
        path, _ = self.prepare("continuous-retry-control", 60)
        _, refs, _ = self.execute("continuous-retry-control", path)
        actual = self.archive(self.blob(self.refs(row["training_run_id"])["checkpoint.last"]))[1]
        self.assertEqual(actual, self.archive(self.blob(refs["checkpoint.last"]))[1])

    def test_search_retry_requires_bounded_policy_and_matching_confirmed_outputs(self) -> None:
        path, original = self.typed("invalid-retry", retry_after=20)
        mutations = [lambda s: s["execution"].update(max_attempts=1),
            lambda s: s["execution"].update(retry_exit_codes=[]),
            lambda s: s["execution"]["training"].pop("outputs"),
            lambda s: s["execution"]["training"]["outputs"][0].update(role="foreign"),
            lambda s: s["execution"]["training"]["outputs"][0].update(max_size_bytes=0),
            lambda s: s["execution"]["training"]["retry_resume"].update(restore_policy="allow_best_effort")]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                spec = copy.deepcopy(original)
                mutate(spec)
                path.write_text(json.dumps(spec), encoding="utf-8")
                self.assertEqual(self.raw_search("execute", "invalid-" + str(index), path).returncode, 2)
                with closing(sqlite3.connect(self.root / "runs.sqlite3")) as db:
                    self.assertEqual(db.execute("SELECT count(*) FROM jyppx_runs WHERE run_id=?",
                                                ("invalid-" + str(index),)).fetchone(), (0,))

    def test_search_retry_missing_checkpoint_unknown_result_and_false_cleanup_remain_fenced(self) -> None:
        path, spec = self.typed("fenced-retry", retry_after=20)
        spec["space"][0]["values"] = [0.05]
        spec["max_trials"] = 1
        spec["execution"]["training"]["retry_resume"]["checkpoint_reference_id"] = "missing.checkpoint"
        path.write_text(json.dumps(spec), encoding="utf-8")
        self.assertNotEqual(self.raw_search("execute", "fenced", path).returncode, 0)
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as db:
            job, = db.execute("SELECT run_id FROM jyppx_run_tags WHERE name='liuding.local.kind' AND value='job'").fetchone()
        local = self.cli("local", "show", "--root", self.root, "--id", job)
        self.assertEqual(len(local["attempts"]), 1)
        attempt = local["attempts"][0]["attempt_id"]
        refs = self.refs(attempt)
        original = json.loads(self.blob(refs["local.result"]))
        self.assertEqual(original["exit_code"], 23)
        self.assertTrue(original["process_tree_cleanup_confirmed"])
        self.assertNotIn("local.training.inputs.2", self.refs(job))
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM jyppx_runs WHERE run_id=?", (job + ".training",)).fetchone(), (0,))
        for fault in ("missing", "false_cleanup", "unknown"):
            with self.subTest(fault=fault):
                if fault == "false_cleanup":
                    altered = copy.deepcopy(original)
                    altered["process_tree_cleanup_confirmed"] = False
                    self.seed_json(attempt, "local.result", altered, replace=True)
                elif fault == "unknown":
                    with closing(sqlite3.connect(self.root / "runs.sqlite3")) as db, db:
                        db.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id='local.result'", (attempt,))
                self.assertNotEqual(self.raw_search("recover", "fenced").returncode, 0)
                with closing(sqlite3.connect(self.root / "runs.sqlite3")) as db:
                    self.assertEqual(db.execute("SELECT count(*) FROM jyppx_runs WHERE parent_run_id=? AND run_id<>?",
                                                (job, job + ".training")).fetchone(), (1,))

    def test_search_retry_rejects_earlier_attempt_output_and_owner_without_retraining(self) -> None:
        path, spec = self.typed("substituted-retry", retry_after=20)
        spec["space"][0]["values"] = [0.05]
        spec["max_trials"] = 1
        path.write_text(json.dumps(spec), encoding="utf-8")
        state = self.cli("search", "execute", "--root", self.root, "--id", "substituted", "--spec", path)
        row = state["best"]
        local = self.cli("local", "show", "--root", self.root, "--id", row["trial_id"])
        first, final = local["attempts"]
        original = self.refs(row["training_run_id"])["checkpoint.last"]["artifact"]["sha256"]
        earlier = self.refs(first["attempt_id"])["checkpoint.last"]["artifact"]["sha256"]
        self.assertNotEqual(original, earlier)
        for fault, reason in (("output", "training_output_mismatch"), ("owner", "training_attempt_mismatch")):
            with self.subTest(fault=fault):
                with closing(sqlite3.connect(self.root / "runs.sqlite3")) as db, db:
                    db.execute("DELETE FROM jyppx_run_artifacts WHERE run_id='substituted' AND "
                               "reference_id IN ('search.result.1','search.result','search.best')")
                    db.execute("UPDATE jyppx_runs SET status=2 WHERE run_id='substituted'")
                    db.execute("UPDATE jyppx_run_artifacts SET digest=? WHERE run_id=? AND reference_id='checkpoint.last'",
                               (earlier if fault == "output" else original, row["training_run_id"]))
                    db.execute("UPDATE jyppx_run_tags SET value=? WHERE run_id=? AND name='liuding.training.owner_attempt'",
                               (first["attempt_id"] if fault == "owner" else final["attempt_id"], row["training_run_id"]))
                rejected = self.raw_search("recover", "substituted")
                self.assertEqual(rejected.returncode, 1, rejected.stdout + rejected.stderr)
                result = json.loads(rejected.stdout)["data"]
                self.assertIsNone(result["best"])
                self.assertEqual(result["trials"][0]["reason"], reason)
                after = self.cli("local", "show", "--root", self.root, "--id", row["trial_id"])
                self.assertEqual(after["attempts"], local["attempts"])

    def test_grid_and_random_bind_native_training_to_identical_model_and_data(self) -> None:
        orders = {}
        for strategy in ("grid", "random"):
            with self.subTest(strategy=strategy):
                self.root = self.directory / ("typed-" + strategy)
                path, spec = self.typed(strategy, strategy)
                self.assertEqual(spec["execution"]["training"]["configuration"], "${trial_configuration}")
                result = self.cli("search", "execute", "--root", self.root, "--id", strategy, "--spec", path)
                self.assertEqual((result["status"], result["confirmed_trials"], result["parallel_trials"]),
                                 ("completed", 3, 1))
                orders[strategy] = [row["parameters"]["learning_rate"] for row in result["trials"]]
                self.assertEqual(sorted(orders[strategy]), [0.005, 0.05, 0.2])
                for row in result["trials"]:
                    self.assertTrue(row["eligible"])
                    inputs = row["training_inputs"]
                    self.assertEqual(inputs["model_id"], "example.affine.v1")
                    self.assertEqual(inputs["data_manifest"], spec["execution"]["training"]["data_manifest"])
                    self.assertEqual(inputs["configuration"]["sha256"], row["config_sha256"])
                    training = self.show_run(row["training_run_id"])
                    refs = {ref["reference_id"]: ref for ref in training["references"]}
                    summary = json.loads(self.blob(refs["training.result"]))
                    self.assertEqual((summary["initial_step"], summary["completed_steps"], summary["resumed"]),
                                     (0, 60, False))
                    self.assertEqual(summary["model_id"], inputs["model_id"])
                    self.assertEqual(summary["configuration_sha256"], row["config_sha256"])
                    self.assertEqual(summary["data_manifest_sha256"], inputs["data_manifest"]["sha256"])
                    self.assertEqual(summary["checkpoint_sha256"], refs["checkpoint.last"]["artifact"]["sha256"])
                    self.assertEqual(summary["validation_mse"], row["metric"]["value"])
                    self.assertEqual(set(refs), {"search.training.configuration", "search.training.data_manifest",
                                                 "checkpoint.last", "training.result"})
                    configuration = self.blob(refs["search.training.configuration"])
                    self.assertEqual((digest(configuration), len(configuration)),
                                     (row["config_sha256"], inputs["configuration"]["size_bytes"]))
                    self.assertEqual(json.loads(configuration)["learning_rate"], row["parameters"]["learning_rate"])
                    manifest, fields = self.archive(self.blob(refs["checkpoint.last"]))
                    self.assertEqual({p["state"]["participant_id"] for p in manifest["participants"]},
                                     {"model.main", "optimizer.main", "run.inputs"})
                    identity = json.loads(fields["run.inputs", "identity"])
                    self.assertEqual(identity, {"model_id": inputs["model_id"],
                        "configuration_sha256": row["config_sha256"],
                        "data_manifest_sha256": inputs["data_manifest"]["sha256"]})
                    # Independent scalar recurrence checks actual native momentum SGD.
                    rate = row["parameters"]["learning_rate"]
                    weight, bias, vw, vb = 0.0, 0.0, 0.0, 0.0
                    for _ in range(60):
                        vw = 0.9 * vw + 2 * (weight - 3) * 11 / 31
                        vb = 0.9 * vb + 2 * (bias - 0.5)
                        weight -= rate * vw
                        bias -= rate * vb
                    self.assertAlmostEqual(summary["weight"], weight, delta=4e-5)
                    self.assertAlmostEqual(summary["bias"], bias, delta=4e-5)
                    expected_mse = (weight - 3) ** 2 * 0.33 + (bias - 0.5) ** 2
                    self.assertAlmostEqual(summary["validation_mse"], expected_mse, delta=5e-6)
                best = min(result["trials"], key=lambda row: row["metric"]["value"])
                self.assertEqual(result["best"], best)
                replay = self.cli("search", "recover", "--root", self.root, "--id", strategy)
                self.assertEqual(replay["trials"], result["trials"])
                self.assertTrue(self.cli("run", "verify", "--root", self.root)["valid"])
        self.assertEqual(orders["grid"], [0.005, 0.05, 0.2])

    def test_explicit_checkpoint_continuation_is_exact_and_changed_config_is_rejected(self) -> None:
        path, _ = self.prepare("partial", 20)
        partial, partial_refs, _ = self.execute("partial-job", path)
        source = partial["attempts"][0]["attempt_id"]
        path, spec = self.typed("continued-search", resume=source)
        self.assertEqual(spec["space"][0]["values"], [0.05])
        self.assertEqual(spec["execution"]["training"]["resume"]["checkpoint"]["run_id"], source)
        result = self.cli("search", "execute", "--root", self.root, "--id", "continued", "--spec", path)
        row = result["trials"][0]
        self.assertTrue(row["eligible"])
        refs = {ref["reference_id"]: ref for ref in self.show_run(row["training_run_id"])["references"]}
        summary = json.loads(self.blob(refs["training.result"]))
        self.assertEqual((summary["initial_step"], summary["completed_steps"], summary["resumed"]), (20, 60, True))
        self.assertEqual(refs["search.training.resume"]["artifact"]["sha256"],
                         partial_refs["checkpoint.last"]["artifact"]["sha256"])
        path, _ = self.prepare("continuous", 60)
        _, continuous_refs, _ = self.execute("continuous-job", path)
        _, resumed_fields = self.archive(self.blob(refs["checkpoint.last"]))
        _, continuous_fields = self.archive(self.blob(continuous_refs["checkpoint.last"]))
        self.assertEqual(resumed_fields, continuous_fields)
        path, spec = self.typed("changed-config", resume=source)
        spec["space"][0]["values"] = [0.2]
        path.write_text(json.dumps(spec), encoding="utf-8")
        process = subprocess.run([str(CLI), "search", "execute", "--root", str(self.root),
            "--id", "changed-config", "--spec", str(path), "--format", "json"], cwd=self.directory,
            capture_output=True, text=True, encoding="utf-8", timeout=45,
            env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        rejected = json.loads(process.stdout)["data"]
        self.assertIsNone(rejected["best"])
        self.assertEqual(rejected["trials"][0]["reason"], "execution_failed")
        failed_job = rejected["trials"][0]["trial_id"]
        attempt = self.cli("local", "show", "--root", self.root, "--id", failed_job)["attempts"][0]["attempt_id"]
        failed_refs = {ref["reference_id"] for ref in self.show_run(attempt)["references"]}
        self.assertNotIn("checkpoint.last", failed_refs)
        self.assertNotIn("training.result", failed_refs)


class LocalTrainingOutputContractTests(unittest.TestCase):
    """Python audits a real C++ training/publication fixture; no model runtime."""
    invoke = SearchTrainingExampleTests.invoke
    cli = SearchTrainingExampleTests.cli
    show_run = SearchTrainingExampleTests.show_run
    blob = SearchTrainingExampleTests.blob
    prepare = LocalTrainingExampleTests.prepare
    archive = LocalTrainingExampleTests.archive
    refs = RetryCheckpointTrainingExampleTests.refs
    raw_local = RetryCheckpointTrainingExampleTests.raw_local
    seed_json = RetryCheckpointTrainingExampleTests.seed_json

    def setUp(self) -> None:
        SearchTrainingExampleTests.setUp(self)
        self.assertIsNotNone(OUTPUT_FIXTURE, "Select the native --output-fixture executable")

    @staticmethod
    def declarations() -> list[dict]:
        return [{"reference_id": "checkpoint.last", "kind": "checkpoint", "role": "last", "max_size_bytes": 65536},
                {"reference_id": "training.result", "kind": "artifact", "role": "training.result", "max_size_bytes": 65536}]

    def contract(self, name: str, *, gated: bool = False, legacy: bool = False) -> tuple[Path, dict, Path, Path]:
        path, spec = self.prepare(name, 60)
        log = self.directory / (name + ".launches")
        gate = self.directory / (name + ".gate")
        spec.update(executable=str(OUTPUT_FIXTURE), executable_sha256=digest(OUTPUT_FIXTURE.read_bytes()),
                    arguments=["--launch-log", str(log)])
        if gated:
            spec["arguments"] += ["--gate", str(gate)]
        if not legacy:
            spec["training"]["outputs"] = self.declarations()
        self.write(path, spec)
        return path, spec, log, gate

    @staticmethod
    def write(path: Path, spec: dict) -> None:
        path.write_text(json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    def snapshot(self) -> tuple:
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database:
            return (database.execute("SELECT * FROM jyppx_runs ORDER BY run_id").fetchall(),
                    database.execute("SELECT * FROM jyppx_run_artifacts ORDER BY run_id,reference_id").fetchall())

    def object_path(self, identity: str) -> Path:
        return self.root / "artifacts" / "objects" / identity[:2] / identity[2:]

    def checked(self, name: str, path: Path, expected: int = 0) -> dict:
        process = self.raw_local("execute", name, path)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        return json.loads(process.stdout)["data"]

    def receipt(self, state: dict) -> tuple[str, dict, dict, dict]:
        attempt = state["attempts"][-1]["attempt_id"]
        refs = self.refs(attempt)
        result = json.loads(self.blob(refs["local.result"]))
        encoded = self.blob(refs["local.training.outputs"])
        self.assertLessEqual(len(encoded), 65536)
        self.assertEqual(result["training_outputs_sha256"], digest(encoded))
        return attempt, refs, result, json.loads(encoded)

    def assert_output_failure(self, state: dict, log: Path, reason: str) -> dict:
        self.assertEqual((state["status"], len(state["attempts"])), ("failed", 1))
        attempt, _, result, receipt = self.receipt(state)
        self.assertEqual((result["status"], result["exit_code"], result["process_tree_cleanup_confirmed"]),
                         ("exited", 0, True))
        self.assertFalse(result["training_outputs_confirmed"])
        self.assertFalse(receipt["confirmed"])
        self.assertIn(reason, [row["failure_reason"] for row in receipt["outputs"]])
        self.assertEqual(log.read_text(encoding="utf-8").splitlines(), [attempt])
        return receipt

    def gated(self, name: str, path: Path, gate: Path, mutate) -> subprocess.CompletedProcess:
        command = [str(CLI), "local", "execute", "--root", str(self.root), "--id", name,
                   "--spec", str(path), "--format", "json"]
        with subprocess.Popen(command, cwd=self.directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")) as process:
            ready = Path(str(gate) + ".ready")
            deadline = time.monotonic() + 15
            while not ready.is_file() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            if not ready.is_file():
                stdout, stderr = process.communicate(timeout=60)
                self.fail("Native publication gate was not reached: " + stdout + stderr)
            try:
                mutate(ready.read_text(encoding="utf-8").strip())
            finally:
                gate.write_text("release\n", encoding="ascii")
            stdout, stderr = process.communicate(timeout=60)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    def test_real_native_outputs_bind_bounded_receipt_and_replay_read_only(self) -> None:
        path, spec, log, _ = self.contract("confirmed")
        state = self.checked("confirmed-job", path)
        attempt, refs, result, receipt = self.receipt(state)
        self.assertTrue(receipt["confirmed"])
        self.assertTrue(result["training_outputs_confirmed"])
        self.assertEqual(receipt["format"], "jyppx.liuding.local_training_outputs")
        self.assertEqual((receipt["job_id"], receipt["attempt_id"], receipt["spec_sha256"]),
                         ("confirmed-job", attempt, state["spec_sha256"]))
        training = copy.deepcopy(spec["training"])
        del training["outputs"]
        self.assertEqual(receipt["training"], training)
        for name in ("intent", "owner"):
            self.assertEqual(receipt[name + "_sha256"], refs["local." + name]["artifact"]["sha256"])
        owner = json.loads(self.blob(refs["local.owner"]))
        self.assertTrue(owner["training_outputs_absent_at_start"])
        native = {key: value for key, value in result.items()
                  if key not in ("training_outputs_sha256", "training_outputs_confirmed")}
        self.assertEqual(receipt["native_result_sha256"], digest(json.dumps(native, sort_keys=True,
                         ensure_ascii=False, separators=(",", ":")).encode()))
        for row in receipt["outputs"]:
            self.assertTrue(row["confirmed"])
            reference = refs[row["reference_id"]]
            self.assertEqual((row["sha256"], row["size_bytes"]),
                             (reference["artifact"]["sha256"], len(self.blob(reference))))
        summary = json.loads(self.blob(refs["training.result"]))
        self.assertEqual(summary["completed_steps"], 60)
        weight = bias = vw = vb = 0.0
        for _ in range(60):
            vw = 0.9 * vw + 2 * (weight - 3) * 11 / 31
            vb = 0.9 * vb + 2 * (bias - 0.5)
            weight -= 0.05 * vw
            bias -= 0.05 * vb
        self.assertAlmostEqual(summary["weight"], weight, delta=3e-5)
        self.assertAlmostEqual(summary["bias"], bias, delta=3e-5)
        manifest, _ = self.archive(self.blob(refs["checkpoint.last"]))
        self.assertEqual({p["state"]["participant_id"] for p in manifest["participants"]},
                         {"model.main", "optimizer.main"})
        before = self.snapshot()
        self.assertEqual(self.cli("local", "recover", "--root", self.root, "--id", "confirmed-job"), state)
        self.assertEqual(self.checked("confirmed-job", path), state)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(log.read_text(encoding="utf-8").splitlines(), [attempt])

    def test_missing_wrong_kind_role_and_size_fail_after_real_exit_zero_without_retry(self) -> None:
        for failure in ("missing_reference", "kind_mismatch", "role_mismatch", "size_limit"):
            with self.subTest(failure=failure):
                self.root = self.directory / failure
                path, spec, log, _ = self.contract(failure)
                spec.update(max_attempts=3, retry_exit_codes=[23])
                output = spec["training"]["outputs"][0]
                if failure == "missing_reference":
                    output["reference_id"] = "never-published"
                elif failure == "kind_mismatch":
                    output["kind"] = "artifact"
                elif failure == "role_mismatch":
                    output["role"] = "best"
                else:
                    output["max_size_bytes"] = 1
                self.write(path, spec)
                state = self.checked(failure, path, 1)
                self.assert_output_failure(state, log, failure)
                before = self.snapshot()
                replay = self.raw_local("recover", failure)
                self.assertEqual(replay.returncode, 1, replay.stdout + replay.stderr)
                self.assertEqual(self.snapshot(), before)

    def test_confirmation_rejects_missing_corrupt_and_false_size_actual_cas(self) -> None:
        for failure in ("missing", "same-size-corrupt", "false-size", "oversize-object"):
            with self.subTest(failure=failure):
                self.root = self.directory / ("actual-cas-" + failure)
                path, _, log, gate = self.contract("actual-cas-" + failure, gated=True)
                def mutate(attempt: str) -> None:
                    reference = self.refs(attempt)["checkpoint.last"]
                    artifact = reference["artifact"]
                    target = self.object_path(artifact["sha256"])
                    if failure == "missing":
                        target.unlink()
                    elif failure == "same-size-corrupt":
                        target.write_bytes(b"x" * artifact["size_bytes"])
                        self.assertEqual(target.stat().st_size, artifact["size_bytes"])
                    elif failure == "oversize-object":
                        target.write_bytes(b"x" * 65537)
                    else:
                        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
                            database.execute("UPDATE jyppx_blobs SET size_bytes=? WHERE digest=?",
                                             (artifact["size_bytes"] - 1, artifact["sha256"]))
                process = self.gated(failure, path, gate, mutate)
                self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
                self.assert_output_failure(json.loads(process.stdout)["data"], log, "verification_unavailable")

    def test_another_attempt_same_name_cannot_satisfy_missing_current_output(self) -> None:
        source_path, _, _, _ = self.contract("other-source")
        source = self.checked("other-source", source_path)["attempts"][0]["attempt_id"]
        source_digest = self.refs(source)["checkpoint.last"]["artifact"]["sha256"]
        path, _, log, gate = self.contract("other-current", gated=True)
        def mutate(attempt: str) -> None:
            with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
                self.assertEqual(database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? "
                    "AND reference_id='checkpoint.last'", (attempt,)).rowcount, 1)
        process = self.gated("other-current", path, gate, mutate)
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        self.assert_output_failure(json.loads(process.stdout)["data"], log, "missing_reference")
        self.assertEqual(self.refs(source)["checkpoint.last"]["artifact"]["sha256"], source_digest)

    def test_preexisting_same_attempt_output_fences_before_any_new_native_launch(self) -> None:
        path, _, log, _ = self.contract("preexisting")
        state = self.checked("preexisting", path)
        attempt = state["attempts"][0]["attempt_id"]
        intent = self.refs(attempt)["local.intent"]["artifact"]["sha256"]
        # Reconstruct an admitted intent with outputs injected before owner
        # claim. This is independent negative-state setup, not product repair.
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id IN "
                             "('local.owner','local.result','local.stdout','local.stderr','local.training.outputs')", (attempt,))
            database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id IN (?,?)", ("preexisting", attempt))
        worker = self.cli("local", "worker", "--root", self.root, "--id", "preexisting",
                          "--attempt", 1, "--intent-sha256", intent)
        self.assertTrue(worker["result_committed"])
        refs = self.refs(attempt)
        result = json.loads(self.blob(refs["local.result"]))
        self.assertEqual((result["status"], result["process_id"], result["exit_code"]),
                         ("launch_error", 0, None))
        self.assertTrue(result["process_tree_cleanup_confirmed"])
        self.assertFalse(json.loads(self.blob(refs["local.owner"]))["training_outputs_absent_at_start"])
        self.assertNotIn("local.training.outputs", refs)
        self.assertEqual(log.read_text(encoding="utf-8").splitlines(), [attempt])
        self.assertEqual(self.raw_local("recover", "preexisting").returncode, 1)

    def test_cleanup_false_or_unknown_precedes_missing_receipt_and_starts_nothing(self) -> None:
        for resources in (False, True):
            for terminal in (False, True):
                for cleanup in (False, None):
                    name = f"cleanup-{resources}-{terminal}-{cleanup}"
                    with self.subTest(name=name):
                        self.root = self.directory / name
                        path, spec, log, _ = self.contract(name)
                        if resources:
                            spec["resources"] = {"capacity": {"cpu_slots": 2, "memory_bytes": 4096, "gpus": {}},
                                                 "request": {"cpu_slots": 1, "memory_bytes": 1024, "gpu": None}}
                            self.write(path, spec)
                        state = self.checked(name, path)
                        attempt, _, result, _ = self.receipt(state)
                        if cleanup is None:
                            del result["process_tree_cleanup_confirmed"]
                        else:
                            result["process_tree_cleanup_confirmed"] = cleanup
                        self.seed_json(attempt, "local.result", result, replace=True)
                        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
                            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id='local.training.outputs'", (attempt,))
                            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id='local.resource.release.1'", (name,))
                            if not terminal:
                                database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id IN (?,?)", (name, attempt))
                        before = self.snapshot()
                        replay = self.raw_local("recover", name)
                        self.assertEqual(replay.returncode, 7, replay.stdout + replay.stderr)
                        self.assertEqual(json.loads(replay.stdout)["data"]["status"], "resource_cleanup_unconfirmed")
                        self.assertEqual(self.snapshot(), before)
                        self.assertEqual(log.read_text(encoding="utf-8").splitlines(), [attempt])

    def test_output_limits_and_projected_receipt_reject_before_job_admission(self) -> None:
        for failure in ("empty", "count", "duplicate", "reserved", "zero", "one-limit", "total-limit", "receipt-limit"):
            with self.subTest(failure=failure):
                path, spec, log, _ = self.contract("limit-" + failure)
                outputs = spec["training"]["outputs"]
                if failure == "empty":
                    spec["training"]["outputs"] = []
                elif failure == "count":
                    spec["training"]["outputs"] = [dict(outputs[0], reference_id=f"out-{i}") for i in range(65)]
                elif failure == "duplicate":
                    outputs[1]["reference_id"] = outputs[0]["reference_id"]
                elif failure == "reserved":
                    outputs[0]["reference_id"] = "local.stdout"
                elif failure == "zero":
                    outputs[0]["max_size_bytes"] = 0
                elif failure == "one-limit":
                    outputs[0]["max_size_bytes"] = (1 << 30) + 1
                elif failure == "total-limit":
                    outputs[0]["max_size_bytes"] = 1 << 30
                else:
                    spec["training"]["outputs"] = [dict(outputs[0], reference_id=f"out-{i:02}-" + "x" * 120,
                                                         role="\x01" * 110, max_size_bytes=1) for i in range(64)]
                    self.assertLess(len(json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()), 65536)
                self.write(path, spec)
                process = self.raw_local("execute", "limit-" + failure, path)
                self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
                if failure == "receipt-limit":
                    self.assertIn("receipt would exceed 64 KiB", process.stdout + process.stderr)
                self.assertFalse(log.exists())
                with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database:
                    self.assertEqual(database.execute("SELECT count(*) FROM jyppx_runs WHERE run_id=? OR parent_run_id=?",
                                     ("limit-" + failure, "limit-" + failure)).fetchone(), (0,))

    def test_sixty_four_observations_keep_receipt_below_64_kib(self) -> None:
        path, spec, log, _ = self.contract("receipt-bound")
        spec["training"]["outputs"] = [dict(self.declarations()[0], reference_id=f"missing-{i:02}-" + "x" * 100,
                                              role="\x01" * 90, max_size_bytes=1) for i in range(64)]
        self.write(path, spec)
        state = self.checked("receipt-bound", path, 1)
        receipt = self.assert_output_failure(state, log, "missing_reference")
        self.assertEqual(len(receipt["outputs"]), 64)
        self.assertGreater(len(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()), 48000)

    def test_replay_rejects_missing_corrupt_and_oversized_receipt_without_repair(self) -> None:
        for failure in ("missing", "same-size-corrupt", "oversized", "oversized-object", "identity"):
            with self.subTest(failure=failure):
                self.root = self.directory / ("receipt-replay-" + failure)
                path, _, log, _ = self.contract("receipt-replay-" + failure)
                state = self.checked(failure, path)
                attempt, refs, result, receipt = self.receipt(state)
                if failure == "missing":
                    with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
                        database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=? AND reference_id='local.training.outputs'", (attempt,))
                elif failure == "same-size-corrupt":
                    target = self.object_path(refs["local.training.outputs"]["artifact"]["sha256"])
                    target.write_bytes(b"x" * target.stat().st_size)
                elif failure == "oversized-object":
                    target = self.object_path(refs["local.training.outputs"]["artifact"]["sha256"])
                    target.write_bytes(b"x" * 65537)
                else:
                    if failure == "oversized":
                        receipt["oversized"] = "x" * 65536
                    else:
                        receipt["attempt_id"] = "another-attempt"
                    identity = self.seed_json(attempt, "local.training.outputs", receipt, replace=True)
                    result["training_outputs_sha256"] = identity
                    self.seed_json(attempt, "local.result", result, replace=True)
                before = self.snapshot()
                replay = self.raw_local("recover", failure)
                self.assertNotEqual(replay.returncode, 0, replay.stdout + replay.stderr)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(log.read_text(encoding="utf-8").splitlines(), [attempt])

    def test_replay_rejects_changed_output_binding_or_same_size_corrupt_cas(self) -> None:
        for failure in ("reference-role", "missing-cas", "same-size-corrupt-cas"):
            with self.subTest(failure=failure):
                self.root = self.directory / failure
                path, _, log, _ = self.contract("output-replay-" + failure)
                state = self.checked(failure, path)
                attempt, refs, _, _ = self.receipt(state)
                reference = refs["checkpoint.last"]
                if failure == "reference-role":
                    with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
                        database.execute("UPDATE jyppx_run_artifacts SET role='other' WHERE run_id=? AND reference_id='checkpoint.last'", (attempt,))
                else:
                    target = self.object_path(reference["artifact"]["sha256"])
                    if failure == "missing-cas":
                        target.unlink()
                    else:
                        target.write_bytes(b"x" * target.stat().st_size)
                before = self.snapshot()
                replay = self.raw_local("recover", failure)
                self.assertNotEqual(replay.returncode, 0, replay.stdout + replay.stderr)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(log.read_text(encoding="utf-8").splitlines(), [attempt])

    def test_native_retry_exit_preserves_retry_policy_and_never_confirms_outputs(self) -> None:
        path, spec, log, _ = self.contract("native-retry")
        spec["arguments"] += ["--exit-code", "23"]
        spec.update(max_attempts=2, retry_exit_codes=[23])
        self.write(path, spec)
        state = self.checked("native-retry", path, 1)
        self.assertEqual((state["status"], len(state["attempts"])), ("failed", 2))
        for attempt in state["attempts"]:
            refs = self.refs(attempt["attempt_id"])
            result = json.loads(self.blob(refs["local.result"]))
            self.assertEqual((result["status"], result["exit_code"]), ("exited", 23))
            self.assertNotIn("training_outputs_sha256", result)
            self.assertNotIn("local.training.outputs", refs)
        self.assertEqual(log.read_text(encoding="utf-8").splitlines(),
                         [item["attempt_id"] for item in state["attempts"]])

    def test_confirmed_native_exit_with_rejected_confirmation_never_retries(self) -> None:
        # Independent reconstruction of the crash boundary after an atomic
        # receipt/result commit and before lifecycle, including late cancel.
        for failure in ("cancelled", "confirmation_deadline", "verification_unavailable"):
            with self.subTest(failure=failure):
                self.root = self.directory / ("confirmation-" + failure)
                path, spec, log, _ = self.contract("confirmation-" + failure)
                spec.update(max_attempts=3, retry_exit_codes=[23])
                self.write(path, spec)
                state = self.checked(failure, path)
                attempt, _, result, receipt = self.receipt(state)
                receipt.update(confirmed=False, failure_reason=failure)
                result["training_outputs_confirmed"] = False
                result["training_outputs_sha256"] = self.seed_json(attempt, "local.training.outputs", receipt, replace=True)
                self.seed_json(attempt, "local.result", result, replace=True)
                with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
                    database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id IN (?,?)", (failure, attempt))
                if failure == "cancelled":
                    self.assertTrue(self.cli("local", "cancel", "--root", self.root, "--id", failure)["cancel_requested"])
                replay = self.raw_local("recover", failure)
                expected = 130 if failure == "cancelled" else 1
                self.assertEqual(replay.returncode, expected, replay.stdout + replay.stderr)
                replay_state = json.loads(replay.stdout)["data"]
                self.assertEqual(replay_state["status"], "cancelled" if failure == "cancelled" else "failed")
                self.assertEqual(len(replay_state["attempts"]), 1)
                observed = replay_state["attempts"][0]["result"]
                self.assertEqual((observed["status"], observed["exit_code"]), ("exited", 0))
                self.assertEqual(log.read_text(encoding="utf-8").splitlines(), [attempt])
                before = self.snapshot()
                self.assertEqual(self.raw_local("recover", failure).returncode, expected)
                self.assertEqual(self.snapshot(), before)

    def test_public_example_opt_in_exact_retry_binds_actual_effective_inputs(self) -> None:
        path = self.directory / "example-opt-in.json"
        self.invoke(WORKER, "--write-local-spec", path, "--store", self.root,
                    "--until-step", 60, "--checkpoint-retry-after", 20, "--confirm-outputs", "yes")
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(spec["training"]["outputs"], self.declarations())
        spec["resources"] = {"capacity": {"cpu_slots": 2, "memory_bytes": 4096, "gpus": {}},
                             "request": {"cpu_slots": 1, "memory_bytes": 1024, "gpu": None}}
        self.write(path, spec)
        state = self.checked("example-opt-in", path)
        self.assertEqual((state["status"], len(state["attempts"])), ("completed", 2))
        source = state["attempts"][0]["attempt_id"]
        self.assertNotIn("local.training.outputs", self.refs(source))
        attempt, refs, result, receipt = self.receipt(state)
        self.assertTrue(receipt["confirmed"])
        self.assertEqual(receipt["training"]["resume"], {"restore_policy": "exact_only", "checkpoint": {
            "run_id": source, "reference_id": "checkpoint.last",
            "sha256": self.refs(source)["checkpoint.last"]["artifact"]["sha256"],
            "size_bytes": self.refs(source)["checkpoint.last"]["artifact"]["size_bytes"]}})
        self.assertEqual(result["training_sha256"], refs["local.training.inputs"]["artifact"]["sha256"])
        summary = json.loads(self.blob(refs["training.result"]))
        self.assertEqual((summary["initial_step"], summary["completed_steps"], summary["resumed"]), (20, 60, True))
        job_refs = self.refs("example-opt-in")
        for index in (1, 2):
            self.assertIn("local.resource.release." + str(index), job_refs)
        before = self.snapshot()
        self.assertEqual(self.cli("local", "recover", "--root", self.root, "--id", "example-opt-in"), state)
        self.assertEqual(self.snapshot(), before)

    def test_legacy_terminal_without_opt_in_remains_read_only_and_needs_no_receipt(self) -> None:
        path, _, log, _ = self.contract("legacy-output", legacy=True)
        state = self.checked("legacy-output", path)
        attempt = state["attempts"][0]["attempt_id"]
        refs = self.refs(attempt)
        result = json.loads(self.blob(refs["local.result"]))
        self.assertNotIn("training_outputs_confirmed", result)
        self.assertNotIn("local.training.outputs", refs)
        del result["process_tree_cleanup_confirmed"]
        self.seed_json(attempt, "local.result", result, replace=True)
        before = self.snapshot()
        self.assertEqual(self.raw_local("recover", "legacy-output").returncode, 0)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(log.read_text(encoding="utf-8").splitlines(), [attempt])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--output-fixture", type=Path)
    arguments, remaining = parser.parse_known_args()
    WORKER = arguments.worker.resolve(strict=True)
    CLI = arguments.cli.resolve(strict=True)
    OUTPUT_FIXTURE = arguments.output_fixture.resolve(strict=True) if arguments.output_fixture else None
    unittest.main(argv=[str(__file__), *remaining])
