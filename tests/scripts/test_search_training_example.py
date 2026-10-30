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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    WORKER = arguments.worker.resolve(strict=True)
    CLI = arguments.cli.resolve(strict=True)
    unittest.main(argv=[str(__file__), *remaining])
