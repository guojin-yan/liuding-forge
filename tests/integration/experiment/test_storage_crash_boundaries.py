# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Actual production SQLite operations interrupted at official commit hooks.

Four operations, two boundaries each. The migration fixture is v1 input; the
operation under test is SQLiteRunStore's actual constructor migration. There
are no synthetic replacement writes, graceful crashes, optional dependencies,
or skipped cases. The child is an already built test executable.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import sqlite3
import subprocess
import tempfile
import threading
import unittest


MAXIMUM_REPORT_BYTES = 256 * 1024
RUN_ID = "run.storage-crash"
EVENT_TABLES = (
    "jyppx_run_event_state", "jyppx_event_streams", "jyppx_events",
    "jyppx_event_batches", "jyppx_event_batch_items", "jyppx_metric_definitions",
    "jyppx_metric_points",
)
LEGACY_TABLES = ("jyppx_runs", "jyppx_run_tags", "jyppx_blobs", "jyppx_run_artifacts")


class StorageCrashBoundaryTests(unittest.TestCase):
    child: Path
    work_root: Path
    timeout: float

    def record(self, stage: str, value):
        self.evidence[stage] = value
        (self.root / "evidence.json").write_text(
            json.dumps(self.evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return value

    def decode(self, output: bytes) -> dict:
        self.assertLessEqual(len(output), MAXIMUM_REPORT_BYTES)
        self.assertTrue(output.endswith(b"\n"), "Storage child report was incomplete")
        value = json.loads(output.decode("utf-8"))
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(value["operation"], self.operation)
        return value

    def complete(self, mode: str, stage: str) -> dict:
        process = subprocess.run(
            [str(self.child), mode, str(self.root), self.operation],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self.root, timeout=self.timeout, check=False, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.assertEqual(process.returncode, 0,
                         f"{mode}: {process.stderr.decode('utf-8', errors='replace')}")
        result = self.decode(process.stdout)
        self.assertEqual(result["phase"], "seeded" if mode == "seed" else "completed")
        self.assertTrue(result["store"]["integrity_ok"], result["store"]["issues"])
        return self.record(stage, result)

    def force_kill(self, boundary: str) -> dict:
        received: queue.Queue = queue.Queue(maxsize=1)
        with (self.root / "crash-stderr.log").open("wb") as diagnostics:
            child = subprocess.Popen(
                [str(self.child), "crash", str(self.root), self.operation, boundary],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostics,
                cwd=self.root, shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

            def read_line() -> None:
                try:
                    received.put(child.stdout.readline(MAXIMUM_REPORT_BYTES + 1))
                except Exception as error:
                    received.put(error)

            reader = threading.Thread(target=read_line, daemon=True)
            reader.start()
            try:
                try:
                    line = received.get(timeout=self.timeout)
                except queue.Empty:
                    self.fail("Production operation did not reach its SQLite hook")
                if isinstance(line, Exception):
                    raise line
                ready = self.decode(line)
                self.assertEqual(ready["phase"], "ready-for-forced-kill")
                self.assertEqual(ready["boundary"], boundary)
                self.assertEqual(ready["hook"],
                                 "sqlite3_commit_hook" if boundary == "before" else "sqlite3_wal_hook")
                self.assertEqual(ready["process_id"], child.pid)
                self.assertEqual(Path(ready["database_path"]).resolve(),
                                 self.root / "store" / "runs.sqlite3")
                self.assertIsNone(child.poll())
                self.assertFalse((self.root / "storage-cleanup.marker").exists())
                statements = "\n".join(ready["statements"])
                self.assertIn("COMMIT", statements)
                expected = {
                    "event": "INSERT INTO jyppx_events",
                    "artifact": "INSERT INTO jyppx_run_artifacts",
                    "finalize": "UPDATE jyppx_runs SET status",
                    "migration": "CREATE TABLE jyppx_events",
                }
                self.assertIn(expected[self.operation], statements)
                if self.operation == "migration":
                    self.assertIn("PRAGMA synchronous = FULL", statements)
                    self.assertIn("PRAGMA user_version = 2", statements)
                    self.assertIn("BEGIN IMMEDIATE", statements)
                else:
                    self.assertTrue(ready["verified_wal_full"])
                self.record("ready", ready)
                # Keep stdin open and kill while the real callback/transaction
                # is on the stack. No destructor or rollback helper runs.
                child.kill()
                code = child.wait(timeout=self.timeout)
                self.assertNotEqual(code, 0)
                if os.name == "posix":
                    self.assertEqual(code, -signal.SIGKILL)
                self.assertFalse((self.root / "storage-cleanup.marker").exists())
                self.record("forced_kill", {"returncode": code, "cleanup_marker": False})
                return ready
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=self.timeout)
                child.stdin.close()
                child.stdout.close()
                reader.join(timeout=self.timeout)

    def facts(self, stage: str) -> dict:
        path = self.root / "store" / "runs.sqlite3"
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True,
                                     timeout=self.timeout)) as connection:
            connection.row_factory = sqlite3.Row
            self.assertEqual([tuple(row) for row in connection.execute("PRAGMA integrity_check")],
                             [("ok",)])
            self.assertEqual(list(connection.execute("PRAGMA foreign_key_check")), [])
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            application = connection.execute("PRAGMA application_id").fetchone()[0]
            journal = connection.execute("PRAGMA journal_mode").fetchone()[0]
            schema = [dict(row) for row in connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE name NOT GLOB 'sqlite_*' ORDER BY type, name")]
            present = {entry["name"] for entry in schema if entry["type"] == "table"}
            tables = {}
            for table in (*LEGACY_TABLES, *EVENT_TABLES):
                tables[table] = None if table not in present else [
                    {key: value.hex() if isinstance(value, bytes) else value
                     for key, value in dict(row).items()}
                    for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY 1, 2')
                ]
            for event in tables["jyppx_events"] or []:
                self.assertEqual(hashlib.sha256(bytes.fromhex(event["canonical"])).hexdigest(),
                                 event["content_digest"])
        self.assertEqual(application, 1247367760)
        self.assertEqual(journal, "wal")
        objects = {}
        for path in (self.root / "store" / "artifacts" / "objects").glob("*/*"):
            if path.is_file():
                content = path.read_bytes()
                digest = path.parent.name + path.name
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)
                objects[digest] = content.hex()
        temporary = sorted(
            path.name for path in (self.root / "store" / "artifacts" / "tmp").iterdir()
            if path.is_file()
        )
        self.assertEqual(temporary, [], "SQLite commit fence must follow completed CAS publication")
        return self.record(stage, {
            "version": version, "application": application, "schema": schema,
            "tables": tables, "objects": objects, "temporary_files": temporary,
        })

    def assert_baseline_preserved(self, initial: dict, actual: dict) -> None:
        before, after = initial["tables"], actual["tables"]
        self.assertEqual(after["jyppx_run_tags"], before["jyppx_run_tags"])
        original_run = before["jyppx_runs"][0]
        recovered_run = after["jyppx_runs"][0]
        for key, value in original_run.items():
            if key not in ("status", "updated_at_ms"):
                self.assertEqual(recovered_run[key], value)
        self.assertTrue(all(row in after["jyppx_run_artifacts"]
                            for row in before["jyppx_run_artifacts"]))
        self.assertTrue(all(row in after["jyppx_blobs"] for row in before["jyppx_blobs"]))
        for digest, content in initial["objects"].items():
            self.assertEqual(actual["objects"][digest], content)
        if before["jyppx_events"]:
            self.assertEqual(after["jyppx_events"][:1], before["jyppx_events"])
            self.assertTrue(all(row in after["jyppx_metric_points"]
                                for row in before["jyppx_metric_points"]))

    def assert_final(self, initial: dict, final: dict, seeded: dict) -> None:
        self.assert_baseline_preserved(initial, final)
        self.assertEqual(final["version"], 2)
        tables = final["tables"]
        self.assertTrue(all(tables[name] is not None for name in EVENT_TABLES))
        run = tables["jyppx_runs"][0]
        self.assertEqual(run["status"], 3 if self.operation == "finalize" else 2)
        self.assertEqual(run["updated_at_ms"],
                         300 if self.operation == "finalize" else 200 if self.operation == "artifact" else 110)
        count = 0 if self.operation == "migration" else 3 if self.operation == "event" else 1
        self.assertEqual(len(tables["jyppx_events"]), count)
        self.assertEqual(len(tables["jyppx_metric_points"]), count)
        if self.operation == "event":
            self.assertEqual([row["run_sequence"] for row in tables["jyppx_events"]], [1, 2, 3])
            self.assertEqual([row["producer_sequence"] for row in tables["jyppx_events"]], [1, 2, 3])
            self.assertEqual(len(tables["jyppx_event_batches"]), 2)
            self.assertEqual(len(tables["jyppx_event_batch_items"]), 3)
            self.assertEqual(tables["jyppx_run_event_state"][0]["last_sequence"], 3)
        references = tables["jyppx_run_artifacts"]
        self.assertEqual(len(references), 2 if self.operation == "artifact" else 1)
        expected_contents = [seeded["confirmed_content"]]
        if self.operation == "artifact":
            expected_contents.append(seeded["target_content"])
        self.assertEqual(final["objects"], {
            hashlib.sha256(content.encode("utf-8")).hexdigest(): content.encode("utf-8").hex()
            for content in expected_contents
        })

    def scenario(self, operation: str, boundary: str) -> None:
        self.operation = operation
        self.root = (self.work_root / f"{operation}-{boundary}").resolve()
        self.root.mkdir()
        self.evidence = {"schema_version": 1, "operation": operation, "boundary": boundary}
        seeded = self.complete("seed", "seeded")
        initial = self.facts("initial_database")
        self.assertEqual(initial["version"], 1 if operation == "migration" else 2)
        self.force_kill(boundary)
        crashed = self.facts("database_after_kill")
        self.assert_baseline_preserved(initial, crashed)
        if boundary == "before":
            self.assertEqual(crashed["version"], initial["version"])
            self.assertEqual(crashed["schema"], initial["schema"])
            self.assertEqual(crashed["tables"], initial["tables"],
                             "An uncommitted real operation leaked partial rows or DDL")
        else:
            self.assert_final(initial, crashed, seeded)

        # put_artifact publishes CAS before its DB relation commits. Recover
        # the legitimate orphan through production GC, preserving confirmed CAS.
        if operation == "artifact":
            target_digest = hashlib.sha256(seeded["target_content"].encode("utf-8")).hexdigest()
            self.assertIn(target_digest, crashed["objects"])
            gc = self.complete("gc", "gc_after_kill")
            self.assertEqual(gc["gc"]["removed_files"], int(boundary == "before"))
            after_gc = self.facts("database_after_gc")
            self.assert_baseline_preserved(initial, after_gc)
            if boundary == "before":
                self.assertEqual(after_gc, initial)
            else:
                self.assertEqual(after_gc, crashed)
        elif operation != "migration":
            inspected = self.complete("inspect", "production_reopened_after_kill")
            self.assertEqual(inspected["store"]["snapshot_through"],
                             3 if operation == "event" and boundary == "after" else 1)

        retried = self.complete("retry", "retried_real_operation")
        final = self.facts("database_after_retry")
        self.assert_final(initial, final, seeded)
        if boundary == "after":
            self.assertEqual(final, crashed)
        if operation == "event":
            self.assertEqual(retried["receipt"]["committed_through"], 3)
            self.assertEqual(retried["receipt"]["batch_replay"], boundary == "after")
            self.assertEqual(retried["receipt"]["receipts"], [
                {"run_sequence": sequence, "disposition": 2 if boundary == "after" else 1}
                for sequence in (2, 3)
            ])
        replayed = self.complete("retry", "idempotent_replay")
        self.assertEqual(self.facts("database_after_replay"), final)
        if operation == "event":
            self.assertTrue(replayed["receipt"]["batch_replay"])
            self.assertEqual([item["disposition"] for item in replayed["receipt"]["receipts"]], [2, 2])
        self.complete("inspect", "fresh_production_integrity")
        self.assertEqual(self.facts("final_database"), final)
        self.assertFalse((self.root / "storage-cleanup.marker").exists())
        self.record("valid", True)

    def test_real_event_batch_commit_boundaries(self) -> None:
        for boundary in ("before", "after"):
            with self.subTest(boundary=boundary):
                self.scenario("event", boundary)

    def test_real_artifact_relationship_commit_boundaries(self) -> None:
        for boundary in ("before", "after"):
            with self.subTest(boundary=boundary):
                self.scenario("artifact", boundary)

    def test_real_finalize_commit_boundaries(self) -> None:
        for boundary in ("before", "after"):
            with self.subTest(boundary=boundary):
                self.scenario("finalize", boundary)

    def test_real_migration_commit_boundaries(self) -> None:
        for boundary in ("before", "after"):
            with self.subTest(boundary=boundary):
                self.scenario("migration", boundary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    arguments = parser.parse_args()
    child = arguments.child.resolve()
    if not child.is_file():
        parser.error("--child must name an already built executable")
    if not 0 < arguments.timeout <= 300:
        parser.error("--timeout must be positive and at most 300 seconds")
    temporary = None
    if arguments.work_dir:
        arguments.work_dir.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="storage-crash-数据-", dir=arguments.work_dir.resolve()))
    else:
        temporary = tempfile.TemporaryDirectory(prefix="jyppx-storage-crash-")
        root = Path(temporary.name) / "storage 数据"
        root.mkdir()
    StorageCrashBoundaryTests.child = child
    StorageCrashBoundaryTests.work_root = root.resolve()
    StorageCrashBoundaryTests.timeout = arguments.timeout
    try:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(StorageCrashBoundaryTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        print(json.dumps({"valid": result.wasSuccessful(), "tests_run": result.testsRun,
                          "scenarios": 8, "skipped": len(result.skipped),
                          "work_root": str(root), "evidence_retained": temporary is None}))
        return 0 if result.wasSuccessful() and result.testsRun == 4 and not result.skipped else 1
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
