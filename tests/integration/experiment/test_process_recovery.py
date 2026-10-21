# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Force-kill a real producer and recover its real file-backed outbox.

The child uses production EventOutbox, SQLiteRunTelemetrySink, and
FileCheckpointStore. Only the supervisor fence and JSON evidence are test code.
No test is skipped, and no graceful shutdown is used for the producer.
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
RUN_ID = "run.process-recovery"
METRIC_ID = "training.validation.loss"


class ProducerProcessRecoveryTests(unittest.TestCase):
    child: Path
    work_root: Path
    timeout: float

    def record(self, stage: str, value):
        self.evidence[stage] = value
        self.evidence_path.write_text(
            json.dumps(self.evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return value

    def decode(self, output: bytes) -> dict:
        self.assertLessEqual(len(output), MAXIMUM_REPORT_BYTES)
        self.assertTrue(output.endswith(b"\n"), "Child report was incomplete")
        value = json.loads(output.decode("utf-8"))
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(Path(value["root"]).resolve(), self.root)
        return value

    def complete(self, mode: str, stage: str) -> dict:
        process = subprocess.run(
            [str(self.child), mode, str(self.root)],
            cwd=self.root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
            check=False,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.assertEqual(
            process.returncode, 0,
            f"{mode} failed: {process.stderr.decode('utf-8', errors='replace')}",
        )
        value = self.decode(process.stdout)
        self.assertEqual(value["mode"], mode)
        self.assertEqual(value["phase"], "completed")
        return self.record(stage, value)

    def kill_at_fence(self, mode: str) -> dict:
        """Read one bounded readiness line with a deadline, then force-kill."""
        received: queue.Queue = queue.Queue(maxsize=1)
        with (self.root / "producer-stderr.log").open("wb") as diagnostics:
            process = subprocess.Popen(
                [str(self.child), mode, str(self.root)],
                cwd=self.root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=diagnostics,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

            def read_report() -> None:
                try:
                    received.put(process.stdout.readline(MAXIMUM_REPORT_BYTES + 1))
                except Exception as error:
                    received.put(error)

            reader = threading.Thread(target=read_report, daemon=True)
            reader.start()
            try:
                try:
                    output = received.get(timeout=self.timeout)
                except queue.Empty:
                    self.fail("Producer did not reach its durable fence before deadline")
                if isinstance(output, Exception):
                    raise output
                ready = self.decode(output)
                self.assertEqual(ready["mode"], mode)
                self.assertEqual(ready["phase"], "ready-for-forced-kill")
                self.assertEqual(ready["process_id"], process.pid)
                self.assertIsNone(process.poll(), "Producer exited before forced kill")
                self.assertFalse((self.root / "producer-cleanup.marker").exists())
                self.assert_checkpoint(ready["checkpoint"], "pending.jyck")
                self.record("ready", ready)

                # stdin remains open: neither EOF nor an exception triggers the
                # process boundary. kill() is SIGKILL / Windows TerminateProcess.
                process.kill()
                returncode = process.wait(timeout=self.timeout)
                self.assertNotEqual(returncode, 0)
                if os.name == "posix":
                    self.assertEqual(returncode, -signal.SIGKILL)
                self.assertFalse(
                    (self.root / "producer-cleanup.marker").exists(),
                    "Producer unwound its stack instead of being forcibly killed",
                )
                self.record("forced_kill", {
                    "process_id": process.pid,
                    "returncode": returncode,
                    "mechanism": "SIGKILL" if os.name == "posix" else "TerminateProcess",
                    "cleanup_marker_present": False,
                })
                return ready
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=self.timeout)
                process.stdin.close()
                process.stdout.close()
                reader.join(timeout=self.timeout)

    def assert_checkpoint(self, evidence: dict, filename: str) -> None:
        path = self.root / filename
        self.assertEqual(Path(evidence["path"]).resolve(), path)
        content = path.read_bytes()
        self.assertGreater(len(content), 0)
        self.assertEqual(evidence["bytes"], len(content))
        self.assertEqual(evidence["sha256"], hashlib.sha256(content).hexdigest())

    def assert_observation(self, value: dict) -> None:
        canonical = bytes.fromhex(value["canonical_hex"])
        self.assertEqual(value["canonical_hex"], canonical.hex())
        self.assertEqual(value["canonical_bytes"], len(canonical))
        digest = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(value["canonical_sha256"], digest)
        self.assertEqual(value["content_digest"], digest)
        envelope = value["envelope"]
        self.assertEqual(envelope["schema_version"], 1)
        self.assertEqual(envelope["correlation"], {
            "project_id": "project.process-recovery",
            "run_id": RUN_ID,
            "attempt_id": "attempt.original",
            "trace_id": "trace.process-recovery",
            "span_id": "",
            "parent_span_id": "",
        })
        self.assertTrue(envelope["identity"]["stream_id"])
        self.assertIsInstance(envelope["time"]["wall_time_unix_ns"], int)
        self.assertGreater(envelope["time"]["wall_time_unix_ns"], 0)
        self.assertIsInstance(envelope["time"]["monotonic_time_ns"], int)
        self.assertTrue(envelope["time"]["clock_id"])
        self.assertEqual(envelope["reliability"], 1)
        self.assertEqual(envelope["payload"]["kind"], 1)
        self.assertFalse(envelope["payload"]["terminal"])
        self.assertEqual(envelope["payload"]["decisions"], [{
            "actor_id": "callback.recovery-audit",
            "action": "request_checkpoint",
            "reason": "Retain the original committed decision across process recovery",
            "effective": True,
        }])
        points = envelope["payload"]["metrics"]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["definition"]["metric_id"], METRIC_ID)
        self.assertEqual(points[0]["context"], {"split": "validation"})
        self.assertEqual(points[0]["status"], 1)

    def assert_outbox(self, state: dict, *, pending: int, enqueued: int,
                      delivered: int, stream: str) -> None:
        self.assertEqual(state["stream_id"], stream)
        self.assertEqual(state["pending_events"], pending)
        self.assertEqual(state["last_enqueued"], enqueued)
        self.assertEqual(state["last_delivered"], delivered)
        self.assertFalse(state["closed"])
        if pending:
            self.assertGreater(state["pending_bytes"], 0)
        else:
            self.assertEqual(state["pending_bytes"], 0)

    def assert_store(self, store: dict, observations: list[dict]) -> None:
        count = len(observations)
        self.assertTrue(store["integrity_ok"], store["integrity_issues"])
        self.assertEqual(store["integrity_issues"], [])
        self.assertEqual(store["run_status"], "running")
        self.assertEqual(store["event_snapshot_through"], count)
        self.assertEqual(store["metric_snapshot_through"], count)
        self.assertEqual(store["events"], [
            {"run_sequence": index, "observation": observation}
            for index, observation in enumerate(observations, start=1)
        ])
        expected_metrics = []
        for index, value in enumerate(observations, start=1):
            self.assert_observation(value)
            envelope = value["envelope"]
            expected_metrics.append({
                "run_sequence": index, "point_index": 0,
                "identity": envelope["identity"], "time": envelope["time"],
                "correlation": envelope["correlation"],
                "attributes": envelope["attributes"],
                "point": envelope["payload"]["metrics"][0], "resource_id": None,
            })
        self.assertEqual(store["metrics"], expected_metrics)
        self.assertEqual(Path(store["database_path"]).resolve(),
                         self.root / "store" / "runs.sqlite3")

    def database_facts(self, observations: list[dict], stage: str) -> dict:
        """Use an independent SQLite reader to check authoritative bytes/counts."""
        path = self.root / "store" / "runs.sqlite3"
        self.assertTrue(path.is_file())
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True,
                                     timeout=self.timeout)) as connection:
            connection.row_factory = sqlite3.Row
            self.assertEqual([tuple(row) for row in connection.execute(
                "PRAGMA integrity_check")], [("ok",)])
            self.assertEqual(list(connection.execute("PRAGMA foreign_key_check")), [])

            def rows(sql: str, parameters: tuple = (RUN_ID,)) -> list[dict]:
                return [
                    {key: value.hex() if isinstance(value, bytes) else value
                     for key, value in dict(row).items()}
                    for row in connection.execute(sql, parameters)
                ]

            facts = {
                "events": rows(
                    "SELECT run_sequence, stream_id, producer_sequence, event_id, "
                    "idempotency_key, payload_kind, content_digest, canonical "
                    "FROM jyppx_events WHERE run_id=? ORDER BY run_sequence"),
                "metric_points": rows(
                    "SELECT run_sequence, point_index, metric_id, definition_version, "
                    "step_kind, step_key, value_status FROM jyppx_metric_points "
                    "WHERE run_id=? ORDER BY run_sequence, point_index"),
                "streams": rows(
                    "SELECT stream_id, last_sequence, terminal FROM jyppx_event_streams "
                    "WHERE run_id=? ORDER BY stream_id"),
                "batches": rows(
                    "SELECT batch_key, request_digest, input_count, committed_through "
                    "FROM jyppx_event_batches WHERE run_id=? ORDER BY batch_key"),
                "batch_items": rows(
                    "SELECT batch_key, ordinal, run_sequence FROM jyppx_event_batch_items "
                    "WHERE run_id=? ORDER BY batch_key, ordinal"),
                "state": rows(
                    "SELECT last_sequence, metric_count, content_bytes, batch_count, "
                    "stream_count FROM jyppx_run_event_state WHERE run_id=?"),
                "definitions": rows(
                    "SELECT metric_id, definition_version, definition_digest, canonical "
                    "FROM jyppx_metric_definitions ORDER BY metric_id, definition_version", ()),
            }
        count = len(observations)
        self.assertEqual(len(facts["events"]), count)
        self.assertEqual(len(facts["metric_points"]), count)
        self.assertEqual(len(facts["batches"]), count)
        self.assertEqual(len(facts["batch_items"]), count)
        self.assertEqual(len(facts["definitions"]), int(count > 0))
        self.assertEqual(len(facts["streams"]), int(count > 0))
        for index, (stored, value) in enumerate(zip(facts["events"], observations), start=1):
            identity = value["envelope"]["identity"]
            self.assertEqual(stored, {
                "run_sequence": index, "stream_id": identity["stream_id"],
                "producer_sequence": identity["sequence"],
                "event_id": identity["event_id"],
                "idempotency_key": identity["idempotency_key"],
                "payload_kind": 1, "content_digest": value["content_digest"],
                "canonical": value["canonical_hex"],
            })
        if count:
            self.assertEqual(facts["streams"], [{
                "stream_id": observations[0]["envelope"]["identity"]["stream_id"],
                "last_sequence": count, "terminal": 0,
            }])
            self.assertEqual(facts["state"], [{
                "last_sequence": count, "metric_count": count,
                "content_bytes": sum(value["canonical_bytes"] for value in observations),
                "batch_count": count, "stream_count": 1,
            }])
        return self.record(stage, facts)

    def scenario(self, name: str, committed_before_kill: bool) -> None:
        self.root = (self.work_root / name).resolve()
        self.root.mkdir()
        self.evidence = {"schema_version": 1, "scenario": name}
        self.evidence_path = self.root / "evidence.json"
        mode = "prepare-ack-loss" if committed_before_kill else "prepare-pending"
        ready = self.kill_at_fence(mode)
        original = ready["observation"]
        self.assert_observation(original)
        envelope = original["envelope"]
        stream = envelope["identity"]["stream_id"]
        self.assertEqual(envelope["identity"]["sequence"], 1)
        self.assertEqual(envelope["payload"]["name"], "training.run_started")
        self.assertEqual(envelope["payload"]["metrics"][0]["value"], 0.25)
        self.assertEqual(ready["internal_dispositions"], [1] if committed_before_kill else [])
        self.assert_outbox(ready["outbox"], pending=1, enqueued=1, delivered=0, stream=stream)

        preexisting = [original] if committed_before_kill else []
        self.assert_store(ready["store"], preexisting)
        inspected = self.complete("inspect", "reopened_after_kill")
        self.assert_store(inspected["store"], preexisting)
        initial_facts = self.database_facts(preexisting, "database_after_kill")

        recovered = self.complete("recover", "recovered")
        self.assert_checkpoint(recovered["source_checkpoint"], "pending.jyck")
        self.assert_checkpoint(recovered["saved_checkpoint"], "delivered.jyck")
        self.assertEqual(recovered["source_checkpoint"], ready["checkpoint"])
        self.assertNotEqual(recovered["created_stream_id"], stream)
        self.assert_outbox(recovered["restored"], pending=1, enqueued=1,
                           delivered=0, stream=stream)
        self.assertEqual(recovered["restored"]["pending_bytes"], ready["outbox"]["pending_bytes"])
        self.assert_outbox(recovered["after"], pending=0, enqueued=1,
                           delivered=1, stream=stream)
        self.assertTrue(recovered["closed"])
        self.assertEqual(recovered["emissions"], [{
            "observation": original,
            "dispositions": [2] if committed_before_kill else [1],
        }])
        self.assertEqual(recovered["exported_events"], int(not committed_before_kill))
        self.assertEqual(recovered["unsupported_events"], 0)
        self.assertEqual(recovered["sampled_out_events"], 0)
        self.assert_store(recovered["store"], [original])
        recovered_facts = self.database_facts([original], "database_after_recovery")
        if committed_before_kill:
            self.assertEqual(recovered_facts, initial_facts)

        # Reopen the same stale pending checkpoint in yet another process.
        replayed = self.complete("recover", "replayed_original_checkpoint")
        self.assertEqual(replayed["source_checkpoint"], ready["checkpoint"])
        self.assertEqual(replayed["emissions"], [{
            "observation": original, "dispositions": [2],
        }])
        self.assert_outbox(replayed["after"], pending=0, enqueued=1,
                           delivered=1, stream=stream)
        self.assertEqual(replayed["exported_events"], 0)
        self.assert_store(replayed["store"], [original])
        self.assertEqual(self.database_facts([original], "database_after_replay"), recovered_facts)
        self.assert_checkpoint(ready["checkpoint"], "pending.jyck")

        advanced = self.complete("advance", "continued_in_new_process")
        self.assert_checkpoint(advanced["source_checkpoint"], "delivered.jyck")
        self.assertEqual(advanced["source_checkpoint"], replayed["saved_checkpoint"])
        self.assertEqual(len(advanced["emissions"]), 1)
        self.assertEqual(advanced["emissions"][0]["dispositions"], [1])
        following = advanced["emissions"][0]["observation"]
        self.assert_observation(following)
        second = following["envelope"]
        self.assertEqual(second["identity"]["stream_id"], stream)
        self.assertEqual(second["identity"]["sequence"], 2)
        self.assertNotEqual(second["identity"]["event_id"], envelope["identity"]["event_id"])
        self.assertNotEqual(second["identity"]["idempotency_key"],
                            envelope["identity"]["idempotency_key"])
        self.assertNotEqual(second["time"]["clock_id"], envelope["time"]["clock_id"])
        self.assertEqual(second["payload"]["name"], "training.epoch_finished")
        self.assertEqual(second["payload"]["metrics"][0]["value"], 0.125)
        self.assertEqual(second["payload"]["metrics"][0]["step"]["value"], 1)
        self.assert_outbox(advanced["restored"], pending=0, enqueued=1,
                           delivered=1, stream=stream)
        self.assert_outbox(advanced["after_enqueue"], pending=1, enqueued=2,
                           delivered=1, stream=stream)
        self.assert_outbox(advanced["after"], pending=0, enqueued=2,
                           delivered=2, stream=stream)
        self.assertEqual(advanced["exported_events"], 1)
        self.assert_store(advanced["store"], [original, following])
        advanced_facts = self.database_facts([original, following], "database_after_continue")
        self.assert_checkpoint(advanced["saved_checkpoint"], "advanced.jyck")

        final = self.complete("verify-final", "restored_delivered_cursors")
        self.assertEqual(final["source_checkpoint"], advanced["saved_checkpoint"])
        self.assert_outbox(final["restored"], pending=0, enqueued=2,
                           delivered=2, stream=stream)
        self.assertEqual(final["emissions"], [])
        self.assertEqual(final["exported_events"], 0)
        self.assert_store(final["store"], [original, following])
        self.assertEqual(self.database_facts([original, following], "database_final"), advanced_facts)
        self.assert_checkpoint(ready["checkpoint"], "pending.jyck")
        self.assertFalse((self.root / "producer-cleanup.marker").exists())
        self.record("valid", True)

    def test_pending_event_survives_forced_producer_death(self) -> None:
        self.scenario("pending-before-acceptance", committed_before_kill=False)

    def test_lost_ack_replays_unchanged_without_duplicate_facts(self) -> None:
        self.scenario("committed-before-ack", committed_before_kill=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, required=True,
                        help="Already built recovery_child executable")
    parser.add_argument("--work-dir", type=Path,
                        help="Keep unique fixture/evidence directories beneath this path")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Deadline in seconds for each child operation")
    arguments = parser.parse_args()
    child = arguments.child.resolve()
    if not child.is_file():
        parser.error("--child must name an existing executable")
    if not 0 < arguments.timeout <= 300:
        parser.error("--timeout must be positive and no greater than 300 seconds")

    temporary = None
    if arguments.work_dir:
        arguments.work_dir.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="producer-recovery-数据-",
                                     dir=arguments.work_dir.resolve()))
    else:
        temporary = tempfile.TemporaryDirectory(prefix="jyppx-producer-recovery-")
        root = Path(temporary.name) / "process 数据"
        root.mkdir()
    ProducerProcessRecoveryTests.child = child
    ProducerProcessRecoveryTests.work_root = root.resolve()
    ProducerProcessRecoveryTests.timeout = arguments.timeout
    try:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProducerProcessRecoveryTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        print(json.dumps({
            "valid": result.wasSuccessful(), "tests_run": result.testsRun,
            "skipped": len(result.skipped), "work_root": str(root),
            "evidence_retained": temporary is None,
        }, ensure_ascii=True))
        return 0 if result.wasSuccessful() and result.testsRun == 2 and not result.skipped else 1
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
