# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independently inspect diagnostic bytes and SQLite after real forced kills."""
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

BOUND = 65536
BOUNDARIES = ("before-cas", "after-cas", "before-reference-commit",
              "after-reference-before-ack")


class DiagnosticRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="diagnostic-", dir=self.work_root))
        self.evidence = {"test": self.id(), "fixture_is_training_result": False}
        self.complete("seed")
        self.checkpoint = self.identity(self.root / "diagnostics.jyck")
        self.expected = (self.root / "expected.json").read_bytes()
        self.digest = hashlib.sha256(self.expected).hexdigest()
        self.object = self.root / "artifacts" / "objects" / self.digest[:2] / self.digest[2:]
        report = json.loads(self.expected)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["outcome"], "failed")
        self.assertEqual(report["identity"], {
            "report_id": "diagnostics-1", "run_id": "run-1", "attempt_id": "attempt-1",
            "model": {"algorithm": "sha256", "hex": "a" * 64},
            "configuration": {"algorithm": "sha256", "hex": "b" * 64},
            "dataset": {"algorithm": "sha256", "hex": "c" * 64},
            "checkpoint": {"algorithm": "sha256", "hex": self.checkpoint["sha256"]}})
        self.assertEqual(report["loss"]["training"],
                         {"state": "measured", "observations": 2**53 + 17, "nan": 2, "infinity": 1})
        self.assertEqual(report["loss"]["validation"],
                         {"state": "measured", "observations": 11, "nan": 1, "infinity": 2})
        gradient = report["gradient"]
        for key, expected in {"update_attempts": 9, "nonfinite_attempts": 3,
                              "amp_skipped_attempts": 2, "state": "measured",
                              "samples": 3, "elements": 12, "last_l2": 4,
                              "mean_l2": 3, "max_l2": 4, "scan_host_seconds": .125,
                              "known_peak_scratch_bytes": 48}.items():
            self.assertEqual(gradient[key], expected)
        for role, batches, seconds in (("training", 7, .75), ("validation", 3, .5)):
            self.assertEqual(report["data_wait"][role],
                             {"state": "measured", "batches": batches,
                              "host_seconds": seconds, "max_host_seconds": .25})
        self.record("report", {"sha256": self.digest, "bytes": len(self.expected),
                               "checkpoint": self.checkpoint, "independent_schema_checked": True})

    def record(self, key, value):
        self.evidence[key] = value
        (self.root / "evidence.json").write_text(
            json.dumps(self.evidence, indent=2) + "\n", encoding="utf-8")
        return value

    def identity(self, path):
        value = path.stat()
        return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": value.st_size, "mtime_ns": value.st_mtime_ns,
                "device": value.st_dev, "file_id": value.st_ino}

    def complete(self, mode, success=True):
        value = subprocess.run([str(self.child), mode, str(self.root)],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=20, shell=False,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertLessEqual(len(value.stdout) + len(value.stderr), BOUND)
        if not success:
            self.assertEqual(value.returncode, 1, value.stderr.decode(errors="replace"))
            self.assertEqual(value.stdout, b"")
            return self.record("rejected-" + mode, {"exit": value.returncode,
                                                    "stderr": value.stderr.decode()})
        self.assertEqual(value.returncode, 0, value.stderr.decode(errors="replace"))
        report = json.loads(value.stdout)
        self.assertEqual(report["phase"], "completed")
        self.assertEqual(report["mode"], mode)
        self.assertTrue(report["integrity_ok"])
        return self.record(mode + "-" + str(len(self.evidence)), report)

    def sql(self, references):
        path = (self.root / "runs.sqlite3").resolve()
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)) as db:
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchall(), [("ok",)])
            self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jyppx_run_artifacts").fetchone(), (references,))
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jyppx_blobs").fetchone(), (references,))
            if references:
                row = db.execute(
                    "SELECT reference_id,role,digest,logical_name,media_type,producer,attached_at_ms "
                    "FROM jyppx_run_artifacts").fetchone()
                self.assertEqual(row, ("diagnostics-1", "training.diagnostics", self.digest,
                                       "training-diagnostics.json",
                                       "application/vnd.jyppx.training-diagnostics+json;version=1",
                                       "training.diagnostics/v1", 20))
                self.assertEqual(db.execute("SELECT size_bytes FROM jyppx_blobs").fetchone(),
                                 (len(self.expected),))

    def kill_at(self, boundary):
        received = queue.Queue(maxsize=1)
        with (self.root / "crash.stderr").open("wb") as errors:
            child = subprocess.Popen([str(self.child), "crash", str(self.root), boundary],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=errors, shell=False,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            reader = threading.Thread(target=lambda: received.put(child.stdout.readline(BOUND + 1)), daemon=True)
            reader.start()
            try:
                line = received.get(timeout=20)
                self.assertLessEqual(len(line), BOUND)
                ready = json.loads(line)
                self.assertEqual(ready, {"phase": "ready-for-forced-kill", "boundary": boundary,
                                         "process_id": child.pid,
                                         "native_calls": int(boundary != "before-cas")})
                self.assertIsNone(child.poll())
                self.sql(int(boundary == "after-reference-before-ack"))
                self.assertEqual(self.object.exists(), boundary != "before-cas")
                self.assertFalse((self.root / "cleanup.marker").exists())
                child.kill()
                code = child.wait(timeout=10)
                self.assertNotEqual(code, 0)
                if os.name == "posix":
                    self.assertEqual(code, -signal.SIGKILL)
                self.assertFalse((self.root / "cleanup.marker").exists())
                self.record("forced-kill", {**ready, "returncode": code, "stack_unwound": False})
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
                child.stdin.close()
                child.stdout.close()
                reader.join(timeout=10)
        self.sql(int(boundary == "after-reference-before-ack"))
        self.assertEqual(self.identity(self.root / "diagnostics.jyck"), self.checkpoint)

    def recover(self, boundary):
        self.kill_at(boundary)
        original = self.identity(self.object) if self.object.exists() else None
        self.complete("replay")
        self.sql(1)
        self.assertEqual(self.object.read_bytes(), self.expected)
        if original:
            self.assertEqual(self.identity(self.object), original)
        before = self.identity(self.object)
        self.complete("replay")
        self.complete("conflict")
        self.sql(1)
        self.complete("gc")
        self.sql(1)
        self.assertEqual(self.identity(self.object), before)
        self.assertEqual(self.object.read_bytes(), self.expected)
        self.assertEqual(self.identity(self.root / "diagnostics.jyck"), self.checkpoint)
        objects = [p for p in (self.root / "artifacts" / "objects").rglob("*") if p.is_file()]
        self.assertEqual(objects, [self.object])
        self.record("valid", True)

    def test_before_cas(self):
        self.recover("before-cas")

    def test_after_cas(self):
        self.recover("after-cas")

    def test_before_reference_commit(self):
        self.recover("before-reference-commit")

    def test_receipt_lost_after_reference_commit(self):
        self.recover("after-reference-before-ack")

    def rejected_checkpoint(self, corrupt):
        self.kill_at("after-cas")
        before = self.identity(self.object)
        path = self.root / "diagnostics.jyck"
        if corrupt:
            path.write_bytes(b"not a checkpoint")
        else:
            path.unlink()
        self.complete("replay", success=False)
        self.sql(0)
        self.assertEqual(self.identity(self.object), before)
        self.assertEqual(self.object.read_bytes(), self.expected)
        self.record("valid", True)

    def test_missing_checkpoint_is_not_reconstructed_from_report(self):
        self.rejected_checkpoint(False)

    def test_corrupt_checkpoint_cannot_attach_report(self):
        self.rejected_checkpoint(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    child = args.child.resolve()
    if not child.is_file():
        parser.error("--child must name a native test executable")
    temporary = None
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="diagnostic-recovery-", dir=args.work_dir.resolve()))
    else:
        temporary = tempfile.TemporaryDirectory(prefix="diagnostic-recovery-")
        root = Path(temporary.name)
    DiagnosticRecoveryTests.child = child
    DiagnosticRecoveryTests.work_root = root
    try:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticRecoveryTests))
        passed = result.wasSuccessful() and result.testsRun == 6 and not result.skipped
        print(json.dumps({"passed": passed, "tests_run": result.testsRun,
                          "skipped": len(result.skipped), "work_root": str(root)}))
        return 0 if passed else 1
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
