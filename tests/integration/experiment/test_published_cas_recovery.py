# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

"""Kill real publication/recovery; inspect SQL and unchanged CAS independently."""
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


SIZE = 2 * 1024 * 1024 + 257
BOUND = 65536


class PublishedCasRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="cas-evidence-", dir=self.work_root))
        self.evidence = {"schema_version": 1, "test": self.id()}
        self.digest = hashlib.sha256(bytes((i * 37 + 11) & 255 for i in range(SIZE))).hexdigest()
        self.object = self.root / "artifacts" / "objects" / self.digest[:2] / self.digest[2:]

    def record(self, name, value):
        self.evidence[name] = value
        (self.root / "evidence.json").write_text(
            json.dumps(self.evidence, indent=2) + "\n", encoding="utf-8")
        return value

    def decode(self, output):
        self.assertLessEqual(len(output), BOUND)
        self.assertTrue(output.endswith(b"\n"))
        value = json.loads(output)
        self.assertEqual(value["schema_version"], 1)
        return value

    def complete(self, mode, name):
        child = subprocess.run([str(self.child), mode, str(self.root)],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=False, timeout=30,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(child.returncode, 0, child.stderr.decode(errors="replace"))
        value = self.decode(child.stdout)
        self.assertEqual(value["phase"], "completed")
        self.assertEqual(value["mode"], mode)
        if mode != "gc-busy":
            self.assertTrue(value["integrity_ok"])
            self.assertEqual(value["sha256"], self.digest)
            self.assertEqual(value["bytes"], SIZE)
        return self.record(name, value)

    def sql(self, references, events, stage):
        path = (self.root / "runs.sqlite3").resolve()
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)) as connection:
            counts = [connection.execute("SELECT COUNT(*) FROM " + name).fetchone()[0]
                      for name in ("jyppx_run_artifacts", "jyppx_events", "jyppx_blobs")]
            self.assertEqual(counts, [references, events, int(references > 0)])
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchall(), [("ok",)])
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            if references:
                row = connection.execute(
                    "SELECT role,digest,logical_name,media_type,producer,attached_at_ms "
                    "FROM jyppx_run_artifacts WHERE reference_id='published-1'").fetchone()
                self.assertEqual(row, ("output.file", self.digest, "output.bin",
                                       "application/octet-stream", "published-cas-recovery-test", 101))
                self.assertEqual(connection.execute(
                    "SELECT size_bytes FROM jyppx_blobs WHERE digest=?", (self.digest,)
                ).fetchone(), (SIZE,))
        self.record(stage, {"references": references, "events": events,
                            "blob_rows": counts[2], "independent_integrity_ok": True})

    def identity(self):
        observed = self.object.stat()
        with self.object.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()
        self.assertEqual(digest, self.digest)
        self.assertEqual(observed.st_size, SIZE)
        return {"volume": observed.st_dev, "file_id": observed.st_ino,
                "mtime_ns": observed.st_mtime_ns, "bytes": observed.st_size,
                "links": observed.st_nlink, "sha256": digest}

    def kill_at(self, mode, boundary, name):
        received = queue.Queue(maxsize=1)
        with (self.root / (name + ".stderr.log")).open("wb") as errors:
            child = subprocess.Popen([str(self.child), mode, str(self.root), boundary],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=errors, shell=False,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            reader = threading.Thread(target=lambda: received.put(child.stdout.readline(BOUND + 1)), daemon=True)
            reader.start()
            try:
                ready = self.decode(received.get(timeout=30))
                self.assertEqual(ready["phase"], "ready-for-forced-kill")
                self.assertEqual(ready["process_id"], child.pid)
                self.assertEqual(ready["boundary"], boundary)
                self.assertEqual(ready["cas_exists"], boundary != "before-cas")
                self.assertEqual(ready["native_calls"], int(boundary == "after-cas"))
                self.assertIsNone(child.poll())
                self.assertFalse((self.root / "cleanup.marker").exists())
                self.sql(0, 0, name + "-uncommitted-sql")
                self.assertTrue(self.complete("gc-busy", name + "-gc-blocked")["writer_blocked"])
                self.assertIsNone(child.poll())
                self.record(name + "-ready", ready)
                child.kill()
                code = child.wait(timeout=30)
                self.assertNotEqual(code, 0)
                if os.name == "posix":
                    self.assertEqual(code, -signal.SIGKILL)
                self.assertFalse((self.root / "cleanup.marker").exists())
                self.record(name + "-kill", {"returncode": code, "stack_unwound": False})
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=30)
                child.stdin.close()
                child.stdout.close()
                reader.join(timeout=30)
        self.sql(0, 0, name + "-rolled-back-sql")

    def test_missing_unpublished_cas_is_not_fabricated(self):
        self.complete("seed", "seed")
        self.kill_at("crash", "before-cas", "publication")
        self.assertFalse(self.object.exists())
        (self.root / "output.bin").unlink()
        result = self.complete("recover", "recovery-refused")
        self.assertFalse(result["recovered"])
        self.assertEqual((result["references"], result["events"]), (0, 0))
        self.sql(0, 0, "final-sql")
        self.assertFalse(self.object.exists())
        self.assertFalse((self.root / "output.bin").exists())
        self.record("valid", True)

    def test_complete_cas_recovers_after_source_and_recovery_process_are_lost(self):
        self.complete("seed", "seed")
        self.kill_at("crash", "after-cas", "publication")
        before = self.record("cas-after-original-kill", self.identity())
        (self.root / "output.bin").unlink()
        self.kill_at("recover-fence", "recovery-before-commit", "recovery")
        self.assertEqual(self.identity(), before)
        first = self.complete("recover", "first-completed-recovery")
        self.assertTrue(first["recovered"])
        self.assertFalse(first["reference_replay"])
        self.assertFalse(first["event_replay"])
        self.sql(2, 1, "recovered-sql")
        replay = self.complete("recover", "replay-in-new-process")
        self.assertTrue(replay["recovered"])
        self.assertTrue(replay["reference_replay"])
        self.assertTrue(replay["event_replay"])
        self.assertEqual(self.complete("gc", "live-gc")["gc_removed"], 0)
        self.sql(2, 1, "final-sql")
        self.assertEqual(self.record("cas-after-replay-and-gc", self.identity()), before)
        self.assertFalse((self.root / "output.bin").exists())
        self.record("valid", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    child = args.child.resolve()
    if not child.is_file():
        parser.error("--child must name the built static-archive test executable")
    temporary = None
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="published-cas-数据-", dir=args.work_dir.resolve()))
    else:
        temporary = tempfile.TemporaryDirectory(prefix="published-cas-")
        root = Path(temporary.name)
    PublishedCasRecoveryTests.child = child
    PublishedCasRecoveryTests.work_root = root
    try:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PublishedCasRecoveryTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        passed = result.wasSuccessful() and result.testsRun == 2 and not result.skipped
        print(json.dumps({"valid": passed, "tests_run": result.testsRun,
                          "skipped": len(result.skipped), "work_root": str(root),
                          "evidence_retained": temporary is None}))
        return 0 if passed else 1
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
