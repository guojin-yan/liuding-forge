# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

"""Kill actual publication processes and audit durable intent/SQL/file states."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import sqlite3
import struct
import subprocess
import tempfile
import threading
import unittest

SIZE = 2 * 1024 * 1024 + 257
BOUND = 65536
PAYLOAD = bytes((i * 37 + 11) & 255 for i in range(SIZE))
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def native_path(path):
    if os.name != "nt":
        return path
    value = str(path.absolute())
    if value.startswith("\\\\?\\"):
        return path
    if value.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + value[2:])
    return Path("\\\\?\\" + value)


class FilePublicationProcessTests(unittest.TestCase):
    def setup_case(self, name):
        self.root = Path(tempfile.mkdtemp(prefix=name + "-", dir=self.work_root))
        self.records = {"schema_version": 1, "test": self.id(), "case": name}
        self.target = self.root / "result.bin"
        self.object = native_path(self.root / "store/artifacts/objects" / DIGEST[:2] / DIGEST[2:])
        self.complete("seed", "seed")

    def record(self, name, value):
        self.records[name] = value
        (self.root / "evidence.json").write_text(
            json.dumps(self.records, indent=2) + "\n", encoding="utf-8")
        return value

    def decode(self, raw):
        self.assertLessEqual(len(raw), BOUND)
        self.assertTrue(raw.endswith(b"\n"))
        value = json.loads(raw)
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
        if mode != "gc":
            self.assertTrue(value["integrity_ok"])
        return self.record(name, value)

    def file_identity(self, path):
        path = native_path(path)
        stat = path.stat()
        with path.open("rb") as stream:
            hash_value = hashlib.sha256()
            for block in iter(lambda: stream.read(65536), b""):
                hash_value.update(block)
        return {"bytes": stat.st_size, "sha256": hash_value.hexdigest(),
                "file_id": stat.st_ino, "volume": stat.st_dev, "mtime_ns": stat.st_mtime_ns}

    def journal(self, connection):
        row = connection.execute(
            "SELECT digest,logical_name,media_type,producer,attached_at_ms "
            "FROM jyppx_run_artifacts WHERE reference_id='publication-1'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[1:], ("file-publication-v1",
                                   "application/vnd.jyppx.file-publication-v1",
                                   "experiment.file-publication", 101))
        path = native_path(self.root / "store/artifacts/objects" / row[0][:2] / row[0][2:])
        data = path.read_bytes()
        self.assertLessEqual(len(data), 4 << 20)
        self.assertEqual(hashlib.sha256(data).hexdigest(), row[0])
        offset = 0

        def number():
            nonlocal offset
            self.assertLessEqual(offset + 8, len(data))
            result, = struct.unpack_from("<Q", data, offset)
            offset += 8
            return result

        def field():
            nonlocal offset
            count = number()
            self.assertLessEqual(count, len(data) - offset)
            result = data[offset:offset + count]
            offset += count
            return result

        self.assertEqual(field(), b"jyppx.file-publication/1")
        self.assertEqual(field(), b"publication-1")
        destination = field().decode("utf-8")
        self.assertEqual(Path(destination), self.target.resolve())
        self.assertEqual(field(), b"result-1")
        self.assertEqual(number(), 101)
        self.assertEqual(field(), b"result-batch-1")
        event = field()
        self.assertEqual(offset, len(data))
        self.assertGreater(len(event), 0)
        return path, event

    def sql(self, prepared, committed, name, writer_blocked=False):
        database = self.root / "store/runs.sqlite3"
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True,
                                     timeout=0.2)) as connection:
            counts = [connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                      for table in ("jyppx_run_artifacts", "jyppx_events", "jyppx_blobs")]
            self.assertEqual(counts, [2 * int(prepared) + 2 * int(committed),
                                      int(committed), 2 * int(prepared)])
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchall(), [("ok",)])
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            if prepared:
                _, event = self.journal(connection)
                pinned = connection.execute(
                    "SELECT digest FROM jyppx_run_artifacts WHERE role='file.publication.content'").fetchall()
                self.assertEqual(pinned, [(DIGEST,)])
                if committed:
                    row = connection.execute(
                        "SELECT digest,role,logical_name,media_type,producer,attached_at_ms "
                        "FROM jyppx_run_artifacts WHERE reference_id='result-1'").fetchone()
                    self.assertEqual(row, (DIGEST, "image.preview", "result.bin",
                                           "application/octet-stream", "file-publication-process-test", 101))
                    event_row = connection.execute(
                        "SELECT stream_id,producer_sequence,event_id,idempotency_key,canonical,content_digest "
                        "FROM jyppx_events").fetchone()
                    self.assertEqual(event_row, ("output-1", 1, "result-event-1", "result-key-1",
                                                 event, hashlib.sha256(event).hexdigest()))
            if writer_blocked:
                # BEGIN IMMEDIATE is also the coordinated GC admission. Probe
                # the real writer lock without launching a second native job.
                with closing(sqlite3.connect(database, timeout=0.1)) as writer:
                    with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                        writer.execute("BEGIN IMMEDIATE")
        self.record(name, {"references": counts[0], "events": counts[1],
                            "blob_rows": counts[2], "writer_blocked": writer_blocked,
                            "independent_integrity_ok": True})

    def kill_at(self, mode, boundary, prepared, file_exists, committed=False):
        marker = self.root / "unwound.marker"
        marker.unlink(missing_ok=True)
        received = queue.Queue(maxsize=1)
        with (self.root / (boundary + ".stderr.log")).open("wb") as errors:
            child = subprocess.Popen([str(self.child), mode, str(self.root), boundary],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=errors, shell=False,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            reader = threading.Thread(target=lambda: received.put(child.stdout.readline(BOUND + 1)), daemon=True)
            reader.start()
            try:
                ready = self.decode(received.get(timeout=30))
                self.assertEqual(ready["phase"], "ready-for-forced-kill")
                self.assertEqual(ready["boundary"], boundary)
                self.assertEqual(ready["process_id"], child.pid)
                self.assertEqual(ready["file_exists"], file_exists)
                self.assertIsNone(child.poll())
                self.assertFalse(marker.exists())
                locked = boundary not in ("after-prepare", "after-complete")
                self.sql(prepared, committed, boundary + "-live-sql", locked)
                self.record(boundary + "-ready", ready)
                child.kill()
                code = child.wait(timeout=30)
                self.assertNotEqual(code, 0)
                if os.name == "posix":
                    self.assertEqual(code, -signal.SIGKILL)
                self.assertFalse(marker.exists())
                self.record(boundary + "-kill", {"returncode": code, "stack_unwound": False})
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=30)
                child.stdin.close()
                child.stdout.close()
                reader.join(timeout=30)
        self.sql(prepared, committed, boundary + "-after-kill")

    def test_uncommitted_preparation_is_not_fabricated_after_source_loss(self):
        for boundary in ("before-payload-cas", "after-payload-cas", "prepare-before-commit"):
            with self.subTest(boundary=boundary):
                self.setup_case(boundary)
                self.kill_at("prepare-fence", boundary, False, False)
                (self.root / "input.bin").unlink()
                result = self.complete("recover", "no-prepared-intent")
                self.assertFalse(result["recovered"])
                self.assertFalse(self.target.exists())
                self.sql(False, False, "final-sql")
                self.complete("gc", "unreferenced-cas-collected")
                self.assertFalse(self.object.exists())
                self.record("valid", True)

    def test_prepared_unreceived_receipt_recovers_using_discovered_intent(self):
        self.setup_case("prepared")
        self.kill_at("prepare-fence", "after-prepare", True, False)
        (self.root / "input.bin").unlink()
        identity = self.file_identity(self.object)
        self.assertEqual(self.complete("gc", "pending-content-retained")["gc_removed"], 0)
        result = self.complete("recover", "completed")
        self.assertTrue(result["recovered"])
        self.assertFalse(result["reference_replay"])
        self.assertFalse(result["event_replay"])
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        self.sql(True, True, "committed-sql")
        self.assertEqual(self.file_identity(self.object), identity)
        self.record("valid", True)

    def test_file_and_database_crashes_recover_without_replacing_published_file(self):
        for boundary in ("before-file-publication", "after-file-publication", "recovery-before-commit"):
            with self.subTest(boundary=boundary):
                self.setup_case(boundary)
                self.complete("prepare", "prepared")
                (self.root / "input.bin").unlink()
                cas = self.file_identity(self.object)
                exists = boundary != "before-file-publication"
                self.kill_at("recover-fence", boundary, True, exists)
                target = self.file_identity(self.target) if exists else None
                self.assertEqual(self.complete("gc", "pending-pins-retained")["gc_removed"], 0)
                result = self.complete("recover", "completed-after-kill")
                self.assertTrue(result["recovered"])
                self.assertFalse(result["reference_replay"])
                self.assertFalse(result["event_replay"])
                self.assertEqual(self.target.read_bytes(), PAYLOAD)
                if target:
                    self.assertEqual(self.file_identity(self.target), target)
                self.assertEqual(self.file_identity(self.object), cas)
                self.assertFalse(list(self.root.glob(".liuding-publication-*.stage")))
                self.sql(True, True, "committed-sql")
                replay = self.complete("recover", "replayed-in-new-process")
                self.assertTrue(replay["reference_replay"])
                self.assertTrue(replay["event_replay"])
                self.sql(True, True, "replayed-sql")
                self.record("valid", True)

    def test_committed_unreceived_receipt_is_replayed_exactly_once(self):
        self.setup_case("committed")
        self.complete("prepare", "prepared")
        (self.root / "input.bin").unlink()
        self.kill_at("recover-fence", "after-complete", True, True, True)
        target = self.file_identity(self.target)
        result = self.complete("recover", "replayed-after-unreceived-receipt")
        self.assertTrue(result["reference_replay"])
        self.assertTrue(result["event_replay"])
        self.assertEqual(self.file_identity(self.target), target)
        self.sql(True, True, "final-sql")
        self.record("valid", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    child = args.child.resolve()
    if not child.is_file():
        parser.error("--child must identify the built native executable")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="file-publication-数据-", dir=args.work_dir.resolve()))
    FilePublicationProcessTests.child = child
    FilePublicationProcessTests.work_root = root
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(FilePublicationProcessTests))
    passed = result.wasSuccessful() and result.testsRun == 4 and not result.skipped
    print(json.dumps({"valid": passed, "tests_run": result.testsRun,
                      "skipped": len(result.skipped), "work_root": str(root),
                      "forced_kill_boundaries": 8, "evidence_retained": True}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
