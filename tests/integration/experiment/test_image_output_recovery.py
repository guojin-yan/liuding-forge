# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

"""Kill actual C++ publication syscalls and replay original evidence identities."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import queue
import sqlite3
import subprocess
import tempfile
import threading
import unittest


class ImageOutputRecoveryTests(unittest.TestCase):
    def _run(self, root, boundary="none"):
        completed = subprocess.run(
            [str(self.child), str(root), boundary], input=b"",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
        result = json.loads(completed.stdout)
        self.assertEqual((result["references"], result["events"]), (2, 1))
        self.assertTrue(result["integrity_ok"])
        self.assertEqual((result["queue_bytes"], result["spool_bytes"]), (0, 0))
        return result

    def _counts(self, root):
        with closing(sqlite3.connect(root / "store/runs.sqlite3")) as database:
            return (
                database.execute("SELECT count(*) FROM jyppx_run_artifacts").fetchone()[0],
                database.execute("SELECT count(*) FROM jyppx_events").fetchone()[0],
            )

    def _kill_at(self, root, boundary, while_waiting=None):
        process = subprocess.Popen(
            [str(self.child), str(root), boundary], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
        )
        lines = queue.Queue()
        reader = threading.Thread(
            target=lambda: lines.put(process.stdout.readline(4097)), daemon=True
        )
        reader.start()
        try:
            ready = json.loads(lines.get(timeout=20))
            self.assertEqual((ready["phase"], ready["boundary"]), ("ready", boundary))
            descriptor = Path(f"/proc/{process.pid}/fd/{ready['descriptor']}")
            observed = descriptor.stat()
            content = descriptor.read_bytes()
            self.assertEqual((observed.st_dev, observed.st_ino, observed.st_nlink, observed.st_size),
                             (ready["device"], ready["inode"], ready["links"], ready["bytes"]))
            ready["content_sha256"] = hashlib.sha256(content).hexdigest()
            ready["named_staging"] = sorted(path.name for path in root.glob("*.liuding-tmp-*"))
            if while_waiting is not None:
                while_waiting()
            self.assertIsNone(process.poll(), "child must still own the paused inode")
            process.kill()
            process.wait(timeout=10)
            self.assertEqual(process.returncode, -9)
            self.assertFalse(descriptor.exists())
            print("native-kill " + json.dumps(ready, sort_keys=True), flush=True)
            return ready
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            reader.join(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()

    def test_unlinked_staging_is_released_after_process_kill(self):
        for boundary in ("staging-created", "staging-writing", "staging-synced", "staging-linked"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory(
                prefix="output-staging-", dir=self.work_root
            ) as directory:
                root = Path(directory)
                ready = self._kill_at(root, boundary)
                linked = boundary == "staging-linked"
                self.assertEqual(ready["links"], int(linked))
                self.assertEqual(ready["named_staging"], [])
                self.assertEqual(sorted(path.name for path in root.iterdir()),
                                 ["preview.png", "store"] if linked else ["store"])
                self.assertEqual(self._counts(root), (0, 0))
                if boundary == "staging-created":
                    self.assertEqual(ready["bytes"], 0)
                elif boundary == "staging-writing":
                    self.assertEqual(ready["bytes"], 13)
                else:
                    self.assertGreater(ready["bytes"], 13)
                recovered = self._run(root)
                output = root / "preview.png"
                published = output.stat()
                if linked:
                    self.assertEqual((published.st_dev, published.st_ino),
                                     (ready["device"], ready["inode"]))
                    self.assertEqual(recovered["digest"], ready["content_sha256"])
                self.assertFalse(recovered["reference_replay"])
                self.assertFalse(recovered["event_replay"])
                repeated = self._run(root)
                self.assertTrue(repeated["reference_replay"])
                self.assertTrue(repeated["event_replay"])
                current = output.stat()
                self.assertEqual((current.st_dev, current.st_ino, current.st_mtime_ns),
                                 (published.st_dev, published.st_ino, published.st_mtime_ns))
                self.assertEqual(repeated["digest"], recovered["digest"])
                self.assertEqual(sorted(path.name for path in root.iterdir()), ["preview.png", "store"])

    def test_native_write_flush_link_and_close_errors_keep_recoverable_state(self):
        for boundary in ("fail-open", "fail-write", "fail-sync", "fail-link", "fail-close", "fail-directory"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory(
                prefix="output-syscall-", dir=self.work_root
            ) as directory:
                root = Path(directory)
                failed = subprocess.run(
                    [str(self.child), str(root), boundary], input=b"",
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, shell=False,
                )
                self.assertEqual(failed.returncode, 1, failed.stdout.decode(errors="replace"))
                self.assertIn(b"publication failed", failed.stderr)
                linked = boundary in ("fail-close", "fail-directory")
                self.assertEqual(sorted(path.name for path in root.iterdir()),
                                 ["preview.png", "store"] if linked else ["store"])
                self.assertEqual(self._counts(root), (0, 0))
                original = (root / "preview.png").stat() if linked else None
                recovered = self._run(root)
                self.assertFalse(recovered["reference_replay"])
                self.assertFalse(recovered["event_replay"])
                if original is not None:
                    current = (root / "preview.png").stat()
                    self.assertEqual((current.st_dev, current.st_ino, current.st_mtime_ns),
                                     (original.st_dev, original.st_ino, original.st_mtime_ns))

    def test_unsupported_filesystem_keeps_named_fallback_explicit(self):
        with tempfile.TemporaryDirectory(prefix="output-fallback-", dir=self.work_root) as directory:
            root = Path(directory)
            result = self._run(root, "fallback")
            self.assertEqual(result["native"], {"anonymous_opens": 1, "named_opens": 1, "link_calls": 1})
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["preview.png", "store"])
        with tempfile.TemporaryDirectory(prefix="output-fallback-kill-", dir=self.work_root) as directory:
            root = Path(directory)
            ready = self._kill_at(root, "fallback-writing")
            self.assertEqual((ready["links"], ready["bytes"]), (1, 13))
            self.assertEqual(len(ready["named_staging"]), 1)
            orphan = root / ready["named_staging"][0]
            before = orphan.stat()
            self.assertEqual(self._counts(root), (0, 0))
            self._run(root)
            after = orphan.stat()
            self.assertEqual((before.st_ino, before.st_size, before.st_mtime_ns),
                             (after.st_ino, after.st_size, after.st_mtime_ns))

    def test_killing_unlinked_writer_preserves_concurrent_publication(self):
        with tempfile.TemporaryDirectory(prefix="output-concurrent-kill-", dir=self.work_root) as directory:
            root = Path(directory)
            competitor = []
            def publish():
                competitor.append(self._run(root))
                competitor.append((root / "preview.png").stat())
            ready = self._kill_at(root, "staging-writing", publish)
            self.assertEqual((ready["links"], ready["named_staging"]), (0, []))
            result = self._run(root)
            self.assertTrue(result["reference_replay"])
            self.assertTrue(result["event_replay"])
            self.assertEqual(result["digest"], competitor[0]["digest"])
            current, original = (root / "preview.png").stat(), competitor[1]
            self.assertEqual((current.st_dev, current.st_ino, current.st_mtime_ns),
                             (original.st_dev, original.st_ino, original.st_mtime_ns))
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["preview.png", "store"])

    def test_publication_and_evidence_crash_replay(self):
        for boundary in ("file", "evidence"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory(
                prefix="output-replay-", dir=self.work_root
            ) as directory:
                root = Path(directory)
                process = subprocess.Popen(
                    [str(self.child), str(root), boundary], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                lines = queue.Queue()
                reader = threading.Thread(
                    target=lambda: lines.put(process.stdout.readline(4097)), daemon=True
                )
                reader.start()
                try:
                    ready = json.loads(lines.get(timeout=20))
                    self.assertEqual(ready["phase"], "ready")
                    self.assertEqual(ready["boundary"], boundary)
                    output = root / "preview.png"
                    original = output.stat()
                    digest = hashlib.sha256(output.read_bytes()).hexdigest()
                    self.assertEqual(ready["digest"], digest)
                    self.assertEqual(ready["bytes"], original.st_size)
                    self.assertIsNone(process.poll(), "child must still be waiting at the boundary")
                    process.kill()
                    process.wait(timeout=10)
                    self.assertNotEqual(process.returncode, 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=10)
                    reader.join(timeout=5)
                    for stream in (process.stdin, process.stdout, process.stderr):
                        stream.close()
                with closing(sqlite3.connect(root / "store/runs.sqlite3")) as database:
                    references = database.execute("SELECT count(*) FROM jyppx_run_artifacts").fetchone()[0]
                    events = database.execute("SELECT count(*) FROM jyppx_events").fetchone()[0]
                self.assertEqual((references, events), (0, 0) if boundary == "file" else (2, 1))
                for attempt in range(2):
                    completed = subprocess.run(
                        [str(self.child), str(root), "none"], input=b"",
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, shell=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
                    result = json.loads(completed.stdout)
                    self.assertEqual(result["phase"], "completed")
                    self.assertEqual(result["digest"], digest)
                    self.assertEqual(result["bytes"], original.st_size)
                    self.assertEqual((result["references"], result["events"]), (2, 1))
                    self.assertTrue(result["integrity_ok"])
                    self.assertEqual((result["queue_bytes"], result["spool_bytes"]), (0, 0))
                    replay = boundary == "evidence" or attempt == 1
                    self.assertEqual(result["reference_replay"], replay)
                    self.assertEqual(result["event_replay"], replay)
                    current = output.stat()
                    self.assertEqual((current.st_dev, current.st_ino, current.st_mtime_ns),
                                     (original.st_dev, original.st_ino, original.st_mtime_ns))
                    self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), digest)
                    self.assertEqual(sorted(path.name for path in root.iterdir()), ["preview.png", "store"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("published", "staging", "faults"), default="published")
    args = parser.parse_args()
    ImageOutputRecoveryTests.child = args.child.resolve(strict=True)
    ImageOutputRecoveryTests.work_root = args.work_dir.resolve()
    ImageOutputRecoveryTests.work_root.mkdir(parents=True, exist_ok=True)
    methods = {
        "published": ["test_publication_and_evidence_crash_replay"],
        "staging": ["test_unlinked_staging_is_released_after_process_kill"],
        "faults": ["test_native_write_flush_link_and_close_errors_keep_recoverable_state",
                   "test_unsupported_filesystem_keeps_named_fallback_explicit",
                   "test_killing_unlinked_writer_preserves_concurrent_publication"],
    }
    unittest.main(argv=[__file__], verbosity=2,
                  defaultTest=["ImageOutputRecoveryTests." + name for name in methods[args.suite]])
