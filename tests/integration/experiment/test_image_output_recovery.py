# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

"""Kill actual C++ output before/after evidence commit, then replay unchanged."""
import argparse
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
                with sqlite3.connect(root / "store/runs.sqlite3") as database:
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
    args = parser.parse_args()
    ImageOutputRecoveryTests.child = args.child.resolve(strict=True)
    ImageOutputRecoveryTests.work_root = args.work_dir.resolve()
    ImageOutputRecoveryTests.work_root.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=[__file__], verbosity=2)
