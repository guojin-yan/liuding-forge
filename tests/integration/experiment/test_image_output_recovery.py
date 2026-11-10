# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

"""Kill actual C++ publication syscalls and replay original evidence identities."""
import argparse
from contextlib import closing
import ctypes
from ctypes import wintypes
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
    def _file_identity(self, path):
        if os.name != "nt":
            observed = path.stat()
            return (observed.st_dev, observed.st_ino, observed.st_nlink, observed.st_size)

        # Python versions expose different Windows volume-ID widths in stat.
        # Read the full native identity independently of the child on all of them.
        class FileInfo(ctypes.Structure):
            _fields_ = [("attributes", wintypes.DWORD),
                        ("created", wintypes.FILETIME), ("accessed", wintypes.FILETIME),
                        ("written", wintypes.FILETIME), ("volume", wintypes.DWORD),
                        ("size_high", wintypes.DWORD), ("size_low", wintypes.DWORD),
                        ("links", wintypes.DWORD), ("index_high", wintypes.DWORD),
                        ("index_low", wintypes.DWORD)]

        class FileId(ctypes.Structure):
            _fields_ = [("volume", ctypes.c_uint64), ("file_id", ctypes.c_ubyte * 16)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(FileInfo)]
        kernel.GetFileInformationByHandle.restype = wintypes.BOOL
        kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                       ctypes.c_void_p, wintypes.DWORD]
        kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.CreateFileW(str(path), 0, 7, None, 3, 0x00200000, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            info, identity = FileInfo(), FileId()
            if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not kernel.GetFileInformationByHandleEx(handle, 18, ctypes.byref(identity),
                                                       ctypes.sizeof(identity)):
                raise ctypes.WinError(ctypes.get_last_error())
            return (identity.volume, int.from_bytes(identity.file_id, "little"), info.links,
                    (info.size_high << 32) | info.size_low)
        finally:
            kernel.CloseHandle(handle)

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
        content = (root / "preview.png").read_bytes()
        self.assertEqual(result["bytes"], len(content))
        self.assertEqual(result["digest"], hashlib.sha256(content).hexdigest())
        if os.name == "nt":
            self.assertEqual((root / "preview.png").stat().st_file_attributes & 0x100, 0)
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
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        lines = queue.Queue()
        reader = threading.Thread(
            target=lambda: lines.put(process.stdout.readline(4097)), daemon=True
        )
        reader.start()
        try:
            ready = json.loads(lines.get(timeout=20))
            self.assertEqual((ready["phase"], ready["boundary"]), ("ready", boundary))
            if os.name == "nt":
                descriptor = Path(ready["staging_path"])
            else:
                descriptor = Path(f"/proc/{process.pid}/fd/{ready['descriptor']}")
            observed = self._file_identity(descriptor)
            inode = ready["inode"] | (ready.get("inode_high", 0) << 64)
            self.assertEqual(observed, (ready["device"], inode, ready["links"], ready["bytes"]))
            ready["inode"] = inode
            if os.name != "nt":
                ready["content_sha256"] = hashlib.sha256(descriptor.read_bytes()).hexdigest()
            ready["named_staging"] = sorted(path.name for path in root.glob("*.liuding-tmp-*"))
            if while_waiting is not None:
                while_waiting()
            self.assertIsNone(process.poll(), "child must still own the paused inode")
            process.kill()
            process.wait(timeout=10)
            self.assertNotEqual(process.returncode, 0)
            if os.name != "nt":
                self.assertEqual(process.returncode, -9)
                self.assertFalse(descriptor.exists())
            elif boundary.startswith("staging-"):
                self.assertFalse(descriptor.exists())
                if boundary == "staging-linked":
                    ready["content_sha256"] = hashlib.sha256(
                        (root / "preview.png").read_bytes()).hexdigest()
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
                self.assertEqual(ready["links"], int(linked) + int(os.name == "nt"))
                self.assertEqual(len(ready["named_staging"]), int(os.name == "nt"))
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
                    self.assertEqual(self._file_identity(output)[:2],
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
        boundaries = ["fail-open", "fail-write", "fail-sync", "fail-link", "fail-close"]
        if os.name != "nt":
            boundaries.append("fail-directory")
        for boundary in boundaries:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory(
                prefix="output-syscall-", dir=self.work_root
            ) as directory:
                root = Path(directory)
                failed = subprocess.run(
                    [str(self.child), str(root), boundary], input=b"",
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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
        boundaries = ("fallback", "volume-query-failed") if os.name == "nt" else ("fallback",)
        for boundary in boundaries:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory(
                prefix="output-fallback-", dir=self.work_root
            ) as directory:
                root = Path(directory)
                result = self._run(root, boundary)
                expected = ({"delete_on_close_opens": 1, "named_opens": 2, "link_calls": 0}
                            if os.name == "nt" else
                            {"anonymous_opens": 1, "named_opens": 1, "link_calls": 1})
                self.assertEqual(result["native"], expected)
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

    def test_replacement_keeps_named_crash_boundary_explicit(self):
        with tempfile.TemporaryDirectory(prefix="output-replace-kill-", dir=self.work_root) as directory:
            root = Path(directory)
            original = self._run(root)
            published = (root / "preview.png").stat()
            ready = self._kill_at(root, "replace-writing")
            self.assertEqual((ready["links"], ready["bytes"]), (1, 13))
            self.assertEqual(len(ready["named_staging"]), 1)
            orphan = root / ready["named_staging"][0]
            self.assertEqual(orphan.stat().st_size, 13)
            current = (root / "preview.png").stat()
            self.assertEqual((current.st_ino, current.st_size, current.st_mtime_ns),
                             (published.st_ino, published.st_size, published.st_mtime_ns))
            self.assertEqual(self._counts(root), (2, 1))
            replay = self._run(root)
            self.assertEqual(replay["digest"], original["digest"])
            self.assertTrue(replay["reference_replay"])
            self.assertTrue(replay["event_replay"])
            self.assertEqual(orphan.stat().st_size, 13)

    def test_killing_unlinked_writer_preserves_concurrent_publication(self):
        with tempfile.TemporaryDirectory(prefix="output-concurrent-kill-", dir=self.work_root) as directory:
            root = Path(directory)
            competitor = []
            def publish():
                competitor.append(self._run(root))
                competitor.append((root / "preview.png").stat())
            ready = self._kill_at(root, "staging-writing", publish)
            self.assertEqual(ready["links"], int(os.name == "nt"))
            self.assertEqual(len(ready["named_staging"]), int(os.name == "nt"))
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
    if os.name == "nt":
        methods["faults"].append("test_replacement_keeps_named_crash_boundary_explicit")
    unittest.main(argv=[__file__], verbosity=2,
                  defaultTest=["ImageOutputRecoveryTests." + name for name in methods[args.suite]])
