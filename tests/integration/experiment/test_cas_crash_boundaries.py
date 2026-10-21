# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Force-kill production CAS upload around actual native publication.

The static test child supervises only a private forwarding symbol. Real native
rename, sync, hashing, upload, recovery, verified reads, and GC remain production
operations. Three boundaries are mandatory; no test is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time
import unittest


MAXIMUM_REPORT_BYTES = 64 * 1024
TARGET_BYTES = 2 * 1024 * 1024 + 257
CONFIRMED = b"Confirmed CAS object predating the interrupted upload"


class CasCrashBoundaryTests(unittest.TestCase):
    child: Path
    work_root: Path
    timeout: float

    @classmethod
    def setUpClass(cls) -> None:
        cls.target_sha256 = hashlib.sha256(
            bytes((index * 37 + 11) & 255 for index in range(TARGET_BYTES))
        ).hexdigest()
        cls.confirmed_sha256 = hashlib.sha256(CONFIRMED).hexdigest()

    def record(self, stage: str, value):
        self.evidence[stage] = value
        (self.root / "evidence.json").write_text(
            json.dumps(self.evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return value

    def decode(self, output: bytes) -> dict:
        self.assertLessEqual(len(output), MAXIMUM_REPORT_BYTES)
        self.assertTrue(output.endswith(b"\n"), "CAS child report was incomplete")
        value = json.loads(output.decode("utf-8"))
        self.assertEqual(value["schema_version"], 1)
        return value

    def complete(self, mode: str, stage: str) -> dict:
        child = subprocess.run(
            [str(self.child), mode, str(self.root)], cwd=self.root,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=self.timeout, check=False, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.assertEqual(child.returncode, 0,
                         f"{mode}: {child.stderr.decode('utf-8', errors='replace')}")
        value = self.decode(child.stdout)
        self.assertEqual(value["phase"], "completed")
        self.assertEqual(value["mode"], mode)
        self.assertTrue(value["store"]["confirmed_exists"])
        self.assertTrue(value["store"]["integrity_ok"])
        return self.record(stage, value)

    def force_kill(self, boundary: str) -> dict:
        received: queue.Queue = queue.Queue(maxsize=1)
        with (self.root / "cas-stderr.log").open("wb") as diagnostics:
            child = subprocess.Popen(
                [str(self.child), "crash", str(self.root), boundary], cwd=self.root,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostics,
                shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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
                    output = received.get(timeout=self.timeout)
                except queue.Empty:
                    self.fail("Private native CAS publication fence was not reached")
                if isinstance(output, Exception):
                    raise output
                ready = self.decode(output)
                self.assertEqual(ready["phase"], "ready-for-forced-kill")
                self.assertEqual(ready["boundary"], boundary)
                self.assertEqual(ready["process_id"], child.pid)
                self.assertTrue(ready["private_seam_observed"])
                self.assertEqual(ready["native_calls"], int(boundary != "before-rename"))
                self.assertEqual(ready["session_closed"], boundary == "after-commit")
                self.assertEqual(ready["confirmed"]["sha256"], self.confirmed_sha256)
                self.assertEqual(ready["confirmed"]["bytes"], len(CONFIRMED))
                self.assertEqual(ready["source"]["exists"], boundary == "before-rename")
                self.assertEqual(ready["destination"]["exists"], boundary != "before-rename")
                active = ready["source"] if boundary == "before-rename" else ready["destination"]
                self.assertEqual(active["bytes"], TARGET_BYTES)
                self.assertEqual(active["sha256"], self.target_sha256)
                self.assertIsNone(child.poll())
                self.assertFalse((self.root / "cas-cleanup.marker").exists())
                self.record("ready", ready)
                child.kill()
                code = child.wait(timeout=self.timeout)
                self.assertNotEqual(code, 0)
                if os.name == "posix":
                    self.assertEqual(code, -signal.SIGKILL)
                self.assertFalse((self.root / "cas-cleanup.marker").exists())
                self.record("forced_kill", {"returncode": code, "cleanup_marker": False})
                return ready
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=self.timeout)
                child.stdin.close()
                child.stdout.close()
                reader.join(timeout=self.timeout)

    @staticmethod
    def file_evidence(path: Path) -> dict:
        content = path.read_bytes()
        return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    def facts(self, stage: str) -> dict:
        objects = {}
        for path in (self.root / "cas" / "objects").glob("*/*"):
            if path.is_file():
                digest = path.parent.name + path.name
                objects[digest] = self.file_evidence(path)
                self.assertEqual(objects[digest]["sha256"], digest)
        temporary = {
            path.name: self.file_evidence(path)
            for path in (self.root / "cas" / "tmp").iterdir() if path.is_file()
        }
        self.assertEqual(objects[self.confirmed_sha256], {
            "bytes": len(CONFIRMED), "sha256": self.confirmed_sha256,
        })
        return self.record(stage, {"objects": objects, "temporary": temporary})

    def scenario(self, boundary: str) -> None:
        self.root = (self.work_root / boundary).resolve()
        self.root.mkdir()
        self.evidence = {"schema_version": 1, "boundary": boundary}
        seeded = self.complete("seed", "seeded")
        self.assertEqual(seeded["confirmed_sha256"], self.confirmed_sha256)
        self.assertEqual(seeded["target_sha256"], self.target_sha256)
        self.assertEqual(seeded["target_bytes"], TARGET_BYTES)
        initial = self.facts("initial_files")
        self.assertEqual(len(initial["objects"]), 1)
        self.assertEqual(initial["temporary"], {})
        ready = self.force_kill(boundary)
        crashed = self.facts("files_after_kill")
        observed = self.complete("inspect", "production_verified_after_kill")
        self.assertEqual(observed["store"]["target_exists"], boundary != "before-rename")
        recovered = self.complete("recover", "recover_recent_crash_files")
        self.assertEqual(recovered["recovery"], {
            "removed_files": 0, "removed_bytes": 0,
            "retained_files": int(boundary == "before-rename"),
        })
        self.assertEqual(self.facts("files_after_recent_recovery"), crashed)

        if boundary == "before-rename":
            self.assertEqual(crashed["objects"], initial["objects"])
            self.assertEqual(len(crashed["temporary"]), 1)
            self.assertEqual(next(iter(crashed["temporary"].values())), {
                "bytes": TARGET_BYTES, "sha256": self.target_sha256,
            })
            abandoned = Path(ready["source"]["path"]).resolve()
            self.assertEqual(abandoned.parent, self.root / "cas" / "tmp")
            self.assertTrue(abandoned.is_file())
            stale = time.time() - 48 * 60 * 60
            os.utime(abandoned, (stale, stale))
            cleaned = self.complete("recover", "recover_actual_stale_crash_upload")
            self.assertEqual(cleaned["recovery"], {
                "removed_files": 1, "retained_files": 0, "removed_bytes": TARGET_BYTES,
            })
            self.assertEqual(self.facts("files_after_stale_cleanup"), initial)
        elif boundary == "after-rename":
            self.assertEqual(crashed["temporary"], {})
            self.assertEqual(len(crashed["objects"]), 2)
            orphan_gc = self.complete("gc-confirmed", "collect_unconfirmed_published_orphan")
            self.assertEqual(orphan_gc["gc"]["removed_files"], 1)
            self.assertEqual(orphan_gc["gc"]["removed_bytes"], TARGET_BYTES)
            self.assertEqual(self.facts("files_after_orphan_gc"), initial)
        else:
            self.assertEqual(crashed["temporary"], {})
            self.assertEqual(len(crashed["objects"]), 2)
            retained = self.complete("gc-both", "preserve_all_confirmed_objects")
            self.assertEqual(retained["gc"]["removed_files"], 0)
            self.assertEqual(self.facts("files_after_live_gc"), crashed)

        retried = self.complete("retry", "repeated_real_upload")
        self.assertEqual(retried["repeated_digest"], self.target_sha256)
        self.assertTrue(retried["store"]["target_exists"])
        final = self.facts("files_after_retry")
        self.assertEqual(final["temporary"], {})
        self.assertEqual(len(final["objects"]), 2)
        self.assertEqual(final["objects"][self.target_sha256], {
            "bytes": TARGET_BYTES, "sha256": self.target_sha256,
        })
        self.complete("retry", "replayed_real_upload")
        self.assertEqual(self.facts("files_after_replay"), final)
        live_gc = self.complete("gc-both", "final_live_set_gc")
        self.assertEqual(live_gc["gc"]["removed_files"], 0)
        self.assertEqual(self.facts("final_files"), final)
        self.assertFalse((self.root / "cas-cleanup.marker").exists())
        self.record("valid", True)

    def test_kill_before_native_rename_recovers_actual_abandoned_upload(self) -> None:
        self.scenario("before-rename")

    def test_kill_after_native_rename_preserves_whole_object_and_collects_orphan(self) -> None:
        self.scenario("after-rename")

    def test_kill_after_production_commit_retains_confirmed_object(self) -> None:
        self.scenario("after-commit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    arguments = parser.parse_args()
    child = arguments.child.resolve()
    if not child.is_file():
        parser.error("--child must name an already built static-archive test executable")
    if not 0 < arguments.timeout <= 300:
        parser.error("--timeout must be positive and at most 300 seconds")
    temporary = None
    if arguments.work_dir:
        arguments.work_dir.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="cas-crash-数据-", dir=arguments.work_dir.resolve()))
    else:
        temporary = tempfile.TemporaryDirectory(prefix="jyppx-cas-crash-")
        root = Path(temporary.name) / "CAS 数据"
        root.mkdir()
    CasCrashBoundaryTests.child = child
    CasCrashBoundaryTests.work_root = root.resolve()
    CasCrashBoundaryTests.timeout = arguments.timeout
    try:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(CasCrashBoundaryTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        print(json.dumps({"valid": result.wasSuccessful(), "tests_run": result.testsRun,
                          "skipped": len(result.skipped), "work_root": str(root),
                          "evidence_retained": temporary is None}))
        return 0 if result.wasSuccessful() and result.testsRun == 3 and not result.skipped else 1
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
