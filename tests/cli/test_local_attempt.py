# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Independent durable-local CLI contracts; Python children are test fixtures."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wait_for(predicate, timeout: float = 15) -> bool:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        result = kernel.WaitForSingleObject(handle, 0)
        kernel.CloseHandle(handle)
        return result == 258
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    try:
        if stat.read_text().rsplit(")", 1)[1].lstrip().startswith("Z "):
            return False
    except FileNotFoundError:
        return False
    return True


def fixture(mode: str, root: Path, arguments: list[str]) -> None:
    if mode == "child":
        if os.name != "nt":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pending = root / "child.pid.pending"
        pending.write_text(str(os.getpid()), encoding="ascii")
        pending.replace(root / "child.pid")
        while True:
            time.sleep(0.01)
    index = int(os.environ["JYPPX_LIUDING_LOCAL_ATTEMPT_INDEX"])
    with (root / "launches.txt").open("a", encoding="utf-8") as stream:
        stream.write(str(index) + "\n")
    if mode in ("hold", "flood"):
        subprocess.Popen([sys.executable, "-I", "-u", str(Path(__file__).resolve()),
                          "--fixture", "child", str(root)], stdin=subprocess.DEVNULL)
        if not wait_for(lambda: (root / "child.pid").is_file()):
            raise RuntimeError("Descendant did not start")
        (root / "ready.json").write_text(json.dumps({"leader": os.getpid(),
            "child": int((root / "child.pid").read_text())}), encoding="ascii")
        while True:
            if mode == "flood":
                os.write(1, b"x" * 4096)
            else:
                time.sleep(0.01)
    if mode == "fail" or (mode == "retry" and index < 2):
        os.write(1, b"failed-output\x00\xff")
        os._exit(23)
    payload = {"argv": arguments, "cwd": str(Path.cwd()), "index": index,
        "job_id": os.environ["JYPPX_LIUDING_LOCAL_JOB_ID"],
        "attempt_id": os.environ["JYPPX_LIUDING_LOCAL_ATTEMPT_ID"],
        "token": os.environ.get("LOCAL_ATTEMPT_FIXTURE"),
        "stdin_eof": sys.stdin.buffer.read() == b""}
    os.write(1, json.dumps(payload, ensure_ascii=True).encode("ascii"))
    os.write(2, b"diagnostic\x00\xfe")


class LocalAttemptTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="liuding-local-attempt-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cwd = self.directory / "工作目录"
        self.cwd.mkdir()
        self.root = self.directory / "store"
        self.spec_path = self.directory / "spec.json"
        self.spec = {"format": "jyppx.liuding.local_run_spec", "schema_version": 1,
            "project_id": "local-contract", "experiment_id": "attempts",
            "executable": str(Path(sys.executable).resolve()),
            "executable_sha256": sha256(Path(sys.executable).resolve()),
            "arguments": ["-I", "-u", str(Path(__file__).resolve()), "--fixture", "inspect", str(self.cwd)],
            "working_directory": str(self.cwd), "environment": {"LOCAL_ATTEMPT_FIXTURE": "literal"},
            "inherit_environment": True, "timeout_ms": 10000, "max_attempts": 3,
            "retry_exit_codes": [23], "max_stdout_bytes": 65536, "max_stderr_bytes": 65536}
        self.write_spec()

    def write_spec(self, mode: str | None = None) -> None:
        if mode:
            self.spec["arguments"][4] = mode
        self.spec_path.write_text(json.dumps(self.spec, ensure_ascii=False), encoding="utf-8")

    def command(self, *arguments: object) -> list[str]:
        return [str(self.cli), *map(str, arguments), "--format", "json"]

    def invoke(self, *arguments: object, expected: int = 0) -> dict:
        result = subprocess.run(self.command(*arguments), capture_output=True,
            text=True, encoding="utf-8", timeout=45, check=False,
            env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def local(self, action: str, *, expected: int = 0) -> dict:
        options = ["--spec", self.spec_path] if action == "execute" else []
        return self.invoke("local", action, "--root", self.root, "--id", "job-one",
                           *options, expected=expected)["data"]

    def start(self) -> subprocess.Popen:
        process = subprocess.Popen(self.command("local", "execute", "--root", self.root,
            "--id", "job-one", "--spec", self.spec_path), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
        self.addCleanup(lambda: self.cleanup_process(process))
        return process

    def cleanup_process(self, process: subprocess.Popen) -> None:
        if process.poll() is None:
            try:
                self.local("cancel")
            except (subprocess.TimeoutExpired, AssertionError):
                process.kill()
        try:
            process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=8)
        # Forced supervisor-death tests deliberately exercise unknown ownership.
        # Cleanup only our known fixture process group, never an arbitrary PID.
        ready = self.cwd / "ready.json"
        if ready.is_file():
            pids = json.loads(ready.read_text())
            if os.name != "nt" and alive(pids["leader"]):
                try:
                    os.killpg(pids["leader"], signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def references(self, run: str) -> dict:
        data = self.invoke("run", "show", "--root", self.root, "--id", run)["data"]
        return {item["reference_id"]: item for item in data["references"]}

    def blob(self, reference: dict) -> bytes:
        artifact = reference["artifact"]
        identity = artifact["sha256"]
        content = (self.root / "artifacts" / "objects" / identity[:2] / identity[2:]).read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), identity)
        self.assertEqual(len(content), artifact["size_bytes"])
        return content

    def launch_count(self) -> int:
        return len((self.cwd / "launches.txt").read_text().splitlines())

    def await_ready(self) -> dict:
        observed = None
        def ready() -> bool:
            nonlocal observed
            try:
                observed = json.loads((self.cwd / "ready.json").read_text())
                return True
            except (FileNotFoundError, json.JSONDecodeError):
                return False
        self.assertTrue(wait_for(ready))
        return observed

    def assert_stopped(self, pids: dict) -> None:
        for pid in pids.values():
            self.assertTrue(wait_for(lambda: not alive(pid)), f"Fixture PID {pid} survived")

    def test_success_literal_argv_raw_cas_and_terminal_replay(self) -> None:
        extra = ["", "with space", "引号\"和\\", "literal & | ; $(x) `x`", "line\nbreak"]
        self.spec["arguments"].extend(extra)
        self.write_spec()
        result = self.local("execute")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["attempts"]), 1)
        attempt = result["attempts"][0]
        refs = self.references(attempt["attempt_id"])
        payload = json.loads(self.blob(refs["local.stdout"]))
        self.assertEqual(payload["argv"], extra)
        self.assertEqual(payload["cwd"], str(self.cwd))
        self.assertEqual(payload["attempt_id"], attempt["attempt_id"])
        self.assertTrue(payload["stdin_eof"])
        self.assertEqual(payload["token"], "literal")
        self.assertEqual(self.blob(refs["local.stderr"]), b"diagnostic\x00\xfe")
        self.assertEqual(json.loads(self.blob(refs["local.result"])), attempt["result"])
        self.assertEqual(self.local("execute")["attempts"], result["attempts"])
        self.assertEqual(self.local("recover")["attempts"], result["attempts"])
        self.assertEqual(self.launch_count(), 1)
        self.assertTrue(self.invoke("run", "verify", "--root", self.root)["data"]["valid"])

    def test_retry_creates_distinct_child_runs_without_overwriting_failure(self) -> None:
        self.write_spec("retry")
        result = self.local("execute")
        self.assertEqual(result["status"], "completed")
        first, second = result["attempts"]
        self.assertNotEqual(first["attempt_id"], second["attempt_id"])
        self.assertEqual(first["result"]["exit_code"], 23)
        self.assertEqual(second["result"]["exit_code"], 0)
        self.assertEqual(self.blob(self.references(first["attempt_id"])["local.stdout"]),
                         b"failed-output\x00\xff")
        for index, item in enumerate((first, second), 1):
            metadata = self.invoke("run", "show", "--root", self.root,
                "--id", item["attempt_id"])["data"]["run"]
            self.assertEqual(metadata["lineage"]["parent_run_id"], "job-one")
            self.assertEqual(metadata["tags"]["liuding.local.index"], str(index))
        self.assertEqual(self.launch_count(), 2)

    def test_retry_budget_is_finite_and_failure_replay_is_idempotent(self) -> None:
        self.write_spec("fail")
        result = self.local("execute", expected=1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["attempts"]), 3)
        self.local("recover", expected=1)
        self.local("execute", expected=1)
        self.assertEqual(self.launch_count(), 3)

    def test_undeclared_exit_code_is_never_retried(self) -> None:
        self.spec["retry_exit_codes"] = [17]
        self.write_spec("fail")
        result = self.local("execute", expected=1)
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(result["attempts"][0]["result"]["exit_code"], 23)

    def test_timeout_cleans_descendants_and_is_not_retried(self) -> None:
        self.spec["timeout_ms"] = 5000
        self.write_spec("hold")
        result = self.local("execute", expected=1)
        self.assertEqual(result["attempts"][0]["result"]["status"], "timed_out")
        self.assertEqual(len(result["attempts"]), 1)
        self.assert_stopped(self.await_ready())

    def test_output_quota_cleans_descendants_and_bounds_raw_artifact(self) -> None:
        self.spec["max_stdout_bytes"] = 129
        self.write_spec("flood")
        result = self.local("execute", expected=1)
        attempt = result["attempts"][0]
        self.assertEqual(attempt["result"]["status"], "output_limit")
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(self.blob(self.references(attempt["attempt_id"])["local.stdout"]), b"x" * 129)
        self.assert_stopped(self.await_ready())

    def test_live_ownership_refuses_recover_and_concurrent_cancel_is_idempotent(self) -> None:
        self.write_spec("hold")
        process = self.start()
        pids = self.await_ready()
        result = self.local("recover", expected=7)
        self.assertEqual(result["status"], "active")
        self.assertTrue(result["owner_active"])
        self.assertTrue(self.local("show")["owner_active"])
        competing = subprocess.run(self.command("local", "execute", "--root", self.root,
            "--id", "job-one", "--spec", self.spec_path), capture_output=True, timeout=15)
        self.assertNotEqual(competing.returncode, 0)
        with ThreadPoolExecutor(max_workers=2) as executor:
            replies = list(executor.map(lambda _: self.local("cancel"), range(2)))
        self.assertTrue(all(reply["cancel_requested"] for reply in replies))
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 130, stdout + stderr)
        final = json.loads(stdout)["data"]
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(final["attempts"][0]["result"]["status"], "cancelled")
        self.assertEqual(self.launch_count(), 1)
        self.assert_stopped(pids)

    def test_caller_crash_does_not_relaunch_an_unconfirmed_attempt(self) -> None:
        self.spec["timeout_ms"] = 3000
        self.write_spec("hold")
        process = self.start()
        pids = self.await_ready()
        process.kill()
        process.communicate(timeout=10)
        state = self.local("recover", expected=7)
        self.assertIn(state["status"], ("active", "recovery_required"))
        self.assertEqual(self.launch_count(), 1)
        if os.name == "nt":
            self.assert_stopped(pids)
            self.assertEqual(self.local("recover", expected=7)["status"], "recovery_required")
        else:
            # The separate worker retains its lock and normal timeout cleanup,
            # then confirms its result even though the dispatch caller is gone.
            self.assertTrue(wait_for(lambda: not alive(pids["leader"]), timeout=10))
            self.assertTrue(wait_for(lambda: self.local("show")["attempts"][0]["result"] is not None))
            recovered = self.local("recover", expected=1)
            self.assertEqual(recovered["status"], "failed")
            self.assertEqual(recovered["attempts"][0]["result"]["status"], "timed_out")
            self.assert_stopped(pids)
        self.assertEqual(self.launch_count(), 1)

    def test_committed_result_recovery_repairs_only_lifecycle(self) -> None:
        original = self.local("execute")
        # Recreate the exact durable boundary after the three-reference commit
        # and before either run transition; no active process exists here.
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("UPDATE jyppx_runs SET status=2")
        repaired = self.local("recover")
        self.assertEqual(repaired["attempts"], original["attempts"])
        self.assertEqual(repaired["status"], "completed")
        self.assertEqual(self.launch_count(), 1)

    def test_unconfirmed_intent_remains_fenced_even_without_an_os_owner(self) -> None:
        original = self.local("execute")
        attempt = original["attempts"][0]["attempt_id"]
        refs = self.references(attempt)
        # A crash before the atomic output/result transaction leaves exactly
        # intent/owner but no confirmed result. Simulate only that boundary.
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("DELETE FROM jyppx_run_artifacts WHERE reference_id IN "
                             "('local.stdout','local.stderr','local.result')")
            database.execute("UPDATE jyppx_runs SET status=2")
        recovered = self.local("recover", expected=7)
        self.assertEqual(recovered["status"], "recovery_required")
        self.assertFalse(recovered["owner_active"])
        self.local("execute", expected=7)
        rejected = subprocess.run(self.command("local", "worker", "--root", self.root,
            "--id", "job-one", "--attempt", 1, "--intent-sha256",
            refs["local.intent"]["artifact"]["sha256"]), capture_output=True, timeout=15)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(self.launch_count(), 1)

    def test_changed_frozen_spec_and_invalid_inputs_fail_before_launch(self) -> None:
        self.local("execute")
        self.spec["timeout_ms"] += 1
        self.write_spec()
        rejected = self.invoke("local", "execute", "--root", self.root, "--id", "job-one",
                               "--spec", self.spec_path, expected=2)
        self.assertEqual(rejected["status"], "error")
        self.assertEqual(rejected["error"]["name"], "invalid_argument")
        self.assertEqual(self.launch_count(), 1)
        mutations = {"max_attempts": 17, "timeout_ms": 0, "retry_exit_codes": [0],
            "unknown_field": True, "executable_sha256": "0" * 64,
            "executable": "python", "environment": {"bad=name": "value"},
            "arguments": ["nul\x00value"]}
        for key, value in mutations.items():
            with self.subTest(field=key):
                invalid = dict(self.spec, **{key: value})
                path = self.directory / (key + ".json")
                path.write_text(json.dumps(invalid), encoding="utf-8")
                root = self.directory / ("invalid-" + key)
                rejected = self.invoke("local", "execute", "--root", root, "--id", "bad-job",
                    "--spec", path, expected=3 if key == "max_attempts" else 2)
                self.assertEqual(rejected["error"]["name"],
                    {"max_attempts": "resource_exhausted", "unknown_field": "serialization"}
                    .get(key, "invalid_argument"))
                self.assertFalse((root / "runs.sqlite3").exists())


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--fixture":
        fixture(sys.argv[2], Path(sys.argv[3]), sys.argv[4:])
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--cli", type=Path, required=True)
        arguments, rest = parser.parse_known_args()
        LocalAttemptTest.cli = arguments.cli.resolve()
        unittest.main(argv=[__file__, *rest])
