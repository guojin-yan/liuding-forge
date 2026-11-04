# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Interrupt a real ResNet18 only after a verified durable checkpoint.

Python supplies images, injects OS faults and checks native/CAS observations.
Suspension models a hung trainer; the native worker owns the timeout. A killed
worker has no result and cannot release its lease, even after test-only cleanup.
This is an interruption fixture, not another training quality experiment.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import tempfile
import time
import unittest

from test_image_recipe import PARTICIPANTS, png


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linux_stat(pid: int) -> dict | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace").rsplit(")", 1)[1].split()
        return {"pid": pid, "parent": int(fields[1]), "group": int(fields[2]),
                "state": fields[0], "created": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def windows_kernel():
    from ctypes import wintypes as wt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    kernel.OpenProcess.restype = wt.HANDLE
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    kernel.CloseHandle.restype = wt.BOOL
    kernel.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
    kernel.WaitForSingleObject.restype = wt.DWORD
    kernel.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]
    kernel.TerminateProcess.restype = wt.BOOL
    kernel.QueryFullProcessImageNameW.argtypes = [
        wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = wt.BOOL
    kernel.GetProcessTimes.argtypes = [wt.HANDLE, *[ctypes.POINTER(wt.FILETIME)] * 4]
    kernel.GetProcessTimes.restype = wt.BOOL
    return kernel


def windows_console_host() -> Path:
    from ctypes import wintypes as wt

    kernel = windows_kernel()
    kernel.GetSystemDirectoryW.argtypes = [wt.LPWSTR, wt.UINT]
    kernel.GetSystemDirectoryW.restype = wt.UINT
    directory = ctypes.create_unicode_buffer(32768)
    length = kernel.GetSystemDirectoryW(directory, len(directory))
    if not length or length >= len(directory):
        raise ctypes.WinError(ctypes.get_last_error())
    system_directory = Path(directory.value).resolve()
    expected = (Path(os.environ["SystemRoot"]) / "System32").resolve()
    if os.path.normcase(str(system_directory)) != os.path.normcase(str(expected)):
        raise AssertionError("Native SystemDirectory differs from SystemRoot/System32")
    return system_directory / "conhost.exe"


def process_parents() -> dict[int, int]:
    if os.name != "nt":
        rows = (linux_stat(int(path.name)) for path in Path("/proc").iterdir()
                if path.name.isdecimal())
        return {row["pid"]: row["parent"] for row in rows if row is not None}
    from ctypes import wintypes as wt

    class Entry(ctypes.Structure):
        _fields_ = [("size", wt.DWORD), ("usage", wt.DWORD), ("pid", wt.DWORD),
                    ("heap", ctypes.c_size_t), ("module", wt.DWORD),
                    ("threads", wt.DWORD), ("parent", wt.DWORD),
                    ("priority", wt.LONG), ("flags", wt.DWORD),
                    ("executable", wt.WCHAR * 260)]

    kernel = windows_kernel()
    kernel.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wt.HANDLE
    for name in ("Process32FirstW", "Process32NextW"):
        function = getattr(kernel, name)
        function.argtypes = [wt.HANDLE, ctypes.POINTER(Entry)]
        function.restype = wt.BOOL
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = Entry()
        entry.size = ctypes.sizeof(entry)
        parents = {}
        available = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while available:
            parents[int(entry.pid)] = int(entry.parent)
            available = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
            raise ctypes.WinError(ctypes.get_last_error())
        return parents
    finally:
        kernel.CloseHandle(snapshot)


class NativeProcess:
    """Pin one verified child identity before injecting a fault; no PID search."""

    def __init__(self, pid: int, parent: int, executable: Path | None = None,
                 *, control: bool = True):
        self.pid, self.handle = pid, None
        self.control_allowed = control
        if process_parents().get(pid) != parent:
            raise AssertionError("Native process has an unexpected OS parent")
        if os.name == "nt":
            from ctypes import wintypes as wt
            self.kernel = windows_kernel()
            access = 0x1000 | 0x100000 | ((0x0800 | 1) if control else 0)
            self.handle = self.kernel.OpenProcess(access, False, pid)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                image, length = ctypes.create_unicode_buffer(32768), wt.DWORD(32768)
                if not self.kernel.QueryFullProcessImageNameW(self.handle, 0, image, ctypes.byref(length)):
                    raise ctypes.WinError(ctypes.get_last_error())
                clocks = [wt.FILETIME() for _ in range(4)]
                if not self.kernel.GetProcessTimes(self.handle, *map(ctypes.byref, clocks)):
                    raise ctypes.WinError(ctypes.get_last_error())
                self.identity = {"pid": pid, "parent": parent, "executable": image.value,
                                 "created": (clocks[0].dwHighDateTime << 32) | clocks[0].dwLowDateTime}
            except BaseException:
                self.close()
                raise
        else:
            # Ubuntu's pidfd remains attached to this exact process after PID
            # reuse. Do not fall back to killing whichever process has a PID.
            self.handle = os.pidfd_open(pid)
            try:
                self.identity = linux_stat(pid)
                if self.identity is None or self.identity["parent"] != parent:
                    raise AssertionError("Native process disappeared during capture")
                self.identity["executable"] = os.readlink(f"/proc/{pid}/exe")
            except BaseException:
                self.close()
                raise
        if process_parents().get(pid) != parent:
            self.close()
            raise AssertionError("Pinned native process has an unexpected OS parent")
        if executable is not None and os.path.normcase(str(Path(self.identity["executable"]).resolve())) != os.path.normcase(str(executable)):
            self.close()
            raise AssertionError("Child executable differs from the frozen native CLI")

    def running(self) -> bool:
        if os.name != "nt":
            return not select.select([self.handle], [], [], 0)[0]
        status = self.kernel.WaitForSingleObject(self.handle, 0)
        if status not in (0, 258):
            raise ctypes.WinError(ctypes.get_last_error())
        return status == 258

    def suspend(self) -> None:
        if not self.control_allowed:
            raise AssertionError("Cannot suspend a read-only process identity")
        if os.name == "nt":
            # Suspend the process atomically, including its native worker
            # threads. The supervising local worker is a different process.
            function = ctypes.WinDLL("ntdll").NtSuspendProcess
            function.argtypes, function.restype = [ctypes.c_void_p], ctypes.c_long
            status = function(self.handle)
            if status < 0:
                raise OSError(f"NtSuspendProcess failed: NTSTATUS {status & 0xffffffff:08x}")
        else:
            signal.pidfd_send_signal(self.handle, signal.SIGSTOP)
            deadline = time.monotonic() + 5
            while True:
                row = linux_stat(self.pid)
                if row and row["created"] == self.identity["created"] and row["state"] in ("T", "t"):
                    break
                if time.monotonic() >= deadline or not self.running():
                    raise AssertionError("Native trainer did not enter the stopped state")
                time.sleep(0.01)

    def kill(self) -> None:
        if not self.control_allowed:
            raise AssertionError("Cannot kill a read-only process identity")
        if not self.running():
            return
        if os.name == "nt":
            if not self.kernel.TerminateProcess(self.handle, 0xe0000397):
                raise ctypes.WinError(ctypes.get_last_error())
            if self.kernel.WaitForSingleObject(self.handle, 10000) != 0:
                raise AssertionError("Killed native process did not stop")
        else:
            signal.pidfd_send_signal(self.handle, signal.SIGKILL)
            if not select.select([self.handle], [], [], 10)[0]:
                raise AssertionError("Killed native process did not stop")

    def close(self) -> None:
        if self.handle is not None:
            if os.name == "nt":
                self.kernel.CloseHandle(self.handle)
            else:
                os.close(self.handle)
            self.handle = None


class LocalRecipeFailureLifecycle(unittest.TestCase):
    cli: Path
    device: str
    attempt_timeout_ms: int

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="liuding-recipe-failure-")
        self.base = Path(self.temporary.name) / "真实故障"
        self.base.mkdir()
        self.store, self.work = self.base / "store", self.base / "training"
        self.spec_path, self.job = self.base / "local.json", "real-training"
        self.environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                CUBLAS_WORKSPACE_CONFIG=":4096:8")
        self.process = self.evidence = self.subreaper_before = None
        self.owned: list[NativeProcess] = []
        self.observation: dict = {"actual_training_processes": 0,
                                  "checkpoint_update_lower_bound": 0,
                                  "checkpoint_file_observed": False,
                                  "checkpoint_container_verified": False,
                                  "configured_update_ceiling": 4000,
                                  "exact_interrupted_update_count_known": False,
                                  "resource_limits_are_logical": True,
                                  "supervisor_killed_processes": []}
        self.command_count = 0
        self.addCleanup(self.cleanup)

    def save(self, name: str, value) -> None:
        if self.evidence:
            (self.evidence / (name + ".json")).write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def command(self, *arguments: object) -> list[str]:
        return [str(self.cli), *map(str, arguments), "--format", "json"]

    def invoke(self, *arguments: object, expected: int = 0) -> dict:
        self.command_count += 1
        result = subprocess.run(self.command(*arguments), capture_output=True,
                                timeout=60, env=self.environment, check=False)
        if self.evidence:
            stem = self.evidence / f"command-{self.command_count:02d}"
            stem.with_suffix(".stdout.raw").write_bytes(result.stdout)
            stem.with_suffix(".stderr.raw").write_bytes(result.stderr)
            self.save(stem.name, {"argv": self.command(*arguments), "returncode": result.returncode,
                                  "expected_returncode": expected})
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        envelope = json.loads(result.stdout)
        return envelope.get("data", envelope)

    def local(self, action: str, expected: int = 0, job: str | None = None) -> dict:
        options = ("--spec", self.spec_path) if action == "execute" else ()
        return self.invoke("local", action, "--root", self.store,
                           "--id", job or self.job, *options, expected=expected)

    def references(self, run: str) -> dict:
        return {row["reference_id"]: row for row in self.invoke(
            "run", "show", "--root", self.store, "--id", run)["references"]}

    def blob(self, reference: dict) -> bytes:
        row = reference["artifact"]
        identity = row["sha256"]
        data = (self.store / "artifacts/objects" / identity[:2] / identity[2:]).read_bytes()
        self.assertEqual(len(data), row["size_bytes"])
        self.assertEqual(hashlib.sha256(data).hexdigest(), identity)
        return data

    def prepare(self, case: str) -> None:
        if output := os.environ.get("JYPPX_LIUDING_RECIPE_EVIDENCE_DIR"):
            evidence = Path(output) / case / self.device
            evidence.mkdir(parents=True, exist_ok=False)
            self.evidence = evidence
        if os.name != "nt":
            # Only this test supervisor adopts/reaps children after worker
            # death. This never proves that the product cleaned an orphan.
            libc = ctypes.CDLL(None, use_errno=True)
            previous = ctypes.c_int()
            if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0 or libc.prctl(36, 1, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "Cannot establish test-only orphan reaping")
            self.subreaper_before = previous.value
        for split, count in (("train", 8), ("validation", 4)):
            for label, name in enumerate(("green", "red")):
                folder = self.base / "images" / split / name
                folder.mkdir(parents=True)
                for number in range(count):
                    (folder / f"{number}.png").write_bytes(png(label, number))
        config = {
            "schema_version": 1, "recipe": "jyppx.liuding.imagefolder.resnet18/v1",
            "train_root": "images/train", "validation_root": "images/validation",
            "data_license": "CC0-1.0; self-authored RGB fixture", "device": self.device,
            "seed": 20261028, "epochs": 1000, "batch_size": 4, "image_size": 32,
            "workers": 2, "prefetch_batches": 2, "learning_rate": 0.01,
            "momentum": 0.9, "weight_decay": 0.0001, "lr_gamma": 0.9,
            "checkpoint_every": 16, "minimum_accuracy": 0.875,
            "max_image_bytes": 1048576, "max_data_bytes": 10485760,
            "max_samples": 1024, "image_budget_bytes": 67108864,
        }
        config_path = self.base / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        plan = self.invoke("recipe", "local-spec", "--file", config_path, "--root", self.work,
                           "--store", self.store, "--spec", self.spec_path,
                           "--executable", self.cli, "--pause-after", 3999,
                           "--timeout-ms", self.attempt_timeout_ms)
        self.spec = plan["specification"]
        memory = 2 << 30
        self.spec["resources"] = {
            "capacity": {"cpu_slots": 1, "memory_bytes": memory,
                         "gpus": {"gpu0": memory} if self.device == "cuda" else {}},
            "request": {"cpu_slots": 1, "memory_bytes": memory,
                        "gpu": {"id": "gpu0", "memory_bytes": memory}
                        if self.device == "cuda" else None},
        }
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        self.assertEqual((self.spec["max_attempts"], self.spec["retry_exit_codes"]), (2, [75]))
        self.save("spec", self.spec)
        self.save("config", config)
        self.observation.update(case=case, timeout_ms=self.attempt_timeout_ms,
                                fault="suspended_real_trainer", test_only_subreaper=os.name != "nt")

    def await_checkpoint_and_suspend(self) -> None:
        started = time.monotonic()
        self.process = subprocess.Popen(self.command("local", "execute", "--root", self.store,
                                       "--id", self.job, "--spec", self.spec_path),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       env=self.environment)
        # A full minute is the default attempt budget. Leave at least 15s for
        # identity capture/inspection; slow hosts can increase it explicitly.
        deadline = started + min(120, self.attempt_timeout_ms / 1000 - 15)
        while not (self.work / "latest.jyck").is_file():
            self.assertIsNone(self.process.poll(), "Trainer ended before a durable checkpoint")
            self.assertLess(time.monotonic(), deadline, "No checkpoint within the declared supervision budget")
            time.sleep(0.01)
        self.observation["first_checkpoint_seconds"] = time.monotonic() - started
        # The native recipe has already trained before this file appears.
        # Identity/inspection failures must not turn that real process into a
        # reported zero-model attempt.
        self.observation["actual_training_processes"] = 1
        self.observation["checkpoint_file_observed"] = True
        state = self.local("show")
        self.assertEqual(len(state["attempts"]), 1)
        attempt = state["attempts"][0]
        self.assertIsNone(attempt["result"])
        self.assertTrue(state["owner_active"])
        self.attempt_id = attempt["attempt_id"]
        references = self.references(self.attempt_id)
        owner = json.loads(self.blob(references["local.owner"]))
        self.assertEqual((owner["job_id"], owner["attempt_id"], owner["spec_sha256"]),
                         (self.job, self.attempt_id, state["spec_sha256"]))
        self.worker = NativeProcess(owner["worker_process_id"], self.process.pid, self.cli)
        self.owned.append(self.worker)
        children = self.capture_children(self.worker, "worker")
        trainers = [row for row in children if row["role"] == "native_cli"]
        hosts = [row for row in children if row["role"] == "system_console_host"]
        self.assertEqual(len(trainers), 1, "Expected exactly one child matching the frozen native CLI")
        self.assertLessEqual(len(hosts), 1, "Unexpected extra worker console hosts")
        self.assertTrue(all(row["role"] in ("native_cli", "system_console_host") for row in children),
                        "Worker has an unknown extra child; see pinned identity evidence")
        selected = trainers[0]
        self.trainer = NativeProcess(selected["pid"], self.worker.pid, self.cli)
        self.owned.append(self.trainer)
        self.assertEqual(self.trainer.identity["created"], selected["created"], "Trainer identity changed")
        self.assertNotIn(self.trainer.pid, (self.process.pid, self.worker.pid, os.getpid()))
        descendants = self.capture_children(self.trainer, "trainer")
        self.assertLessEqual(len(descendants), 1, "Unexpected extra trainer subprocesses")
        self.assertTrue(all(row["role"] == "system_console_host" for row in descendants),
                        "Trainer has an unknown subprocess; see pinned identity evidence")
        if os.name != "nt":
            self.assertEqual(self.trainer.identity["group"], self.trainer.pid)
            arguments = Path(f"/proc/{self.trainer.pid}/cmdline").read_bytes().split(b"\0")
            self.assertEqual(arguments[1:3], [b"recipe", b"local-run"])
        self.trainer.suspend()
        self.assertTrue(self.trainer.running())
        self.observation["suspended_seconds_after_start"] = time.monotonic() - started
        self.observation["captured_native_processes"] = [p.identity for p in self.owned]
        self.observation["checkpoint_sha256"] = sha256(self.work / "latest.jyck")
        checkpoint = self.invoke("checkpoint", "inspect", "--path", self.work / "latest.jyck")
        self.assertTrue(checkpoint["valid"])
        self.assertEqual({p["state"]["participant_id"] for p in checkpoint["manifest"]["participants"]}, PARTICIPANTS)
        self.observation["checkpoint_update_lower_bound"] = 16
        self.observation["checkpoint_container_verified"] = True
        self.save("checkpoint", checkpoint)
        self.save("owner", owner)
        self.workspace_before = self.workspace_snapshot()
        self.assertFalse((self.work / "final.jyck").exists())
        self.assertFalse((self.work / "result.json").exists())
        self.assertTrue(self.worker.running())
        self.observation["fault_injection_completed"] = True

    def capture_children(self, parent: NativeProcess, label: str) -> list[dict]:
        """Capture all direct children; only the observed exact OS host is allowed.

        Non-CLI identities receive query/wait access and close immediately.
        They can never enter the fixture's kill/suspend ownership list.
        """
        host = windows_console_host() if os.name == "nt" else None
        rows: list[dict] = []
        self.observation[label + "_direct_children"] = rows
        parents = process_parents()
        for pid, owner in parents.items():
            if owner != parent.pid:
                continue
            child = NativeProcess(pid, parent.pid, control=False)
            try:
                row = dict(child.identity)
                image = os.path.normcase(str(Path(row["executable"]).resolve()))
                row["role"] = (
                    "native_cli" if image == os.path.normcase(str(self.cli)) else
                    "system_console_host" if host is not None and image == os.path.normcase(str(host)) else
                    "unexpected")
                row["read_only_capture"] = True
                rows.append(row)
                self.assertGreaterEqual(row["created"], parent.identity["created"])
                self.assertTrue(child.running(), "Observed child exited during identity capture")
                if row["role"] == "system_console_host":
                    # The permitted host is a leaf. Capture any additional
                    # descendants before rejecting them, rather than hiding an
                    # arbitrary program behind a trusted host image name.
                    extras = []
                    row["direct_children"] = extras
                    for extra_pid, extra_parent in process_parents().items():
                        if extra_parent == child.pid:
                            extra = NativeProcess(extra_pid, child.pid, control=False)
                            try:
                                extras.append(dict(extra.identity))
                            finally:
                                extra.close()
                    self.assertEqual(extras, [], "System console host has an unexpected descendant")
            finally:
                child.close()
                self.save(label + "-direct-children", rows)
        return rows

    def workspace_snapshot(self) -> dict:
        return {str(path.relative_to(self.work)): [path.stat().st_size, path.stat().st_mtime_ns]
                for path in self.work.rglob("*") if path.is_file()}

    def probe(self, *, recover: bool = False, expected: int = 7) -> dict:
        path = self.base / "competitor.json"
        if not path.exists():
            probe = json.loads(json.dumps(self.spec))
            probe["arguments"][probe["arguments"].index("--root") + 1] = str(self.base / "never-trained")
            probe["training"]["model_id"] = "wrong-model"
            path.write_text(json.dumps(probe), encoding="utf-8")
        options = () if recover else ("--spec", path)
        return self.invoke("local", "recover" if recover else "execute", "--root", self.store,
                           "--id", "competitor", *options, expected=expected)

    def assert_no_success_or_retry(self, state: dict) -> tuple[dict, dict]:
        self.assertEqual(len(state["attempts"]), 1)
        attempt = state["attempts"][0]
        self.assertEqual(attempt["attempt_id"], self.attempt_id)
        self.assertTrue(attempt["resource_reserved"])
        references, job_refs = self.references(self.attempt_id), self.references(self.job)
        self.assertTrue({"checkpoint.last", "model.weights", "training.result",
                         "local.training.outputs"}.isdisjoint(references))
        self.assertNotIn("local.training.inputs.2", job_refs)
        self.assertNotIn("local.resource.lease.2", job_refs)
        self.assertFalse((self.work / "final.jyck").exists())
        self.assertFalse((self.work / "result.json").exists())
        self.assertEqual(sha256(self.work / "latest.jyck"), self.observation["checkpoint_sha256"])
        for name in ("local.intent", "local.owner", "local.training.inputs", "local.result", "local.stdout", "local.stderr"):
            if self.evidence and name in references:
                (self.evidence / (name + ".raw")).write_bytes(self.blob(references[name]))
        return references, job_refs

    def collect_execution(self, expected: int) -> dict:
        stdout, stderr = self.process.communicate(timeout=self.attempt_timeout_ms / 1000 + 30)
        if self.evidence:
            (self.evidence / "execute.stdout.raw").write_bytes(stdout)
            (self.evidence / "execute.stderr.raw").write_bytes(stderr)
        self.assertEqual(self.process.returncode, expected, stdout + stderr)
        state = json.loads(stdout)["data"]
        self.save("execution", state)
        return state

    def test_timeout_after_checkpoint_cleans_native_process_and_releases_lease(self) -> None:
        self.prepare("timeout")
        self.await_checkpoint_and_suspend()
        blocked = self.probe()
        self.assertEqual((blocked["status"], blocked["attempts"]), ("resource_unavailable", []))
        state = self.collect_execution(1)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["run_status"], "failed")
        self.assertFalse(state["cancel_requested"])
        attempt, result = state["attempts"][0], state["attempts"][0]["result"]
        self.assertEqual(result["status"], "timed_out")
        self.assertIsNone(result["exit_code"])
        self.assertIsNone(result["signal"])
        self.assertEqual(result["process_id"], self.trainer.pid)
        self.assertTrue(result["process_tree_cleanup_confirmed"])
        self.assertTrue(attempt["resource_released"])
        self.assertFalse(result.get("training_outputs_confirmed", False))
        self.assertTrue(all(not child.running() for child in self.owned))
        references, job_refs = self.assert_no_success_or_retry(state)
        release = json.loads(self.blob(job_refs["local.resource.release.1"]))
        self.assertEqual(release["lease_sha256"], job_refs["local.resource.lease.1"]["artifact"]["sha256"])
        for role in ("intent", "owner", "result", "stdout", "stderr"):
            self.assertEqual(release[role + "_sha256"], references["local." + role]["artifact"]["sha256"])
        for action in ("recover", "execute"):
            self.assertEqual(self.local(action, expected=1), state)
        self.assertEqual(self.workspace_snapshot(), self.workspace_before)
        admitted = self.probe(recover=True, expected=1)
        self.assertEqual(admitted["status"], "failed")
        self.assertEqual(len(admitted["attempts"]), 1)
        rejection = admitted["attempts"][0]
        self.assertEqual(rejection["result"]["exit_code"], 2)
        self.assertTrue(rejection["result"]["process_tree_cleanup_confirmed"])
        self.assertTrue(rejection["resource_released"])
        self.assertFalse((self.base / "never-trained").exists())
        self.invoke("run", "verify", "--root", self.store)
        self.save("release", release)
        self.save("admitted-rejection", admitted)
        self.observation.update(native_timeout_cleanup_confirmed=True, logical_lease_released=True,
                                workspace_metadata_replay_equal=True, post_timeout_native_processes_alive=False)

    def assert_unknown(self, state: dict) -> None:
        self.assertEqual(state["status"], "recovery_required")
        self.assertEqual(state["run_status"], "running")
        self.assertFalse(state["owner_active"])
        self.assertEqual(len(state["attempts"]), 1)
        attempt = state["attempts"][0]
        self.assertEqual(attempt["run_status"], "running")
        self.assertIsNone(attempt["result"])
        self.assertTrue(attempt["resource_reserved"])
        self.assertFalse(attempt["resource_released"])
        references, job_refs = self.assert_no_success_or_retry(state)
        self.assertTrue({"local.result", "local.stdout", "local.stderr"}.isdisjoint(references))
        self.assertNotIn("local.resource.release.1", job_refs)

    def test_killed_worker_remains_unknown_and_retains_lease_after_os_cleanup(self) -> None:
        self.prepare("worker-kill")
        self.await_checkpoint_and_suspend()
        blocked = self.probe()
        self.assertEqual((blocked["status"], blocked["attempts"]), ("resource_unavailable", []))
        self.worker.kill()  # Kill the persisted owner, not the dispatch caller.
        self.observation["fault"] = "killed_owner_worker_after_suspending_real_trainer"
        state = self.collect_execution(7)
        self.assert_unknown(state)
        self.assertFalse(self.worker.running())
        self.observation["trainer_alive_before_supervisor_cleanup"] = self.trainer.running()
        if os.name == "nt":
            self.assertFalse(self.trainer.running(), "Native kill-on-close Job retained its trainer")
        # POSIX's nested trainer owns a different process group. If it survives
        # worker death, only the test supervisor stops/reaps it. This is not a
        # native cleanup confirmation and must never release the durable lease.
        self.supervisor_cleanup()
        self.assertTrue(all(not child.running() for child in self.owned))
        for action in ("recover", "execute", "recover"):
            self.assertEqual(self.local(action, expected=7), state)
        cancellation = self.local("cancel")
        self.assertTrue(cancellation["cancel_requested"])
        after_cancel = self.local("recover", expected=7)
        self.assert_unknown(after_cancel)
        self.assertEqual(self.local("execute", expected=7), after_cancel)
        denied = self.probe(recover=True)
        self.assertEqual((denied["status"], denied["attempts"]), ("resource_unavailable", []))
        self.assertEqual(self.workspace_snapshot(), self.workspace_before)
        self.invoke("run", "verify", "--root", self.store)
        self.save("unknown-after-cancel", after_cancel)
        self.save("blocked-after-os-cleanup", denied)
        self.observation.update(native_cleanup_result_available=False, logical_lease_released=False,
                                unknown_attempt_still_fenced_after_os_cleanup=True,
                                workspace_metadata_replay_equal=True, post_supervisor_cleanup_native_processes_alive=False)

    def supervisor_cleanup(self) -> None:
        for child in reversed(self.owned):
            if child.running():
                child.kill()
                self.observation["supervisor_killed_processes"].append(child.pid)
        if os.name != "nt":
            deadline = time.monotonic() + 10
            for child in reversed(self.owned):
                while True:
                    try:
                        pid, _ = os.waitpid(child.pid, os.WNOHANG)
                    except ChildProcessError:
                        break  # Already reaped by its actual native parent.
                    if pid == child.pid:
                        break
                    self.assertLess(time.monotonic(), deadline, "Could not reap an adopted native process")
                    time.sleep(0.01)

    def cleanup(self) -> None:
        try:
            if self.process and self.process.poll() is None:
                try:
                    self.local("cancel")
                    self.process.communicate(timeout=10)
                except (AssertionError, subprocess.TimeoutExpired):
                    self.supervisor_cleanup()
                    self.process.kill()
                    self.process.communicate(timeout=10)
            self.supervisor_cleanup()
            self.assertTrue(all(not child.running() for child in self.owned), "Test retained an owned native process")
        finally:
            self.save("observation", self.observation)
            for child in self.owned:
                child.close()
            if self.subreaper_before is not None:
                libc = ctypes.CDLL(None, use_errno=True)
                if libc.prctl(36, self.subreaper_before, 0, 0, 0) != 0:
                    raise OSError(ctypes.get_errno(), "Cannot restore test supervisor policy")
            self.temporary.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-ms", dest="attempt_timeout_ms", type=int, default=60000)
    args, rest = parser.parse_known_args()
    if not 60000 <= args.attempt_timeout_ms <= 300000:
        parser.error("--timeout-ms must be between 60000 and 300000")
    LocalRecipeFailureLifecycle.cli = args.cli.resolve()
    LocalRecipeFailureLifecycle.device = args.device
    LocalRecipeFailureLifecycle.attempt_timeout_ms = args.attempt_timeout_ms
    unittest.main(argv=[__file__, *rest])
