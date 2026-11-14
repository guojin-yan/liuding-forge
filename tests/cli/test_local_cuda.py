# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Independent schema and actual GPU placement/admission contracts; no training."""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import unittest
from unittest.mock import patch


sys.dont_write_bytecode = True
module_spec = importlib.util.spec_from_file_location(
    "local_cuda_attempts", Path(__file__).with_name("test_local_attempt.py"))
ATTEMPTS = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(ATTEMPTS)
UUID = "GPU-00000000-0000-0000-0000-000000000000"
search_spec = importlib.util.spec_from_file_location(
    "local_cuda_search", Path(__file__).with_name("test_local_search.py"))
SEARCH = importlib.util.module_from_spec(search_spec)
search_spec.loader.exec_module(SEARCH)


def resources(uuid: str, capacity: int = 1 << 30) -> dict:
    return {"capacity": {"cpu_slots": 1, "memory_bytes": 1 << 30, "gpus": {uuid: capacity}},
            "request": {"cpu_slots": 1, "memory_bytes": 64 << 20,
                        "gpu": {"id": uuid, "memory_bytes": 64 << 20}}}


class CudaSchemaTests(ATTEMPTS.LocalAttemptTest):
    def test_cuda_schema_refuses_invalid_or_ambiguous_binding_before_store(self) -> None:
        original = copy.deepcopy(self.spec)
        valid = dict(original, cuda_device={"uuid": UUID}, resources=resources(UUID))
        invalid = [None, False, [], {}, {"uuid": 0}, {"uuid": UUID, "unknown": 1}]
        invalid.extend({"uuid": uuid} for uuid in ("0", "GPU-00000000", UUID.lower(),
            UUID[:-1] + "A", "MIG-" + UUID[4:], UUID + "," + UUID, UUID + "\x00"))
        invalid.extend(dict(uuid=UUID, minimum_compute_capability=value)
                       for value in (None, False, {}, [], [7], [7, 5, 0], [0, 0], [True, 0], [7, -1], [7, 5.0]))
        invalid.extend(dict(uuid=UUID, minimum_free_memory_bytes=value)
                       for value in (False, 0, -1, 1.5, "1"))
        plans = [dict(valid, cuda_device=value) for value in invalid]
        plans += [dict(original, cuda_device={"uuid": UUID}),
                  dict(valid, resources=self.resource_request()),
                  dict(valid, resources=self.resource_request(gpu=False)),
                  dict(valid, environment={"cuda_visible_devices": "0"}),
                  dict(valid, environment_refs={"CUDA_VISIBLE_DEVICES":
                      {"source": "process_environment", "name": "CUDA_MASK"}})]
        aliases = copy.deepcopy(valid)
        aliases["resources"]["capacity"]["gpus"]["gpu0"] = 1 << 30
        plans.append(aliases)
        for index, plan in enumerate(plans):
            with self.subTest(index=index):
                self.spec = plan
                self.write_spec()
                self.invoke("local", "execute", "--root", self.root, "--id", "job-one",
                            "--spec", self.spec_path, expected=2)
                self.assertFalse(self.root.exists())
                self.assertFalse((self.cwd / "launches.txt").exists())

    def test_cuda_mask_enters_effective_environment_without_changing_parent(self) -> None:
        original = copy.deepcopy(self.spec)
        self.spec.update(cuda_device={"uuid": UUID}, resources=resources(UUID))
        self.write_spec()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "parent-mask"}):
            bound = self.invoke("local", "environment", "--spec", self.spec_path)["data"]
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "parent-mask")
        self.spec = original
        self.spec["environment"]["CUDA_VISIBLE_DEVICES"] = UUID
        self.write_spec()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "another-parent-mask"}):
            explicit = self.invoke("local", "environment", "--spec", self.spec_path)["data"]
        self.assertEqual(bound, explicit)
        self.assertFalse(self.root.exists())


class CudaPlacementTests(ATTEMPTS.LocalAttemptTest):
    def setUp(self) -> None:
        super().setUp()
        inventory = self.invoke("local", "cuda-devices")["data"]
        self.assertEqual(inventory["source"], "nvml")
        self.assertTrue(inventory["devices"], "CUDA acceptance requires an actual NVIDIA GPU")
        self.device = inventory["devices"][0]
        self.uuid = self.device["uuid"]
        self.spec.update(cuda_device={"uuid": self.uuid,
            "minimum_compute_capability": self.device["compute_capability"]},
            resources=resources(self.uuid, self.device["total_memory_bytes"]))
        self.write_spec()

    def native_fixture(self) -> None:
        executable = self.cwd / ("cuda-probe.exe" if os.name == "nt" else "cuda-probe")
        shutil.copy2(self.cuda_fixture, executable)
        self.spec.update(executable=str(executable), executable_sha256=ATTEMPTS.sha256(executable),
                         arguments=[], max_attempts=1, retry_exit_codes=[])
        self.write_spec()

    @contextmanager
    def occupied_memory(self):
        ready = self.directory / "gpu-holder-ready.json"
        process = subprocess.Popen([str(self.cuda_fixture), "--hold", str(ready)],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=self.uuid),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            def ready_or_exited():
                if process.poll() is not None:
                    return True
                try:
                    return json.loads(ready.read_text())["uuid"] == self.uuid
                except (FileNotFoundError, json.JSONDecodeError):
                    return False
            self.assertTrue(ATTEMPTS.wait_for(ready_or_exited))
            self.assertIsNone(process.poll(), "Bounded CUDA holder failed before readiness")
            self.assertEqual(json.loads(ready.read_text())["uuid"], self.uuid)
            current = self.invoke("local", "cuda-devices")["data"]["devices"]
            free = next(d["free_memory_bytes"] for d in current if d["uuid"] == self.uuid)
            self.assertLess(free, self.spec["cuda_device"]["minimum_free_memory_bytes"])
            yield
        finally:
            ready.unlink(missing_ok=True)
            try:
                out, err = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                out, err = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, out + err)
            # WDDM/NVML can publish the released context memory after the
            # holder has exited. Keep the original admission threshold and
            # wait for the observable prerequisite before the next launch.
            def memory_released():
                devices = self.invoke("local", "cuda-devices")["data"]["devices"]
                free = next(d["free_memory_bytes"] for d in devices if d["uuid"] == self.uuid)
                return free >= self.spec["cuda_device"]["minimum_free_memory_bytes"]
            self.assertTrue(ATTEMPTS.wait_for(memory_released),
                            "NVML did not report the holder's GPU memory release")

    def set_memory_margin(self) -> None:
        devices = self.invoke("local", "cuda-devices")["data"]["devices"]
        free = next(d["free_memory_bytes"] for d in devices if d["uuid"] == self.uuid)
        self.assertGreater(free, 1 << 30)
        self.spec["cuda_device"]["minimum_free_memory_bytes"] = free - (128 << 20)
        self.write_spec()

    def test_cuda_absent_uuid_capability_and_memory_refuse_before_store(self) -> None:
        valid = copy.deepcopy(self.spec)
        invalid = [dict(valid, cuda_device={"uuid": UUID}, resources=resources(UUID)),
                   dict(valid, cuda_device={"uuid": self.uuid, "minimum_compute_capability": [99, 0]}),
                   dict(valid, cuda_device={"uuid": self.uuid,
                       "minimum_free_memory_bytes": self.device["total_memory_bytes"] + 1})]
        excessive = copy.deepcopy(valid)
        excessive["resources"]["capacity"]["gpus"][self.uuid] += 1
        invalid.append(excessive)
        for index, plan in enumerate(invalid):
            with self.subTest(index=index):
                self.spec = plan
                self.write_spec()
                self.invoke("local", "execute", "--root", self.root, "--id", "job-one",
                            "--spec", self.spec_path, expected=2 if index == 0 else 3)
                self.assertFalse(self.root.exists())
                self.assertFalse((self.cwd / "launches.txt").exists())

    def test_cuda_native_device_matches_uuid_and_mask_with_cpu_and_os_boundaries(self) -> None:
        self.native_fixture()
        self.spec["cpu_affinity"] = [ATTEMPTS.cpu_ids()[0]]
        # The CUDA 12.9 Windows driver context exceeds 512 MiB of process
        # commit even for this 64 KiB device allocation. Keep a hard 1 GiB
        # boundary for the successful placement case.
        host_limit = 1 << 30 if os.name == "nt" else 512 << 20
        if os.name == "nt":
            self.spec["os_memory_limit"] = {"bytes": host_limit}
        else:
            parent = os.environ.get("JYPPX_LIUDING_TEST_CGROUP_PARENT")
            self.assertTrue(parent, "CUDA resource acceptance requires a delegated memory parent")
            self.spec["os_memory_limit"] = {"bytes": host_limit, "cgroup_parent": parent}
        self.write_spec()
        # The effective hash must include the runner's UUID override.
        self.spec["environment_sha256"] = self.invoke(
            "local", "environment", "--spec", self.spec_path)["data"]["sha256"]
        self.write_spec()
        result = self.local("execute")
        attempt = result["attempts"][0]
        receipt = attempt["result"]["cuda_placement"]
        self.assertEqual(receipt["source"], "nvml_prelaunch")
        self.assertEqual(receipt["uuid"], self.uuid)
        observed = json.loads(self.blob(self.references(attempt["attempt_id"])["local.stdout"]))
        self.assertEqual(observed["uuid"], self.uuid)
        self.assertEqual(observed["visible_devices"], self.uuid)
        self.assertEqual(observed["device_count"], 1)
        self.assertEqual(observed["logical_device"], 0)
        self.assertEqual(observed["compute_capability"], receipt["compute_capability"])
        # CUDA runtime usable memory and physical NVML memory are distinct counters.
        self.assertGreater(observed["total_memory_bytes"], 0)
        self.assertTrue(attempt["resource_released"])
        self.assertEqual(attempt["result"]["memory_boundary"]["limit_bytes"], host_limit)
        if os.name == "nt":
            self.assertLessEqual(attempt["result"]["memory_boundary"]["peak_bytes"], host_limit)
        self.assertEqual(attempt["result"]["cpu_affinity"]["cpu_ids"], self.spec["cpu_affinity"])
        print(json.dumps({"case": "native_cuda_placement", "nvml_prelaunch": receipt,
            "native_cuda": observed, "memory_boundary": attempt["result"]["memory_boundary"],
            "cpu_affinity": attempt["result"]["cpu_affinity"],
            "process_tree_cleanup_confirmed": attempt["result"]["process_tree_cleanup_confirmed"],
            "resource_released": attempt["resource_released"]}), flush=True)
        Path(self.spec["executable"]).unlink()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "unavailable"}):
            for action in ("show", "recover", "execute"):
                self.assertEqual(self.local(action), result)
        self.assertEqual(self.launch_count(), 1)

    def test_cuda_retry_memory_drift_preserves_confirmed_prefix(self) -> None:
        self.set_memory_margin()
        original = self.committed_retry_prefix()
        references = self.references("job-one")
        with self.occupied_memory():
            for action in ("execute", "recover"):
                options = ["--spec", self.spec_path] if action == "execute" else []
                self.invoke("local", action, "--root", self.root, "--id", "job-one", *options, expected=3)
                self.assertEqual(self.local("show"), original)
                self.assertEqual(self.references("job-one"), references)
                self.assertEqual(self.launch_count(), 1)
        completed = self.local("recover")
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["attempts"][0], original["attempts"][0])
        self.assertEqual(self.launch_count(), 2)

    def test_cuda_reserved_worker_rechecks_and_releases_original_lease(self) -> None:
        self.spec["max_attempts"] = 1
        self.set_memory_margin()
        original = self.local("execute")
        attempt = original["attempts"][0]["attempt_id"]
        lease = self.references("job-one")["local.resource.lease.1"]
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("PRAGMA foreign_keys=ON")
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=?", (attempt,))
            database.execute("DELETE FROM jyppx_runs WHERE run_id=?", (attempt,))
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id='job-one' "
                             "AND reference_id='local.resource.release.1'")
            database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id='job-one'")
        with self.occupied_memory():
            failed = self.local("recover", expected=1)
        observed = failed["attempts"][0]
        self.assertEqual(observed["result"]["status"], "launch_error")
        self.assertEqual(observed["result"]["process_id"], 0)
        self.assertIsNone(observed["result"]["cuda_placement"])
        self.assertTrue(observed["result"]["process_tree_cleanup_confirmed"])
        self.assertTrue(observed["resource_released"])
        self.assertEqual(self.references("job-one")["local.resource.lease.1"], lease)
        self.assertEqual(self.local("recover", expected=1), failed)
        self.assertEqual(self.launch_count(), 1)

    def test_cuda_invalid_receipts_are_refused_without_relaunch(self) -> None:
        original = self.local("execute")
        attempt = original["attempts"][0]["attempt_id"]
        result = original["attempts"][0]["result"]
        receipt = result["cuda_placement"]
        invalid = [None, {}, [], dict(receipt, source="sampled"), dict(receipt, uuid=UUID),
                   dict(receipt, logical_device=1), dict(receipt, logical_device=False),
                   dict(receipt, free_memory_bytes=0), dict(receipt, total_memory_bytes=0),
                   dict(receipt, compute_capability=[1, 0]), dict(receipt, compute_capability=[7]),
                   dict(receipt, extra=1)]
        for value in invalid:
            with self.subTest(receipt=value):
                self.replace_result(attempt, dict(result, cuda_placement=value))
                self.invoke("local", "show", "--root", self.root, "--id", "job-one",
                            expected=3 if isinstance(value, dict) and value.get("logical_device") == 1 else 2)
                self.assertEqual(self.launch_count(), 1)
        legacy = dict(result)
        legacy.pop("cuda_placement")
        self.replace_result(attempt, legacy)
        self.invoke("local", "show", "--root", self.root, "--id", "job-one", expected=2)


class CudaSearchTests(SEARCH.LocalSearchTests):
    def test_cuda_uuid_is_frozen_in_each_grid_random_trial_and_terminal_replay(self) -> None:
        device = self.invoke("local", "cuda-devices")["data"]["devices"][0]
        uuid = device["uuid"]
        self.spec["execution"].update(cuda_device={"uuid": uuid},
            resources=resources(uuid, device["total_memory_bytes"]))
        self.spec["configuration"]["case"] = "cuda_visibility"
        for strategy in ("grid", "random"):
            self.search_id = "cuda-" + strategy
            self.spec["strategy"] = strategy
            completed = self.search("execute")
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["confirmed_trials"], 3)
            for trial in completed["trials"]:
                job = self.invoke("local", "show", "--root", self.root, "--id", trial["trial_id"])["data"]
                self.assertEqual(job["attempts"][0]["result"]["cuda_placement"]["uuid"], uuid)
                self.assertTrue(job["attempts"][0]["resource_released"])
                frozen, _ = SEARCH.committed_reference(self.root, trial["trial_id"], "local.spec")
                self.assertEqual(json.loads(frozen)["cuda_device"], {"uuid": uuid})
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "unavailable"}):
                for action in ("show", "recover", "execute"):
                    self.assertEqual(self.search(action)["best"], completed["best"])
        self.assertEqual([row["cuda_visible_devices"] for row in self.launches()], [uuid] * 6)


HARDWARE = False


def load_tests(loader, tests, pattern):
    selected = (CudaPlacementTests, CudaSearchTests) if HARDWARE else (CudaSchemaTests,)
    return unittest.TestSuite(cls(name) for cls in selected for name in sorted(cls.__dict__)
                             if name.startswith("test_cuda_"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--hardware", action="store_true")
    args, remaining = parser.parse_known_args()
    HARDWARE = args.hardware
    CudaSchemaTests.cli = CudaPlacementTests.cli = CudaSearchTests.cli = args.cli.resolve()
    CudaPlacementTests.cuda_fixture = args.fixture.resolve() if args.fixture else None
    if HARDWARE and CudaPlacementTests.cuda_fixture is None:
        parser.error("--hardware requires the native --fixture")
    unittest.main(argv=[sys.argv[0], *remaining])
