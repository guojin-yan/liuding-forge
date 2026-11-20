# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Native ROCm physical placement, admission and durable replay contracts."""
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
    "rocm_local_attempts", Path(__file__).with_name("test_local_attempt.py"))
ATTEMPTS = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(ATTEMPTS)
search_spec = importlib.util.spec_from_file_location(
    "rocm_local_search", Path(__file__).with_name("test_local_search.py"))
SEARCH = importlib.util.module_from_spec(search_spec)
search_spec.loader.exec_module(SEARCH)
MASKS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
ID = "AMD-0000-0a-00-0-" + "1" * 32


def resources(identity, capacity=1 << 30):
    return {"capacity": {"cpu_slots": 1, "memory_bytes": 1 << 30,
                         "gpus": {identity: capacity}},
            "request": {"cpu_slots": 1, "memory_bytes": 64 << 20,
                        "gpu": {"id": identity, "memory_bytes": 64 << 20}}}


class RocmSchemaTests(ATTEMPTS.LocalAttemptTest):
    def test_rocm_invalid_schema_fails_before_store(self):
        original = copy.deepcopy(self.spec)
        device = {"id": ID, "ordinal": 0, "architecture": "gfx1100"}
        valid = dict(original, rocm_device=device, resources=resources(ID))
        invalid = [None, False, [], {}, dict(device, extra=True)]
        invalid += [dict(device, id=value) for value in
                    ("0", "GPU-" + "0" * 36, ID.lower(), ID[:-1] + "A", ID + "\x00",
                     ID.replace("-00-0-", "-20-0-"), ID[:17] + "0" * 32)]
        invalid += [dict(device, ordinal=value) for value in (True, -1, 64, "0", 0.5)]
        invalid += [dict(device, architecture=value) for value in (None, 0, "", "gfx", "sm_89", "g" * 256)]
        invalid += [dict(device, minimum_free_memory_bytes=value) for value in (False, 0, -1, "1", 0.5)]
        plans = [dict(valid, rocm_device=value) for value in invalid]
        plans += [dict(original, rocm_device=device),
                  dict(valid, cuda_device={"uuid": "GPU-00000000-0000-0000-0000-000000000000"}),
                  dict(valid, resources=resources("another"))]
        for name in MASKS:
            plans.append(dict(valid, environment={name.lower() if os.name == "nt" else name: "0"}))
            plans.append(dict(valid, environment_refs={name: {
                "source": "process_environment", "name": "TEST_MASK"}}))
        for number, plan in enumerate(plans):
            with self.subTest(number=number):
                self.spec = plan
                self.write_spec()
                self.invoke("local", "execute", "--root", self.root, "--id", "job-one",
                            "--spec", self.spec_path,
                            expected=3 if isinstance(plan.get("rocm_device"), dict)
                            and plan["rocm_device"].get("ordinal") == 64 else 2)
                self.assertFalse(self.root.exists())
                self.assertFalse((self.cwd / "launches.txt").exists())

    def test_rocm_environment_digest_contains_all_owned_masks(self):
        original = copy.deepcopy(self.spec)
        self.spec.update(rocm_device={"id": ID, "ordinal": 3, "architecture": "gfx1100"},
                         resources=resources(ID))
        self.write_spec()
        with patch.dict(os.environ, {name: "parent-mask" for name in MASKS}):
            bound = self.invoke("local", "environment", "--spec", self.spec_path)["data"]
            self.assertTrue(all(os.environ[name] == "parent-mask" for name in MASKS))
            self.spec = original
            self.spec["environment"].update({name: "3" for name in MASKS})
            self.write_spec()
            explicit = self.invoke("local", "environment", "--spec", self.spec_path)["data"]
        self.assertEqual(bound, explicit)
        self.assertFalse(self.root.exists())


class RocmPlacementTests(ATTEMPTS.LocalAttemptTest):
    def setUp(self):
        super().setUp()
        self.assertTrue(all(name not in os.environ for name in MASKS),
                        "Physical ROCm qualification requires an unmasked launcher")
        inventory = self.invoke("local", "rocm-devices")["data"]
        self.assertEqual(inventory["source"], "hip")
        self.assertTrue(inventory["devices"], "ROCm acceptance requires an actual AMD GPU")
        self.device = inventory["devices"][0]
        self.identity = self.device["id"]
        self.spec.update(rocm_device={name: self.device[name] for name in ("id", "ordinal", "architecture")},
                         resources=resources(self.identity, self.device["total_memory_bytes"]))
        self.write_spec()

    def native_fixture(self):
        executable = self.cwd / ("rocm-probe.exe" if os.name == "nt" else "rocm-probe")
        shutil.copy2(self.rocm_fixture, executable)
        self.spec.update(executable=str(executable), executable_sha256=ATTEMPTS.sha256(executable),
                         arguments=[], max_attempts=1, retry_exit_codes=[], timeout_ms=30000)
        self.write_spec()

    def test_rocm_identity_ordinal_architecture_memory_and_masks_refuse_before_store(self):
        valid = copy.deepcopy(self.spec)
        device = valid["rocm_device"]
        invalid = [dict(valid, rocm_device=dict(device, id=ID), resources=resources(ID)),
                   dict(valid, rocm_device=dict(device, ordinal=63)),
                   dict(valid, rocm_device=dict(device, architecture="gfx9999")),
                   dict(valid, rocm_device=dict(device,
                       minimum_free_memory_bytes=self.device["total_memory_bytes"] + 1))]
        excessive = copy.deepcopy(valid)
        excessive["resources"]["capacity"]["gpus"][self.identity] += 1
        invalid.append(excessive)
        for index, plan in enumerate(invalid):
            with self.subTest(index=index):
                self.spec = plan
                self.write_spec()
                self.invoke("local", "execute", "--root", self.root, "--id", "job-one",
                            "--spec", self.spec_path, expected=2 if index < 3 else 3)
                self.assertFalse(self.root.exists())
        for name in MASKS:
            with self.subTest(mask=name), patch.dict(os.environ, {name: "0"}):
                self.invoke("local", "rocm-devices", expected=2)
                self.assertFalse(self.root.exists())

    def test_rocm_native_child_matches_physical_identity_and_terminal_replay(self):
        self.native_fixture()
        self.spec["cpu_affinity"] = [ATTEMPTS.cpu_ids()[0]]
        if os.name == "nt":
            self.spec["os_memory_limit"] = {"bytes": 2 << 30}
        self.write_spec()
        self.spec["environment_sha256"] = self.invoke(
            "local", "environment", "--spec", self.spec_path)["data"]["sha256"]
        self.write_spec()
        completed = self.local("execute")
        attempt = completed["attempts"][0]
        receipt = attempt["result"]["rocm_placement"]
        self.assertEqual(receipt["id"], self.identity)
        self.assertEqual(receipt["source"], "hip_prelaunch")
        self.assertEqual(receipt["memory_source"], "hip_wddm" if os.name == "nt" else "hip")
        observed = json.loads(self.blob(self.references(attempt["attempt_id"])["local.stdout"]))
        for name in ("pci_bus_id", "uuid", "architecture"):
            self.assertEqual(observed[name], self.device[name])
        self.assertEqual(observed["device_count"], 1)
        self.assertEqual(observed["logical_device"], 0)
        self.assertTrue(observed["allocation_verified"])
        self.assertEqual(observed["visibility"], {name: str(self.device["ordinal"]) for name in MASKS})
        self.assertTrue(attempt["resource_released"])
        self.assertTrue(attempt["result"]["process_tree_cleanup_confirmed"])
        self.assertEqual(attempt["result"]["cpu_affinity"]["cpu_ids"], self.spec["cpu_affinity"])
        print(json.dumps({"case": "rocm_native_placement", "receipt": receipt,
                          "native_child": observed, "memory_boundary": attempt["result"]["memory_boundary"]}), flush=True)
        Path(self.spec["executable"]).unlink()
        with patch.dict(os.environ, {name: "unavailable" for name in MASKS}):
            for action in ("show", "recover", "execute"):
                self.assertEqual(self.local(action), completed)
        self.assertEqual(self.launch_count(), 1)

    def set_memory_margin(self):
        free = self.invoke("local", "rocm-devices")["data"]["devices"][0]["free_memory_bytes"]
        self.assertGreater(free, 4 << 30)
        # Leave room for simultaneous launcher/worker HIP contexts; the
        # independent holder touches 2 GiB to cross this 1 GiB margin.
        self.spec["rocm_device"]["minimum_free_memory_bytes"] = free - (1 << 30)
        self.write_spec()

    @contextmanager
    def occupied_memory(self):
        ready = self.directory / "rocm-holder.json"
        process = subprocess.Popen([str(self.rocm_fixture), "--hold", str(ready)],
            env=dict(os.environ, **{name: str(self.device["ordinal"]) for name in MASKS}),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertTrue(ATTEMPTS.wait_for(lambda: ready.exists() or process.poll() is not None))
            self.assertIsNone(process.poll(), "Bounded HIP holder exited before readiness")
            self.assertEqual(json.loads(ready.read_text())["uuid"], self.device["uuid"])
            free = self.invoke("local", "rocm-devices")["data"]["devices"][0]["free_memory_bytes"]
            self.assertLess(free, self.spec["rocm_device"]["minimum_free_memory_bytes"])
            yield
        finally:
            ready.unlink(missing_ok=True)
            try:
                out, err = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                out, err = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, out + err)
            self.assertTrue(ATTEMPTS.wait_for(lambda: self.invoke("local", "rocm-devices")["data"]
                ["devices"][0]["free_memory_bytes"] >= self.spec["rocm_device"]["minimum_free_memory_bytes"]))

    def test_rocm_retry_memory_drift_preserves_confirmed_prefix(self):
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

    def test_rocm_reserved_worker_rechecks_and_releases_lease(self):
        self.spec["max_attempts"] = 1
        self.set_memory_margin()
        original = self.local("execute")
        attempt = original["attempts"][0]["attempt_id"]
        lease = self.references("job-one")["local.resource.lease.1"]
        with closing(sqlite3.connect(self.root / "runs.sqlite3")) as database, database:
            database.execute("PRAGMA foreign_keys=ON")
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id=?", (attempt,))
            database.execute("DELETE FROM jyppx_runs WHERE run_id=?", (attempt,))
            database.execute("DELETE FROM jyppx_run_artifacts WHERE run_id='job-one' AND reference_id='local.resource.release.1'")
            database.execute("UPDATE jyppx_runs SET status=2 WHERE run_id='job-one'")
        with self.occupied_memory():
            failed = self.local("recover", expected=1)
        observed = failed["attempts"][0]
        self.assertEqual(observed["result"]["status"], "launch_error")
        self.assertEqual(observed["result"]["process_id"], 0)
        self.assertIsNone(observed["result"]["rocm_placement"])
        self.assertTrue(observed["result"]["process_tree_cleanup_confirmed"])
        self.assertTrue(observed["resource_released"])
        self.assertEqual(self.references("job-one")["local.resource.lease.1"], lease)
        self.assertEqual(self.local("recover", expected=1), failed)
        self.assertEqual(self.launch_count(), 1)

    def test_rocm_invalid_receipts_refuse_replay_without_relaunch(self):
        completed = self.local("execute")
        attempt = completed["attempts"][0]["attempt_id"]
        result = completed["attempts"][0]["result"]
        receipt = result["rocm_placement"]
        invalid = [None, {}, [], dict(receipt, source="sampled"), dict(receipt, id=ID),
                   dict(receipt, memory_source="process_local"),
                   dict(receipt, ordinal=False), dict(receipt, ordinal=63),
                   dict(receipt, architecture="gfx9999"), dict(receipt, free_memory_bytes=0),
                   dict(receipt, total_memory_bytes=0), dict(receipt, extra=True)]
        for value in invalid:
            with self.subTest(receipt=value):
                self.replace_result(attempt, dict(result, rocm_placement=value))
                self.invoke("local", "show", "--root", self.root, "--id", "job-one", expected=2)
                self.assertEqual(self.launch_count(), 1)
        missing = dict(result)
        missing.pop("rocm_placement")
        self.replace_result(attempt, missing)
        self.invoke("local", "show", "--root", self.root, "--id", "job-one", expected=2)


class RocmSearchTests(SEARCH.LocalSearchTests):
    def test_rocm_placement_is_frozen_in_grid_random_trials_and_terminal_replay(self):
        device = self.invoke("local", "rocm-devices")["data"]["devices"][0]
        binding = {name: device[name] for name in ("id", "ordinal", "architecture")}
        self.spec["execution"].update(rocm_device=binding,
            resources=resources(device["id"], device["total_memory_bytes"]))
        self.spec["configuration"]["case"] = "rocm_visibility"
        for strategy in ("grid", "random"):
            self.search_id = "rocm-" + strategy
            self.spec["strategy"] = strategy
            completed = self.search("execute")
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["confirmed_trials"], 3)
            for trial in completed["trials"]:
                job = self.invoke("local", "show", "--root", self.root, "--id", trial["trial_id"])["data"]
                self.assertEqual(job["attempts"][0]["result"]["rocm_placement"]["id"], device["id"])
                self.assertTrue(job["attempts"][0]["resource_released"])
                frozen, _ = SEARCH.committed_reference(self.root, trial["trial_id"], "local.spec")
                self.assertEqual(json.loads(frozen)["rocm_device"], binding)
            with patch.dict(os.environ, {name: "unavailable" for name in MASKS}):
                for action in ("show", "recover", "execute"):
                    self.assertEqual(self.search(action)["best"], completed["best"])
        self.assertEqual([row["rocm_visibility"] for row in self.launches()],
                         [{name: str(device["ordinal"]) for name in MASKS}] * 6)


HARDWARE = False


def load_tests(loader, tests, pattern):
    selected = (RocmPlacementTests, RocmSearchTests) if HARDWARE else (RocmSchemaTests,)
    return unittest.TestSuite(cls(name) for cls in selected for name in sorted(cls.__dict__)
                             if name.startswith("test_rocm_"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--hardware", action="store_true")
    args, remaining = parser.parse_known_args()
    HARDWARE = args.hardware
    RocmSchemaTests.cli = RocmPlacementTests.cli = RocmSearchTests.cli = args.cli.resolve()
    RocmPlacementTests.rocm_fixture = args.fixture.resolve() if args.fixture else None
    if HARDWARE and RocmPlacementTests.rocm_fixture is None:
        parser.error("--hardware requires a native --fixture")
    unittest.main(argv=[sys.argv[0], *remaining])
