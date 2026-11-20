# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reject incomparable hardware/process measurements and invalid training runs."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/benchmarks"))
import run_rocm_baseline as runner


def sample() -> dict:
    return {
        "schema_version": 1,
        "kind": "jyppx-rocm-training-benchmark-sample",
        "mode": "rocm-fp16-multi-sgd-amp",
        "workload": {
            "id": "jyppx.rocm.training.mlp/v1", "digest": "fnv1a64:0123456789abcdef",
            "seed": 23, "batch_size": 2, "steps": 20, "sample_count": 40,
            "input_features": 4, "hidden_features": 8, "output_features": 2,
            "worker_count": 1, "prefetch_batches": 2,
        },
        "build_fingerprint": {
            "jyppx_version": "fixture", "git_commit": "fixture", "compiler": "fixture",
            "cxx_standard": "202002", "os_arch": "Windows-x86_64", "build_type": "Release",
            "libtorch_version": "fixture", "libtorch_build": "rocm-fixture",
            "accelerator_runtime": "rocm:available:1", "dependency_lock_digest": "fixture",
        },
        "rocm": {
            "device_index": 0, "device_name": "Fixture GPU", "hip_uuid": "00112233445566778899aabbccddeeff",
            "pci_bus_id": "0000:03:00.0", "architecture": "gfx1100",
            "hip_driver_api_version": 1, "hip_runtime_version": 2, "compile_hip_version": 3,
            "total_memory_bytes": 4096,
        },
        "runtime": {"intraop_threads": 1, "interop_threads": 1},
        "result": {
            "samples_per_second": 100, "mean_step_milliseconds": 20,
            "peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048,
            "global_step": 20, "optimizer_steps": [20, 20],
            "grad_scaler": {"enabled": True, "checkpoint_ready": True, "scale": 128.0},
        },
    }


class RocmBaselineContract(unittest.TestCase):
    def report(self, samples: list[dict] | None = None, warmup: dict | None = None) -> dict:
        value = sample()
        mode = value["mode"]
        return runner.build_report(
            {mode: samples or [copy.deepcopy(value) for _ in range(5)]},
            {mode: [warmup or copy.deepcopy(value)]}, "training", value,
            {"display_driver": "fixture"}, "fixture-sha",
        )

    def test_preserves_five_independent_samples_and_protocol(self):
        values = [sample() for _ in range(5)]
        for index, value in enumerate(values):
            value["result"]["samples_per_second"] += index
        report = self.report(values)
        self.assertEqual(report["metrics"][0]["samples"], [100, 101, 102, 103, 104])
        self.assertTrue(report["metadata"]["protocol"]["hip_synchronize_timing_boundary"])
        self.assertTrue(report["environment_key"].startswith("jyppx-rocm-training-v1:"))

    def test_rejects_changed_uuid_pci_driver_build_and_threads(self):
        for section, field, changed in (
            ("rocm", "hip_uuid", "112233445566778899aabbccddeeff00"),
            ("rocm", "pci_bus_id", "0000:04:00.0"),
            ("rocm", "hip_runtime_version", 9),
            ("build_fingerprint", "git_commit", "different"),
            ("runtime", "intraop_threads", 2),
        ):
            with self.subTest(field=field):
                values = [sample() for _ in range(5)]
                values[-1][section][field] = changed
                with self.assertRaises(ValueError):
                    self.report(values)

    def test_rejects_changed_warmup_and_workload(self):
        changed = sample()
        changed["workload"]["digest"] = "fnv1a64:fedcba9876543210"
        with self.assertRaises(ValueError):
            self.report(warmup=changed)

    def test_rejects_non_rocm_invalid_uuid_and_boolean_ordinals(self):
        for section, field, changed in (
            ("build_fingerprint", "accelerator_runtime", "cuda:available:1"),
            ("rocm", "hip_uuid", "0" * 32),
            ("rocm", "device_index", False),
            ("rocm", "pci_bus_id", ""),
        ):
            with self.subTest(field=field):
                value = sample()
                value[section][field] = changed
                with self.assertRaises(ValueError):
                    runner.validate_identity(value)

    def test_rejects_skipped_steps_scaler_overflow_and_invalid_memory(self):
        for field, changed in (
            ("optimizer_steps", [20, 19]), ("global_step", 19),
            ("grad_scaler", {"enabled": True, "checkpoint_ready": True, "scale": 64.0}),
            ("peak_reserved_bytes", 8192), ("peak_allocated_bytes", 3000),
            ("samples_per_second", float("nan")),
        ):
            with self.subTest(field=field):
                value = sample()
                value["result"][field] = changed
                with self.assertRaises(ValueError):
                    runner.validate_sample(value, "training", value["mode"])

    def test_rejects_too_few_processes(self):
        with self.assertRaises(ValueError):
            self.report([sample() for _ in range(4)])

    def test_process_timeout_keeps_partial_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaises(RuntimeError):
                runner.run_native(Path(sys.executable), ["-c", "import time; print('partial', flush=True); time.sleep(5)"], output, "timeout", 0.5)
            self.assertIn("partial", (output / "timeout.stdout.json").read_text())
            self.assertIn('"timeout"', (output / "timeout.process.json").read_text())

    def test_failed_process_preserves_stderr(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaises(RuntimeError):
                runner.run_native(Path(sys.executable), ["-c", "import sys; sys.stderr.write('failure'); sys.exit(7)"], output, "failed", 5)
            self.assertEqual((output / "failed.stderr.log").read_text(), "failure")
            self.assertIn('"exit_code": 7', (output / "failed.process.json").read_text())


if __name__ == "__main__":
    unittest.main()
