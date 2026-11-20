# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Ensure short, incomplete or inconsistent runs cannot claim stability."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/benchmarks"))
import run_rocm_stability as runner
from test_rocm_baseline import sample as training_sample


def sample():
    identity = training_sample()
    identity = {k: identity[k] for k in ("schema_version", "build_fingerprint", "rocm", "runtime")}
    identity["kind"] = "jyppx-rocm-benchmark-identity"
    identity["rocm"]["total_memory_bytes"] = 2**30
    memory = {"allocated_bytes": 2**21, "reserved_bytes": 2**22, "peak_allocated_bytes": 2**21, "peak_reserved_bytes": 2**22, "device_free_bytes": 2**29, "device_total_bytes": 2**30, "rss_bytes": 2**20, "peak_rss_bytes": 2**21}
    released = {**memory, "allocated_bytes": 2**20}
    return {"schema_version": 1, "kind": "jyppx-rocm-stability-sample", "identity": identity, "precision": "fp16", "requested_seconds": 2, "elapsed_seconds": 2.1, "committed_steps": 7, "checkpoint_restores": 1, "exact_continuation_steps": 3, "pressure_bytes": 2**20, "allocator_quota_oom_recovered": True, "allocator_quota_bytes": 64*1024*1024, "allocator_rejected_bytes": 128*1024*1024, "shape_rejection_recovered": True, "injected_overflow_skips": 1, "final_loss": 0.01, "initial_memory": memory, "final_memory": copy.deepcopy(memory), "released_memory": released, "snapshots": [{**memory, "elapsed_seconds": 1.5, "committed_steps": 5, "checkpoint_bytes": 1024}]}


class RocmStabilityEvidence(unittest.TestCase):
    def validate(self, value):
        runner.validate_report(value, "fp16", 2, 1, sample()["identity"])

    def test_accepts_complete_report_and_zero_loss(self):
        self.validate(sample())
        value = sample()
        value["final_loss"] = 0
        self.validate(value)

    def test_rejects_short_duration_missing_restore_and_recovery(self):
        for field, changed in (("elapsed_seconds", 1.9), ("checkpoint_restores", 0), ("exact_continuation_steps", 2), ("allocator_quota_oom_recovered", False), ("shape_rejection_recovered", False), ("injected_overflow_skips", 0), ("final_loss", float("nan"))):
            with self.subTest(field=field):
                value = sample()
                value[field] = changed
                with self.assertRaises(ValueError):
                    self.validate(value)

    def test_rejects_missing_snapshots_and_wrong_payload(self):
        value = sample()
        value["snapshots"] = []
        with self.assertRaises(ValueError):
            self.validate(value)
        value = sample()
        value["pressure_bytes"] += 1
        with self.assertRaises(ValueError):
            self.validate(value)

    def test_rejects_device_change_and_unreleased_pressure(self):
        value = sample()
        value["identity"]["rocm"]["pci_bus_id"] = "0000:04:00.0"
        with self.assertRaises(ValueError):
            self.validate(value)
        value = sample()
        value["released_memory"]["allocated_bytes"] = value["final_memory"]["allocated_bytes"]
        with self.assertRaises(ValueError):
            self.validate(value)

    def test_rejects_false_resource_peaks(self):
        value = sample()
        value["final_memory"]["peak_rss_bytes"] = 1
        with self.assertRaises(ValueError):
            self.validate(value)


if __name__ == "__main__":
    unittest.main()
