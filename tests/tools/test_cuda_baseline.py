# Copyright 2026 JYPPX Training Framework Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the hardware-independent CUDA baseline report logic."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import run_cuda_baseline  # noqa: E402


def _sample(mode: str, throughput: float = 1000.0) -> dict[str, object]:
    optimizer_count = run_cuda_baseline.EXPECTED_OPTIMIZERS[mode]
    return {
        "schema_version": 1,
        "kind": "jyppx-cuda-training-benchmark-sample",
        "mode": mode,
        "workload": {
            "id": "jyppx.cuda.training.mlp/v1",
            "digest": "fnv1a64:0123456789abcdef",
            "seed": 7,
            "batch_size": 4,
            "steps": 10,
            "sample_count": 40,
            "input_features": 8,
            "hidden_features": 16,
            "output_features": 2,
            "worker_count": 1,
            "prefetch_batches": 2,
        },
        "build_fingerprint": {
            "jyppx_version": "0.1.0",
            "git_commit": "fixture",
            "compiler": "GNU 11.4",
            "cxx_standard": "202002",
            "os_arch": "Linux-x86_64",
            "build_type": "Release",
            "libtorch_version": "2.5.1",
            "libtorch_build": "cuda-12.4-cxx11-abi1",
            "accelerator_runtime": "cuda:available:1",
            "dependency_lock_digest": "fixture",
        },
        "cuda": {
            "device_index": 0,
            "device_name": "Fixture GPU",
            "compute_capability": "7.5",
            "compile_toolkit_version": 12040,
        },
        "runtime": {"intraop_threads": 1, "interop_threads": 1},
        "result": {
            "samples_per_second": throughput,
            "mean_step_milliseconds": 4.0,
            "peak_allocated_bytes": 1024,
            "peak_reserved_bytes": 2048,
            "global_step": 10,
            "optimizer_steps": [10] * optimizer_count,
            "grad_scaler": (
                {
                    "enabled": True,
                    "scale": 128.0,
                    "checkpoint_ready": True,
                }
                if mode == "cuda-fp16-multi-sgd-amp"
                else None
            ),
        },
    }


def _optimizer_sample(
    mode: str, throughput: float = 1000000.0
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "jyppx-cuda-optimizer-benchmark-sample",
        "mode": mode,
        "workload": {
            "id": "jyppx.cuda.optimizer.dense/v1",
            "digest": "fnv1a64:fedcba9876543210",
            "seed": 7,
            "steps": 10,
            "parameter_count": 8,
            "parameter_elements": 2048,
            "dtype": "float32",
            "gradient_policy": "fixed-dense-per-parameter/v1",
            "option_profile": "stateful-representative/v1",
        },
        "build_fingerprint": _sample(
            "cuda-fp32-single-sgd"
        )["build_fingerprint"],
        "cuda": _sample("cuda-fp32-single-sgd")["cuda"],
        "runtime": {"intraop_threads": 1, "interop_threads": 1},
        "result": {
            "parameter_elements_per_second": throughput,
            "mean_step_milliseconds": 2.0,
            "peak_allocated_bytes": 4096,
            "peak_reserved_bytes": 8192,
            "completed_steps": 10,
            "parameters_finite": True,
            "parameters_changed": True,
        },
    }


def _gpu_identity() -> dict[str, str]:
    return {
        "index": "0",
        "name": "Fixture GPU",
        "uuid": "GPU-fixture",
        "driver_version": "580.0",
        "power.limit": "160.00",
        "persistence_mode": "Enabled",
    }


class CudaBaselineTest(unittest.TestCase):
    """Covers sample invariants and environment-key construction."""

    def test_builds_two_mode_report_with_eight_metrics(self) -> None:
        measured = {
            mode: [_sample(mode, 1000.0 + index) for index in range(5)]
            for mode in run_cuda_baseline.TRAINING_MODES
        }
        warmups = {mode: [_sample(mode)] for mode in measured}
        report = run_cuda_baseline.build_raw_report(
            measured, warmups, _gpu_identity(), [{"status": "recorded"}]
        )
        self.assertTrue(
            report["environment_key"].startswith("jyppx-cuda-training-v1:")
        )
        self.assertEqual(len(report["metrics"]), 8)
        self.assertTrue(all(len(metric["samples"]) == 5 for metric in report["metrics"]))

    def test_rejects_incomplete_optimizer_cycle(self) -> None:
        sample = _sample("cuda-fp16-multi-sgd-amp")
        sample["result"]["optimizer_steps"][1] = 9
        with self.assertRaisesRegex(ValueError, "optimizer step counter"):
            run_cuda_baseline.validate_sample(sample)

    def test_builds_eight_mode_optimizer_report(self) -> None:
        measured = {
            mode: [_optimizer_sample(mode, 1000000.0 + index) for index in range(5)]
            for mode in run_cuda_baseline.OPTIMIZER_MODES
        }
        warmups = {mode: [_optimizer_sample(mode)] for mode in measured}
        report = run_cuda_baseline.build_raw_report(
            measured,
            warmups,
            _gpu_identity(),
            [{"status": "recorded"}],
            suite="optimizer",
        )
        self.assertTrue(
            report["environment_key"].startswith("jyppx-cuda-optimizer-v1:")
        )
        self.assertEqual(len(report["metrics"]), 32)
        self.assertTrue(
            all(len(metric["samples"]) == 5 for metric in report["metrics"])
        )

    def test_rejects_optimizer_without_semantic_postconditions(self) -> None:
        sample = _optimizer_sample("cuda-fp32-optimizer-lamb")
        sample["result"]["parameters_changed"] = False
        with self.assertRaisesRegex(ValueError, "did not update"):
            run_cuda_baseline.validate_sample(sample)

    def test_rejects_mode_from_another_suite(self) -> None:
        sample = _optimizer_sample("cuda-fp32-optimizer-adam")
        with self.assertRaisesRegex(ValueError, "expected suite training"):
            run_cuda_baseline.validate_sample(
                sample, expected_suite="training"
            )

    def test_rejects_cross_process_workload_change(self) -> None:
        first = _sample("cuda-fp32-single-sgd")
        changed = copy.deepcopy(first)
        changed["workload"]["digest"] = "fnv1a64:changed"
        with self.assertRaisesRegex(ValueError, "changed identity"):
            run_cuda_baseline.build_raw_report(
                {"cuda-fp32-single-sgd": [first, changed, first]},
                {"cuda-fp32-single-sgd": [first]},
                _gpu_identity(),
                [],
            )

    def test_power_limit_changes_environment_key(self) -> None:
        mode = "cuda-fp32-single-sgd"
        measured = {mode: [_sample(mode) for _ in range(3)]}
        warmups = {mode: [_sample(mode)]}
        first = run_cuda_baseline.build_raw_report(
            measured, warmups, _gpu_identity(), []
        )
        changed_gpu = _gpu_identity()
        changed_gpu["power.limit"] = "120.00"
        second = run_cuda_baseline.build_raw_report(
            measured, warmups, changed_gpu, []
        )
        self.assertNotEqual(first["environment_key"], second["environment_key"])

    def test_rejects_nonzero_process_visible_device(self) -> None:
        original = sys.argv
        try:
            sys.argv = [
                "run_cuda_baseline.py",
                "--benchmark",
                __file__,
                "--output-dir",
                str(Path(__file__).parent / "unused-output"),
                "--device-index",
                "1",
            ]
            with self.assertRaisesRegex(ValueError, "must be 0"):
                run_cuda_baseline.main()
        finally:
            sys.argv = original


if __name__ == "__main__":
    unittest.main()
