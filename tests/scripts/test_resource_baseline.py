# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0.
"""Negative resource evidence/provenance checks and real subprocess bounds."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmarks"))
import run_resource_baseline as runner  # noqa: E402

COMMIT = "a" * 40


def sample(profile: str = "storage") -> dict:
    descriptor, operations, size = runner.workload(profile)
    metrics = {}
    for name in runner.COMMON_METRICS | runner.PROFILE_METRICS[profile]:
        if name.startswith("peak_cuda"):
            metrics[name] = {"status": "unavailable", "reason": "CPU execution", "unit": "bytes",
                             "direction": "lower", "gate": False}
            continue
        unit = ("seconds" if name.endswith("_seconds") else "elements" if name == "parameter_elements"
                else name.removeprefix("throughput_").replace("_per_", "/") if name.startswith("throughput_")
                else "bytes")
        diagnostic = name in ("load_seconds", "warmup_seconds", "calibration_conversion_seconds") or (
            profile == "serving" and name == "publish_seconds")
        metrics[name] = {"status": "available", "value": size[0] if name == "output_bytes" else 100,
                         "unit": unit, "direction": "higher" if name.startswith("throughput_") else "lower",
                         "gate": not diagnostic, "scope": "fixture"}
    return {
        "schema_version": 1, "kind": "jyppx-resource-benchmark-sample", "profile": profile,
        "workload": {"id": profile + "/v1", "descriptor": descriptor,
                     "digest": "sha256:" + hashlib.sha256(runner.canonical(descriptor).encode()).hexdigest()},
        "build_fingerprint": {"git_commit": COMMIT, "jyppx_version": "0.3.0", "compiler": "GNU 13.3",
            "cxx_standard": "202002", "os_arch": "Linux-x86_64", "build_type": "Release",
            "libtorch_version": "2.5.1", "libtorch_build": "cpu-abi1", "accelerator_runtime": "cpu",
            "dependency_lock_digest": "fixture-lock"},
        "execution": {"device": "cpu", "path": "fixture-public-api", "timing": "steady_clock",
                      "intraop_threads": 1, "interop_threads": 1},
        "validation": {"expected_operations": operations, "completed_operations": operations,
                       "finite_output": True, "content_verified": True, "output_sum": 1.0},
        "metrics": metrics,
    }


def statistics_report(commit: str = COMMIT) -> dict:
    row = sample()
    row["build_fingerprint"]["git_commit"] = commit
    environment = runner.environment_identity({"processor": "fixture"}, {"CMAKE_CXX_FLAGS": "-O3"},
                                              {"storage": row}, {}, None)
    return {"schema_version": 1, "kind": "jyppx-resource-benchmark-statistics",
        "source_provenance": {"git_commit": commit, "clean": True}, "environment": environment,
        "environment_key": "sha256:" + hashlib.sha256(runner.canonical(environment).encode()).hexdigest(),
        **runner.summarize({"storage": [copy.deepcopy(row) for _ in range(5)]}, 10)}


class ResourceEvidenceTest(unittest.TestCase):
    def test_accepts_complete_fixed_profile(self) -> None:
        runner.validate_sample(sample(), "storage", "cpu", COMMIT)

    def test_rejects_stale_binary_and_dirty_build_identity(self) -> None:
        for identity in ("b" * 40, COMMIT + "+dirty", "unknown"):
            row = sample()
            row["build_fingerprint"]["git_commit"] = identity
            with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, "binary/source"):
                runner.validate_sample(row, "storage", "cpu", COMMIT)

    def test_rejects_partial_or_nonfinite_work(self) -> None:
        for field, value in (("completed_operations", 7), ("finite_output", False), ("content_verified", False)):
            row = sample()
            row["validation"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "exact workload"):
                runner.validate_sample(row, "storage", "cpu", COMMIT)

    def test_workload_cannot_be_relabelled_with_a_new_digest(self) -> None:
        row = sample()
        row["workload"]["descriptor"]["objects"] = 1
        row["workload"]["digest"] = "sha256:" + hashlib.sha256(
            runner.canonical(row["workload"]["descriptor"]).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "fixed workload"):
            runner.validate_sample(row, "storage", "cpu", COMMIT)

    def test_unavailable_cannot_be_a_fabricated_zero_pass(self) -> None:
        row = sample()
        row["metrics"]["peak_cuda_allocated_bytes"]["value"] = 0
        with self.assertRaisesRegex(ValueError, "fabricated value"):
            runner.validate_sample(row, "storage", "cpu", COMMIT)

    def test_rejects_missing_metric_and_disabled_stability_gate(self) -> None:
        row = sample()
        del row["metrics"]["publish_seconds"]
        with self.assertRaisesRegex(ValueError, "metric set"):
            runner.validate_sample(row, "storage", "cpu", COMMIT)
        row = sample()
        row["metrics"]["publish_seconds"]["gate"] = False
        with self.assertRaisesRegex(ValueError, "stability gate"):
            runner.validate_sample(row, "storage", "cpu", COMMIT)

    def test_rejects_nonfinite_metric_and_excessive_output(self) -> None:
        row = sample()
        row["metrics"]["publish_seconds"]["value"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            runner.validate_sample(row, "storage", "cpu", COMMIT)
        row = sample()
        row["metrics"]["output_bytes"]["value"] = 1 << 40
        with self.assertRaisesRegex(ValueError, "size exceeds"):
            runner.validate_sample(row, "storage", "cpu", COMMIT)

    def test_rejects_cpu_fallback_labelled_as_cuda(self) -> None:
        row = sample("resnet18")
        with self.assertRaisesRegex(ValueError, "device mismatch"):
            runner.validate_sample(row, "resnet18", "cuda:0", COMMIT)

    def test_new_commit_compares_without_rewriting_reference_provenance(self) -> None:
        reference = runner.make_reference(statistics_report())
        preserved = copy.deepcopy(reference)
        current = statistics_report("b" * 40)
        comparison = runner.compare_reference(current, reference)
        self.assertTrue(comparison["passed"])
        self.assertTrue(comparison["reference_commit_differs"])
        self.assertEqual(reference, preserved)
        self.assertEqual(comparison["reference_source_provenance"]["git_commit"], COMMIT)
        self.assertEqual(current["source_provenance"]["git_commit"], "b" * 40)

    def test_environment_change_cannot_claim_no_regression(self) -> None:
        reference = runner.make_reference(statistics_report())
        current = statistics_report("b" * 40)
        current["environment"]["host"]["processor"] = "different CPU"
        with self.assertRaisesRegex(ValueError, "environment/workload differs"):
            runner.compare_reference(current, reference)

    def test_cv_failure_cannot_create_reference(self) -> None:
        rows = [sample() for _ in range(5)]
        rows[-1]["metrics"]["publish_seconds"]["value"] *= 10
        report = statistics_report()
        report.update(runner.summarize({"storage": rows}, 10))
        self.assertFalse(report["stability"]["passed"])
        self.assertIn("storage.publish_seconds", report["stability"]["unstable_metrics"])
        self.assertIn("storage.peak_cuda_allocated_bytes", report["stability"]["unavailable_metrics"])
        with self.assertRaisesRegex(ValueError, "unstable samples"):
            runner.make_reference(report)

    def test_real_regression_fails_and_weakened_reference_is_rejected(self) -> None:
        reference = runner.make_reference(statistics_report())
        current = statistics_report("b" * 40)
        current["metrics"]["storage.publish_seconds"]["median"] *= 1.5
        result = runner.compare_reference(current, reference)
        self.assertFalse(result["passed"])
        self.assertIn("storage.publish_seconds", result["failed_metrics"])
        reference["metrics"]["storage.publish_seconds"]["allowed_regression_percent"] = 100
        with self.assertRaisesRegex(ValueError, "allowance was modified"):
            runner.compare_reference(current, reference)


class ResourceProcessTest(unittest.TestCase):
    def test_retains_log_and_metadata_for_protocol_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "protocol"
            with self.assertRaises(ValueError):
                runner.run_process([sys.executable, "-c", "print('not json')"], directory,
                                   timeout=5, max_output_bytes=4096)
            self.assertIn("not json", (directory / "stdout.log").read_text())
            metadata = json.loads((directory / "process.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["returncode"], 0)

    def test_kills_hung_process_and_retains_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "timeout"
            with self.assertRaisesRegex(ValueError, "process timeout"):
                runner.run_process([sys.executable, "-c", "import time; time.sleep(30)"], directory,
                                   timeout=0.25, max_output_bytes=4096)
            metadata = json.loads((directory / "process.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertLess(metadata["elapsed_seconds"], 10)

    def test_bounds_both_stdout_and_stderr(self) -> None:
        for stream in ("stdout", "stderr"):
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "flood"
                code = f"import sys; sys.{stream}.write('x'*1048576); sys.{stream}.flush()"
                with self.assertRaisesRegex(ValueError, "exceeded output bound"):
                    runner.run_process([sys.executable, "-c", code], directory,
                                       timeout=5, max_output_bytes=4096)
                self.assertLessEqual((directory / (stream + ".log")).stat().st_size, 4096)
                self.assertEqual(json.loads((directory / "process.json").read_text())["status"], "failed")

    def test_success_captures_json_and_existing_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "sample"
            self.assertEqual(runner.run_process([sys.executable, "-c", 'print("{\\"sample\\":1}")'],
                             directory, timeout=5, max_output_bytes=4096), {"sample": 1})
            before = (directory / "sample.json").read_bytes()
            with self.assertRaises(FileExistsError):
                runner.run_process([sys.executable, "-c", "print('{}')"], directory,
                                   timeout=5, max_output_bytes=4096)
            self.assertEqual((directory / "sample.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
