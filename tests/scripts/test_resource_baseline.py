# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0.
"""Negative resource evidence/provenance checks and real subprocess bounds."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmarks"))
import run_resource_baseline as runner  # noqa: E402

COMMIT = "a" * 40
STORAGE_ID = "fixture-volume-ext4-sync"
GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


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


def cuda_sample() -> dict:
    row = sample("resnet18")
    row["build_fingerprint"].update(libtorch_build="cu124-abi1", accelerator_runtime="cuda:available:12.4")
    row["execution"].update(device="cuda:0", timing="steady_clock plus CUDA synchronize",
        cuda={"device_index": 0, "device_name": "fixture GPU", "device_uuid": GPU_UUID,
              "compute_capability": "7.5", "compile_toolkit_version": 12040,
              "runtime_version": 12040, "driver_api_version": 12040})
    for name in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        row["metrics"][name] = {"status": "available", "value": 4096, "unit": "bytes",
                                 "direction": "lower", "gate": True, "scope": "fixture"}
    return row


def gpu_fixture(power: str = "160.00", persistence: str = "Enabled") -> dict:
    completed = subprocess.CompletedProcess([], 0,
        f"{GPU_UUID}, fixture GPU, 550.54.14, 6144, 00000000:01:00.0, {power}, {persistence}\n", "")
    with patch.object(runner.subprocess, "run", return_value=completed):
        return runner.gpu_identity(GPU_UUID)


def statistics_report(commit: str = COMMIT, *, row: dict | None = None, host_os: str = "Linux",
                      gpu: dict | None = None, storage_id: str | None = STORAGE_ID) -> dict:
    row = copy.deepcopy(row) if row is not None else sample()
    row["build_fingerprint"]["git_commit"] = commit
    name = row["profile"]
    environment = runner.environment_identity({"processor": "fixture", "os": host_os},
        {"CMAKE_CXX_FLAGS": "-O3"}, {name: row}, {}, gpu, storage_id)
    return {"schema_version": 1, "kind": "jyppx-resource-benchmark-statistics",
        "source_provenance": {"git_commit": commit, "git_tree": "c" * 40, "clean": True,
            "binaries": {name: {"sha256": "d" * 64, "path": "historical/unavailable/binary"}},
            "runner_sha256": "e" * 64, "cmake_cache_sha256": "f" * 64,
            "build_fingerprints": {name: copy.deepcopy(row["build_fingerprint"])}}, "environment": environment,
        "environment_key": "sha256:" + hashlib.sha256(runner.canonical(environment).encode()).hexdigest(),
        **runner.summarize({name: [copy.deepcopy(row) for _ in range(5)]}, 10)}


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
        current["source_provenance"]["git_tree"] = "1" * 40
        current["source_provenance"]["runner_sha256"] = "2" * 64
        current["source_provenance"]["cmake_cache_sha256"] = "3" * 64
        current["source_provenance"]["binaries"]["storage"]["sha256"] = "4" * 64
        current["source_provenance"]["build_fingerprints"]["storage"]["jyppx_version"] = "0.4.0"
        comparison = runner.compare_reference(current, reference)
        self.assertTrue(comparison["passed"])
        self.assertTrue(comparison["reference_commit_differs"])
        self.assertEqual(reference, preserved)
        self.assertEqual(comparison["reference_source_provenance"]["git_commit"], COMMIT)
        self.assertEqual(current["source_provenance"]["git_commit"], "b" * 40)

    def test_reference_requires_every_source_provenance_field(self) -> None:
        complete = runner.make_reference(statistics_report())
        for field in ("git_commit", "git_tree", "clean", "binaries", "build_fingerprints",
                      "runner_sha256", "cmake_cache_sha256"):
            reference = copy.deepcopy(complete)
            del reference["source_provenance"][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                runner.compare_reference(statistics_report("b" * 40), reference)
        reference = copy.deepcopy(complete)
        reference["source_provenance"] = {"git_commit": COMMIT}
        with self.assertRaisesRegex(ValueError, "git_tree"):
            runner.compare_reference(statistics_report(), reference)

    def test_rejects_dirty_and_malformed_reference_provenance(self) -> None:
        complete = runner.make_reference(statistics_report())
        for field, value in (("clean", False), ("clean", 1), ("git_tree", "bad"),
                ("git_commit", COMMIT + "+dirty"), ("git_tree", 123), ("runner_sha256", "f" * 63),
                ("cmake_cache_sha256", "G" * 64)):
            reference = copy.deepcopy(complete)
            reference["source_provenance"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                runner.compare_reference(statistics_report(), reference)

    def test_historical_binary_and_fingerprint_must_be_complete_and_consistent(self) -> None:
        complete = runner.make_reference(statistics_report())
        mutations = (
            ("binaries", {}),
            ("binaries", {"storage": {"path": "old/binary", "sha256": "invalid"}}),
            ("binaries", {"storage": {"sha256": "d" * 64}}),
            ("build_fingerprints", {}),
            ("build_fingerprints", {"storage": {**sample()["build_fingerprint"], "git_commit": "b" * 40}}),
            ("build_fingerprints", {"storage": {**sample()["build_fingerprint"], "build_type": "Debug"}}),
            ("build_fingerprints", {"storage": {**sample()["build_fingerprint"], "compiler": "other compiler"}}),
        )
        for field, value in mutations:
            reference = copy.deepcopy(complete)
            reference["source_provenance"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                runner.compare_reference(statistics_report(), reference)
        reference = copy.deepcopy(complete)
        del reference["source_provenance"]["build_fingerprints"]["storage"]["dependency_lock_digest"]
        with self.assertRaisesRegex(ValueError, "dependency_lock_digest"):
            runner.compare_reference(statistics_report(), reference)

    def test_reference_publication_checks_provenance_too(self) -> None:
        report = statistics_report()
        report["source_provenance"]["clean"] = False
        with self.assertRaisesRegex(ValueError, "clean"):
            runner.make_reference(report)

    def test_unavailable_supported_platform_rss_is_partial_evidence(self) -> None:
        for host_os in ("Windows", "Linux"):
            row = sample()
            row["metrics"]["peak_rss_bytes"] = {"status": "unavailable", "unit": "bytes",
                "direction": "lower", "gate": False, "reason": "OS RSS measurement failed"}
            runner.validate_sample(row, "storage", "cpu", COMMIT)
            report = statistics_report(row=row, host_os=host_os)
            completeness = runner.resource_completeness(report["environment"], report["metrics"])
            self.assertTrue(report["stability"]["passed"])
            self.assertFalse(completeness["passed"])
            self.assertEqual(completeness["missing_metrics"], ["storage.peak_rss_bytes"])
            with self.subTest(os=host_os), self.assertRaisesRegex(ValueError, "incomplete resource"):
                runner.make_reference(report)
            reference = runner.make_reference(statistics_report(host_os=host_os))
            reference["environment"] = copy.deepcopy(report["environment"])
            reference["environment_key"] = report["environment_key"]
            reference["metrics"]["storage.peak_rss_bytes"] = copy.deepcopy(row["metrics"]["peak_rss_bytes"])
            with self.subTest(comparison_os=host_os), self.assertRaisesRegex(ValueError, "incomplete resource"):
                runner.compare_reference(report, reference)

    def test_cpu_cuda_inapplicability_does_not_block_complete_rss_evidence(self) -> None:
        report = statistics_report()
        completeness = runner.resource_completeness(report["environment"], report["metrics"])
        self.assertTrue(completeness["passed"])
        self.assertEqual(completeness["required_metrics"], ["storage.peak_rss_bytes"])
        self.assertTrue(runner.compare_reference(report, runner.make_reference(report))["passed"])

    def test_io_profiles_require_explicit_stable_storage_identity(self) -> None:
        for name in ("storage", "checkpoint", "telemetry"):
            with self.subTest(profile=name), self.assertRaisesRegex(ValueError, "storage-environment-id"):
                runner.environment_identity({"os": "Linux"}, {}, {name: sample(name)}, {}, None)
        for value in ("", " ", "volume\nother", "x" * 257):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "storage environment ID"):
                runner.storage_identifier(["storage"], value)
        self.assertIsNone(runner.storage_identifier(["resnet18"], None))

    def test_different_storage_environment_cannot_claim_no_regression(self) -> None:
        reference = runner.make_reference(statistics_report(storage_id="local-ntfs-volume-F-sync"))
        current = statistics_report("b" * 40, storage_id="other-ntfs-volume-G-sync")
        self.assertNotEqual(reference["environment_key"], current["environment_key"])
        with self.assertRaisesRegex(ValueError, "environment/workload differs"):
            runner.compare_reference(current, reference)

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


class ResourceGpuIdentityTest(unittest.TestCase):
    def test_queries_actual_uuid_and_binds_power_and_persistence(self) -> None:
        completed = subprocess.CompletedProcess([], 0,
            f"{GPU_UUID}, fixture GPU, 550.54.14, 6144, 00000000:01:00.0, 160.00, Enabled\n", "")
        with patch.object(runner.subprocess, "run", return_value=completed) as query:
            gpu = runner.gpu_identity(GPU_UUID)
        command = query.call_args.args[0]
        self.assertIn("--id=" + GPU_UUID, command)
        self.assertIn("power.limit,persistence_mode", next(arg for arg in command if arg.startswith("--query-gpu=")))
        self.assertEqual(gpu["power.limit"], {"status": "available", "value": 160.0, "unit": "watts"})
        self.assertEqual(gpu["persistence_mode"], {"status": "available", "value": "enabled"})
        runner.validate_sample(cuda_sample(), "resnet18", "cuda:0", COMMIT)
        reference = runner.make_reference(statistics_report(row=cuda_sample(), gpu=gpu))
        for changed in (gpu_fixture("100.00"), gpu_fixture(persistence="Disabled")):
            current = statistics_report("b" * 40, row=cuda_sample(), gpu=changed)
            self.assertNotEqual(reference["environment_key"], current["environment_key"])
            with self.subTest(gpu=changed), self.assertRaisesRegex(ValueError, "environment/workload differs"):
                runner.compare_reference(current, reference)

    def test_unsupported_gpu_controls_are_explicit_and_never_unknown_to_unknown_pass(self) -> None:
        for power, persistence, name in (("[Not Supported]", "Enabled", "power.limit"),
                                        ("160.00", "N/A", "persistence_mode")):
            gpu = gpu_fixture(power, persistence)
            self.assertEqual(gpu[name]["status"], "unavailable")
            self.assertNotIn("value", gpu[name])
            self.assertIn("nvidia-smi", gpu[name]["reason"])
            report = statistics_report(row=cuda_sample(), gpu=gpu)
            self.assertFalse(runner.comparison_eligibility(report["environment"])["passed"])
            with self.subTest(control=name), self.assertRaisesRegex(ValueError, "unavailable GPU controls"):
                runner.make_reference(report)
            reference = runner.make_reference(statistics_report(row=cuda_sample(), gpu=gpu_fixture()))
            reference["environment"] = copy.deepcopy(report["environment"])
            reference["environment_key"] = report["environment_key"]
            with self.subTest(comparison_control=name), self.assertRaisesRegex(ValueError, "unavailable GPU controls"):
                runner.compare_reference(report, reference)

    def test_bad_gpu_readings_and_query_errors_do_not_become_measurements(self) -> None:
        for power in ("nan", "inf", "0", "-1", "unknown"):
            with self.subTest(power=power), self.assertRaises(ValueError):
                gpu_fixture(power)
        with self.assertRaisesRegex(ValueError, "persistence_mode"):
            gpu_fixture(persistence="unknown")
        with patch.object(runner.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.gpu_identity(GPU_UUID)


class ResourceMainTest(unittest.TestCase):
    def fixture_run(self, temporary: str, row: dict, *, gpu: dict | None = None,
                    write_reference: bool = True) -> tuple[int, Path, Path]:
        root = Path(temporary)
        source, build, output = root / "source", root / "build", root / "run"
        source.mkdir()
        build.mkdir()
        (build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={source.resolve()}\nJYPPX_LIUDING_BUILD_BENCHMARKS:BOOL=ON\n", encoding="utf-8")
        binary = build / "fixture-binary"
        binary.write_bytes(b"fixture, never executed")
        reference = root / "reference.json"
        args = ["--source-root", str(source), "--build-dir", str(build), "--output", str(output),
                "--profiles", row["profile"], "--storage-environment-id", STORAGE_ID,
                "--device", row["execution"]["device"]]
        if write_reference:
            args.extend(("--write-reference", str(reference)))
        identity = {"git_commit": COMMIT, "git_tree": "c" * 40, "clean": True}
        with patch.object(runner, "source_identity", side_effect=lambda root: identity.copy()), \
                patch.object(runner, "cpu_identity", return_value={"os": "Windows", "processor": "fixture"}), \
                patch.object(runner, "executable", return_value=binary), \
                patch.object(runner, "gpu_identity", return_value=gpu), \
                patch.object(runner, "run_process", side_effect=lambda *args, **kwargs: copy.deepcopy(row)) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            status = runner.main(args)
        self.assertEqual(run.call_count, 6)
        return status, output, reference

    def test_supported_platform_rss_failure_retains_report_and_never_publishes_reference(self) -> None:
        row = sample()
        row["metrics"]["peak_rss_bytes"] = {"status": "unavailable", "unit": "bytes",
            "direction": "lower", "gate": False, "reason": "GetProcessMemoryInfo failed"}
        for write_reference in (False, True):
            with self.subTest(reference=write_reference), tempfile.TemporaryDirectory() as temporary:
                status, output, reference = self.fixture_run(temporary, row, write_reference=write_reference)
                self.assertEqual(status, 1)
                self.assertEqual(json.loads((output / "run.json").read_text())["status"], "failed")
                statistics = json.loads((output / "statistics.json").read_text())
                self.assertTrue(statistics["stability"]["passed"])
                self.assertFalse(statistics["resource_completeness"]["passed"])
                self.assertEqual(statistics["metrics"]["storage.peak_rss_bytes"]["reason"], "GetProcessMemoryInfo failed")
                self.assertFalse(reference.exists())

    def test_valid_cpu_run_publishes_reference_with_full_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, output, reference = self.fixture_run(temporary, sample())
            self.assertEqual(status, 0)
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "passed")
            statistics = json.loads((output / "statistics.json").read_text())
            self.assertTrue(statistics["resource_completeness"]["passed"])
            self.assertTrue(runner.compare_reference(statistics, json.loads(reference.read_text()))["passed"])

    def test_unknown_gpu_configuration_retains_evidence_without_reference(self) -> None:
        for write_reference in (False, True):
            with self.subTest(reference=write_reference), tempfile.TemporaryDirectory() as temporary:
                status, output, reference = self.fixture_run(temporary, cuda_sample(), gpu=gpu_fixture("N/A"),
                                                            write_reference=write_reference)
                self.assertEqual(status, 1 if write_reference else 0)
                statistics = json.loads((output / "statistics.json").read_text())
                self.assertTrue(statistics["stability"]["passed"])
                self.assertTrue(statistics["resource_completeness"]["passed"])
                self.assertFalse(statistics["comparison_eligibility"]["passed"])
                self.assertFalse(reference.exists())

    def test_missing_storage_id_fails_before_source_or_subprocess_work(self) -> None:
        with patch.object(runner, "source_identity") as source, patch.object(runner, "run_process") as run:
            with self.assertRaisesRegex(ValueError, "storage-environment-id"):
                runner.main(["--build-dir", "unused", "--output", "unused", "--profiles", "storage"])
        source.assert_not_called()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
