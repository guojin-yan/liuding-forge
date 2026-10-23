"""Portable TSan policy/protocol tests; never configure, build, or invoke CTest."""

import copy
import io
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tsan_lane", ROOT / "scripts/validation/tsan_lane.py")
LANE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LANE)


class TSanPolicyTest(unittest.TestCase):
    def test_real_source_policy_retains_all_concurrency_owners(self):
        expected = LANE.source_policy(ROOT)
        self.assertEqual(len(LANE.SUITES), 15)
        self.assertEqual(len(LANE.TARGETS), 7)
        required = (
            "ServingTest.ConcurrentLeaseBoundTimedDrainAndNoPrematureUnload",
            "ServingTest.BatchingPreservesOrderOwnsSnapshotsAndSeparatesScopes",
            "ServingTest.InBatchCancellationDoesNotCancelPeers",
            "ServingTest.QueueByteCountDeadlineAndShutdownBounds",
            "ServingTest.QueuedDeadlineExpiresAndNeverEntersLaterBatch",
            "SQLiteRunStoreTest.RejectsBusyWriterThenRetriesWithoutPartialIdentity",
            "SQLiteRunStoreTest.CoordinatesIndependentHandlesAndGcUsingOneDatabaseLock",
            "SQLiteRunEventTest.DeadlineAndCancellationInterruptCompetingSqliteWriterWait",
            "EventOutboxTest.CancellationAndCloseHaveExplicitBoundaries",
            "EventOutboxTelemetryTest.CancellationAndAbsoluteDeadlineRetainTheStagedNode",
        )
        self.assertTrue(set(required).issubset(expected))
        self.assertEqual({name.split(".")[0] for name in expected}, set(LANE.SUITES))
        self.assertFalse(any("Cuda" in name or "DeathTest" in name for name in expected))

    def test_source_suite_missing_or_disabled_is_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "case.cpp"
            with mock.patch.dict(LANE.SUITES, {"Owner": ("case.cpp", "owner_target")}, clear=True):
                source.write_text("TEST(Other, Case) {}", encoding="utf-8")
                with self.assertRaisesRegex(LANE.LaneError, "required suite"):
                    LANE.source_policy(root)
                source.write_text("TEST(Owner, DISABLED_Case) {}", encoding="utf-8")
                with self.assertRaisesRegex(LANE.LaneError, "disabled"):
                    LANE.source_policy(root)

    def test_options_reject_zero_exit_or_suppressed_reports(self):
        for options in ("exitcode=0", "halt_on_error=0", "report_bugs=0", "suppressions=local.txt",
                        "report_atomic_races=0", "report_atomic_races=no", "report_atomic_races=FALSE",
                        "report_thread_leaks=off", "report_signal_unsafe=unexpected",
                        "ignore_noninstrumented_modules=1", "history_size=2:history_size=4"):
            with self.subTest(options=options), self.assertRaises(LANE.LaneError):
                LANE.tsan_options(options)
        options = LANE.tsan_options("history_size=4")
        for required in ("halt_on_error=1", "exitcode=66", "report_bugs=1", "history_size=4"):
            self.assertIn(required, options)


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.build = Path(self.temporary.name)
        self.expected = LANE.source_policy(ROOT)
        for target in LANE.TARGETS:
            (self.build / target).touch()
        self.document = {
            "kind": "ctestInfo", "version": {"major": 1, "minor": 0},
            "tests": [{"name": name, "command": [str(self.build / target), f"--gtest_filter={name}", "--gtest_also_run_disabled_tests"],
                       "properties": []} for name, target in self.expected.items()],
        }

    def test_complete_discovery_with_actual_case_mapping_is_accepted(self):
        selected = LANE.validate_discovery(self.document, self.expected, self.build)
        self.assertEqual({item["name"] for item in selected}, set(self.expected))

    def test_empty_selection_and_missing_required_owner_are_rejected(self):
        complete = copy.deepcopy(self.document)
        self.document["tests"] = []
        with self.assertRaisesRegex(LANE.LaneError, "no tests"):
            LANE.validate_discovery(self.document, self.expected, self.build)
        self.document = complete
        self.document["tests"] = [test for test in self.document["tests"] if not test["name"].startswith("ServingTest.")]
        with self.assertRaisesRegex(LANE.LaneError, "missing=.*ServingTest"):
            LANE.validate_discovery(self.document, self.expected, self.build)

    def test_missing_single_sqlite_case_is_rejected(self):
        name = "SQLiteRunEventTest.DeadlineAndCancellationInterruptCompetingSqliteWriterWait"
        self.document["tests"] = [test for test in self.document["tests"] if test["name"] != name]
        with self.assertRaisesRegex(LANE.LaneError, "DeadlineAndCancellation"):
            LANE.validate_discovery(self.document, self.expected, self.build)

    def test_wrong_binary_filter_disabled_or_runtime_override_is_rejected(self):
        for mutation in ("binary", "filter", "disabled", "environment", "list_only", "duplicate"):
            document = copy.deepcopy(self.document)
            test = document["tests"][0]
            if mutation == "binary":
                test["command"][0] = str(self.build / "unbuilt_target")
            elif mutation == "filter":
                test["command"][1] = "--gtest_filter=Other.Case"
            elif mutation == "disabled":
                test["properties"] = [{"name": "DISABLED", "value": True}]
            elif mutation == "environment":
                test["properties"] = [{"name": "ENVIRONMENT", "value": ["TSAN_OPTIONS=exitcode=0"]}]
            elif mutation == "list_only":
                test["command"].append("--gtest_list_tests")
            else:
                document["tests"].append(copy.deepcopy(test))
            with self.subTest(mutation=mutation), self.assertRaises(LANE.LaneError):
                LANE.validate_discovery(document, self.expected, self.build)

    def test_invalid_discovery_stops_before_the_execution_command(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            lane = LANE.Lane(source)
            compiler = str(source / "clang++")
            cache = {"CMAKE_BUILD_TYPE": "Debug", "JYPPX_LIUDING_ENABLE_TSAN": "ON",
                     "CMAKE_HOME_DIRECTORY": str(source), "CMAKE_CXX_COMPILER": compiler}
            cache.update({f"JYPPX_LIUDING_BUILD_{component}": "ON" for component in LANE.ENABLED})
            calls = []

            def command(stage, argv, **kwargs):
                calls.append(stage)
                return 0, json.dumps({"kind": "ctestInfo", "version": {"major": 1}, "tests": []})

            with mock.patch.object(LANE, "source_policy", return_value=self.expected), \
                 mock.patch.object(lane, "preflight", return_value=(compiler, str(source / "share/cmake/Torch"), str(source), 2)), \
                 mock.patch.object(LANE, "read_cache", return_value=cache), \
                 mock.patch.object(LANE, "validate_torch", return_value={}), \
                 mock.patch.object(LANE, "digest", return_value="fixture"), \
                 mock.patch.object(LANE.shutil, "which", return_value=None), \
                 mock.patch.object(lane, "command", side_effect=command), \
                 mock.patch.object(LANE.subprocess, "run", side_effect=AssertionError("no tools allowed")), \
                 mock.patch.object(LANE.subprocess, "Popen", side_effect=AssertionError("no tools allowed")):
                with self.assertRaisesRegex(LANE.LaneError, "no tests"):
                    lane.run_lane()
            self.assertIn("00-test-discovery", calls)
            self.assertNotIn("03-ctest", calls)


class JUnitAuditTest(unittest.TestCase):
    def audit(self, names=("Owner.A", "Owner.B"), skip=None, failure=None, status=None, code=0, log=""):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "junit.xml"
            root = ET.Element("testsuite")
            for name in names:
                case = ET.SubElement(root, "testcase", name=name, status=status or "run")
                if name == skip:
                    ET.SubElement(case, "skipped")
                if name == failure:
                    ET.SubElement(case, "failure", message="failed")
            ET.ElementTree(root).write(path, encoding="utf-8")
            return LANE.audit_junit(path, ["Owner.A", "Owner.B"], code, log)

    def test_only_complete_execution_can_pass(self):
        self.assertTrue(self.audit()["passed"])
        for arguments in ({"names": ()}, {"names": ("Owner.A",)}, {"names": ("Owner.A", "Owner.A")},
                          {"skip": "Owner.B"}, {"failure": "Owner.B"}, {"status": "notrun"}, {"code": 8},
                          {"log": "WARNING: ThreadSanitizer: data race"}, {"log": "FATAL: ThreadSanitizer: unexpected memory mapping"}):
            with self.subTest(arguments=arguments):
                self.assertFalse(self.audit(**arguments)["passed"])

    def test_actual_nonzero_exit_and_skip_are_retained(self):
        result = self.audit(skip="Owner.B", code=8)
        self.assertEqual(result["ctest_exit_code"], 8)
        self.assertEqual(result["skipped"], ["Owner.B"])
        self.assertFalse(result["passed"])

    def test_ctest_junit_wrapper_and_textual_skip_are_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "junit.xml"
            path.write_text('<testsuites><testsuite><testcase name="Owner.A" status="run">'
                            '<system-out>[  SKIPPED ] Owner.A</system-out>'
                            '</testcase></testsuite></testsuites>', encoding="utf-8")
            result = LANE.audit_junit(path, ["Owner.A"], 0, "")
            self.assertFalse(result["passed"])
            self.assertEqual(result["skipped"], ["Owner.A"])


class DependencyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def package(self, name, version):
        prefix = self.root / name
        torch = prefix / "share/cmake/Torch"
        caffe = prefix / "share/cmake/Caffe2"
        torch.mkdir(parents=True)
        caffe.mkdir(parents=True)
        (prefix / "lib").mkdir()
        (torch / "TorchConfig.cmake").write_text("# fixture", encoding="utf-8")
        (caffe / "Caffe2Config.cmake").write_text("# fixture", encoding="utf-8")
        (prefix / "build-version").write_text(version, encoding="utf-8")
        for name in ("libtorch.so", "libtorch_cpu.so", "libc10.so"):
            (prefix / "lib" / name).write_bytes(b"fixture")
        return prefix, torch

    def test_cached_cpu_wins_over_ambient_cuda_and_explicit_conflict_fails(self):
        cpu, torch = self.package("cpu", "2.5.1+cpu")
        cuda, cuda_torch = self.package("cuda", "2.5.1+cu124")
        cache = {"Torch_DIR": str(torch)}
        self.assertEqual(LANE.select_torch(cache, {"CMAKE_PREFIX_PATH": str(cuda)}), torch.resolve())
        with self.assertRaisesRegex(LANE.LaneError, "disagrees"):
            LANE.select_torch(cache, {"JYPPX_TSAN_TORCH_DIR": str(cuda_torch)})
        record = LANE.validate_torch(torch, cache)
        self.assertEqual(record["prefix"], str(cpu))
        self.assertEqual(len(record["files"]), 6)

    def test_cuda_package_missing_file_and_stale_caffe2_are_rejected(self):
        cpu, torch = self.package("cpu", "2.5.1+cpu")
        cuda, cuda_torch = self.package("cuda", "2.5.1+cu124")
        with self.assertRaisesRegex(LANE.LaneError, "CPU LibTorch"):
            LANE.validate_torch(cuda_torch, {})
        with self.assertRaisesRegex(LANE.LaneError, "Caffe2_DIR"):
            LANE.validate_torch(torch, {"Caffe2_DIR": str(cuda / "share/cmake/Caffe2")})
        with self.assertRaisesRegex(LANE.LaneError, "TORCH_LIBRARY"):
            LANE.validate_torch(torch, {"TORCH_LIBRARY": str(cuda / "lib/libtorch.so")})
        (cpu / "lib/libtorch_cuda.so").touch()
        with self.assertRaisesRegex(LANE.LaneError, "CPU LibTorch"):
            LANE.validate_torch(torch, {})
        (cpu / "lib/libtorch_cuda.so").unlink()
        (cpu / "lib/libc10.so").unlink()
        with self.assertRaisesRegex(LANE.LaneError, "incomplete"):
            LANE.validate_torch(torch, {})

    def test_ambiguous_prefixes_and_conflicting_instrumentation_are_rejected(self):
        cpu, _ = self.package("cpu", "2.5.1+cpu")
        cuda, _ = self.package("cuda", "2.5.1+cu124")
        with self.assertRaisesRegex(LANE.LaneError, "exactly one"):
            LANE.select_torch({}, {"CMAKE_PREFIX_PATH": f"{cpu};{cuda}"})
        for cache, environment in (({"JYPPX_LIUDING_ENABLE_COVERAGE": "ON"}, {}),
                                   ({"JYPPX_LIUDING_ENABLE_ASAN": "ON"}, {}),
                                   ({"JYPPX_LIUDING_ENABLE_UBSAN": "ON"}, {}),
                                   ({"CMAKE_CXX_FLAGS": "--coverage"}, {}),
                                   ({}, {"CXXFLAGS": "-fsanitize=address,thread"}),
                                   ({}, {"LDFLAGS": "-fno-sanitize=thread"})):
            with self.subTest(cache=cache, environment=environment), self.assertRaises(LANE.LaneError):
                LANE.validate_instrumentation(cache, environment)
        LANE.validate_instrumentation({"JYPPX_LIUDING_ENABLE_TSAN": "ON", "CMAKE_CXX_FLAGS": "-fsanitize=thread"}, {})


class EvidenceTest(unittest.TestCase):
    def test_sigterm_handler_records_a_lane_termination_without_global_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            lane = LANE.Lane(Path(directory))
            with mock.patch.object(lane, "_signal_active_group") as signal_group:
                with self.assertRaises(LANE.LaneTermination) as raised:
                    lane._signal_handler(signal.SIGTERM, None)
            signal_group.assert_called_once_with(signal.SIGTERM)
            self.assertEqual(raised.exception.signum, signal.SIGTERM)
            self.assertEqual(lane._signal_requested, signal.SIGTERM)

    def test_early_failure_retains_metadata_without_fabricated_ctest_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            lane = LANE.Lane(Path(directory))
            with mock.patch.object(lane, "run_lane", side_effect=LANE.LaneError("fixture preflight failure")), \
                 mock.patch.object(LANE.platform, "system", return_value="Windows"), \
                 mock.patch.object(LANE.subprocess, "run", side_effect=AssertionError("no tools allowed")), \
                 mock.patch.object(LANE.subprocess, "Popen", side_effect=AssertionError("no tools allowed")):
                self.assertEqual(lane.execute(), 2)
            record = json.loads((lane.run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failed")
            self.assertIsNone(record["ctest_exit_code"])
            self.assertEqual(record["exit_code"], 2)
            self.assertIn("fixture preflight failure", record["error"])
            self.assertEqual(record, json.loads((lane.output / "metadata.json").read_text(encoding="utf-8")))

    def test_command_records_the_actual_failed_ctest_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            lane = LANE.Lane(Path(directory))
            lane.run_dir.mkdir(parents=True)
            process = mock.Mock()
            process.stdout = io.StringIO("fixture failing test output\n")
            process.returncode = 8
            process.poll.return_value = 8
            process.wait.return_value = 8
            with mock.patch.object(LANE.subprocess, "Popen", return_value=process), \
                 mock.patch.object(LANE.subprocess, "run", side_effect=AssertionError("no tools allowed")):
                code, output = lane.command("03-ctest", ["fixture-ctest"], check=False)
            self.assertEqual(code, 8)
            self.assertIn("failing test", output)
            metadata = json.loads((lane.run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["ctest_exit_code"], 8)
            self.assertEqual(metadata["commands"][0]["exit_code"], 8)

    def test_source_change_invalidates_an_otherwise_successful_run(self):
        with tempfile.TemporaryDirectory() as directory:
            lane = LANE.Lane(Path(directory))

            def completed_fixture():
                lane.metadata["source_before"] = {"commit": "a", "clean": True}

            with mock.patch.object(lane, "run_lane", side_effect=completed_fixture), \
                 mock.patch.object(LANE.platform, "system", return_value="Windows"), \
                 mock.patch.object(LANE, "source_snapshot", return_value={"commit": "b", "clean": True}), \
                 mock.patch.object(LANE.subprocess, "run", side_effect=AssertionError("no tools allowed")), \
                 mock.patch.object(LANE.subprocess, "Popen", side_effect=AssertionError("no tools allowed")):
                self.assertEqual(lane.execute(), 2)
            record = json.loads((lane.run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failed")
            self.assertTrue(record["source_changed"])

    @unittest.skipUnless(os.name == "posix", "process-group assertions require POSIX")
    def test_real_helper_descendant_is_killed_without_late_marker(self):
        """A cancelled lane command must stop a child that outlives its parent."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "late-marker"
            helper = root / "parent.py"
            child_code = (
                "import time; from pathlib import Path; time.sleep(0.8); "
                f"Path({str(marker)!r}).write_text('late', encoding='utf-8')"
            )
            helper.write_text(
                "import subprocess, sys, time\n"
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
                "time.sleep(30)\n",
                encoding="utf-8",
            )
            lane = LANE.Lane(root)
            lane.run_dir.mkdir(parents=True)
            outcome = {}

            def invoke():
                try:
                    outcome["result"] = lane.command("fixture-process-group", [sys.executable, str(helper)], check=False)
                except BaseException as error:  # pragma: no cover - only reports a fixture failure
                    outcome["error"] = error

            worker = threading.Thread(target=invoke)
            worker.start()
            deadline = time.monotonic() + 5
            while lane._active_pgid is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(lane._active_pgid)
            lane._signal_active_group(signal.SIGTERM)
            worker.join(8)
            self.assertFalse(worker.is_alive())
            self.assertNotIn("error", outcome)
            self.assertLess(outcome["result"][0], 0)
            time.sleep(1.0)
            self.assertFalse(marker.exists(), "descendant continued after process-group cancellation")
            record = json.loads((lane.run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(record["commands"][0]["cleanup"]["confirmed"])


if __name__ == "__main__":
    unittest.main()
