"""Exercise coverage-lane failure handling without building or running CTest."""
from __future__ import annotations

import json
import hashlib
import ast
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


SOURCE = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.name == "posix" and shutil.which("bash") and shutil.which("git"),
                     "Linux shell-lane protocol test")
class CoverageLaneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="jyppx coverage lane ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "scripts/validation"
        scripts.mkdir(parents=True)
        for name in ("run-coverage.sh", "coverage_gate.py", "coverage_summary.py", "coverage_dependencies.py"):
            shutil.copyfile(SOURCE / "scripts/validation" / name, scripts / name)
        fixtures = SOURCE / "tests/fixtures/coverage"
        shutil.copyfile(fixtures / "target-policy.json", self.root / "coverage-policy.json")
        shutil.copyfile(fixtures / "target-summary.json", self.root / "sample.json")
        (self.root / "include/jyppx").mkdir(parents=True)
        (self.root / "src/jyppx").mkdir(parents=True)
        for entry in json.loads((fixtures / "target-summary.json").read_text())["files"]:
            source = self.root / entry["filename"]
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("// Synthetic protocol source; never instrumented.\n", encoding="utf-8")
        self.dependency_cache = self.root / "fixture-dependency-inputs.cmake"
        self.dependency_cache.write_text("# Synthetic reviewed dependency inputs; no real SDK.\n", encoding="utf-8")
        spec = importlib.util.spec_from_file_location("synthetic_offline_fixture_factory",
            SOURCE / "tests/scripts/test_coverage_dependencies.py")
        assert spec and spec.loader
        factory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(factory)
        self.offline = factory.synthetic_inputs(self.root / "out/offline fixture", self.dependency_cache)
        helper_path = scripts / "coverage_dependencies.py"
        helper_tree = ast.parse(helper_path.read_text(encoding="utf-8"))
        for node in helper_tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                    target.id == "PINNED_FETCHCONTENT" for target in node.targets):
                node.value = ast.parse(repr(self.offline["locks"]), mode="eval").body
                break
        else:
            self.fail("Synthetic protocol helper pin assignment not found")
        helper_path.write_text("# SYNTHETIC protocol copy: fixture archive hashes, never release inputs.\n" +
            ast.unparse(ast.fix_missing_locations(helper_tree)) + "\n", encoding="utf-8")
        (self.root / ".gitignore").write_text("out/\n", encoding="utf-8")
        (self.root / "marker.txt").write_text("committed\n", encoding="utf-8")
        self.output = self.root / "out/coverage/linux/report"
        self.build = self.root / "out/coverage/linux/build"
        torch = self.root / "out/cpu torch/share/cmake/Torch"
        torch.mkdir(parents=True)
        (torch / "TorchConfig.cmake").touch()
        self.bin = self.root / "fixture-bin"
        self.bin.mkdir()
        stub = '''#!/usr/bin/env python3
import json, os, pathlib, signal, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["JYPPX_FIXTURE_COMMAND_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps([name, *args]) + "\\n")
with open(os.environ["JYPPX_FIXTURE_PARALLEL_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"name": name, "args": args,
        "CTEST_PARALLEL_LEVEL": os.environ.get("CTEST_PARALLEL_LEVEL"),
        "CMAKE_BUILD_PARALLEL_LEVEL": os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")}) + "\\n")
if os.environ.get("JYPPX_FIXTURE_WAIT_AT") == name:
    pathlib.Path(os.environ["JYPPX_FIXTURE_READY"]).write_text(name)
    while True:
        signal.pause()
if name == "cmake":
    if "--build" not in args:
        build = pathlib.Path(args[args.index("-B") + 1])
        build.mkdir(parents=True, exist_ok=True)
        entries = []
        for argument in args:
            if argument.startswith("-D") and "=" in argument:
                key, value = argument[2:].split("=", 1)
                entries.append(key + ("" if ":" in key else ":STRING") + "=" + value)
        override = os.environ.get("JYPPX_FIXTURE_CACHE_OVERRIDE")
        if override:
            key = override.split(":", 1)[0]
            entries = [entry for entry in entries if not entry.startswith(key + ":")]
            if not override.endswith(":MISSING"):
                entries.append(override)
        (build / "CMakeCache.txt").write_text("\\n".join(entries) + "\\n", encoding="utf-8")
    sys.exit(int(os.environ.get("JYPPX_FIXTURE_BUILD_EXIT" if "--build" in args
                               else "JYPPX_FIXTURE_CONFIGURE_EXIT", "0")))
if name == "ctest":
    if os.environ.get("JYPPX_FIXTURE_REAL_CTEST"):
        sys.exit(subprocess.run([os.environ["JYPPX_FIXTURE_REAL_CTEST"], *args]).returncode)
    pathlib.Path(args[args.index("--output-junit") + 1]).write_text(
        '<testsuite tests="1" failures="0"/>', encoding="utf-8")
    sys.exit(int(os.environ.get("JYPPX_FIXTURE_CTEST_EXIT", "0")))
if "--version" in args:
    print("gcovr fixture")
    sys.exit(0)
if os.environ.get("JYPPX_FIXTURE_GCOV_EXIT"):
    sys.exit(int(os.environ["JYPPX_FIXTURE_GCOV_EXIT"]))
if "--json-summary" in args:
    data = json.loads(pathlib.Path(os.environ["JYPPX_FIXTURE_SUMMARY"]).read_text())
    if os.environ.get("JYPPX_FIXTURE_BAD_COVERAGE"):
        data["files"][0]["line_covered"] = 1
        data["files"][0]["line_percent"] = 10.0
        data["line_covered"] = 19
        data["line_percent"] = 100 * 19 / 30
    pathlib.Path(args[args.index("--json-summary") + 1]).write_text(json.dumps(data))
    if os.environ.get("JYPPX_FIXTURE_MUTATE_OFFLINE_SOURCE"):
        pathlib.Path(os.environ["JYPPX_FIXTURE_MUTATE_OFFLINE_SOURCE"]).write_text("changed fixture source")
else:
    pathlib.Path(args[args.index("--output") + 1]).write_text("<coverage/>")
'''
        for name in ("cmake", "ctest", "gcovr"):
            path = self.bin / name
            path.write_text(stub, encoding="utf-8")
            path.chmod(0o755)
        # This lane test stubs the native tools rather than creating GCC data.
        # The counter parser's real binary and XML contracts have separate tests.
        (scripts / "coverage_hits.py").write_text('''
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["JYPPX_FIXTURE_COMMAND_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(["coverage_hits.py", *args]) + "\\n")
mode = args[0]
status = int(os.environ.get("JYPPX_FIXTURE_COUNTER_EXIT" if mode == "prepare"
                            else "JYPPX_FIXTURE_XML_EXIT", "0"))
if status:
    sys.exit(status)
if mode == "validate-xml":
    assert pathlib.Path(args[args.index("--receipt") + 1]).is_file()
    assert pathlib.Path(args[args.index("--xml") + 1]).is_file()
pathlib.Path(args[args.index("--output") + 1]).write_text('{"synthetic_protocol_only": true}')
print("68719476736" if mode == "prepare" else '{"synthetic_protocol_only": true}')
''', encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Fixture",
                        "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"],
                       check=True)
        self.environment = {**os.environ,
                            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
                            "CMAKE_PREFIX_PATH": str(torch.parents[2]),
                            "JYPPX_FIXTURE_COMMAND_LOG": str(self.root / "out/commands.jsonl"),
                            "JYPPX_FIXTURE_PARALLEL_LOG": str(self.root / "out/parallel.jsonl"),
                            "JYPPX_FIXTURE_SUMMARY": str(self.root / "sample.json")}
        for key in tuple(self.environment):
            if key.startswith("JYPPX_COVERAGE_") or key in {"JYPPX_LIUDING_BUILD_PARALLEL_LEVEL", "JYPPX_LIUDING_MODELS_REFERENCE_DIR"}:
                del self.environment[key]
        self.environment.update({"JYPPX_COVERAGE_DEPENDENCY_CACHE": str(self.dependency_cache),
                                 "JYPPX_COVERAGE_DEPENDENCY_CACHE_SHA256": hashlib.sha256(self.dependency_cache.read_bytes()).hexdigest(),
                                 "JYPPX_COVERAGE_OFFLINE_INPUTS": str(self.offline["inputs_file"]),
                                 "JYPPX_COVERAGE_OFFLINE_INPUTS_SHA256": hashlib.sha256(self.offline["inputs_file"].read_bytes()).hexdigest()})

    def run_lane(self, **environment: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["bash", str(self.root / "scripts/validation/run-coverage.sh")],
                              cwd=self.root.parent, env={**self.environment, **environment},
                              capture_output=True, text=True, timeout=30)

    def metadata(self) -> dict[str, str]:
        return dict(line.split("=", 1) for line in
                    (self.output / "coverage-metadata.txt").read_text().splitlines())

    def commands(self) -> list[list[str]]:
        return [json.loads(line) for line in (self.root / "out/commands.jsonl").read_text().splitlines()]

    def test_only_own_counters_and_cache_contribute(self) -> None:
        self.build.mkdir(parents=True)
        own = self.build / "old.gcda"
        own.touch()
        other = self.root / "out/another-cache/keep.gcda"
        other.parent.mkdir(parents=True)
        other.touch()
        reference = str(self.root / "external model reference")
        result = self.run_lane(JYPPX_COVERAGE_EXCLUDE_PACKAGING="1",
                               JYPPX_LIUDING_MODELS_REFERENCE_DIR=reference)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertFalse(own.exists())
        self.assertTrue(other.exists())
        commands = self.commands()
        configure = next(row for row in commands if row[0] == "cmake" and "-S" in row)
        self.assertIn("-DTorch_DIR=" + self.environment["CMAKE_PREFIX_PATH"] + "/share/cmake/Torch", configure)
        self.assertIn("-DJYPPX_LIUDING_MODELS_REFERENCE_DIR=" + reference, configure)
        self.assertEqual(self.metadata()["models_reference_dir"], reference)
        for owner in ("DATA_IMAGE", "MODELS", "SERVING", "VISION_RENDER", "ONNX_EXPORT",
                      "ONNX_RUNTIME", "ONNX_SERVING", "BUNDLE_SIGNING", "TELEMETRY_NATIVE", "TELEMETRY_HTTP"):
            self.assertIn("-DJYPPX_LIUDING_BUILD_" + owner + "=ON", configure)
        for row in commands:
            if row[0] == "gcovr" and "--version" not in row:
                self.assertEqual(row[-1], str(self.build))
                self.assertEqual(row[row.index("-j") + 1], "2")
                self.assertIn("--gcov-suspicious-hits-threshold=68719476736", row)
        self.assertEqual(self.metadata()["coverage_hit_validation"], "passed")
        self.assertEqual(configure[configure.index("-C") + 1], str(self.dependency_cache))
        self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED:BOOL=ON", configure)
        self.assertIn("-DFETCHCONTENT_UPDATES_DISCONNECTED:BOOL=ON", configure)
        for name, row in self.offline["document"]["fetchcontent"].items():
            option = self.offline["locks"][name]["cache_option"]
            self.assertIn("-D" + option + ":PATH=" + Path(row["source_root"]).as_posix(), configure)
        self.assertEqual(self.metadata()["dependency_cache_sha256"], hashlib.sha256(self.dependency_cache.read_bytes()).hexdigest())
        ctest = next(row for row in commands if row[0] == "ctest")
        self.assertIn("--no-tests=error", ctest)
        self.assertEqual(ctest[ctest.index("--parallel") + 1], "1")
        self.assertEqual(ctest[ctest.index("-LE") + 1], "packaging")
        self.assertEqual(self.metadata()["ctest_excluded_label"], "packaging")
        self.assertEqual(self.metadata()["coverage_profile"], "v1-selected-cpu")
        self.assertEqual(self.metadata()["ctest_excluded_test_regex"], "")

    @unittest.skipUnless(shutil.which("ctest"), "Real CTest selection contract")
    def test_explicit_exclusion_never_executes_excluded_command(self) -> None:
        self.build.mkdir(parents=True)
        allowed = self.build / "allowed.txt"
        excluded = self.build / "excluded.txt"
        quote = lambda value: '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
        command = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('executed')"
        tests = [("AllowedCoverageFixture", allowed), ("ExcludedCoverageFixture", excluded)]
        (self.build / "CTestTestfile.cmake").write_text("\n".join(
            "add_test(" + " ".join(map(quote, (name, sys.executable, "-c", command, marker))) + ")"
            for name, marker in tests) + "\n", encoding="utf-8")
        pattern = "^ExcludedCoverageFixture$"
        result = self.run_lane(JYPPX_COVERAGE_EXCLUDE_TEST_REGEX=pattern,
                               JYPPX_FIXTURE_REAL_CTEST=shutil.which("ctest"))
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(allowed.read_text(), "executed")
        self.assertFalse(excluded.exists())
        self.assertEqual(self.metadata()["ctest_excluded_test_regex"], pattern)
        self.assertEqual(self.metadata()["coverage_profile"], "v1-selected-cpu-filtered")
        self.assertEqual(self.metadata()["ctest_exit"], "0")

    def test_multiline_exclusion_cannot_start_collection(self) -> None:
        result = self.run_lane(JYPPX_COVERAGE_EXCLUDE_TEST_REGEX="fixture\nctest_exit=0")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.metadata()["last_stage"], "preflight")
        self.assertEqual(self.metadata()["ctest_exit"], "not_run")
        self.assertEqual(self.metadata()["coverage_profile"], "v1-selected-cpu-filtered")
        self.assertFalse((self.root / "out/commands.jsonl").exists())

    def test_ctest_failure_still_collects_and_fails(self) -> None:
        result = self.run_lane(JYPPX_FIXTURE_CTEST_EXIT="8")
        self.assertEqual(result.returncode, 8, result.stderr)
        self.assertTrue((self.output / "coverage-targets.json").is_file())
        self.assertEqual(self.metadata()["ctest_exit"], "8")
        self.assertEqual(self.metadata()["policy_exit"], "0")
        self.assertEqual(self.metadata()["lane_exit"], "8")

    def test_interrupted_ctest_records_signal_failure(self) -> None:
        ready = self.root / "out/ctest-ready"
        environment = {**self.environment, "JYPPX_FIXTURE_WAIT_AT": "ctest",
                       "JYPPX_FIXTURE_READY": str(ready)}
        process = subprocess.Popen(["bash", str(self.root / "scripts/validation/run-coverage.sh")],
            cwd=self.root.parent, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists(), "Synthetic CTest did not reach its blocking point")
            os.killpg(process.pid, signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 128 + signal.SIGTERM, stdout + stderr)
            metadata = self.metadata()
            self.assertEqual(metadata["lane_exit"], "143")
            self.assertEqual(metadata["termination_signal"], "TERM")
            self.assertEqual(metadata["last_stage"], "ctest")
            self.assertEqual(metadata["ctest_exit"], "not_run")
            self.assertEqual(metadata["policy_exit"], "not_run")
            for name in ("ctest.exit", "policy.exit", "coverage-ctest.xml", "coverage.xml",
                         "coverage-summary.json", "coverage-targets.json"):
                self.assertFalse((self.output / name).exists(), name)
            self.assertFalse(any(row[0] == "gcovr" and "--version" not in row
                                 for row in self.commands()))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=10)

    def test_dependency_inputs_missing_or_changed_prevent_configure(self) -> None:
        for override in ({"JYPPX_COVERAGE_DEPENDENCY_CACHE": ""},
                         {"JYPPX_COVERAGE_DEPENDENCY_CACHE_SHA256": "0" * 64}):
            with self.subTest(override=override):
                result = self.run_lane(**override)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(any(row[0] == "cmake" for row in self.commands()))

    def test_parallelism_cannot_exceed_two(self) -> None:
        for count in ("0", "3", "999999999999999999999999999999999999"):
            with self.subTest(count=count):
                result = self.run_lane(JYPPX_LIUDING_BUILD_PARALLEL_LEVEL=count)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(any(row[0] == "cmake" for row in self.commands()))

    def test_inherited_parallelism_is_overridden_for_ctest_and_build(self) -> None:
        result = self.run_lane(CTEST_PARALLEL_LEVEL="999", CMAKE_BUILD_PARALLEL_LEVEL="999")
        self.assertEqual(result.returncode, 0, result.stderr)
        entries = [json.loads(line) for line in (self.root / "out/parallel.jsonl").read_text().splitlines()]
        ctest = next(row for row in entries if row["name"] == "ctest")
        self.assertEqual(ctest["CTEST_PARALLEL_LEVEL"], "1")
        self.assertEqual(ctest["args"][ctest["args"].index("--parallel") + 1], "1")
        build = next(row for row in entries if row["name"] == "cmake" and "--build" in row["args"])
        self.assertEqual(build["CMAKE_BUILD_PARALLEL_LEVEL"], "2")
        self.assertEqual(build["args"][build["args"].index("--parallel") + 1], "2")

    def test_offline_descriptor_missing_or_bad_digest_prevents_configure(self) -> None:
        for override in ({"JYPPX_COVERAGE_OFFLINE_INPUTS": ""},
                         {"JYPPX_COVERAGE_OFFLINE_INPUTS_SHA256": "0" * 64}):
            with self.subTest(override=override):
                result = self.run_lane(**override)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(any(row[0] == "cmake" for row in self.commands()))

    def test_missing_local_fetchcontent_archive_prevents_configure(self) -> None:
        Path(self.offline["document"]["fetchcontent"]["googletest"]["archive"]).unlink()
        result = self.run_lane()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(any(row[0] == "cmake" for row in self.commands()))
        self.assertEqual(self.metadata()["last_stage"], "preflight")

    def test_offline_cache_type_value_or_missing_option_prevents_build(self) -> None:
        for override in ("FETCHCONTENT_FULLY_DISCONNECTED:BOOL=OFF",
                         "FETCHCONTENT_UPDATES_DISCONNECTED:STRING=ON",
                         "FETCHCONTENT_FULLY_DISCONNECTED:MISSING",
                         "FETCHCONTENT_SOURCE_DIR_GOOGLETEST:PATH=/wrong-source"):
            with self.subTest(override=override):
                result = self.run_lane(JYPPX_FIXTURE_CACHE_OVERRIDE=override)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(any("--build" in row or row[0] == "ctest" for row in self.commands()))
                self.assertEqual(self.metadata()["last_stage"], "ownership_profile")

    def test_local_dependency_changes_during_collection_reject(self) -> None:
        target = Path(self.offline["document"]["fetchcontent"]["googletest"]["source_root"]) / "CMakeLists.txt"
        result = self.run_lane(JYPPX_FIXTURE_MUTATE_OFFLINE_SOURCE=str(target))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.metadata()["last_stage"], "verify_dependencies")
        self.assertEqual(self.metadata()["ctest_exit"], "0")
        self.assertEqual(self.metadata()["policy_exit"], "0")

    def test_real_policy_failure_is_not_a_collection_pass(self) -> None:
        result = self.run_lane(JYPPX_FIXTURE_BAD_COVERAGE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.metadata()["ctest_exit"], "0")
        self.assertNotEqual(self.metadata()["policy_exit"], "0")
        self.assertEqual(self.metadata()["lane_exit"], str(result.returncode))

    def test_failed_build_preserves_counters_and_marks_unrun(self) -> None:
        self.build.mkdir(parents=True)
        counter = self.build / "keep.gcda"
        counter.touch()
        result = self.run_lane(JYPPX_FIXTURE_BUILD_EXIT="7")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertTrue(counter.exists())
        self.assertEqual(self.metadata()["last_stage"], "build")
        self.assertEqual(self.metadata()["ctest_exit"], "not_run")
        self.assertEqual(self.metadata()["policy_exit"], "not_run")

    def test_gcov_failure_retains_metadata(self) -> None:
        result = self.run_lane(JYPPX_FIXTURE_GCOV_EXIT="3")
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(self.metadata()["last_stage"], "gcov_xml")
        self.assertEqual(self.metadata()["policy_exit"], "not_run")

    def test_counter_sanity_failure_prevents_all_report_passes(self) -> None:
        result = self.run_lane(JYPPX_FIXTURE_COUNTER_EXIT="2")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.metadata()["last_stage"], "counter_sanity")
        self.assertEqual(self.metadata()["coverage_hit_validation"], "not_run")
        self.assertEqual(self.metadata()["policy_exit"], "not_run")
        self.assertFalse(any(row[0] == "gcovr" and "--version" not in row
                             for row in self.commands()))
        self.assertFalse((self.output / "coverage-targets.json").exists())

    def test_invalid_xml_hits_prevent_summary_and_policy(self) -> None:
        result = self.run_lane(JYPPX_FIXTURE_XML_EXIT="2")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.metadata()["last_stage"], "validate_hits")
        self.assertEqual(self.metadata()["coverage_hit_validation"], "not_run")
        self.assertEqual(self.metadata()["policy_exit"], "not_run")
        reports = [row for row in self.commands() if row[0] == "gcovr" and "--version" not in row]
        self.assertEqual(len(reports), 1)
        self.assertIn("--xml-pretty", reports[0])
        self.assertFalse((self.output / "coverage-summary.json").exists())
        self.assertFalse((self.output / "coverage-targets.json").exists())

    def test_dirty_source_cannot_begin_collection(self) -> None:
        (self.root / "marker.txt").write_text("changed\n", encoding="utf-8")
        result = self.run_lane()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.metadata()["dirty"], "true")
        self.assertEqual(self.metadata()["ctest_exit"], "not_run")
        self.assertFalse(any(row[0] in ("cmake", "ctest") for row in self.commands()))

    def test_preflight_failure_replaces_prior_success_metadata(self) -> None:
        for arguments in ({"JYPPX_LIUDING_BUILD_PARALLEL_LEVEL": "bad"},
                          {"JYPPX_COVERAGE_EXCLUDE_PACKAGING": "2"},
                          {"CMAKE_PREFIX_PATH": str(self.root / "missing torch")}):
            with self.subTest(arguments=arguments):
                first = self.run_lane()
                self.assertEqual(first.returncode, 0, first.stderr)
                self.assertEqual(self.metadata()["lane_exit"], "0")
                second = self.run_lane(**arguments)
                self.assertEqual(second.returncode, 2, second.stderr)
                self.assertEqual(self.metadata()["lane_exit"], "2")
                self.assertEqual(self.metadata()["last_stage"], "preflight")
                self.assertEqual(self.metadata()["ctest_exit"], "not_run")
                self.assertEqual(self.metadata()["policy_exit"], "not_run")
                self.assertFalse((self.output / "ctest.exit").exists())
                self.assertFalse((self.output / "policy.exit").exists())


if __name__ == "__main__":
    unittest.main()
