"""Exercise coverage-lane failure handling without building or running CTest."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
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
        for name in ("run-coverage.sh", "coverage_gate.py", "coverage_summary.py"):
            shutil.copyfile(SOURCE / "scripts/validation" / name, scripts / name)
        fixtures = SOURCE / "tests/fixtures/coverage"
        shutil.copyfile(fixtures / "target-policy.json", self.root / "coverage-policy.json")
        shutil.copyfile(fixtures / "target-summary.json", self.root / "sample.json")
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
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["JYPPX_FIXTURE_COMMAND_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps([name, *args]) + "\\n")
if name == "cmake":
    sys.exit(int(os.environ.get("JYPPX_FIXTURE_BUILD_EXIT" if "--build" in args
                               else "JYPPX_FIXTURE_CONFIGURE_EXIT", "0")))
if name == "ctest":
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
else:
    pathlib.Path(args[args.index("--output") + 1]).write_text("<coverage/>")
'''
        for name in ("cmake", "ctest", "gcovr"):
            path = self.bin / name
            path.write_text(stub, encoding="utf-8")
            path.chmod(0o755)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Fixture",
                        "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"],
                       check=True)
        self.environment = {**os.environ,
                            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
                            "CMAKE_PREFIX_PATH": str(torch.parents[2]),
                            "JYPPX_FIXTURE_COMMAND_LOG": str(self.root / "out/commands.jsonl"),
                            "JYPPX_FIXTURE_SUMMARY": str(self.root / "sample.json")}
        for key in tuple(self.environment):
            if key.startswith("JYPPX_COVERAGE_") or key == "JYPPX_BUILD_PARALLEL_LEVEL":
                del self.environment[key]

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
        result = self.run_lane(JYPPX_COVERAGE_EXCLUDE_PACKAGING="1")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertFalse(own.exists())
        self.assertTrue(other.exists())
        commands = self.commands()
        configure = next(row for row in commands if row[0] == "cmake" and "-S" in row)
        self.assertIn("-DTorch_DIR=" + self.environment["CMAKE_PREFIX_PATH"] + "/share/cmake/Torch", configure)
        for owner in ("MODELS", "QUANTIZATION", "SERVING"):
            self.assertIn("-DJYPPX_BUILD_" + owner + "=ON", configure)
        for row in commands:
            if row[0] == "gcovr" and "--version" not in row:
                self.assertEqual(row[-1], str(self.build))
                self.assertEqual(row[row.index("-j") + 1], "2")
        ctest = next(row for row in commands if row[0] == "ctest")
        self.assertIn("--no-tests=error", ctest)
        self.assertEqual(ctest[ctest.index("-LE") + 1], "packaging")
        self.assertEqual(self.metadata()["ctest_excluded_label"], "packaging")

    def test_ctest_failure_still_collects_and_fails(self) -> None:
        result = self.run_lane(JYPPX_FIXTURE_CTEST_EXIT="8")
        self.assertEqual(result.returncode, 8, result.stderr)
        self.assertTrue((self.output / "coverage-targets.json").is_file())
        self.assertEqual(self.metadata()["ctest_exit"], "8")
        self.assertEqual(self.metadata()["policy_exit"], "0")
        self.assertEqual(self.metadata()["lane_exit"], "8")

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

    def test_dirty_source_cannot_begin_collection(self) -> None:
        (self.root / "marker.txt").write_text("changed\n", encoding="utf-8")
        result = self.run_lane()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.metadata()["dirty"], "true")
        self.assertEqual(self.metadata()["ctest_exit"], "not_run")
        self.assertFalse(any(row[0] in ("cmake", "ctest") for row in self.commands()))

    def test_preflight_failure_replaces_prior_success_metadata(self) -> None:
        for arguments in ({"JYPPX_BUILD_PARALLEL_LEVEL": "bad"},
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
