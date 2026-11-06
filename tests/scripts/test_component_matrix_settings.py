# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0.

"""Exercise nested CMake settings with a real, locally locked dependency."""

from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
CMAKE = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")
HELPER = (REPO / "cmake/ComponentMatrixSettings.cmake").as_posix()


@unittest.skipUnless(CMAKE, "CMake is required")
class ComponentMatrixSettingsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="liuding-matrix-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = dict(os.environ)
        self.env.pop("CMAKE_BUILD_PARALLEL_LEVEL", None)
        self.script = self.root / "settings.cmake"
        self.script.write_text(
            f'include("{HELPER}")\n'
            'jyppx_write_matrix_dependency_cache("${OUTPUT_CACHE}")\n'
            'jyppx_component_matrix_settings(config_args build_args)\n'
            'file(WRITE "${OUTPUT_ARGS}" "${build_args}")\n', encoding="utf-8")

    def settings(self, *options, environment_jobs=None):
        env = dict(self.env)
        if environment_jobs is not None:
            env["CMAKE_BUILD_PARALLEL_LEVEL"] = environment_jobs
        return subprocess.run(
            [CMAKE, f"-DOUTPUT_CACHE={self.root / 'dependencies.cmake'}",
             f"-DOUTPUT_ARGS={self.root / 'arguments.txt'}", *options,
             "-P", str(self.script)], env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            encoding="utf-8", errors="replace")

    def test_offline_source_survives_nested_configure_and_build(self):
        locked = self.root / "locked source with spaces"
        locked.mkdir()
        (locked / "CMakeLists.txt").write_text(
            'add_custom_target(locked_probe ALL COMMAND "${CMAKE_COMMAND}" '
            '-E touch "${CMAKE_BINARY_DIR}/locked-built")\n', encoding="utf-8")
        result = self.settings(
            "-DFETCHCONTENT_FULLY_DISCONNECTED:BOOL=ON",
            "-DFETCHCONTENT_UPDATES_DISCONNECTED:BOOL=ON",
            "-DFETCHCONTENT_UPDATES_DISCONNECTED_LOCKED:BOOL=ON",
            "-DFETCHCONTENT_UPDATES_DISCONNECTED_LITERAL:STRING=literal ]=] ]==]",
            f"-DFETCHCONTENT_SOURCE_DIR_LOCKED:PATH={locked.as_posix()}",
            f"-DFETCHCONTENT_BASE_DIR:PATH={(self.root / 'parent-mutable').as_posix()}")
        self.assertEqual(result.returncode, 0, result.stdout)
        child = self.root / "child"
        child.mkdir()
        (child / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\n'
            'project(matrix_dependency_probe LANGUAGES NONE)\n'
            'include(FetchContent)\n'
            'FetchContent_Declare(locked URL "${CMAKE_CURRENT_SOURCE_DIR}/missing.tar")\n'
            'FetchContent_MakeAvailable(locked)\n'
            'if(NOT FETCHCONTENT_FULLY_DISCONNECTED OR '
            'NOT FETCHCONTENT_UPDATES_DISCONNECTED OR '
            'NOT FETCHCONTENT_UPDATES_DISCONNECTED_LOCKED)\n'
            '  message(FATAL_ERROR "Offline policy was lost")\n'
            'endif()\n'
            'if(NOT "${FETCHCONTENT_UPDATES_DISCONNECTED_LITERAL}" '
            'STREQUAL [===[literal ]=] ]==]]===])\n'
            '  message(FATAL_ERROR "Cache value was altered")\n'
            'endif()\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/dependency-build-root.txt" '
            '"${FETCHCONTENT_BASE_DIR}")\n', encoding="utf-8")
        build = self.root / "child-build"
        configured = subprocess.run(
            [CMAKE, "-C", str(self.root / "dependencies.cmake"),
             "-S", str(child), "-B", str(build)], env=self.env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            encoding="utf-8", errors="replace")
        self.assertEqual(configured.returncode, 0, configured.stdout)
        self.assertEqual(Path((build / "dependency-build-root.txt").read_text()), build / "_deps")
        built = subprocess.run(
            [CMAKE, "--build", str(build), "--parallel", "1"], env=self.env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            encoding="utf-8", errors="replace")
        self.assertEqual(built.returncode, 0, built.stdout)
        self.assertTrue((build / "locked-built").is_file())

    def test_job_budget_priority(self):
        cases = [((), None, "2"), ((), "1", "1"),
                 (("-DJYPPX_MATRIX_BUILD_JOBS=1",), "2", "1"),
                 (("-DJYPPX_MATRIX_BUILD_JOBS=",), "1", "1")]
        for options, environment_jobs, expected in cases:
            with self.subTest(options=options, environment_jobs=environment_jobs):
                result = self.settings(*options, environment_jobs=environment_jobs)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertEqual((self.root / "arguments.txt").read_text(), f"--parallel;{expected}")

    def test_ctest_registration_keeps_invalid_list_budget_intact(self):
        source = self.root / "registered"
        source.mkdir()
        for name in ("cmake/ComponentMatrixSettings.cmake",
                     "cmake/TestOptionalComponentMatrix.cmake",
                     "cmake/tests/TestPreviewComponentMatrix.cmake"):
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, target)
        registrations = (REPO / "tests/CMakeLists.txt").read_text(encoding="utf-8")
        blocks = []
        for name in ("OptionalComponentMatrix", "PreviewComponentMatrix"):
            match = re.search(r"add_test\(\s*NAME " + name + r"\s.*?\n\s*\)",
                              registrations, re.DOTALL)
            self.assertIsNotNone(match, name)
            blocks.append(match.group())
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\n'
            'project(matrix_registration_probe LANGUAGES NONE)\n'
            'enable_testing()\n'
            'set(JYPPX_LIUDING_MATRIX_BUILD_JOBS "1;2")\n'
            + "\n".join(blocks) + "\n", encoding="utf-8")
        build = self.root / "registered-build"
        configured = subprocess.run(
            [CMAKE, "-S", str(source), "-B", str(build)], env=self.env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace", check=False)
        self.assertEqual(configured.returncode, 0, configured.stdout)
        ctest = str(Path(CMAKE).with_name("ctest.exe" if os.name == "nt" else "ctest"))
        tested = subprocess.run(
            [ctest, "--test-dir", str(build), "-C", "Release", "--output-on-failure"],
            env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace", check=False)
        self.assertNotEqual(tested.returncode, 0, tested.stdout)
        self.assertEqual(tested.stdout.count("Component matrix build jobs must be a positive integer"),
                         2, tested.stdout)

    def test_invalid_budget_and_missing_dependency_cache_fail_before_build(self):
        cases = [((f"-DJYPPX_MATRIX_BUILD_JOBS={value}",), None)
                 for value in ("0", "-1", "1;2", "bad")]
        cases += [((), "bad"),
                  ((f"-DJYPPX_MATRIX_DEPENDENCY_CACHE={self.root / 'missing.cmake'}",), None)]
        for options, environment_jobs in cases:
            with self.subTest(options=options, environment_jobs=environment_jobs):
                result = self.settings(*options, environment_jobs=environment_jobs)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("Component matrix", result.stdout)

    def installed_fixture(self):
        source = self.root / "installable"
        source.mkdir()
        config = source / "jyppxConfig.cmake"
        config.write_text("# Minimal package fixture.\n", encoding="utf-8")
        notices = ["nlohmann-json-3.12.0-LICENSE.MIT",
                   "sqlite-3.54.1-copyright-blessing.txt",
                   "sqlite-copyright-page.txt"]
        source.joinpath("CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\n'
            'project(installable_fixture LANGUAGES NONE)\n'
            'install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/jyppxConfig.cmake" '
            'DESTINATION lib/cmake/jyppx)\n'
            + 'install(FILES\n'
            + "\n".join(f'  "{(REPO / "third_party/licenses" / name).as_posix()}"'
                        for name in notices)
            + '\n  DESTINATION share/doc/jyppx/licenses)\n', encoding="utf-8")
        consumer = self.root / "consumer"
        consumer.mkdir()
        consumer.joinpath("CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\n'
            'project(consumer_fixture LANGUAGES NONE)\n'
            'add_custom_target(first ALL COMMAND "${CMAKE_COMMAND}" -E '
            'touch "${CMAKE_BINARY_DIR}/first-built")\n'
            'add_custom_target(second ALL COMMAND "${CMAKE_COMMAND}" -E '
            'touch "${CMAKE_BINARY_DIR}/second-built")\n'
            'enable_testing()\n'
            'add_test(NAME minimal_consumer COMMAND "${CMAKE_COMMAND}" -E true)\n',
            encoding="utf-8")
        build = self.root / "installed-build"
        configured = subprocess.run(
            [CMAKE, "-S", str(source), "-B", str(build)], env=self.env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False, encoding="utf-8", errors="replace")
        self.assertEqual(configured.returncode, 0, configured.stdout)
        return build, consumer

    def run_installed_script(self, name, build, consumer, *options,
                             environment_jobs=None):
        env = dict(self.env)
        if environment_jobs is not None:
            env["CMAKE_BUILD_PARALLEL_LEVEL"] = environment_jobs
        script = REPO / "cmake" / name
        return subprocess.run(
            [CMAKE, "--trace-expand", f"--trace-source={script}",
             f"-DJYPPX_SOURCE_DIR={REPO.as_posix()}",
             f"-DJYPPX_LIUDING_BUILD_DIR={build.as_posix()}",
             f"-DJYPPX_CONSUMER_SOURCE_DIR={consumer.as_posix()}",
             f"-DJYPPX_LIUDING_TEST_ROOT={(build / name).as_posix()}",
             "-DJYPPX_LIUDING_TEST_CONFIG=Release",
             f"-DJYPPX_TORCH_DIR={(self.root / 'unused-torch').as_posix()}",
             *options, "-P", str(script)], env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            encoding="utf-8", errors="replace")

    def test_installed_consumers_inherit_actual_build_budget(self):
        build, consumer = self.installed_fixture()
        cases = [((), "1", "1"),
                 (("-DJYPPX_MATRIX_BUILD_JOBS=1",), "2", "1")]
        for name in ("TestInstalledTargets.cmake", "TestInstalledComponentMatrix.cmake"):
            for options, environment_jobs, expected in cases:
                with self.subTest(script=name, options=options):
                    result = self.run_installed_script(
                        name, build, consumer, *options,
                        environment_jobs=environment_jobs)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    commands = re.findall(r"execute_process\(COMMAND [^\n]* --build [^\n]*",
                                          result.stdout)
                    self.assertEqual(len(commands), 10 if "Matrix" in name else 1,
                                     result.stdout)
                    for command in commands:
                        self.assertRegex(command, r" --parallel[ ;]" + expected + r"(?:[ ;]|\))")
                    self.assertNotRegex(result.stdout, r" --install [^\n]* --parallel[ ;]")
                    self.assertTrue((build / name / "b" / "CMakeCache.txt").is_file()
                                    if "Matrix" not in name else
                                    (build / name / "b" / "sdk" / "CMakeCache.txt").is_file())

    def test_installed_invalid_budget_keeps_existing_prefix(self):
        build = self.root / "unconfigured"
        build.mkdir()
        consumer = self.root / "unused-consumer"
        consumer.mkdir()
        for name in ("TestInstalledTargets.cmake", "TestInstalledComponentMatrix.cmake"):
            sentinel = build / name / "i" / "preserve-existing.txt"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_text("existing install", encoding="utf-8")
            for options, environment_jobs in [(("-DJYPPX_MATRIX_BUILD_JOBS=1;2",), "1"),
                                               ((), "bad")]:
                with self.subTest(script=name, options=options):
                    result = self.run_installed_script(
                        name, build, consumer, *options,
                        environment_jobs=environment_jobs)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("Component matrix build jobs must be a positive integer",
                                  result.stdout)
                    self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing install")

    def test_installed_ctest_registration_preserves_invalid_list_budget(self):
        source = self.root / "installed-registrations"
        source.mkdir()
        for name in ("ComponentMatrixSettings.cmake", "TestInstalledTargets.cmake",
                     "TestInstalledComponentMatrix.cmake"):
            target = source / "cmake" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / "cmake" / name, target)
        registrations = (REPO / "tests/CMakeLists.txt").read_text(encoding="utf-8")
        blocks = []
        for name in ("MigratedInstallConsumer", "ComponentInstallMatrixConsumer"):
            match = re.search(r"add_test\(\s*NAME " + name + r"\s.*?\n\s*\)",
                              registrations, re.DOTALL)
            self.assertIsNotNone(match, name)
            blocks.append(match.group())
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\n'
            'project(installed_registration_probe LANGUAGES NONE)\n'
            'enable_testing()\n'
            'set(JYPPX_LIUDING_MATRIX_BUILD_JOBS "1;2")\n'
            + "\n".join(blocks) + "\n", encoding="utf-8")
        build = self.root / "installed-registered-build"
        configured = subprocess.run(
            [CMAKE, "-S", str(source), "-B", str(build)], env=self.env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace", check=False)
        self.assertEqual(configured.returncode, 0, configured.stdout)
        ctest = str(Path(CMAKE).with_name("ctest.exe" if os.name == "nt" else "ctest"))
        tested = subprocess.run(
            [ctest, "--test-dir", str(build), "-C", "Release", "--output-on-failure"],
            env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace", check=False)
        self.assertNotEqual(tested.returncode, 0, tested.stdout)
        self.assertEqual(tested.stdout.count("Component matrix build jobs must be a positive integer"),
                         2, tested.stdout)


if __name__ == "__main__":
    unittest.main()
