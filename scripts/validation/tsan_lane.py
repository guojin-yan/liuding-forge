#!/usr/bin/env python3
"""Run the fixed Linux/Clang/CPU TSan lane and retain evidence on failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET


class LaneError(RuntimeError):
    pass


# Deliberately exclude CUDA, death/crash subprocess tests, and unrelated suites.
# Literal declarations in these files are also the expected discovery contract.
SUITES = {
    "DataPipelineTest": ("tests/unit/data/test_data_pipeline.cpp", "jyppx_test_data"),
    "DataPipelineCheckpointTest": ("tests/unit/data/test_data_pipeline.cpp", "jyppx_test_data"),
    "DataPipelineWorkerTest": ("tests/unit/data/test_data_pipeline.cpp", "jyppx_test_data"),
    "IterableDataPipelineTest": ("tests/unit/data/test_iterable_data_pipeline.cpp", "jyppx_test_data"),
    "IterableDataPipelineCheckpointTest": ("tests/unit/data/test_iterable_data_pipeline.cpp", "jyppx_test_data"),
    "BatchMemoryPipelineTest": ("tests/unit/data/test_batch_memory.cpp", "jyppx_test_data"),
    "ValueSchemaPipelineTest": ("tests/unit/data/test_value_schema.cpp", "jyppx_test_data"),
    "ConcurrentStressTest": ("tests/integration/test_concurrent_stress.cpp", "jyppx_test_concurrent_stress"),
    "ServingTest": ("tests/unit/serving/test_serving.cpp", "jyppx_test_serving"),
    "SQLiteRunStoreTest": ("tests/unit/experiment/test_run_store.cpp", "jyppx_test_experiment_run_store"),
    "SQLiteRunEventTest": ("tests/unit/experiment/test_run_events.cpp", "jyppx_test_experiment_run_events"),
    "TelemetryExporterTest": ("tests/unit/experiment/test_telemetry_exporters.cpp", "jyppx_test_experiment_telemetry"),
    "EventOutboxTest": ("tests/unit/training/test_event_sink.cpp", "jyppx_test_training_event_sink"),
    "EventOutboxTelemetryTest": ("tests/unit/training/test_event_sink.cpp", "jyppx_test_training_event_sink"),
    "TrainingTelemetryAdapterTest": ("tests/unit/training/test_telemetry_adapter.cpp", "jyppx_test_training_event_sink"),
}
TARGETS = list(dict.fromkeys(target for _, target in SUITES.values()))
TEST_REGEX = "^(" + "|".join(SUITES) + r")\."
ENABLED = ("CORE", "INTEROP", "DATA", "NN", "OPTIM", "TRAINING", "EXPERIMENT", "SERVING", "TESTS")
DISABLED = ("EXAMPLES", "BENCHMARKS", "FUZZERS", "DOCS", "TOOLS", "MODELS", "QUANTIZATION")
TSAN_DIAGNOSTIC = re.compile(r"(?:WARNING|FATAL):\s*ThreadSanitizer|ThreadSanitizer: reported [1-9]", re.I)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}-{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_policy(source: Path) -> dict[str, str]:
    expected = {}
    for suite, (relative, target) in SUITES.items():
        text = (source / relative).read_text(encoding="utf-8")
        names = re.findall(r"\bTEST(?:_F|_P)?\(\s*" + re.escape(suite) + r"\s*,\s*(\w+)\s*\)", text)
        if not names or len(names) != len(set(names)):
            raise LaneError(f"required suite {suite} is empty or has duplicate declarations in {relative}")
        for name in names:
            if name.startswith("DISABLED_"):
                raise LaneError(f"required test {suite}.{name} is disabled")
            expected[f"{suite}.{name}"] = target
    return dict(sorted(expected.items()))


def read_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"([^/#][^:=]*):[^=]*=(.*)$", line)
        if match:
            result[match[1]] = match[2]
    return result


def is_on(value: str) -> bool:
    return value.upper() not in ("", "0", "OFF", "FALSE", "NO", "N", "IGNORE", "NOTFOUND") and not value.endswith("-NOTFOUND")


def validate_instrumentation(cache: dict[str, str], environment: dict[str, str]) -> None:
    for key in ("JYPPX_ENABLE_COVERAGE", "JYPPX_ENABLE_ASAN", "JYPPX_ENABLE_UBSAN"):
        if is_on(cache.get(key, "OFF")):
            raise LaneError(f"fixed TSan cache has conflicting {key}; preserve/review it before reuse")
    flags = [(key, value) for key, value in cache.items() if "FLAGS" in key]
    flags += [(key, environment.get(key, "")) for key in ("CFLAGS", "CXXFLAGS", "LDFLAGS")]
    for key, value in flags:
        if re.search(r"--coverage|-fprofile|-ftest-coverage|-fno-sanitize", value):
            raise LaneError(f"conflicting instrumentation in {key}: {value}")
        for sanitizers in re.findall(r"-fsanitize=([^\s]+)", value):
            if set(sanitizers.split(",")) != {"thread"}:
                raise LaneError(f"non-TSan instrumentation in {key}: {value}")


def torch_directory(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "TorchConfig.cmake").is_file():
        return candidate
    return candidate / "share/cmake/Torch"


def select_torch(cache: dict[str, str], environment: dict[str, str]) -> Path:
    explicit = environment.get("JYPPX_TSAN_TORCH_DIR")
    cached = cache.get("Torch_DIR")
    if explicit:
        selected = torch_directory(explicit)
    elif cached and not cached.endswith("-NOTFOUND"):
        # A CUDA prefix inherited from another lane must not replace this cache.
        selected = torch_directory(cached)
    else:
        prefixes = re.split("[;:]" if os.name != "nt" else ";", environment.get("CMAKE_PREFIX_PATH", ""))
        candidates = {torch_directory(item) for item in prefixes if item and (torch_directory(item) / "TorchConfig.cmake").is_file()}
        if len(candidates) != 1:
            raise LaneError("select exactly one CPU LibTorch with JYPPX_TSAN_TORCH_DIR or CMAKE_PREFIX_PATH")
        selected = candidates.pop()
    if cached and not cached.endswith("-NOTFOUND") and torch_directory(cached) != selected:
        raise LaneError("Torch_DIR in the fixed cache disagrees with the selected CPU package")
    return selected.resolve()


def validate_torch(torch_dir: Path, cache: dict[str, str]) -> dict:
    if torch_dir.parts[-3:] != ("share", "cmake", "Torch"):
        raise LaneError("LibTorch must have the official <prefix>/share/cmake/Torch layout")
    prefix = torch_dir.parents[2]
    files = [torch_dir / "TorchConfig.cmake", prefix / "share/cmake/Caffe2/Caffe2Config.cmake", prefix / "build-version"]
    files += [prefix / "lib" / name for name in ("libtorch.so", "libtorch_cpu.so", "libc10.so")]
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(prefix):
            raise LaneError(f"CPU LibTorch package is incomplete or escapes its prefix: {path}")
    version = (prefix / "build-version").read_text(encoding="utf-8").strip()
    if "+cpu" not in version.lower() or list((prefix / "lib").glob("*cuda*")):
        raise LaneError(f"TSan requires an official CPU LibTorch package, found {version}")
    expected_paths = {"Torch_DIR": torch_dir, "Caffe2_DIR": prefix / "share/cmake/Caffe2", "TORCH_LIBRARY": prefix / "lib/libtorch.so"}
    for key, expected in expected_paths.items():
        value = cache.get(key)
        if value and not value.endswith("-NOTFOUND") and Path(value).resolve() != expected.resolve():
            raise LaneError(f"{key} in the fixed cache points outside the selected CPU package")
    return {"prefix": str(prefix), "torch_dir": str(torch_dir), "version": version,
            "files": {str(path.relative_to(prefix)): digest(path) for path in files}}


def tsan_options(value: str) -> str:
    required = {"halt_on_error": "1", "report_bugs": "1", "exitcode": "66"}
    options = {}
    for item in filter(None, value.split(":")):
        key, separator, setting = item.partition("=")
        if not separator or key in options:
            raise LaneError("TSAN_OPTIONS must contain unique key=value options")
        if key in required and setting != required[key]:
            raise LaneError(f"TSAN_OPTIONS cannot weaken {key}")
        if key == "suppressions" or key.startswith("ignore_") or (key.startswith("report_") and setting in ("0", "false")):
            raise LaneError(f"TSAN_OPTIONS cannot suppress reports with {key}")
        options[key] = setting
    options = {"second_deadlock_stack": "1", **options, **required}
    return ":".join(f"{key}={setting}" for key, setting in options.items())


def validate_discovery(document: dict, expected: dict[str, str], build: Path) -> list[dict]:
    if document.get("kind") != "ctestInfo" or document.get("version", {}).get("major") != 1:
        raise LaneError("unsupported CTest JSON discovery protocol")
    tests = document.get("tests", [])
    names = [test.get("name") for test in tests]
    if not names:
        raise LaneError("TSan discovery selected no tests")
    if len(names) != len(set(names)) or set(names) != set(expected):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected), key=str)
        raise LaneError(f"TSan discovery differs from required source suites: missing={missing}, extra={extra}")
    for test in tests:
        name = test["name"]
        command = test.get("command", [])
        if not command:
            raise LaneError(f"discovered test has no executable: {name}")
        executable = Path(command[0]).resolve()
        allowed = {f"--gtest_filter={name}", "--gtest_also_run_disabled_tests"}
        if (not executable.is_file() or not executable.is_relative_to(build.resolve())
                or executable.name != expected[name] or command.count(f"--gtest_filter={name}") != 1
                or any(argument not in allowed for argument in command[1:])):
            raise LaneError(f"wrong executable or GoogleTest filter for {name}: {command}")
        for prop in test.get("properties", []):
            if prop["name"] == "DISABLED" and is_on(str(prop["value"])):
                raise LaneError(f"CTest disabled required test {name}")
            if prop["name"] in ("ENVIRONMENT", "ENVIRONMENT_MODIFICATION"):
                if any(key in str(prop["value"]) for key in ("TSAN_OPTIONS", "LD_LIBRARY_PATH", "LD_PRELOAD", "GTEST_")):
                    raise LaneError(f"CTest overrides sanitizer/runtime environment for {name}")
    return tests


def audit_junit(path: Path, selected: list[str], ctest_exit: int, log: str) -> dict:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    names = [case.get("name") for case in cases]
    skipped, failed = [], []
    for case in cases:
        name = case.get("name")
        if (case.find("skipped") is not None or case.get("status", "run").lower() not in ("run", "passed")
                or re.search(r"\[\s*SKIPPED\s*\]", "".join(case.itertext()))):
            skipped.append(name)
        if case.find("failure") is not None or case.find("error") is not None:
            failed.append(name)
    complete = len(names) == len(set(names)) and set(names) == set(selected)
    diagnostic = bool(TSAN_DIAGNOSTIC.search(log + path.read_text(encoding="utf-8")))
    return {"passed": ctest_exit == 0 and complete and not skipped and not failed and not diagnostic,
            "ctest_exit_code": ctest_exit, "executed_count": len(cases), "complete_selection": complete,
            "missing": sorted(set(selected) - set(names)), "skipped": skipped, "failed": failed,
            "tsan_diagnostic": diagnostic}


def source_snapshot(source: Path) -> dict:
    def git(*arguments: str) -> str:
        return subprocess.run(["git", "-C", str(source), *arguments], check=True, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8").stdout.rstrip("\r\n")
    status = git("status", "--porcelain=v1", "--untracked-files=normal", "--ignore-submodules=none")
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
            "clean": not status, "porcelain": status,
            "dependency_manifest_sha256": digest(source / "dependency-manifest.json")}


class Lane:
    def __init__(self, source: Path):
        self.source = source.resolve()
        self.build = self.source / "out/build/tsan-linux"
        self.output = self.source / "out/sanitizer/linux/tsan"
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{time.time_ns()}"
        self.run_dir = self.output / "runs" / identifier
        self.environment = dict(os.environ)
        self.metadata = {"schema_version": 1, "started_utc": now(), "status": "running",
                         "source_root": str(self.source), "build_root": str(self.build), "run_dir": str(self.run_dir),
                         "platform": platform.platform(), "python": sys.version, "targets": TARGETS,
                         "required_suites": list(SUITES), "test_regex": TEST_REGEX, "ctest_exit_code": None,
                         "commands": [], "scope": "selected CPU concurrency suites; not full project/CUDA coverage"}

    def save(self) -> None:
        write_json(self.run_dir / "metadata.json", self.metadata)
        write_json(self.output / "metadata.json", self.metadata)

    def command(self, stage: str, argv: list[str], check: bool = True, json_output: bool = False) -> tuple[int, str]:
        log = self.run_dir / f"{stage}.log"
        record = {"stage": stage, "argv": argv, "started_utc": now(), "log": str(log), "exit_code": None}
        self.metadata["commands"].append(record)
        self.save()
        process = None
        try:
            if json_output:
                # Keep stderr out of CTest's machine-readable stdout.
                with (self.run_dir / f"{stage}.json").open("w", encoding="utf-8") as output, log.open("w", encoding="utf-8") as errors:
                    process = subprocess.run(argv, cwd=self.source, env=self.environment, stdin=subprocess.DEVNULL,
                                             stdout=output, stderr=errors, text=True, encoding="utf-8")
                text = (self.run_dir / f"{stage}.json").read_text(encoding="utf-8")
            else:
                with log.open("w", encoding="utf-8") as output:
                    process = subprocess.Popen(argv, cwd=self.source, env=self.environment, stdin=subprocess.DEVNULL,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                    for line in process.stdout:
                        output.write(line)
                        print(line, end="", flush=True)
                    process.wait()
                text = log.read_text(encoding="utf-8")
            record["exit_code"] = process.returncode
            if stage == "03-ctest":
                self.metadata["ctest_exit_code"] = process.returncode
        finally:
            if process is not None and hasattr(process, "poll") and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process is not None:
                record["exit_code"] = process.returncode
            record["finished_utc"] = now()
            self.save()
        if check and process.returncode:
            raise LaneError(f"{stage} exited {process.returncode}; see {log}")
        return process.returncode, text

    def preflight(self) -> tuple[str, str, str, int]:
        if platform.system() != "Linux":
            raise LaneError("the TSan lane runs only on Linux; --policy is portable and does not run tools")
        snapshot = source_snapshot(self.source)
        self.metadata["source_before"] = snapshot
        if not snapshot["clean"]:
            raise LaneError("TSan evidence requires a clean committed source tree")
        for program in ("cmake", "ctest", "git", "ldd"):
            if not shutil.which(program):
                raise LaneError(f"required tool is missing: {program}")
        cache_file = self.build / "CMakeCache.txt"
        cache = read_cache(cache_file)
        self.metadata["cache_before_sha256"] = digest(cache_file) if cache_file.is_file() else None
        if cache.get("CMAKE_HOME_DIRECTORY") and Path(cache["CMAKE_HOME_DIRECTORY"]).resolve() != self.source:
            raise LaneError("fixed CMake cache belongs to a different source checkout")
        validate_instrumentation(cache, self.environment)
        compiler = self.environment.get("JYPPX_TSAN_CXX") or cache.get("CMAKE_CXX_COMPILER") or "clang++"
        compiler = shutil.which(compiler)
        if not compiler:
            raise LaneError("clang++ with ThreadSanitizer support is required")
        if cache.get("CMAKE_CXX_COMPILER") and Path(cache["CMAKE_CXX_COMPILER"]).resolve() != Path(compiler).resolve():
            raise LaneError("refusing to replace the compiler in the fixed TSan cache")
        _, version = self.command("compiler-version", [compiler, "--version"])
        if "clang" not in version.lower():
            raise LaneError("TSan lane requires Clang")
        self.metadata["compiler"] = {"path": compiler, "version": version.strip()}
        torch_dir = select_torch(cache, self.environment)
        dependency = validate_torch(torch_dir, cache)
        self.metadata["libtorch"] = dependency
        prefix = dependency["prefix"]
        self.environment["LD_LIBRARY_PATH"] = str(Path(prefix) / "lib") + (":" + self.environment["LD_LIBRARY_PATH"] if self.environment.get("LD_LIBRARY_PATH") else "")
        if self.environment.get("LD_PRELOAD"):
            raise LaneError("LD_PRELOAD must be unset for the TSan lane")
        self.environment["TSAN_OPTIONS"] = tsan_options(self.environment.get("TSAN_OPTIONS", ""))
        if any(key.startswith("GTEST_") for key in self.environment):
            raise LaneError("unset GTEST_* environment overrides for the TSan lane")
        self.metadata["tsan_options"] = self.environment["TSAN_OPTIONS"]
        raw_parallelism = self.environment.get("JYPPX_BUILD_PARALLEL_LEVEL", "2")
        if not raw_parallelism.isdecimal() or int(raw_parallelism) < 1:
            raise LaneError("JYPPX_BUILD_PARALLEL_LEVEL must be a positive integer")
        parallelism = int(raw_parallelism)
        self.metadata["build_parallelism"] = parallelism
        self.metadata["ctest_parallelism"] = parallelism
        for tool in ("cmake", "ctest"):
            _, version = self.command(f"{tool}-version", [tool, "--version"])
            self.metadata[f"{tool}_version"] = version.strip()
        return compiler, str(torch_dir), prefix, parallelism

    def run_lane(self) -> None:
        compiler, torch_dir, prefix, parallelism = self.preflight()
        expected = source_policy(self.source)
        self.metadata["expected_tests"] = list(expected)
        configure = ["cmake", "-S", str(self.source), "-B", str(self.build), "-DCMAKE_BUILD_TYPE=Debug",
                     f"-DCMAKE_CXX_COMPILER={compiler}", f"-DTorch_DIR={torch_dir}",
                     f"-DCaffe2_DIR={prefix}/share/cmake/Caffe2", f"-DCMAKE_PREFIX_PATH={prefix}",
                     "-DJYPPX_ENABLE_TSAN=ON", "-DJYPPX_ENABLE_COVERAGE=OFF", "-DJYPPX_ENABLE_ASAN=OFF",
                     "-DJYPPX_ENABLE_UBSAN=OFF", "-DJYPPX_MODELS_REFERENCE_DIR=", "-DJYPPX_TEST_OFFICIAL_TELEMETRY=OFF",
                     "-DJYPPX_TEST_CLI_CUDA=OFF"]
        configure += [f"-DJYPPX_BUILD_{component}=ON" for component in ENABLED]
        configure += [f"-DJYPPX_BUILD_{component}=OFF" for component in DISABLED]
        self.command("01-configure", configure)
        cache_file = self.build / "CMakeCache.txt"
        cache = read_cache(cache_file)
        validate_instrumentation(cache, self.environment)
        if not is_on(cache.get("JYPPX_ENABLE_TSAN", "OFF")) or cache.get("CMAKE_BUILD_TYPE") != "Debug":
            raise LaneError("configured cache is not a Debug TSan build")
        if Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != self.source or Path(cache.get("CMAKE_CXX_COMPILER", "")).resolve() != Path(compiler).resolve():
            raise LaneError("configured source/compiler differs from preflight")
        if any(not is_on(cache.get(f"JYPPX_BUILD_{component}", "OFF")) for component in ENABLED):
            raise LaneError("configured cache is missing a required component")
        self.metadata["libtorch_after_configure"] = validate_torch(Path(torch_dir), cache)
        self.metadata["cache_configured_sha256"] = digest(cache_file)
        fetched = {"jyppx_sqlite": "sqlite3.c", "googletest": "googletest/src/gtest-all.cc",
                   "jyppx_nlohmann_json": "include/nlohmann/json.hpp"}
        dependency_base = Path(cache.get("FETCHCONTENT_BASE_DIR", str(self.build / "_deps")))
        self.metadata["fetched_dependencies"] = {}
        for name, relative in fetched.items():
            override = cache.get(f"FETCHCONTENT_SOURCE_DIR_{name.upper()}")
            directory = Path(override) if override else dependency_base / f"{name}-src"
            path = directory / relative
            self.metadata["fetched_dependencies"][name] = {"path": str(path), "sha256": digest(path)}
        self.command("02-build", ["cmake", "--build", str(self.build), "--config", "Debug", "--target", *TARGETS, "--parallel", str(parallelism)])
        runner = []
        self.metadata["address_layout"] = "native"
        if shutil.which("setarch"):
            code, _ = self.command("address-layout-probe", ["setarch", platform.machine(), "-R", "true"], check=False)
            if code == 0:
                runner = ["setarch", platform.machine(), "-R"]
                self.metadata["address_layout"] = "setarch-no-randomize"
        base = [*runner, "ctest", "--test-dir", str(self.build), "--build-config", "Debug", "--tests-regex", TEST_REGEX]
        _, discovery = self.command("00-test-discovery", [*base, "--show-only=json-v1"], json_output=True)
        selected = validate_discovery(json.loads(discovery), expected, self.build)
        self.metadata["selected_tests"] = [test["name"] for test in selected]
        self.metadata["selected_count"] = len(selected)
        write_json(self.run_dir / "selected-tests.json", selected)
        (self.run_dir / "selected-tests.txt").write_text("\n".join(self.metadata["selected_tests"]) + "\n", encoding="utf-8")
        binaries = {}
        for executable in sorted({test["command"][0] for test in selected}):
            binary = Path(executable)
            _, linkage = self.command(f"ldd-{binary.name}", ["ldd", executable])
            if "not found" in linkage or re.search(r"lib(?:torch_cuda|c10_cuda|cuda|cudart|cublas|cudnn)", linkage):
                raise LaneError(f"unresolved or CUDA runtime dependency for {binary.name}")
            torch_libraries = re.findall(r"\b(lib(?:torch[^\s]*|c10[^\s]*)\.so[^\s]*)\s+=>\s+(\S+)", linkage)
            if not torch_libraries or any(not Path(location).resolve().is_relative_to(Path(prefix) / "lib") for _, location in torch_libraries):
                raise LaneError(f"runtime LibTorch dependencies do not belong to the selected CPU package: {binary.name}")
            binaries[executable] = {"sha256": digest(binary), "torch_dependencies": dict(torch_libraries)}
        self.metadata["binaries"] = binaries
        code, log = self.command("03-ctest", [*base, "--output-on-failure", "--no-tests=error", "--parallel", str(parallelism),
                                                  "--timeout", "300", "--output-junit", str(self.run_dir / "junit.xml")], check=False)
        self.metadata["results"] = audit_junit(self.run_dir / "junit.xml", self.metadata["selected_tests"], code, log)
        if not self.metadata["results"]["passed"]:
            raise LaneError("TSan execution failed, skipped tests, or did not execute the complete validated selection")

    def execute(self) -> int:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        exit_code = 2
        lock = None
        try:
            self.save()
            # Only one process may use the fixed incremental cache at a time.
            if platform.system() == "Linux":
                import fcntl
                lock = (self.output / "lane.lock").open("a")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.run_lane()
            self.metadata["status"] = "passed"
            exit_code = 0
        except Exception as error:
            self.metadata["status"] = "failed"
            self.metadata["error"] = f"{type(error).__name__}: {error}"
            print(self.metadata["error"], file=sys.stderr)
        except KeyboardInterrupt:
            self.metadata["status"] = "interrupted"
            self.metadata["error"] = "interrupted by user"
            exit_code = 130
        finally:
            if "source_before" in self.metadata:
                try:
                    self.metadata["source_after"] = source_snapshot(self.source)
                    if self.metadata["source_after"] != self.metadata["source_before"]:
                        self.metadata["status"] = "failed"
                        self.metadata["source_changed"] = True
                        self.metadata.setdefault("error", "source identity changed during TSan execution")
                        exit_code = 2
                except (OSError, subprocess.SubprocessError) as error:
                    self.metadata["source_after_error"] = str(error)
                    self.metadata["status"] = "failed"
                    exit_code = 2
            try:
                cache = self.build / "CMakeCache.txt"
                self.metadata["cache_after_sha256"] = digest(cache) if cache.is_file() else None
                if "cache_configured_sha256" in self.metadata and self.metadata["cache_after_sha256"] != self.metadata["cache_configured_sha256"]:
                    self.metadata["cache_changed"] = True
                    self.metadata["status"] = "failed"
                    self.metadata.setdefault("error", "CMake cache changed after its validated configuration")
                    exit_code = 2
            except OSError as error:
                self.metadata["cache_after_error"] = str(error)
                self.metadata["status"] = "failed"
                exit_code = 2
            self.metadata["finished_utc"] = now()
            self.metadata["exit_code"] = exit_code
            self.save()
            if lock:
                lock.close()
            print(f"TSan report: {self.run_dir}")
        return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="store_true", help="print source-derived selection only; no configure/build/CTest")
    arguments = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    if arguments.policy:
        print(json.dumps({"required_suites": list(SUITES), "targets": TARGETS, "test_regex": TEST_REGEX,
                          "tests": source_policy(source)}, indent=2))
        return 0
    return Lane(source).execute()


if __name__ == "__main__":
    sys.exit(main())
