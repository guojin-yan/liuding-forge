# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0.

"""Check GNU, Clang and clang-cl commands from the compiler policy."""

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[3]
CMAKE = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")
ROOT = Path(os.environ.get(
    "JYPPX_COMPILER_FLAGS_TEST_ROOT", REPO / "out/build/compiler-flags"))
MSVC_FRONTEND = os.environ.get("JYPPX_COMPILER_FLAGS_TEST_FRONTEND") == "MSVC"


@unittest.skipUnless(CMAKE and shutil.which("ninja"), "CMake and Ninja are required")
class CompilerConfigurationFlagsTest(unittest.TestCase):
    def setUp(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="profile-", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        source.mkdir()
        (source / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
        (source / "codec.c").write_text("int codec(void) { return 0; }\n", encoding="utf-8")
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\n'
            'project(compiler_flags_contract LANGUAGES C CXX)\n'
            'add_library(jyppx_liuding_foundation STATIC main.cpp)\n'
            'add_executable(jyppx_cli main.cpp)\n'
            'add_library(jyppx_native_telemetry_codec STATIC codec.c)\n'
            f'include("{(REPO / "cmake/CompilerOptions.cmake").as_posix()}")\n',
            encoding="utf-8")
        self.source = source

    def configure(self, generator, *options):
        build = self.root / "build"
        args = [CMAKE, "-S", str(self.source), "-B", str(build), "-G", generator,
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", *options]
        for language in ("C", "CXX"):
            compiler = os.environ.get(f"JYPPX_COMPILER_FLAGS_TEST_{language}")
            if compiler:
                args.append(f"-DCMAKE_{language}_COMPILER={compiler}")
        env = dict(os.environ)
        for name in ("CFLAGS", "CXXFLAGS"):
            env.pop(name, None)
        result = subprocess.run(args, env=env, text=True, encoding="utf-8",
                                errors="replace", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stdout)
        commands = json.loads((build / "compile_commands.json").read_text())
        self.assertTrue(commands)
        return [shlex.split(row["command"]) for row in commands]

    def assert_profile(self, commands, optimization, debug, ndebug=False,
                       sanitizers=False):
        self.assertEqual(len(commands), 3, commands)
        for argv in commands:
            with self.subTest(command=argv):
                optimization_pattern = r"/O[12dx]" if MSVC_FRONTEND else r"-O(?:[0-3sgz]|fast)"
                debug_pattern = r"/Z[7iI]" if MSVC_FRONTEND else r"-g[0-3]?"
                optimizations = [a for a in argv if re.fullmatch(optimization_pattern, a)]
                debugging = [a for a in argv if re.fullmatch(debug_pattern, a)]
                self.assertEqual(optimizations[-1:] or [None], [optimization])
                self.assertEqual(debugging[-1:] or [None], [debug])
                self.assertEqual(any(a in argv for a in ("-DNDEBUG", "/DNDEBUG")), ndebug)
                if any(a.endswith("main.cpp") for a in argv):
                    warning = "/clang:-Wall" if MSVC_FRONTEND else "-Wall"
                    self.assertIn(warning, argv)
                    if MSVC_FRONTEND:
                        self.assertNotIn("-Wall", argv)
                        self.assertNotIn("/Wall", argv)
                if sanitizers:
                    self.assertIn("-fsanitize=address", argv)
                    self.assertIn("-fsanitize=undefined", argv)
                    self.assertIn("-fno-omit-frame-pointer", argv)

    @unittest.skipIf(MSVC_FRONTEND, "GNU sanitizer combination requires the GNU frontend")
    def test_custom_debug_sanitizer_profile_keeps_requested_flags(self):
        commands = self.configure(
            "Ninja", "-DCMAKE_BUILD_TYPE=Debug",
            "-DCMAKE_CXX_FLAGS_DEBUG=-O1 -g1 -fno-omit-frame-pointer",
            "-DCMAKE_C_FLAGS_DEBUG=-O1 -g1 -fno-omit-frame-pointer",
            "-DJYPPX_LIUDING_ENABLE_ASAN=ON", "-DJYPPX_LIUDING_ENABLE_UBSAN=ON")
        self.assert_profile(commands, "-O1", "-g1", sanitizers=True)

    @unittest.skipUnless(MSVC_FRONTEND, "clang-cl frontend profile")
    def test_custom_clang_cl_debug_profile_keeps_requested_flags(self):
        commands = self.configure(
            "Ninja", "-DCMAKE_BUILD_TYPE=Debug",
            "-DCMAKE_CXX_FLAGS_DEBUG=/O1 /Z7",
            "-DCMAKE_C_FLAGS_DEBUG=/O1 /Z7")
        self.assert_profile(commands, "/O1", "/Z7")

    def test_custom_release_profile_keeps_toolchain_choices(self):
        flags = "/O1 /DNDEBUG" if MSVC_FRONTEND else "-Os -g0 -DNDEBUG"
        commands = self.configure(
            "Ninja", "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CXX_FLAGS_RELEASE={flags}",
            f"-DCMAKE_C_FLAGS_RELEASE={flags}")
        self.assert_profile(commands, "/O1" if MSVC_FRONTEND else "-Os",
                            None if MSVC_FRONTEND else "-g0", ndebug=True)

    def test_multi_config_retains_cmake_defaults(self):
        commands = self.configure("Ninja Multi-Config")
        profiles = (
                ("Debug", None, "-g", False),
                ("Release", "-O3", None, True),
                ("RelWithDebInfo", "-O2", "-g", True))
        if MSVC_FRONTEND:
            profiles = (("Debug", "/Od", "/Zi", False),
                        ("Release", "/O2", None, True),
                        ("RelWithDebInfo", "/O2", "/Zi", True))
        for config, optimization, debug, ndebug in profiles:
            selected = [argv for argv in commands if any(
                a.startswith("-DCMAKE_INTDIR=") and config in a for a in argv)]
            with self.subTest(configuration=config):
                self.assert_profile(selected, optimization, debug, ndebug)


if __name__ == "__main__":
    unittest.main()
