# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Exercise real transitive ELF loading without an inherited environment."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


class LinuxTorchRuntimeSearchContract(unittest.TestCase):
    def test_build_and_relocated_install_resolve_transitive_sdk_libraries(self):
        with tempfile.TemporaryDirectory(prefix="liuding runtime ") as temporary:
            root = Path(temporary)
            source, build, sdk = root / "source", root / "build", root / "external SDK"
            source.mkdir()
            sources = {
                "sysdeps": "int fixture_sysdeps() { return 40; }\n",
                "math": "int fixture_sysdeps(); int fixture_math() { return fixture_sysdeps() + 1; }\n",
                "hip": "int fixture_math(); int fixture_hip() { return fixture_math() + 1; }\n",
                "torch": "int fixture_hip(); int fixture_torch() { return fixture_hip(); }\n",
                "app": "int fixture_torch(); int main() { return fixture_torch() == 42 ? 0 : 1; }\n",
            }
            for name, content in sources.items():
                (source / (name + ".cpp")).write_text(content, encoding="utf-8")
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.18)\n"
                "project(runtime_sdk_fixture LANGUAGES CXX)\n"
                "foreach(name IN ITEMS sysdeps math hip torch)\n"
                '  add_library(fixture_${name} SHARED "${name}.cpp")\n'
                "  set_target_properties(fixture_${name} PROPERTIES\n"
                '    OUTPUT_NAME liuding_fixture_${name} BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")\n'
                "endforeach()\n"
                'set_target_properties(fixture_sysdeps PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${SDK}/lib/rocm_sysdeps/lib")\n'
                'set_target_properties(fixture_math PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${SDK}/lib/host-math/lib")\n'
                'set_target_properties(fixture_hip fixture_torch PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${SDK}/lib")\n'
                "target_link_libraries(fixture_math PRIVATE fixture_sysdeps)\n"
                "target_link_libraries(fixture_hip PRIVATE fixture_math)\n"
                "target_link_libraries(fixture_torch PRIVATE fixture_hip)\n",
                encoding="utf-8",
            )
            cmake = os.environ.get("CMAKE_COMMAND", "cmake")

            def run(*arguments):
                result = subprocess.run([cmake, *map(str, arguments)], capture_output=True, timeout=90)
                self.assertEqual(result.returncode, 0, (result.stdout, result.stderr))

            compiler = os.environ.get("JYPPX_RUNTIME_TEST_CXX", "c++")
            run("-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CXX_COMPILER=" + compiler, "-DSDK=" + str(sdk))
            run("--build", build, "--parallel", "2")
            tools = root / "tools"
            tools.mkdir()
            shutil.copy2(source / "app.cpp", tools / "app.cpp")
            (tools / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.18)\n"
                "project(runtime_search_contract LANGUAGES CXX)\n"
                "add_library(hip::amdhip64 SHARED IMPORTED)\n"
                'set_property(TARGET hip::amdhip64 PROPERTY IMPORTED_LOCATION "${SDK}/lib/libliuding_fixture_hip.so")\n'
                "add_library(torch_hip SHARED IMPORTED)\n"
                'set(TORCH_LIBRARY "${SDK}/lib/libliuding_fixture_torch.so")\n'
                'set_property(TARGET torch_hip PROPERTY IMPORTED_LOCATION "${TORCH_LIBRARY}")\n'
                f'include("{ROOT.as_posix()}/cmake/CopyTorchRuntime.cmake")\n'
                "foreach(name IN ITEMS control resolved)\n"
                "  add_executable(${name} app.cpp)\n"
                "  target_link_libraries(${name} PRIVATE torch_hip hip::amdhip64)\n"
                '  target_link_options(${name} PRIVATE "LINKER:-rpath-link,${SDK}/lib"\n'
                '    "LINKER:-rpath-link,${SDK}/lib/host-math/lib"\n'
                '    "LINKER:-rpath-link,${SDK}/lib/rocm_sysdeps/lib")\n'
                "endforeach()\n"
                # RPATH alone cannot resolve children of a DSO with RUNPATH.
                'set_property(TARGET control PROPERTY BUILD_RPATH "${SDK}/lib;${SDK}/lib/host-math/lib;${SDK}/lib/rocm_sysdeps/lib")\n'
                'target_link_options(control PRIVATE "LINKER:--disable-new-dtags")\n'
                'set_property(TARGET resolved PROPERTY INSTALL_RPATH "$ORIGIN/../lib")\n'
                "jyppx_add_torch_runtime_dependency(resolved)\n"
                'get_target_property(paths resolved INSTALL_RPATH)\n'
                'if(NOT "$ORIGIN/../lib" IN_LIST paths)\n'
                '  message(FATAL_ERROR "caller install path was discarded")\n'
                "endif()\n"
                "install(TARGETS resolved RUNTIME DESTINATION bin)\n",
                encoding="utf-8",
            )
            build = root / "tools build"
            run("-S", tools, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CXX_COMPILER=" + compiler, "-DSDK=" + str(sdk))
            run("--build", build, "--parallel", "2")
            control = subprocess.run([str(build / "control")], env={}, capture_output=True, timeout=15)
            self.assertNotEqual(control.returncode, 0)
            self.assertIn(b"libliuding_fixture_", control.stderr)
            result = subprocess.run([str(build / "resolved")], env={}, capture_output=True, timeout=15)
            self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"", b""))
            prefix, relocated = root / "installed", root / "relocated install"
            run("--install", build, "--prefix", prefix)
            shutil.move(str(prefix), relocated)
            result = subprocess.run([str(relocated / "bin/resolved")], env={}, capture_output=True, timeout=15)
            self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"", b""))
            # A missing SDK child must stop configuration before any app builds.
            (sdk / "lib/rocm_sysdeps/lib/libliuding_fixture_sysdeps.so").unlink()
            rejected = root / "missing SDK build"
            result = subprocess.run([
                cmake, "-S", str(tools), "-B", str(rejected),
                "-DCMAKE_CXX_COMPILER=" + compiler, "-DSDK=" + str(sdk),
            ], capture_output=True, timeout=90)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"unresolved runtime dependencies", result.stderr)
            self.assertFalse((rejected / "resolved").exists())


if __name__ == "__main__":
    unittest.main()
