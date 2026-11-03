#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Exercise real CMake staging and regeneration with small synthetic DLL files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class TorchRuntimeStagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="liuding-runtime-staging-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.sdk = self.root / "SDK with spaces"
        self.sdk.mkdir()
        self.output = self.root / "runtime"
        self.cmake = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")
        self.assertTrue(self.cmake, "CMake is required by this build-tool contract")
        (self.sdk / "torch.dll").write_bytes(b"synthetic torch runtime\0")

    def invoke(self, *arguments: str, success: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run([str(self.cmake), *map(str, arguments)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=90)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout.decode(errors="replace"))
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout.decode(errors="replace"))
        return result

    def stage(self, success: bool = True) -> subprocess.CompletedProcess:
        return self.invoke(f"-DJYPPX_RUNTIME_SOURCE_DIR={self.sdk}",
                           f"-DJYPPX_RUNTIME_OUTPUT_DIR={self.output}",
                           "-P", ROOT / "cmake/StageTorchRuntime.cmake", success=success)

    def inventory(self) -> dict:
        return json.loads((self.output / "liuding-forge-torch-runtime.json").read_bytes())

    def test_exact_inventory_and_unchanged_staging_keep_hardlinks(self) -> None:
        self.stage()
        before = (self.output / "torch.dll").stat()
        self.stage()
        after = (self.output / "torch.dll").stat()
        self.assertEqual((before.st_ino, before.st_mtime_ns),
                         (after.st_ino, after.st_mtime_ns))
        raw = (self.sdk / "torch.dll").read_bytes()
        self.assertEqual(self.inventory(), {
            "schema": "liuding.torch-runtime-staging/v1", "scope": "staged-torch-dlls",
            "files": [{"name": "torch.dll", "bytes": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest()}],
        })

    def test_switch_removes_only_owned_unchanged_old_dll(self) -> None:
        (self.sdk / "old.dll").write_bytes(b"old runtime")
        self.stage()
        (self.output / "other-component.dll").write_bytes(b"independent runtime")
        (self.sdk / "old.dll").unlink()
        (self.sdk / "new.dll").write_bytes(b"new runtime")
        self.stage()
        self.assertFalse((self.output / "old.dll").exists())
        self.assertEqual((self.output / "other-component.dll").read_bytes(), b"independent runtime")
        self.assertEqual([row["name"] for row in self.inventory()["files"]],
                         ["new.dll", "torch.dll"])

    def test_modified_old_dll_rejects_before_other_output_mutation(self) -> None:
        (self.sdk / "old.dll").write_bytes(b"old runtime")
        self.stage()
        (self.sdk / "old.dll").unlink()
        (self.output / "old.dll").unlink()
        (self.output / "old.dll").write_bytes(b"externally replaced")
        (self.sdk / "new.dll").write_bytes(b"new runtime")
        old_report = (self.output / "liuding-forge-torch-runtime.json").read_bytes()
        self.assertIn(b"was modified", self.stage(success=False).stdout)
        self.assertFalse((self.output / "new.dll").exists())
        self.assertEqual((self.output / "old.dll").read_bytes(), b"externally replaced")
        self.assertEqual((self.output / "liuding-forge-torch-runtime.json").read_bytes(), old_report)

    def test_new_sdk_bytes_replace_destination_without_modifying_old_hardlink(self) -> None:
        self.stage()
        old_link = self.root / "old-sdk-link.dll"
        os.link(self.sdk / "torch.dll", old_link)
        replacement = self.sdk / "replacement"
        replacement.write_bytes(b"updated immutable SDK file")
        replacement.replace(self.sdk / "torch.dll")
        self.stage()
        self.assertEqual((self.output / "torch.dll").read_bytes(), b"updated immutable SDK file")
        self.assertEqual(old_link.read_bytes(), b"synthetic torch runtime\0")

    def test_overlapping_directories_rejected_before_output_creation(self) -> None:
        self.output = self.sdk / "nested runtime"
        self.assertIn(b"must not overlap", self.stage(success=False).stdout)
        self.assertFalse(self.output.exists())
        self.output = self.sdk
        self.stage(success=False)
        self.assertEqual((self.sdk / "torch.dll").read_bytes(), b"synthetic torch runtime\0")

    def test_invalid_ownership_cannot_remove_an_outside_file(self) -> None:
        self.stage()
        sentinel = self.root / "outside.dll"
        sentinel.write_bytes(b"must remain")
        index = self.output / "liuding-forge-torch-runtime.files"
        old_report = (self.output / "liuding-forge-torch-runtime.json").read_bytes()
        for record in ("a" * 64 + " ../outside.dll", "a" * 64 + " /outside.dll",
                       "a" * 63 + " missing.dll", "a" * 64 + " one.dll\n" + "a" * 64 + " ONE.dll"):
            with self.subTest(record=record):
                index.write_text("liuding-forge-torch-runtime-v1\n" + record + "\n", encoding="ascii")
                self.stage(success=False)
                self.assertEqual(sentinel.read_bytes(), b"must remain")
                self.assertEqual((self.output / "liuding-forge-torch-runtime.json").read_bytes(), old_report)

    def test_missing_sdk_and_empty_sdk_fail_without_output(self) -> None:
        (self.sdk / "torch.dll").unlink()
        self.stage(success=False)
        self.assertFalse(self.output.exists())
        self.sdk.rmdir()
        self.stage(success=False)
        self.assertFalse(self.output.exists())

    def test_incremental_build_detects_added_removed_and_switched_sdk_files(self) -> None:
        project = self.root / "project"
        project.mkdir()
        # Exercise the Windows-only build graph on either host with text files;
        # no compiler, executable or synthetic DLL is loaded.
        project.joinpath("CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.18)\nproject(runtime_contract NONE)\n"
            "set(WIN32 TRUE)\n"
            f'set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")\n'
            'set(TORCH_LIBRARY "${FIXTURE_SDK}/torch.lib")\n'
            f'include("{ROOT.as_posix()}/cmake/CopyTorchRuntime.cmake")\n'
            "add_custom_target(runtime_contract ALL)\n"
            "jyppx_add_torch_runtime_dependency(runtime_contract)\n",
            encoding="utf-8",
        )
        build = self.root / "build"
        arguments = ["-S", project, "-B", build, f"-DFIXTURE_SDK={self.sdk}"]
        ninja = os.environ.get("JYPPX_TEST_NINJA") or shutil.which("ninja")
        if ninja:
            arguments.extend(["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"])
        self.invoke(*arguments)
        self.invoke("--build", build, "--config", "Release", "--parallel", "1")
        outputs = list(build.rglob("liuding-forge-torch-runtime.json"))
        self.assertEqual(len(outputs), 1)
        self.output = outputs[0].parent
        (self.sdk / "added.dll").write_bytes(b"added without manual configure")
        self.invoke("--build", build, "--config", "Release", "--parallel", "1")
        self.assertTrue((self.output / "added.dll").is_file())
        (self.sdk / "added.dll").unlink()
        self.invoke("--build", build, "--config", "Release", "--parallel", "1")
        self.assertFalse((self.output / "added.dll").exists())
        other_sdk = self.root / "other SDK"
        other_sdk.mkdir()
        (other_sdk / "switched.dll").write_bytes(b"switched SDK")
        self.invoke("-S", project, "-B", build, f"-DFIXTURE_SDK={other_sdk}")
        self.invoke("--build", build, "--config", "Release", "--parallel", "1")
        self.assertFalse((self.output / "torch.dll").exists())
        self.assertEqual([row["name"] for row in self.inventory()["files"]], ["switched.dll"])


if __name__ == "__main__":
    unittest.main()
