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
        self.extra_source = self.root / "HIP SDK" / "bin"
        self.extra_source.mkdir(parents=True)
        self.extra_manifest = self.root / "hip-runtime.files"
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
        arguments = [f"-DJYPPX_RUNTIME_SOURCE_DIR={self.sdk}",
                     f"-DJYPPX_RUNTIME_OUTPUT_DIR={self.output}"]
        if self.extra_manifest.exists():
            arguments.extend((f"-DJYPPX_RUNTIME_EXTRA_DLL_LIST={self.extra_manifest}",
                              f"-DJYPPX_RUNTIME_EXTRA_DLL_SOURCE_DIR={self.extra_source}"))
        return self.invoke(*arguments, "-P", ROOT / "cmake/StageTorchRuntime.cmake",
                           success=success)

    def write_extra_manifest(self, *files: Path) -> None:
        self.extra_manifest.write_text("".join(f"{path}\n" for path in files), encoding="utf-8")

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

    def test_rocm_images_keep_relative_layout_and_owned_removal(self) -> None:
        image = self.sdk / "aotriton.images/amd-gfx110x/flash/attn_fwd.zip"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"synthetic device archive")
        signature = image.parents[1] / "__signature__"
        signature.write_bytes(b"synthetic signature")
        self.stage()
        self.assertEqual((self.output / image.relative_to(self.sdk)).read_bytes(), image.read_bytes())
        self.assertEqual((self.output / signature.relative_to(self.sdk)).read_bytes(), signature.read_bytes())
        image.unlink()
        self.stage()
        self.assertFalse((self.output / image.relative_to(self.sdk)).exists())
        self.assertTrue((self.output / signature.relative_to(self.sdk)).exists())

    def test_additional_sdk_dlls_are_owned_updated_and_removed(self) -> None:
        hip = self.extra_source / "amdhip64_7.dll"
        hip.write_bytes(b"synthetic HIP runtime")
        self.write_extra_manifest(hip)
        self.stage()
        first = hip.read_bytes()
        report = self.inventory()
        self.assertEqual(report["scope"], "staged-torch-runtime-sdk-closure")
        self.assertEqual([row["name"] for row in report["files"]],
                         ["amdhip64_7.dll", "torch.dll"])
        self.assertEqual((self.output / hip.name).read_bytes(), first)

        hip.write_bytes(b"updated synthetic HIP runtime")
        self.stage()
        self.assertEqual((self.output / hip.name).read_bytes(), hip.read_bytes())
        self.assertNotEqual((self.output / hip.name).read_bytes(), first)

        self.write_extra_manifest()
        self.stage()
        self.assertFalse((self.output / hip.name).exists())
        self.assertEqual([row["name"] for row in self.inventory()["files"]], ["torch.dll"])

    def test_additional_sdk_dll_collision_is_rejected_before_mutation(self) -> None:
        self.stage()
        old_report = (self.output / "liuding-forge-torch-runtime.json").read_bytes()
        collision = self.extra_source / "TORCH.dll"
        collision.write_bytes(b"conflicting runtime")
        self.write_extra_manifest(collision)
        self.assertIn(b"Duplicate runtime DLL name", self.stage(success=False).stdout)
        self.assertEqual((self.output / "torch.dll").read_bytes(),
                         (self.sdk / "torch.dll").read_bytes())
        self.assertEqual((self.output / "liuding-forge-torch-runtime.json").read_bytes(),
                         old_report)

    def test_additional_source_overlap_is_rejected_before_output_creation(self) -> None:
        hip = self.extra_source / "amdhip64_7.dll"
        hip.write_bytes(b"synthetic HIP runtime")
        self.write_extra_manifest(hip)
        self.output = self.extra_source / "nested-runtime"
        self.assertIn(b"must not overlap", self.stage(success=False).stdout)
        self.assertFalse(self.output.exists())
        self.assertEqual(hip.read_bytes(), b"synthetic HIP runtime")

    def write_math_data(self, root: Path) -> list[Path]:
        paths = [root / "hipblaslt/library/gfx1100/TensileLibrary_lazy_gfx1100.dat.zlib",
                 root / "hipblaslt/library/gfx1100/Kernels.so-000-gfx1100.hsaco",
                 root / "rocblas/library/TensileLibrary_lazy_gfx1100.dat"]
        for index, path in enumerate(paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"math data {index}".encode())
        return paths

    def test_bundled_math_data_keeps_layout_hashes_and_owned_removal(self) -> None:
        paths = self.write_math_data(self.sdk)
        self.stage()
        rows = {row["name"]: row for row in self.inventory()["files"]}
        self.assertEqual(self.inventory()["scope"], "staged-torch-runtime")
        for path in paths:
            name = path.relative_to(self.sdk).as_posix()
            self.assertEqual((self.output / name).read_bytes(), path.read_bytes())
            self.assertEqual(rows[name]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        independent = self.output / "rocblas/library/application.dat"
        independent.write_bytes(b"independent data")
        paths[0].unlink()
        self.stage()
        self.assertFalse((self.output / paths[0].relative_to(self.sdk)).exists())
        self.assertEqual(independent.read_bytes(), b"independent data")

    def test_sdk_math_data_is_staged_and_removed_when_opt_out(self) -> None:
        hip = self.extra_source / "hipblaslt.dll"
        hip.write_bytes(b"synthetic math runtime")
        self.write_extra_manifest(hip)
        paths = self.write_math_data(self.extra_source)
        self.stage()
        for path in paths:
            self.assertEqual((self.output / path.relative_to(self.extra_source)).read_bytes(),
                             path.read_bytes())
        self.write_extra_manifest()
        self.stage()
        self.assertEqual([row["name"] for row in self.inventory()["files"]], ["torch.dll"])
        for path in paths:
            self.assertFalse((self.output / path.relative_to(self.extra_source)).exists())

    def test_math_collision_and_modified_stale_data_preserve_previous_output(self) -> None:
        path = self.write_math_data(self.sdk)[0]
        self.stage()
        before = (self.output / "liuding-forge-torch-runtime.json").read_bytes()
        hip = self.extra_source / "hipblaslt.dll"
        hip.write_bytes(b"synthetic math runtime")
        self.write_extra_manifest(hip)
        self.write_math_data(self.extra_source)
        self.assertIn(b"Duplicate runtime DLL name", self.stage(success=False).stdout)
        self.assertFalse((self.output / hip.name).exists())
        self.extra_manifest.unlink()
        target = self.output / path.relative_to(self.sdk)
        target.unlink()  # Never write through a staged SDK hardlink.
        target.write_bytes(b"application replacement")
        path.unlink()
        self.assertIn(b"was modified", self.stage(success=False).stdout)
        self.assertEqual(target.read_bytes(), b"application replacement")
        self.assertEqual((self.output / "liuding-forge-torch-runtime.json").read_bytes(), before)

    def link_directory(self, target: Path, link: Path) -> None:
        link.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c",
                                     "mklink", "/J", str(link), str(target)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout.decode(errors="replace"))
        else:
            link.symlink_to(target, target_is_directory=True)

    def test_math_data_rejects_destination_link_and_source_escape(self) -> None:
        paths = self.write_math_data(self.extra_source)
        hip = self.extra_source / "hipblaslt.dll"
        hip.write_bytes(b"synthetic math runtime")
        self.write_extra_manifest(hip)
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel.dat"
        sentinel.write_bytes(b"must remain")
        self.link_directory(outside, self.output / "hipblaslt")
        self.assertIn(b"parent is a symlink", self.stage(success=False).stdout)
        self.assertFalse((self.output / "torch.dll").exists())
        self.assertEqual(sentinel.read_bytes(), b"must remain")
        self.output = self.root / "second-runtime"
        shutil.rmtree(paths[0].parents[1])
        self.link_directory(outside, self.extra_source / "hipblaslt/library")
        self.assertIn(b"outside its selected source", self.stage(success=False).stdout)
        self.assertFalse(self.output.exists())

    def test_math_subtree_does_not_relax_additional_dll_manifest_boundary(self) -> None:
        nested = self.extra_source / "hipblaslt/library/injected.dll"
        nested.parent.mkdir(parents=True)
        nested.write_bytes(b"not an approved DLL import")
        self.write_extra_manifest(nested)
        self.assertIn(b"outside its selected source directory", self.stage(success=False).stdout)
        self.assertFalse(self.output.exists())

    def test_rocm_device_archives_keep_dll_parent_layout(self) -> None:
        self.sdk = self.sdk / "lib"
        self.sdk.mkdir()
        (self.sdk / "torch_hip.dll").write_bytes(b"synthetic HIP backend")
        packs = self.sdk.parent / ".kpack"
        packs.mkdir()
        (packs / "torch_gfx1100.kpack").write_bytes(b"synthetic HIP device archive")
        self.output = self.root / "stage/bin"
        self.stage()
        output_pack = self.output.parent / ".kpack/torch_gfx1100.kpack"
        self.assertEqual(output_pack.read_bytes(), b"synthetic HIP device archive")
        (packs / "torch_gfx1100.kpack").unlink()
        (packs / "torch_gfx1200.kpack").write_bytes(b"another device archive")
        self.stage()
        self.assertFalse(output_pack.exists())
        self.assertEqual((output_pack.parent / "torch_gfx1200.kpack").read_bytes(),
                         b"another device archive")

    def prepare_sdk_archives(self) -> tuple[Path, Path]:
        self.sdk = self.sdk / "lib"
        self.sdk.mkdir()
        (self.sdk / "torch_hip.dll").write_bytes(b"synthetic HIP backend")
        torch_packs = self.sdk.parent / ".kpack"
        torch_packs.mkdir()
        (torch_packs / "torch_gfx1100.kpack").write_bytes(b"Torch device code")
        hip = self.extra_source / "rocblas.dll"
        hip.write_bytes(b"synthetic rocBLAS")
        self.write_extra_manifest(hip)
        sdk_packs = self.extra_source.parent / ".kpack"
        sdk_packs.mkdir()
        (sdk_packs / "blas_lib_gfx1100.kpack").write_bytes(b"BLAS device code")
        self.output = self.root / "stage/bin"
        return torch_packs, sdk_packs

    def test_sdk_archives_merge_and_disable_without_removing_torch_or_other_files(self) -> None:
        _, sdk_packs = self.prepare_sdk_archives()
        self.stage()
        packs = self.output.parent / ".kpack"
        report = json.loads((packs / "liuding-forge-torch-kpack.json").read_bytes())
        self.assertEqual(report["scope"], "staged-torch-sdk-kpacks")
        self.assertEqual({row["name"] for row in report["files"]},
                         {"blas_lib_gfx1100.kpack", "torch_gfx1100.kpack"})
        self.assertEqual((packs / "blas_lib_gfx1100.kpack").read_bytes(), b"BLAS device code")
        (packs / "independent.kpack").write_bytes(b"unowned device code")
        (sdk_packs / "blas_lib_gfx1100.kpack").unlink()
        (sdk_packs / "blas_lib_gfx1200.kpack").write_bytes(b"replacement device code")
        self.stage()
        self.assertFalse((packs / "blas_lib_gfx1100.kpack").exists())
        self.assertEqual((packs / "blas_lib_gfx1200.kpack").read_bytes(), b"replacement device code")
        self.write_extra_manifest()
        self.stage()
        self.assertFalse((packs / "blas_lib_gfx1200.kpack").exists())
        self.assertEqual((packs / "torch_gfx1100.kpack").read_bytes(), b"Torch device code")
        self.assertEqual((packs / "independent.kpack").read_bytes(), b"unowned device code")

    def test_sdk_archive_conflict_and_modified_stale_archive_are_rejected(self) -> None:
        torch_packs, sdk_packs = self.prepare_sdk_archives()
        self.stage()
        packs = self.output.parent / ".kpack"
        report = packs / "liuding-forge-torch-kpack.json"
        before = report.read_bytes()
        collision = sdk_packs / "TORCH_gfx1100.kpack"
        collision.write_bytes(b"conflicting device code")
        arguments = [f"-DJYPPX_RUNTIME_SOURCE_DIR={torch_packs}",
                     f"-DJYPPX_RUNTIME_OUTPUT_DIR={packs}",
                     f"-DJYPPX_RUNTIME_EXTRA_KPACK_DIR={sdk_packs}",
                     "-DJYPPX_RUNTIME_KPACK_ONLY=ON", "-P", ROOT / "cmake/StageTorchRuntime.cmake"]
        self.assertIn(b"Duplicate runtime DLL name", self.invoke(*arguments, success=False).stdout)
        self.assertEqual(report.read_bytes(), before)
        collision.unlink()
        stale = packs / "blas_lib_gfx1100.kpack"
        stale.unlink()  # Do not modify the SDK through a staged hardlink.
        stale.write_bytes(b"application replacement")
        (sdk_packs / stale.name).unlink()
        (sdk_packs / "new.kpack").write_bytes(b"new archive")
        self.assertIn(b"was modified", self.invoke(*arguments, success=False).stdout)
        self.assertEqual(stale.read_bytes(), b"application replacement")
        self.assertFalse((packs / "new.kpack").exists())
        self.assertEqual(report.read_bytes(), before)
        arguments[1] = f"-DJYPPX_RUNTIME_OUTPUT_DIR={sdk_packs / 'nested'}"
        self.assertIn(b"must not overlap", self.invoke(*arguments, success=False).stdout)
        self.assertFalse((sdk_packs / "nested").exists())

    def test_incremental_build_tracks_sdk_archives_and_opt_out(self) -> None:
        _, sdk_packs = self.prepare_sdk_archives()
        math_paths = self.write_math_data(self.extra_source)
        project = self.root / "project"
        project.mkdir()
        # Supply a synthetic already-scanned import closure; never load text DLLs.
        project.joinpath("CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.18)\nproject(runtime_contract NONE)\n"
            "set(WIN32 TRUE)\n"
            "option(JYPPX_LIUDING_STAGE_ROCM_RUNTIME_DLLS \"\" ON)\n"
            "add_library(hip::amdhip64 INTERFACE IMPORTED)\n"
            f'set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")\n'
            f'set(TORCH_LIBRARY "{self.sdk.as_posix()}/torch.lib")\n'
            'set_property(GLOBAL PROPERTY JYPPX_WINDOWS_ROCM_RUNTIME_SCANNED TRUE)\n'
            f'set_property(GLOBAL PROPERTY JYPPX_WINDOWS_ROCM_RUNTIME_DLLS "{self.extra_source.as_posix()}/rocblas.dll")\n'
            f'set_property(GLOBAL PROPERTY JYPPX_WINDOWS_ROCM_RUNTIME_MANIFEST "{self.extra_manifest.as_posix()}")\n'
            f'set_property(GLOBAL PROPERTY JYPPX_WINDOWS_ROCM_RUNTIME_SOURCE_DIR "{self.extra_source.as_posix()}")\n'
            f'include("{ROOT.as_posix()}/cmake/CopyTorchRuntime.cmake")\n'
            "add_custom_target(runtime_contract ALL)\n"
            "jyppx_add_torch_runtime_dependency(runtime_contract)\n", encoding="utf-8")
        build = self.root / "build"
        configure = ["-S", project, "-B", build]
        ninja = os.environ.get("JYPPX_TEST_NINJA") or shutil.which("ninja")
        if ninja:
            configure.extend(["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"])
        self.invoke(*configure)
        build_args = ["--build", build, "--config", "Release", "--parallel", "1"]
        self.invoke(*build_args)
        packs = next(build.parent.rglob("liuding-forge-torch-kpack.json")).parent
        target = packs / "blas_lib_gfx1100.kpack"
        self.assertEqual(target.read_bytes(), b"BLAS device code")
        runtime = next(build.rglob("liuding-forge-torch-runtime.json")).parent
        math_target = runtime / math_paths[0].relative_to(self.extra_source)
        self.assertEqual(math_target.read_bytes(), math_paths[0].read_bytes())
        new_math = self.extra_source / "hipblaslt/library/gfx1100/new.dat.zlib"
        new_math.write_bytes(b"new math metadata")
        self.invoke(*build_args)
        new_math_target = runtime / new_math.relative_to(self.extra_source)
        self.assertEqual(new_math_target.read_bytes(), b"new math metadata")
        replacement = new_math.with_name("replacement")
        replacement.write_bytes(b"updated metadata")
        replacement.replace(new_math)
        self.invoke(*build_args)
        self.assertEqual(new_math_target.read_bytes(), b"updated metadata")
        new_math.unlink()
        self.invoke(*build_args)
        self.assertFalse(new_math_target.exists())
        replacement = sdk_packs / "replacement"
        replacement.write_bytes(b"updated code")
        replacement.replace(sdk_packs / target.name)
        self.invoke(*build_args)
        self.assertEqual(target.read_bytes(), b"updated code")
        (sdk_packs / "solver_lib_gfx1100.kpack").write_bytes(b"solver code")
        self.invoke(*build_args)
        self.assertEqual((packs / "solver_lib_gfx1100.kpack").read_bytes(), b"solver code")
        (sdk_packs / target.name).unlink()
        self.invoke(*build_args)
        self.assertFalse(target.exists())
        self.invoke(*configure, "-DJYPPX_LIUDING_STAGE_ROCM_RUNTIME_DLLS=OFF")
        self.invoke(*build_args)
        self.assertFalse((packs / "solver_lib_gfx1100.kpack").exists())
        self.assertFalse(math_target.exists())
        self.assertEqual((packs / "torch_gfx1100.kpack").read_bytes(), b"Torch device code")

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
