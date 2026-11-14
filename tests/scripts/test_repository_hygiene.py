#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Run the real hygiene gate against isolated Git trees, without a build."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class RepositoryHygieneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="liuding-hygiene-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.cmake = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")
        self.assertTrue(self.cmake, "CMake is required by this maintenance test")
        for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md",
                     "scripts/telemetry/native_schema.lock.json"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        generated = "src/jyppx/liuding/telemetry_native/detail/telemetry_proto"
        shutil.copytree(ROOT / generated, self.root / generated)
        self.git("init", "--quiet")
        self.git("add", ".")

    def git(self, *arguments):
        return subprocess.run(["git", *arguments], cwd=self.root, check=True,
                              capture_output=True)

    def add(self, name, contents):
        path = (self.root / name).resolve()
        self.assertTrue(path.is_relative_to(self.root))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        self.git("add", "--force", "--", name)

    def check(self, passed, diagnostic=""):
        result = subprocess.run(
            [self.cmake, "-DJYPPX_SOURCE_DIR=" + str(self.root), "-P",
             str(ROOT / "cmake/CheckRepositoryHygiene.cmake")],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        output = result.stdout + result.stderr
        if passed:
            self.assertEqual(result.returncode, 0, output)
        else:
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn(diagnostic, output)

    def test_reusable_sources_and_reviewed_small_fixtures_are_allowed(self):
        self.add("recipes/detection.json", '{"source":"datasets/train.json"}')
        self.add("scripts/interop/dependencies.json", json.dumps({
            "url": "https://example.org/package.whl", "sha256": "0" * 64}))
        self.add("tests/fixtures/model.pt", "small reviewed model fixture")
        self.add("tests/cmake/build_provenance/expected.json", "{}")
        self.check(True)

    def test_nested_generated_files_are_rejected(self):
        cases = (
            ("docs/session.log", "tracked runtime artifact"),
            ("scripts/__pycache__/helper.pyc", "tracked build/cache/dependency path"),
            ("scripts/cache/CMakeCache.txt", "tracked build/cache/dependency path"),
            ("tools/cache/helper.DLL", "tracked binary or downloaded archive"),
            ("scripts/dependencies/package.whl", "tracked binary or downloaded archive"),
            ("recipes/model.onnx", "tracked model/checkpoint outside reviewed fixtures"),
            ("outputs/report.json", "tracked generated/runtime path"),
            ("scripts/cache/core.dmp", "tracked runtime artifact"),
        )
        for name, diagnostic in cases:
            with self.subTest(path=name):
                self.add(name, "synthetic generated artifact")
                self.check(False, diagnostic)
                self.git("rm", "--quiet", "--force", "--", name)

    def test_machine_paths_are_rejected_in_portable_configuration(self):
        for value in ("F:/GitSpace/project/python.exe",
                      r"C:\Users\developer\python.exe",
                      "/home/developer/env/bin/python",
                      "/Users/developer/env/bin/python",
                      "file:///tmp/package.whl"):
            with self.subTest(value=value):
                self.add("scripts/interop/environment.json",
                         json.dumps({"executable": value}))
                self.check(False, "machine-local path in portable metadata")
                self.git("rm", "--quiet", "--force", "--", "scripts/interop/environment.json")


if __name__ == "__main__":
    unittest.main()
