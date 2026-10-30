# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent byte, exact-rational and filesystem checks of native preparation."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "paired_affine_reference", ROOT / "tests/reference/paired_affine.py")
REFERENCE = importlib.util.module_from_spec(SPEC)
_previous_bytecode_setting = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    SPEC.loader.exec_module(REFERENCE)
finally:
    sys.dont_write_bytecode = _previous_bytecode_setting
del _previous_bytecode_setting


class NativePairedDataPreparation(unittest.TestCase):
    cli: Path
    example: Path | None

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="liuding-native-paired-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.contract = json.loads((ROOT / "recipes/multi-io/paired-affine.dataset.json")
                                   .read_text(encoding="utf-8"))

    def invoke(self, *arguments: object, expected: int = 0,
               environment: dict | None = None) -> dict:
        result = subprocess.run(
            [str(self.cli), "data", "prepare", *(str(value) for value in arguments),
             "--format", "json"], capture_output=True, text=True, encoding="utf-8",
            timeout=30, check=False, env=environment)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        envelope = json.loads(result.stdout)
        self.assertEqual(envelope["schema_version"], 1)
        self.assertEqual(envelope["status"], "ok" if expected == 0 else "error")
        return envelope

    def prepare(self, root: Path, **options: object) -> Path:
        result = self.invoke("--name", "paired-affine", "--data-root", root,
                             **options)["data"]
        self.assertEqual(result["dataset_id"], self.contract["dataset_id"])
        destination = Path(result["destination"])
        self.assertEqual(result["preparation"], json.loads(
            (destination / "preparation.json").read_text(encoding="utf-8")))
        return destination

    def test_frozen_bytes_independent_reference_and_exact_targets(self) -> None:
        destination = self.prepare(self.root / "双输入 数据")
        reference = REFERENCE.prepare(self.root / "reference", self.contract)
        features = {}
        for split in ("train", "validation"):
            content = (destination / f"{split}.jsonl").read_bytes()
            self.assertEqual(content, (reference / f"{split}.jsonl").read_bytes())
            frozen = self.contract["files"][split]
            self.assertEqual(hashlib.sha256(content).hexdigest(), frozen["sha256"])
            self.assertEqual(len(content), frozen["bytes"])
            self.assertTrue(content.endswith(b"\n"))
            self.assertNotIn(b"\r", content)
            rows = content.splitlines(keepends=True)
            self.assertEqual(len(rows), frozen["samples"])
            self.assertEqual(max(map(len, rows)), frozen["max_record_bytes_observed"])
            self.assertLessEqual(max(map(len, rows)), self.contract["max_record_bytes"])
            features[split] = set()
            classes = [0, 0]
            for index, line in enumerate(rows):
                row = json.loads(line)
                self.assertEqual(set(row), {"id", "left", "right", "class_id", "regression"})
                self.assertEqual(row["id"], f"{split}:{index:06d}")
                self.assertEqual([len(row[name]) for name in ("left", "right", "regression")], [2, 2, 1])
                a, b, c, d = map(Fraction, row["left"] + row["right"])
                self.assertTrue(all(-1 <= value <= 1 and (value * 1024).denominator == 1
                                    for value in (a, b, c, d)))
                self.assertEqual(row["class_id"], int(a + b / 2 + c - d / 4 >= 0))
                self.assertEqual(Fraction(row["regression"][0]),
                                 (4 * a - 2 * b + 3 * c + d + Fraction(1, 2)) / 8)
                features[split].add((a, b, c, d))
                classes[row["class_id"]] += 1
            self.assertEqual(classes, frozen["class_counts"])
            self.assertEqual(len(features[split]), frozen["samples"])
        self.assertFalse(features["train"] & features["validation"])
        manifest = json.loads((destination / "preparation.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest, json.loads((reference / "preparation.json").read_text(encoding="utf-8")))
        self.assertEqual(manifest["files"], self.contract["files"])
        self.assertEqual(manifest["data_bytes"], 173375)
        self.assertEqual(manifest["max_data_bytes"], self.contract["max_data_bytes"])
        replay = self.prepare(self.root / "second")
        for filename in ("train.jsonl", "validation.jsonl", "preparation.json"):
            self.assertEqual((replay / filename).read_bytes(), (destination / filename).read_bytes())

    def test_environment_root_explicit_precedence_and_closed_arguments(self) -> None:
        environment = dict(os.environ, JYPPX_LIUDING_DATA_ROOT=str(self.root / "环境路径"))
        result = self.invoke("--name", "paired-affine", environment=environment)["data"]
        self.assertEqual(Path(result["destination"]), self.root / "环境路径/paired-affine-v1")
        self.prepare(self.root / "explicit", environment=environment)
        environment.pop("JYPPX_LIUDING_DATA_ROOT")
        for arguments in ((), ("--name", "unknown"), ("--name", "paired-affine"),
                          ("--name", "paired-affine", "--data-root", self.root, "--unknown", "1"),
                          ("--name", "paired-affine", "--name", "paired-affine"),
                          ("--data-root", self.root)):
            with self.subTest(arguments=arguments):
                self.invoke(*arguments, expected=2, environment=environment)

    def test_product_runs_with_no_python_on_path(self) -> None:
        environment = dict(os.environ)
        for key in list(environment):
            if key.upper() == "PATH":
                del environment[key]
        system_paths = [str(self.cli.parent)]
        if os.name == "nt":
            system_paths.append(str(Path(os.environ["SystemRoot"]) / "System32"))
        environment.update(PATH=os.pathsep.join(system_paths),
                           PYTHONHOME=str(self.root / "missing-python"),
                           PYTHONPATH=str(self.root / "missing-python-modules"),
                           JYPPX_LIUDING_PYTHON=str(self.root / "missing-python.exe"))
        self.prepare(self.root / "without-python", environment=environment)

    def test_existing_outputs_and_caller_owned_files_are_preserved(self) -> None:
        destination = self.prepare(self.root / "existing")
        before = {entry.name: entry.read_bytes() for entry in destination.iterdir()}
        marker = destination.parent / "caller-owned.txt"
        marker.write_bytes(b"keep")
        self.invoke("--name", "paired-affine", "--data-root", destination.parent, expected=3)
        self.assertEqual(before, {entry.name: entry.read_bytes() for entry in destination.iterdir()})
        self.assertEqual(marker.read_bytes(), b"keep")
        for kind in ("directory", "file"):
            root = self.root / kind
            root.mkdir()
            target = root / "paired-affine-v1"
            if kind == "directory":
                target.mkdir()
            else:
                target.write_bytes(b"caller data")
            self.invoke("--name", "paired-affine", "--data-root", root, expected=3)
            self.assertEqual(list(root.iterdir()), [target])
            if kind == "file":
                self.assertEqual(target.read_bytes(), b"caller data")

    def test_repository_output_is_rejected_before_creation(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".paired-native-reject-", dir=ROOT) as parent:
            target = Path(parent) / "not-created"
            self.invoke("--name", "paired-affine", "--data-root", target, expected=2)
            self.assertFalse(target.exists())
        self.invoke("--name", "paired-affine", "--data-root", ROOT, expected=2)

    def test_symbolic_link_ancestors_and_destinations_are_rejected(self) -> None:
        target = self.root / "real"
        target.mkdir()
        linked = self.root / "linked"
        try:
            linked.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Creating symlinks is unavailable: {error}")
        self.invoke("--name", "paired-affine", "--data-root", linked / "new", expected=2)
        self.assertEqual(list(target.iterdir()), [])
        root = self.root / "dangling"
        root.mkdir()
        (root / "paired-affine-v1").symlink_to(self.root / "missing", target_is_directory=True)
        self.invoke("--name", "paired-affine", "--data-root", root, expected=3)
        self.assertTrue((root / "paired-affine-v1").is_symlink())

    def test_concurrent_publication_keeps_one_complete_dataset(self) -> None:
        root = self.root / "contended"
        command = [str(self.cli), "data", "prepare", "--name", "paired-affine",
                   "--data-root", str(root), "--format", "json"]
        processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                     for _ in range(4)]
        try:
            results = [(process, process.communicate(timeout=30)) for process in processes]
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=10)
        self.assertEqual([process.returncode for process, _ in results].count(0), 1, results)
        self.assertTrue(all(process.returncode in (0, 3) for process, _ in results), results)
        destination = root / "paired-affine-v1"
        self.assertEqual(list(root.iterdir()), [destination])
        for split in ("train", "validation"):
            self.assertEqual(hashlib.sha256((destination / f"{split}.jsonl").read_bytes()).hexdigest(),
                             self.contract["files"][split]["sha256"])

    def test_cpp_teaching_example(self) -> None:
        if self.example is None:
            self.skipTest("The optional example target is disabled")
        result = subprocess.run([str(self.example), str(self.root / "example")],
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("sample id: train:000000", result.stdout)
        self.assertIn("positional model inputs:", result.stdout)
        self.assertIn("supervision targets:", result.stdout)
        for split in ("train", "validation"):
            content = (self.root / "example/paired-affine-v1" / f"{split}.jsonl").read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), self.contract["files"][split]["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--example", type=Path)
    arguments, remaining = parser.parse_known_args()
    NativePairedDataPreparation.cli = arguments.cli.resolve()
    NativePairedDataPreparation.example = arguments.example.resolve() if arguments.example else None
    unittest.main(argv=[__file__, *remaining])
