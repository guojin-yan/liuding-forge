#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Validate paired-input data against independent exact-rational targets."""

import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests/reference/paired_affine.py"
SPEC = importlib.util.spec_from_file_location("prepare_paired_affine", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
_previous_bytecode_setting = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    SPEC.loader.exec_module(MODULE)
finally:
    sys.dont_write_bytecode = _previous_bytecode_setting
del _previous_bytecode_setting


class PreparePairedAffineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="liuding-paired-affine-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.contract = json.loads(MODULE.CONTRACT.read_text(encoding="utf-8"))

    def test_exact_rational_targets_shapes_split_disjointness_and_frozen_bytes(self):
        destination = MODULE.prepare(self.root, self.contract)
        manifest = json.loads((destination / "preparation.json").read_text(encoding="utf-8"))
        features = {}
        for split, count in [("train", 1024), ("validation", 256)]:
            content = (destination / f"{split}.jsonl").read_bytes()
            expected = self.contract["files"][split]
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected["sha256"])
            self.assertEqual(len(content), expected["bytes"])
            self.assertEqual(manifest["files"][split], expected)
            rows = [json.loads(line) for line in content.splitlines()]
            self.assertEqual(len(rows), count)
            self.assertEqual(len({row["id"] for row in rows}), count)
            features[split] = set()
            classes = [0, 0]
            for index, row in enumerate(rows):
                self.assertEqual(set(row), {"id", "left", "right", "class_id", "regression"})
                self.assertEqual(row["id"], f"{split}:{index:06d}")
                self.assertEqual([len(row[key]) for key in ["left", "right", "regression"]], [2, 2, 1])
                a, b, c, d = [Fraction(value) for value in row["left"] + row["right"]]
                self.assertTrue(all(-1 <= v <= 1 and (v * 1024).denominator == 1
                                    for v in [a, b, c, d]))
                self.assertEqual(row["class_id"], int(a + b / 2 + c - d / 4 >= 0))
                self.assertEqual(Fraction(row["regression"][0]),
                                 (4 * a - 2 * b + 3 * c + d + Fraction(1, 2)) / 8)
                classes[row["class_id"]] += 1
                features[split].add((a, b, c, d))
            self.assertEqual(classes, expected["class_counts"])
            self.assertEqual(len(features[split]), count)
        self.assertFalse(features["train"] & features["validation"])
        self.assertEqual(manifest["data_bytes"], 173375)

    def test_different_output_roots_produce_identical_files(self):
        first = MODULE.prepare(self.root / "first", self.contract)
        second = MODULE.prepare(self.root / "second", self.contract)
        for name in ["train.jsonl", "validation.jsonl", "preparation.json"]:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_existing_dataset_and_unrelated_files_are_preserved(self):
        destination = MODULE.prepare(self.root, self.contract)
        marker = self.root / "caller-owned.txt"
        marker.write_bytes(b"keep")
        before = (destination / "train.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            MODULE.prepare(self.root, self.contract)
        self.assertEqual((destination / "train.jsonl").read_bytes(), before)
        self.assertEqual(marker.read_bytes(), b"keep")

    def test_contract_drift_and_wrong_digest_never_publish(self):
        for key, replacement in [("seed", 0), ("samples", {"train": 1, "validation": 1}),
                                 ("dataset_id", "other/v1")]:
            contract = copy.deepcopy(self.contract)
            contract[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.prepare(self.root, contract)
            self.assertEqual(list(self.root.iterdir()), [])
        contract = copy.deepcopy(self.contract)
        contract["files"]["validation"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            MODULE.prepare(self.root, contract)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_partial_write_failure_cleans_only_owned_staging(self):
        marker = self.root / "caller-owned.txt"
        marker.write_bytes(b"keep")

        def failed_write(staging, split, count):
            (staging / "partial").write_bytes(b"partial")
            raise OSError("injected short write")

        with patch.object(MODULE, "write_split", side_effect=failed_write):
            with self.assertRaisesRegex(OSError, "short write"):
                MODULE.prepare(self.root, self.contract)
        self.assertEqual(list(self.root.iterdir()), [marker])
        self.assertEqual(marker.read_bytes(), b"keep")

    def test_repository_output_is_rejected_before_creation(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            MODULE.prepare(ROOT / "paired-affine-forbidden-output", self.contract)
        self.assertFalse((ROOT / "paired-affine-forbidden-output").exists())



if __name__ == "__main__":
    unittest.main()
