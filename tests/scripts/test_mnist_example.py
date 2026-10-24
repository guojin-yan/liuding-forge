#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Run the actual example on generated IDX data; no downloads or quality claim."""

import argparse
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest


def write_idx(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for prefix, count in (("train", 137), ("t10k", 19)):
        images = bytes((index * 17 + index // 784) % 256 for index in range(count * 784))
        (directory / f"{prefix}-images-idx3-ubyte").write_bytes(struct.pack(">IIII", 2051, count, 28, 28) + images)
        (directory / f"{prefix}-labels-idx1-ubyte").write_bytes(struct.pack(">II", 2049, count) + bytes(index % 10 for index in range(count)))


class MnistExampleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="liuding-mnist-example-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Spaces and shell metacharacters are literal filesystem names.
        self.cache = self.root / "cache & literal $value %PATH%"
        self.directory = self.cache / "mnist"
        write_idx(self.directory)

    def run_example(self, *arguments, root=None):
        environment = os.environ.copy()
        environment.pop("JYPPX_DATA_ROOT", None)
        if root is not None:
            environment["JYPPX_DATA_ROOT"] = str(root)
        return subprocess.run([str(BINARY), *map(str, arguments)], cwd=self.root,
                              env=environment, text=True, encoding="utf-8", errors="replace",
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)

    def assert_training(self, result, epochs):
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"completed epochs: {epochs}", result.stdout)
        self.assertIn(f"training steps: {3 * epochs}", result.stdout)
        for field in ("training loss", "validation accuracy"):
            match = re.search(re.escape(field) + r": (\S+)", result.stdout)
            self.assertIsNotNone(match, result.stdout)
            value = float(match.group(1))
            self.assertTrue(math.isfinite(value), result.stdout)
            if field == "validation accuracy":
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)
        self.assertIn("(offline)", result.stdout)

    def test_environment_root_executes_training(self):
        result = self.run_example("--offline", "--epochs", 1, "--device", DEVICE, root=self.cache)
        self.assert_training(result, 1)

    def test_explicit_directory_overrides_environment_and_positional_epochs(self):
        device = "cuda:0" if DEVICE == "cuda" else DEVICE
        result = self.run_example(self.directory, 2, "--device", device, root=self.root / "wrong-root")
        self.assert_training(result, 2)

    def test_help_and_missing_root_do_not_create_cache(self):
        result = self.run_example("--help")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Usage:", result.stdout)
        result = self.run_example("--epochs", 1)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("JYPPX_DATA_ROOT", result.stdout)

    def test_missing_cache_is_actionable_and_not_created(self):
        missing = self.root / "missing-cache"
        result = self.run_example("--offline", "--epochs", 1, root=missing)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("prepare_mnist.py", result.stdout)
        self.assertFalse(missing.exists())

    def test_invalid_arguments_fail_before_reading_files(self):
        for arguments in (("--epochs", "0"), ("--epochs", "-1"), ("--epochs", "1junk"),
                          ("--epochs", "18446744073709551616"), ("--epochs",), ("--unknown",)):
            with self.subTest(arguments=arguments):
                result = self.run_example(*arguments)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertNotIn("missing local MNIST", result.stdout)

    def test_invalid_idx_is_rejected_before_training(self):
        image_path = self.directory / "train-images-idx3-ubyte"
        original = image_path.read_bytes()
        for malformed in (b"short", struct.pack(">IIII", 2051, 0, 28, 28),
                          original[:16], struct.pack(">IIII", 2051, 138, 28, 28) + original[16:]):
            with self.subTest(size=len(malformed)):
                image_path.write_bytes(malformed)
                result = self.run_example(self.directory, 1, "--device", DEVICE)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("MNIST IDX", result.stdout)
                self.assertNotIn("training loss:", result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve()
    DEVICE = arguments.device
    unittest.main(argv=[str(__file__), *remaining])
