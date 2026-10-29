#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Execute the public C++ checkpoint/ONNX example without Python model tooling."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


class DeploymentExampleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="liuding-deployment-example-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_example(self, *arguments):
        return subprocess.run(
            [str(BINARY), *map(str, arguments)], cwd=self.root,
            text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)

    def test_checkpoint_multi_io_dynamic_batches_and_cpu_trace(self):
        output = self.root / "部署 sample & literal $value %PATH%"
        result = self.run_example("--output", output)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("All outputs match the independent scalar oracle.", result.stdout)
        self.assertGreater((output / "checkpoint.jyck").stat().st_size, 0)
        export = json.loads((output / "export-report.json").read_text(encoding="utf-8"))
        onnx_digest = hashlib.sha256((output / "model.onnx").read_bytes()).hexdigest()
        self.assertEqual(export["model"]["digest"]["hex"], onnx_digest)
        self.assertEqual([item["name"] for item in export["inputs"]], ["left", "right"])
        self.assertEqual([item["name"] for item in export["outputs"]], ["calibrated", "difference"])
        state = {item["name"]: item for item in export["state"]}
        self.assertEqual({name: item["role"] for name, item in state.items()},
                         {"state.gain": "parameter", "state.offset": "buffer"})
        for name, values in (("state.gain", (0.5, -1.0, 2.0)), ("state.offset", (-0.25, 0.5, 1.0))):
            self.assertEqual(state[name]["digest"]["hex"], hashlib.sha256(struct.pack("<3f", *values)).hexdigest())
        self.assertEqual(export["minimum_batch"], 1)
        self.assertEqual(export["maximum_batch"], 8)
        self.assertEqual({axis["name"]: (axis["axis"], axis["symbol"]) for axis in export["dynamic_axes"]},
                         {name: (0, "batch") for name in ("left", "right", "calibrated", "difference")})
        report = json.loads((output / "verification-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["model_sha256"], onnx_digest)
        self.assertEqual(report["verification"]["numerical"], "passed")
        self.assertTrue(report["verification"]["requested_provider_observed"])
        self.assertFalse(report["runtime"]["allow_cpu_fallback"])
        self.assertEqual(report["runtime"]["requested_provider"], "CPUExecutionProvider")
        self.assertEqual([case["id"] for case in report["cases"]], ["batch-1", "batch-2", "batch-5"])
        for batch, case in zip((1, 2, 5), report["cases"]):
            self.assertEqual(len(case["inputs"]), 2)
            self.assertEqual(len(case["outputs"]), 2)
            for tensor in case["inputs"]:
                self.assertEqual(tensor["shape"], [batch, 3])
            left = [index / 4 - 1 for index in range(batch * 3)]
            right = [index / 8 + 0.125 for index in range(batch * 3)]
            gain, offset = (0.5, -1.0, 2.0), (-0.25, 0.5, 1.0)
            oracle = {
                "calibrated": [max(0, value * gain[index % 3] + right[index] + offset[index % 3])
                               for index, value in enumerate(left)],
                "difference": [a - b for a, b in zip(left, right)],
            }
            for comparison in case["outputs"]:
                self.assertEqual(comparison["mismatched_elements"], 0)
                self.assertEqual(comparison["actual"]["shape"], [batch, 3])
                self.assertTrue(math.isfinite(comparison["maximum_absolute_error"]))
                expected = comparison["expected"]
                payload = struct.pack(f"<{batch * 3}f", *oracle[expected["name"]])
                self.assertEqual(expected["sha256"], hashlib.sha256(payload).hexdigest())
        profile = (output / "runtime-profile.json").read_bytes()
        self.assertEqual(report["profile"]["sha256"], hashlib.sha256(profile).hexdigest())
        self.assertEqual(set(report["profile"]["kernel_events"]), {"CPUExecutionProvider"})
        self.assertGreater(sum(report["profile"]["kernel_events"]["CPUExecutionProvider"].values()), 0)
        self.assertIsInstance(json.loads(profile), list)

    def test_existing_directory_is_preserved(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "sentinel.txt"
        sentinel.write_bytes(b"keep existing bytes")
        result = self.run_example("--output", output)
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertEqual(sentinel.read_bytes(), b"keep existing bytes")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_help_and_invalid_arguments_create_no_artifacts(self):
        result = self.run_example("--help")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Usage:", result.stdout)
        for arguments in ((), ("--unknown",), ("--output",), ("--output", "relative"),
                          ("--output", self.root / "unused", "--unknown")):
            with self.subTest(arguments=arguments):
                result = self.run_example(*arguments)
                self.assertEqual(result.returncode, 2, result.stdout)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_parent_is_not_created(self):
        missing = self.root / "missing-parent"
        result = self.run_example("--output", missing / "output")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve()
    unittest.main(argv=[str(__file__), *remaining])
