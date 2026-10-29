# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native-generated ONNX report registration; Python is test orchestration only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class OnnxDeliveryCliTest(unittest.TestCase):
    cli: Path
    exporter: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_temp = tempfile.TemporaryDirectory(prefix="liuding-native-report-")
        cls.addClassCleanup(cls.fixture_temp.cleanup)
        cls.fixture = Path(cls.fixture_temp.name)
        environment = dict(os.environ, JYPPX_LIUDING_ONNX_REPORT_FIXTURE=str(cls.fixture))
        process = subprocess.run(
            [str(cls.exporter), "--gtest_filter=NativeOnnxReport.CapturesExactBytesStateAndResolvedAxesWithoutRuntimeClaims"],
            env=environment, capture_output=True, text=True, encoding="utf-8",
            timeout=120, shell=False, check=False,
        )
        if process.returncode != 0:
            raise AssertionError(process.stdout + process.stderr)
        cls.model = (cls.fixture / "model.onnx").read_bytes()
        cls.report = (cls.fixture / "report.json").read_bytes()

    def invoke(self, *arguments: object, expected: int = 0) -> dict:
        process = subprocess.run(
            [str(self.cli), *(str(value) for value in arguments), "--format", "json"],
            capture_output=True, text=True, encoding="utf-8", timeout=60,
            shell=False, check=False,
        )
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        return json.loads(process.stdout)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="liuding-report-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "runs 测试"
        self.model_path = self.directory / "model.onnx"
        self.report_path = self.directory / "report.json"
        self.model_path.write_bytes(self.model)
        self.report_path.write_bytes(self.report)
        metadata = {
            "schema_version": 1, "run_id": "training-source",
            "project_id": "native", "experiment_id": "delivery",
            "display_name": "已完成训练", "status": "running",
            "created_at_unix_ms": 100, "lineage": {
                "source_digest": "a" * 64, "config_digest": "b" * 64,
                "data_manifest_digest": "c" * 64,
            },
        }
        path = self.directory / "metadata.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        self.invoke("run", "start", "--root", self.root, "--metadata", path)
        self.invoke("run", "transition", "--root", self.root, "--id", "training-source",
                    "--status", "completed", "--time-ms", 200)

    def register(self, expected: int = 0, identity: str = "export-attempt") -> dict:
        return self.invoke(
            "model", "onnx", "register", "--path", self.model_path,
            "--report", self.report_path, "--report-sha256", hashlib.sha256(self.report).hexdigest(),
            "--root", self.root, "--id", identity, "--reference", "deployment", expected=expected,
        )

    def runs(self) -> list[dict]:
        return self.invoke("run", "list", "--root", self.root)["data"]["runs"]

    def test_registers_complete_pair_with_parent_lineage_and_idempotent_replay(self) -> None:
        first = self.register()["data"]
        self.assertEqual(first["source_run_id"], "training-source")
        self.assertFalse(first["runtime_executed"])
        self.assertFalse(first["source_authenticated"])
        self.assertEqual(first["official_checker"], "passed")
        artifacts = first["artifacts"]
        self.assertEqual(len(artifacts), 2)
        self.assertEqual(artifacts[0]["sha256"], hashlib.sha256(self.model).hexdigest())
        self.assertEqual(artifacts[1]["sha256"], hashlib.sha256(self.report).hexdigest())
        self.assertEqual(self.register()["data"], first)
        runs = {entry["run_id"]: entry for entry in self.runs()}
        self.assertEqual(runs["training-source"]["status"], "completed")
        self.assertEqual(runs["training-source"]["updated_at_unix_ms"], 200)
        self.assertEqual(runs["export-attempt"]["status"], "completed")
        self.assertEqual(runs["export-attempt"]["lineage"]["parent_run_id"], "training-source")
        self.invoke("run", "verify", "--root", self.root)

    def test_modified_model_or_report_fails_before_creating_export_run(self) -> None:
        self.model_path.write_bytes(self.model[:-1] + bytes([self.model[-1] ^ 1]))
        self.assertEqual(self.register(expected=2)["error"]["name"], "serialization")
        self.model_path.write_bytes(self.model)
        self.report_path.write_bytes(self.report + b" ")
        self.assertEqual(self.register(expected=2)["error"]["name"], "serialization")
        self.assertEqual(len(self.runs()), 1)

    def test_source_lineage_mismatch_and_source_id_reuse_are_rejected(self) -> None:
        self.assertEqual(self.register(expected=2, identity="training-source")["error"]["name"], "invalid_argument")
        report = json.loads(self.report)
        report["source"]["config_digest"] = "d" * 64
        self.report = json.dumps(report).encode()
        self.report_path.write_bytes(self.report)
        self.assertEqual(self.register(expected=2)["error"]["name"], "compatibility")
        self.assertEqual(len(self.runs()), 1)

    def test_unknown_fields_and_false_runtime_proof_are_rejected(self) -> None:
        report = json.loads(self.report)
        report["verification"]["runtime"] = "passed"
        self.report = json.dumps(report).encode()
        self.report_path.write_bytes(self.report)
        self.register(expected=2)
        self.assertEqual(len(self.runs()), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    options = parser.parse_args()
    OnnxDeliveryCliTest.cli = options.cli.resolve(strict=True)
    OnnxDeliveryCliTest.exporter = options.exporter.resolve(strict=True)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(OnnxDeliveryCliTest))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
