# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Independent known vectors for C++ ORT verification and durable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import unittest

from test_onnx_delivery import OnnxDeliveryCliTest


class OnnxVerificationCliTest(unittest.TestCase):
    setUpClass = classmethod(OnnxDeliveryCliTest.setUpClass.__func__)
    setUp = OnnxDeliveryCliTest.setUp
    invoke = OnnxDeliveryCliTest.invoke
    runs = OnnxDeliveryCliTest.runs

    def suite(self, wrong: bool = False) -> None:
        tensors, payload, cases = {}, bytearray(), []
        for batch in (1, 2, 5):
            values = [float((index % 9) - 4) / 2 for index in range(batch * 3)]
            expected = []
            for row in range(batch):
                for output in range(2):
                    # Fixed native fixture uses arange(6)/7 and bias [0.25,-0.5].
                    total = sum(values[row * 3 + col] * (output * 3 + col) / 7
                                for col in range(3))
                    expected.append(total + (0.25 if output == 0 else -0.5))
            if wrong:
                expected[0] += 2
            for role, shape, data in (("input", [batch, 3], values),
                                      ("output", [batch, 2], expected)):
                name = f"b{batch}.{role}"
                start = len(payload)
                payload += struct.pack(f"<{len(data)}f", *data)
                tensors[name] = {"dtype": "F32", "shape": shape,
                                 "data_offsets": [start, len(payload)]}
            cases.append({"id": f"batch-{batch}",
                          "inputs": [{"name": "input", "tensor": f"b{batch}.input"}],
                          "expected": [{"name": "output", "tensor": f"b{batch}.output"}]})
        header = json.dumps(tensors, separators=(",", ":")).encode()
        header += b" " * (-len(header) % 8)
        self.tensor_path = self.directory / "cases.safetensors"
        self.tensor_path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
        self.case_path = self.directory / "cases.json"
        self.case_path.write_text(json.dumps({
            "format": "jyppx.liuding.onnx_verification_cases", "schema_version": 1,
            "reference": "independent-closed-form-linear-v1",
            "absolute_tolerance": 1e-6, "relative_tolerance": 1e-5, "cases": cases,
        }), encoding="utf-8")

    def verify(self, identity: str = "verification-1", expected: int = 0,
               provider: str = "CPUExecutionProvider") -> dict:
        return self.invoke(
            "model", "onnx", "verify", "--path", self.model_path,
            "--report", self.report_path, "--report-sha256", hashlib.sha256(self.report).hexdigest(),
            "--cases", self.case_path, "--cases-sha256", hashlib.sha256(self.case_path.read_bytes()).hexdigest(),
            "--tensors", self.tensor_path, "--tensors-sha256", hashlib.sha256(self.tensor_path.read_bytes()).hexdigest(),
            "--provider", provider, "--root", self.root, "--id", identity,
            "--reference", "deployment", expected=expected,
        )

    def artifact(self, receipt: dict) -> bytes:
        digest = receipt["sha256"]
        data = (self.root / "artifacts/objects" / digest[:2] / digest[2:]).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
        return data

    def test_real_cpu_numerics_profile_and_complete_evidence_are_durable(self) -> None:
        self.suite()
        result = self.verify()["data"]
        self.assertTrue(result["runtime_executed"])
        self.assertEqual(result["numerical"], "passed")
        self.assertTrue(result["requested_provider_observed"])
        self.assertFalse(result["source_authenticated"])
        self.assertEqual(len(result["artifacts"]), 6)
        artifacts = {item["role"]: self.artifact(item) for item in result["artifacts"]}
        report = json.loads(artifacts["onnx.verification_report"])
        self.assertEqual(report, result["report"])
        execution = report["execution"]
        self.assertEqual([case["id"] for case in execution["cases"]], ["batch-1", "batch-2", "batch-5"])
        self.assertEqual(execution["profile"]["sha256"], hashlib.sha256(artifacts["onnx.runtime_profile"]).hexdigest())
        self.assertGreater(execution["profile"]["kernel_events"]["CPUExecutionProvider"]["Gemm"], 0)
        self.assertEqual(execution["verification"]["task_quality"], "not_checked")
        self.assertEqual(report["export_report_sha256"], hashlib.sha256(self.report).hexdigest())
        runs = {run["run_id"]: run for run in self.runs()}
        self.assertEqual(runs["verification-1"]["status"], "completed")
        self.assertEqual(runs["verification-1"]["lineage"]["parent_run_id"], "training-source")
        self.assertEqual(runs["training-source"]["updated_at_unix_ms"], 200)
        self.invoke("run", "verify", "--root", self.root)
        self.verify(expected=2)

    def test_mismatch_persists_failure_without_changing_the_training_run(self) -> None:
        self.suite(wrong=True)
        result = self.verify(expected=4)["data"]
        self.assertEqual(result["numerical"], "failed")
        self.assertEqual(len(result["artifacts"]), 6)
        report = result["report"]["execution"]
        self.assertGreater(report["verification"]["mismatched_elements"], 0)
        runs = {run["run_id"]: run for run in self.runs()}
        self.assertEqual(runs["verification-1"]["status"], "failed")
        self.assertEqual(runs["training-source"]["status"], "completed")
        self.invoke("run", "verify", "--root", self.root)

    def test_missing_provider_and_unknown_suite_fields_create_no_attempt(self) -> None:
        self.suite()
        self.verify(provider="UnavailableExecutionProvider", expected=2)
        document = json.loads(self.case_path.read_text())
        document["unrecognized"] = 1
        self.case_path.write_text(json.dumps(document))
        self.verify(expected=2)
        self.assertEqual(len(self.runs()), 1)

    def test_invalid_names_nonfinite_and_source_mismatch_fail_closed(self) -> None:
        self.suite()
        document = json.loads(self.case_path.read_text())
        document["cases"][0]["expected"][0]["name"] = "wrong"
        self.case_path.write_text(json.dumps(document))
        self.verify(expected=2)
        self.assertEqual(len(self.runs()), 1)
        self.suite()
        data = bytearray(self.tensor_path.read_bytes())
        header_size = struct.unpack_from("<Q", data)[0]
        struct.pack_into("<f", data, 8 + header_size, float("inf"))
        self.tensor_path.write_bytes(data)
        self.verify(expected=2)
        self.assertEqual(len(self.runs()), 1)
        self.suite()
        report = json.loads(self.report)
        report["source"]["config_digest"] = "e" * 64
        self.report = json.dumps(report).encode()
        self.report_path.write_bytes(self.report)
        self.verify(expected=2)
        self.assertEqual(len(self.runs()), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    options = parser.parse_args()
    OnnxVerificationCliTest.cli = options.cli.resolve(strict=True)
    OnnxVerificationCliTest.exporter = options.exporter.resolve(strict=True)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(OnnxVerificationCliTest))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
