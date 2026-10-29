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
               provider: str = "CPUExecutionProvider",
               model_arguments: tuple | None = None) -> dict:
        model_arguments = model_arguments or ("--path", self.model_path)
        return self.invoke(
            "model", "onnx", "verify", *model_arguments,
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

    def bundle(self, version: int = 1) -> tuple:
        self.bundle_path = self.directory / "bundle 测试"
        self.bundle_path.mkdir()
        files = []

        def add(name: str, role: int, data: bytes, path: str,
                media: str = "application/json") -> None:
            target = self.bundle_path / path
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(data)
            files.append({"id": name, "path": path, "role": role, "artifact": {
                "digest": {"algorithm": "sha256", "hex": hashlib.sha256(data).hexdigest()},
                "size_bytes": len(data), "logical_name": name, "media_type": media,
                "producer": "independent.bundle.fixture",
            }})

        manifest = {
            "format": "jyppx.model_bundle", "schema_version": version, "bundle_id": "onnx.cli.fixture",
            "build": self.invoke("--version")["data"]["build"],
            "compression": "none", "encryption": "none", "files": files,
            "models": [{"id": "primary", "format": "onnx", "file_ids": ["model"], "tensors": []},
                       {"id": "opaque", "format": "onnx", "file_ids": ["opaque"], "tensors": []}],
            "labels": None, "weight_import_reports": [], "signatures": [],
            "runtime": {"backend_id": "fixture.declaration", "backend_version": "1.0.0",
                        "environment_lock": {"algorithm": "sha256", "hex": "a" * 64},
                        "dependencies": [], "devices": ["cpu"], "dtypes": ["float32"],
                        "os_arch": ["fixture.arch"], "dynamic_shapes": True},
        }
        add("model", 2, self.model, "models/model.onnx", "application/onnx")
        add("opaque", 2, b"not an ONNX graph", "models/opaque.onnx", "application/onnx")
        for name, role in (("input_schema", 3), ("output_schema", 3), ("preprocessing", 4),
                           ("postprocessing", 4), ("evaluation", 6), ("sbom", 7), ("provenance", 8)):
            add(name, role, json.dumps({"fixture": name, "schema_version": 1}).encode(),
                f"metadata/{name}.json")
            manifest[name] = {"schema_id": f"fixture.{name}", "schema_version": 1, "file_id": name}
        if version == 2:
            manifest["evidence"] = []
            for name, data, path, media in (
                ("export-report", self.report, "evidence/export-report.json", "application/json"),
                ("cases", self.case_path.read_bytes(), "evidence/cases.json", "application/json"),
                ("tensors", self.tensor_path.read_bytes(), "evidence/cases.safetensors", "application/safetensors"),
            ):
                add(name, 10, data, path, media)
                manifest["evidence"].append({"schema_id": f"fixture.{name}",
                                              "schema_version": 1, "file_id": name})
        # Original noncanonical whitespace must survive in the recorded artifact.
        self.manifest_bytes = (json.dumps(manifest, indent=2) + "\n\n").encode()
        (self.bundle_path / "manifest.json").write_bytes(self.manifest_bytes)
        return ("--bundle", self.bundle_path, "--bundle-sha256",
                hashlib.sha256(self.manifest_bytes).hexdigest(), "--model-id", "primary")

    def test_bundle_model_executes_and_binds_original_manifest_for_pass_and_failure(self) -> None:
        self.suite()
        arguments = self.bundle()
        result = self.verify(model_arguments=arguments)["data"]
        self.assertEqual(result["numerical"], "passed")
        self.assertEqual(len(result["artifacts"]), 7)
        bundle = result["input_bundle"]
        self.assertEqual(bundle["model_id"], "primary")
        self.assertEqual(bundle["manifest_sha256"], hashlib.sha256(self.manifest_bytes).hexdigest())
        self.assertFalse(bundle["authenticated"])
        self.assertFalse(bundle["descriptor_payloads_validated"])
        self.assertFalse(bundle["runtime_requirements_checked"])
        self.assertEqual(result["report"]["schema_version"], 2)
        self.assertEqual(result["report"]["input_bundle"], bundle)
        artifacts = {item["role"]: self.artifact(item) for item in result["artifacts"]}
        self.assertEqual(artifacts["onnx.input_bundle_manifest"], self.manifest_bytes)
        self.assertEqual(artifacts["onnx.model"], self.model)
        self.suite(wrong=True)
        failed = self.verify(identity="bundle-failed", expected=4, model_arguments=arguments)["data"]
        self.assertEqual(failed["numerical"], "failed")
        self.assertEqual(failed["input_bundle"], bundle)
        self.assertEqual(len(failed["artifacts"]), 7)
        self.invoke("run", "verify", "--root", self.root)

    def test_bundle_identity_selection_and_input_modes_fail_before_attempt(self) -> None:
        self.suite()
        arguments = self.bundle()
        self.verify(model_arguments=(*arguments, "--path", self.model_path), expected=2)
        self.verify(model_arguments=("--path", self.model_path, "--model-id", "primary"), expected=2)
        bad_digest = list(arguments)
        bad_digest[3] = "e" * 64
        self.verify(model_arguments=tuple(bad_digest), expected=2)
        for variant in ("missing", "opaque"):
            self.verify(model_arguments=(*arguments[:-1], variant), expected=2)
        self.assertEqual(len(self.runs()), 1)

    def test_packed_v2_bundle_executes_and_rejects_tampered_evidence(self) -> None:
        self.suite()
        self.bundle(version=2)
        output = self.directory / "published v2"
        packed = self.invoke("model", "bundle", "pack", "--manifest", self.bundle_path / "manifest.json",
                             "--manifest-sha256", hashlib.sha256(self.manifest_bytes).hexdigest(),
                             "--source", self.bundle_path, "--output", output)["data"]
        self.assertEqual(packed["format"], "jyppx.model_bundle/v2")
        self.assertEqual(len(packed["manifest"]["evidence"]), 3)
        arguments = ("--bundle", output, "--bundle-sha256", packed["manifest_sha256"],
                     "--model-id", "primary")
        result = self.verify(model_arguments=arguments)["data"]
        self.assertEqual(result["numerical"], "passed")
        self.assertTrue(result["runtime_executed"])
        self.assertFalse(result["input_bundle"]["descriptor_payloads_validated"])
        artifacts = {item["role"]: self.artifact(item) for item in result["artifacts"]}
        self.assertEqual(artifacts["onnx.input_bundle_manifest"], (output / "manifest.json").read_bytes())
        self.assertEqual((output / "evidence/cases.safetensors").read_bytes(), self.tensor_path.read_bytes())
        evidence = output / "evidence/export-report.json"
        data = evidence.read_bytes()
        evidence.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        self.verify(identity="tampered-evidence", model_arguments=arguments, expected=2)
        self.assertEqual(len(self.runs()), 2)

    def test_bundle_rejects_tampered_selected_or_unselected_files(self) -> None:
        self.suite()
        arguments = self.bundle()
        model = self.bundle_path / "models/model.onnx"
        model.write_bytes(self.model[:-1] + bytes([self.model[-1] ^ 1]))
        self.verify(model_arguments=arguments, expected=2)
        model.write_bytes(self.model)
        unselected = self.bundle_path / "metadata/evaluation.json"
        data = unselected.read_bytes()
        unselected.write_bytes(data[:-1] + b" ")
        self.verify(model_arguments=arguments, expected=2)
        self.assertEqual(len(self.runs()), 1)

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
