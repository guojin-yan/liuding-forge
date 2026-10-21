# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Installed product CLI bundle contracts using independently encoded data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


def digest(value: bytes) -> dict:
    return {"algorithm": "sha256", "hex": hashlib.sha256(value).hexdigest()}


class ModelBundleCliTest(unittest.TestCase):
    cli: Path

    def invoke(self, *arguments: object, expected: int = 0) -> dict:
        process = subprocess.run(
            [str(self.cli), *(str(value) for value in arguments), "--format", "json"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
            shell=False, check=False,
        )
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["schema_version"], 1)
        return result

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="jyppx-cli-bundle-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "bundle 测试"
        self.root.mkdir()
        header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
        header += b" " * (-len(header) % 8)
        self.weights = struct.pack("<Q", len(header)) + header + struct.pack("<2f", 1.25, -2.5)
        self.document = {
            "format": "jyppx.model_bundle", "schema_version": 1, "bundle_id": "cli.fixture",
            "build": self.invoke("--version")["data"]["build"],
            "compression": "none", "encryption": "none", "files": [],
            "models": [{"id": "primary", "format": "safetensors", "file_ids": ["weights"],
                        "tensors": [{"name": "weight", "dtype": "F32", "shape": [2],
                                     "device_intent": "cpu", "shard_file_id": "weights"}]}],
            "labels": None, "weight_import_reports": [], "signatures": [],
            "runtime": {"backend_id": "fixture.adapter", "backend_version": "1.0.0",
                        "environment_lock": digest(b"independent fixture lock"),
                        "dependencies": [{"name": "libtorch", "version": "2.5.1"}],
                        "devices": ["cpu"], "dtypes": ["float32"],
                        "os_arch": ["windows-x86_64", "linux-x86_64"], "dynamic_shapes": False},
        }
        self.add_file("weights", "weights.safetensors", 1, self.weights, "application/safetensors")
        for name, role in (("input_schema", 3), ("output_schema", 3), ("preprocessing", 4),
                           ("postprocessing", 4), ("evaluation", 6), ("sbom", 7), ("provenance", 8)):
            value = json.dumps({"owner_schema": name, "version": 1}).encode()
            self.add_file(name, name + ".json", role, value, "application/json")
            self.document[name] = {"schema_id": "fixture." + name, "schema_version": 1, "file_id": name}
        self.save()

    def add_file(self, identity: str, path: str, role: int, value: bytes, media: str) -> None:
        (self.root / path).write_bytes(value)
        self.document["files"].append({
            "id": identity, "path": path, "role": role,
            "artifact": {"digest": digest(value), "size_bytes": len(value),
                         "logical_name": identity, "media_type": media, "producer": "cli.fixture"},
        })

    def save(self) -> None:
        self.encoded = json.dumps(self.document, sort_keys=True, separators=(",", ":")).encode()
        (self.root / "manifest.json").write_bytes(self.encoded)

    def test_complete_inventory_and_unsigned_state_are_explicit(self) -> None:
        for action in ("inspect", "verify"):
            data = self.invoke("model", "bundle", action, "--path", self.root)["data"]
            self.assertTrue(data["valid"])
            self.assertEqual(data["verified_tensors"], 1)
            self.assertEqual(data["integrity_only_models"], 0)
            self.assertEqual(data["signature_status"], "unsigned")
            self.assertFalse(data["authenticated"])
            self.assertFalse(data["runtime_executed"])
            self.assertFalse(data["descriptor_payloads_validated"])
            self.assertEqual(data["manifest_sha256"], digest(self.encoded)["hex"])
            self.assertEqual(data["verified_file_bytes"], sum(entry["artifact"]["size_bytes"]
                                                                  for entry in self.document["files"]))
            self.assertEqual(data["manifest"]["bundle_id"], "cli.fixture")

    def test_signature_metadata_does_not_establish_authentication(self) -> None:
        self.document["signatures"] = [{"algorithm": "ed25519", "key_id": "untrusted.fixture", "signature_hex": "ab" * 64}]
        self.save()
        data = self.invoke("model", "bundle", "verify", "--path", self.root)["data"]
        self.assertEqual(data["signature_status"], "present_unverified")
        self.assertFalse(data["authenticated"])
        result = self.invoke("model", "bundle", "verify", "--path", self.root,
                             "--require-trusted", "true", expected=2)
        self.assertEqual(result["error"]["name"], "compatibility")

    def test_tampered_shard_and_unknown_file_fail_complete_verification(self) -> None:
        path = self.root / "weights.safetensors"
        path.write_bytes(self.weights[:-1] + bytes([self.weights[-1] ^ 1]))
        result = self.invoke("model", "bundle", "verify", "--path", self.root, expected=2)
        self.assertEqual(result["error"]["name"], "serialization")
        path.write_bytes(self.weights)
        (self.root / "unexpected.json").write_bytes(b"{}")
        self.invoke("model", "bundle", "inspect", "--path", self.root, expected=2)

    def test_tensor_mapping_and_unsafe_manifest_path_are_rejected(self) -> None:
        self.document["models"][0]["tensors"][0]["shape"] = [1, 2]
        self.save()
        self.invoke("model", "bundle", "verify", "--path", self.root, expected=2)
        self.document["models"][0]["tensors"][0]["shape"] = [2]
        self.document["files"][0]["path"] = "../weights.safetensors"
        self.save()
        self.invoke("model", "bundle", "verify", "--path", self.root, expected=2)

    def test_onnx_bytes_are_integrity_only_without_runtime_claim(self) -> None:
        (self.root / "weights.safetensors").unlink()
        self.document["files"].pop(0)
        self.add_file("opaque_model", "model.onnx", 2, b"opaque fixture, deliberately not protobuf", "application/onnx")
        self.document["models"] = [{"id": "opaque", "format": "onnx", "file_ids": ["opaque_model"], "tensors": []}]
        self.save()
        data = self.invoke("model", "bundle", "verify", "--path", self.root)["data"]
        self.assertEqual(data["verified_tensors"], 0)
        self.assertEqual(data["integrity_only_models"], 1)
        self.assertFalse(data["runtime_executed"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    options = parser.parse_args()
    ModelBundleCliTest.cli = options.cli.resolve(strict=True)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ModelBundleCliTest))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
