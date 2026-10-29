# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Installed product CLI bundle contracts using independently encoded data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        (self.root / path).write_bytes(value)
        self.document["files"].append({
            "id": identity, "path": path, "role": role,
            "artifact": {"digest": digest(value), "size_bytes": len(value),
                         "logical_name": identity, "media_type": media, "producer": "cli.fixture"},
        })

    def save(self) -> None:
        self.encoded = json.dumps(self.document, sort_keys=True, separators=(",", ":")).encode()
        (self.root / "manifest.json").write_bytes(self.encoded)

    def add_evidence(self) -> None:
        self.document["schema_version"] = 2
        self.document["evidence"] = []
        for identity, suffix, payload, media in (
            ("trace", ".json", b'{"kernel":"fixture-only"}', "application/json"),
            ("cases", ".safetensors", self.weights, "application/safetensors"),
            ("export_report", ".json", b'{"status":"fixture-only"}', "application/json"),
        ):
            self.add_file(identity, "evidence/" + identity + suffix, 10, payload, media)
            self.document["evidence"].append(
                {"file_id": identity, "schema_id": "fixture." + identity, "schema_version": 1})
        self.save()

    def pack(self, *extra: object, expected: int = 0) -> dict:
        return self.invoke("model", "bundle", "pack", "--manifest", self.root / "manifest.json",
                           "--manifest-sha256", digest(self.encoded)["hex"], "--source", self.root,
                           "--output", self.root.parent / "published 包", *extra, expected=expected)

    def assert_unpublished(self) -> None:
        self.assertFalse((self.root.parent / "published 包").exists())
        self.assertEqual(list(self.root.parent.glob(".jyppx-bundle-stage-*")), [])

    def test_complete_inventory_and_unsigned_state_are_explicit(self) -> None:
        for action in ("inspect", "verify"):
            data = self.invoke("model", "bundle", action, "--path", self.root)["data"]
            self.assertTrue(data["valid"])
            self.assertEqual(data["format"], "jyppx.model_bundle/v1")
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

    def test_pack_v1_publishes_complete_bytes_and_can_be_verified(self) -> None:
        data = self.pack()["data"]
        self.assertEqual(data["operation"], "pack")
        self.assertEqual(data["format"], "jyppx.model_bundle/v1")
        self.assertEqual(data["source_manifest_sha256"], digest(self.encoded)["hex"])
        published = self.root.parent / "published 包"
        verified = self.invoke("model", "bundle", "verify", "--path", published)["data"]
        self.assertEqual(data["manifest_sha256"], verified["manifest_sha256"])
        self.assertNotIn("evidence", data["manifest"])
        for entry in self.document["files"]:
            self.assertEqual((published / entry["path"]).read_bytes(),
                             (self.root / entry["path"]).read_bytes())

    def test_pack_v2_keeps_evidence_and_distinguishes_input_manifest_bytes(self) -> None:
        self.add_evidence()
        self.encoded = json.dumps(self.document, ensure_ascii=False, indent=2).encode() + b"\n"
        (self.root / "manifest.json").write_bytes(self.encoded)
        data = self.pack()["data"]
        self.assertEqual(data["format"], "jyppx.model_bundle/v2")
        self.assertEqual(data["source_manifest_sha256"], digest(self.encoded)["hex"])
        published = self.root.parent / "published 包"
        self.assertEqual(data["manifest_sha256"], digest((published / "manifest.json").read_bytes())["hex"])
        self.assertNotEqual(data["source_manifest_sha256"], data["manifest_sha256"])
        self.assertEqual([ref["file_id"] for ref in data["manifest"]["evidence"]],
                         ["cases", "export_report", "trace"])
        self.assertFalse(data["authenticated"])
        self.assertFalse(data["runtime_executed"])
        self.assertFalse(data["descriptor_payloads_validated"])
        self.assertFalse(data["runtime_requirements_checked"])
        for entry in self.document["files"]:
            self.assertEqual((published / entry["path"]).read_bytes(),
                             (self.root / entry["path"]).read_bytes())
        verified = self.invoke("model", "bundle", "verify", "--path", published)["data"]
        self.assertEqual(verified["manifest"], data["manifest"])

    def test_pack_rejects_wrong_manifest_identity_and_changed_evidence(self) -> None:
        self.add_evidence()
        (self.root / "manifest.json").write_bytes(self.encoded + b" ")
        result = self.pack(expected=2)
        self.assertEqual(result["error"]["name"], "serialization")
        self.assert_unpublished()
        self.save()
        (self.root / "evidence/trace.json").write_bytes(b'{"kernel":"tampered-only"}')
        self.pack(expected=2)
        self.assert_unpublished()

    def test_pack_never_overwrites_an_existing_bundle(self) -> None:
        self.add_evidence()
        self.pack()
        published = self.root.parent / "published 包"
        before = {p.relative_to(published): p.read_bytes() for p in published.rglob("*") if p.is_file()}
        self.assertEqual(self.pack(expected=3)["error"]["name"], "io")
        self.assertEqual(before, {p.relative_to(published): p.read_bytes()
                                  for p in published.rglob("*") if p.is_file()})
        self.assertEqual(list(self.root.parent.glob(".jyppx-bundle-stage-*")), [])

    def test_pack_rejects_missing_payload_and_manifest_schema_drift(self) -> None:
        self.add_evidence()
        path = self.root / "evidence/trace.json"
        payload = path.read_bytes()
        path.unlink()
        self.assertEqual(self.pack(expected=3)["error"]["name"], "io")
        self.assert_unpublished()
        path.write_bytes(payload)
        original = json.loads(self.encoded)
        for change in ("future", "duplicate_owner", "wrong_role", "v1", "future_version", "missing_evidence"):
            with self.subTest(change=change):
                self.document = json.loads(json.dumps(original))
                if change == "future":
                    self.document["future"] = []
                elif change == "duplicate_owner":
                    self.document["evidence"].append(self.document["evidence"][0])
                elif change == "wrong_role":
                    next(f for f in self.document["files"] if f["id"] == "trace")["role"] = 9
                elif change == "v1":
                    self.document["schema_version"] = 1
                elif change == "future_version":
                    self.document["schema_version"] = 3
                else:
                    del self.document["evidence"]
                self.save()
                self.pack(expected=2)
                self.assert_unpublished()

    def test_pack_cannot_establish_signature_trust_from_manifest_metadata(self) -> None:
        self.add_evidence()
        self.document["signatures"] = [{"algorithm": "ed25519", "key_id": "untrusted.fixture",
                                        "signature_hex": "ab" * 64}]
        self.save()
        self.pack("--require-trusted", "true", expected=2)
        self.assert_unpublished()
        data = self.pack()["data"]
        self.assertEqual(data["signature_status"], "present_unverified")
        self.assertFalse(data["authenticated"])

    def test_pack_rejects_hardlinked_source(self) -> None:
        self.add_evidence()
        source = self.root / "evidence/trace.json"
        os.link(source, self.root.parent / "trace-alias.json")
        self.assertEqual(self.pack(expected=3)["error"]["name"], "io")
        self.assert_unpublished()

    def test_pack_rejects_source_directory_symlink(self) -> None:
        self.add_evidence()
        original = self.root / "evidence"
        target = self.root.parent / "real-evidence"
        original.rename(target)
        try:
            original.symlink_to(target, target_is_directory=True)
        except OSError as error:
            self.skipTest("No permission to create directory symlinks: " + str(error))
        self.assertEqual(self.pack(expected=3)["error"]["name"], "io")
        self.assert_unpublished()


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
