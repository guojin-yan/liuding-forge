# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Signed native ONNX deployment snapshots, runtime execution and rejection order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unittest

from test_bundle_signing import BundleSigningCliTest, SigningFiles
from test_onnx_delivery import OnnxDeliveryCliTest
from test_onnx_verification import OnnxVerificationCliTest


class SignedOnnxCliTest(unittest.TestCase):
    cli: Path
    exporter: Path
    setUpClass = classmethod(OnnxDeliveryCliTest.setUpClass.__func__)
    setUp = OnnxDeliveryCliTest.setUp
    invoke = BundleSigningCliTest.invoke
    assert_redacted = BundleSigningCliTest.assert_redacted
    runs = OnnxDeliveryCliTest.runs
    suite = OnnxVerificationCliTest.suite
    delivery_metadata = OnnxVerificationCliTest.delivery_metadata
    pack = OnnxVerificationCliTest.pack
    packed_arguments = OnnxVerificationCliTest.packed_arguments
    smoke = OnnxVerificationCliTest.smoke
    verify_contained = OnnxVerificationCliTest.verify_contained
    verify = OnnxVerificationCliTest.verify
    artifact = OnnxVerificationCliTest.artifact
    refresh_bundle = OnnxVerificationCliTest.refresh_bundle

    def signed_bundle(self) -> tuple[SigningFiles, tuple]:
        self.suite()
        self.delivery_metadata()
        self.pack()
        files = SigningFiles(self.directory)
        original = self.bundle_path
        encoded = (original / "manifest.json").read_bytes()
        self.bundle_path = self.directory / "signed ONNX 包"
        self.invoke("model", "bundle", "sign", "--source", original,
                    "--manifest-sha256", hashlib.sha256(encoded).hexdigest(),
                    "--output", self.bundle_path, "--private-key-file", files.key,
                    *files.arguments())
        self.assertEqual((original / "manifest.json").read_bytes(), encoded)
        return files, self.packed_arguments()

    def artifact_snapshot(self) -> dict[str, str]:
        directory = self.root / "artifacts/objects"
        return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob("*") if path.is_file()}

    def assert_signature_rejection(self, result: dict) -> None:
        self.assertEqual(result["error"]["name"], "serialization")
        self.assertEqual(result["error"]["operation"], "signature.trust")

    def test_signed_smoke_uses_only_authenticated_contained_snapshot(self) -> None:
        files, arguments = self.signed_bundle()
        self.root.rename(self.directory / "unavailable-training-store")
        for path in (self.model_path, self.report_path, self.case_path, self.tensor_path, self.metadata_path):
            path.unlink()
        data = self.smoke((*arguments, *files.arguments()))["data"]
        self.assertEqual(data["numerical"], "passed")
        self.assertTrue(data["runtime_executed"])
        self.assertEqual(data["input_bundle"]["signature_status"], "trusted")
        self.assertTrue(data["input_bundle"]["authenticated"])
        self.assertTrue(data["input_bundle"]["runtime_requirements_checked"])
        self.assertEqual(data["report"]["schema_version"], 4)
        self.assertEqual(data["report"]["input_bundle"], data["input_bundle"])
        self.assertFalse(data["report"]["source_run_lineage_verified"])
        self.assertGreater(data["report"]["execution"]["profile"]["kernel_events"]["CPUExecutionProvider"]["Gemm"], 0)
        self.assertFalse(self.root.exists())

    def test_signed_verify_persists_exact_signed_manifest_and_trust_state(self) -> None:
        files, arguments = self.signed_bundle()
        data = self.verify_contained((*arguments, *files.arguments()))["data"]
        self.assertEqual(data["numerical"], "passed")
        self.assertTrue(data["input_bundle"]["authenticated"])
        self.assertTrue(data["report"]["source_run_lineage_verified"])
        artifacts = {row["role"]: self.artifact(row) for row in data["artifacts"]}
        self.assertEqual(artifacts["onnx.input_bundle_manifest"], self.manifest_bytes)
        self.assertEqual(json.loads(artifacts["onnx.verification_report"]), data["report"])
        self.assertTrue(json.loads(artifacts["onnx.verification_report"])["input_bundle"]["authenticated"])
        self.assertEqual(artifacts["onnx.runtime_lock"], (self.bundle_path / "evidence/runtime-lock.json").read_bytes())
        self.assertEqual(next(row for row in self.runs() if row["run_id"] == "contained-verify")["status"], "completed")
        self.invoke("run", "verify", "--root", self.root)

    def test_bad_unknown_and_additional_signatures_fail_before_session_or_attempt(self) -> None:
        files, arguments = self.signed_bundle()
        original = self.manifest_bytes
        artifacts = self.artifact_snapshot()
        for change in ("bad", "unknown", "additional"):
            with self.subTest(change=change):
                document = json.loads(original)
                signature = document["signatures"][0]
                if change == "bad":
                    signature["signature_hex"] = "00" * 64
                elif change == "unknown":
                    signature["key_id"] = "foreign.untrusted"
                else:
                    document["signatures"].append({**signature, "key_id": "foreign.untrusted"})
                (self.bundle_path / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
                updated = self.refresh_bundle(arguments)
                trusted = (*updated, *files.arguments(), "--require-trusted", "false")
                # The nonexistent provider would fail Session setup. A precise
                # signature.trust error proves the retained bundle is rejected
                # before that boundary, even after an explicit unsigned opt-out.
                self.assert_signature_rejection(self.smoke(trusted, expected=2,
                                                         provider="UnavailableExecutionProvider"))
                self.assert_signature_rejection(self.verify_contained(trusted, expected=2))
                self.assertEqual(len(self.runs()), 1)
                self.assertEqual(self.artifact_snapshot(), artifacts)

    def test_rehashed_payload_tampering_still_fails_signature_before_runtime(self) -> None:
        files, arguments = self.signed_bundle()
        artifacts = self.artifact_snapshot()
        value = self.tensor_path.read_bytes()
        altered = value[:-1] + bytes([value[-1] ^ 1])
        updated = self.refresh_bundle(arguments, ("tensors", altered))
        trusted = (*updated, *files.arguments())
        self.assert_signature_rejection(self.smoke(trusted, expected=2,
                                                  provider="UnavailableExecutionProvider"))
        self.assert_signature_rejection(self.verify_contained(trusted, expected=2))
        self.assertEqual(len(self.runs()), 1)
        self.assertEqual(self.artifact_snapshot(), artifacts)

    def test_revoked_and_digest_mismatched_policy_fail_before_runtime(self) -> None:
        files, arguments = self.signed_bundle()
        artifacts = self.artifact_snapshot()
        files.policy.write_bytes(files.policy_bytes + b" ")
        result = self.smoke((*arguments, *files.arguments()), expected=2,
                            provider="UnavailableExecutionProvider")
        self.assertEqual(result["error"]["operation"], "policy.digest")
        files.write_policy(revoked=True)
        trusted = (*arguments, *files.arguments())
        self.assert_signature_rejection(self.smoke(trusted, expected=2,
                                                  provider="UnavailableExecutionProvider"))
        self.assert_signature_rejection(self.verify_contained(trusted, expected=2))
        self.assertEqual(len(self.runs()), 1)
        self.assertEqual(self.artifact_snapshot(), artifacts)

    def test_unsigned_default_refuses_and_explicit_false_admits_only_unsigned(self) -> None:
        self.suite()
        self.delivery_metadata()
        self.pack()
        files = SigningFiles(self.directory)
        arguments = self.packed_arguments()
        result = self.smoke((*arguments, *files.arguments()), expected=2,
                            provider="UnavailableExecutionProvider")
        self.assertEqual(result["error"]["operation"], "signature.required")
        data = self.smoke((*arguments, *files.arguments(), "--require-trusted", "false"))["data"]
        self.assertEqual(data["numerical"], "passed")
        self.assertEqual(data["input_bundle"]["signature_status"], "unsigned")
        self.assertFalse(data["input_bundle"]["authenticated"])
        self.assertEqual(len(self.runs()), 1)

    def test_bare_onnx_cannot_silently_ignore_bundle_trust_options(self) -> None:
        self.suite()
        files = SigningFiles(self.directory)
        artifacts = self.artifact_snapshot()
        for options in (files.arguments(), ("--require-trusted", "false"),
                        ("--trust-policy", files.policy),
                        ("--trust-policy-sha256", "a" * 64)):
            with self.subTest(options=options):
                result = self.verify(model_arguments=("--path", self.model_path, *options), expected=2)
                self.assertEqual(result["error"]["name"], "invalid_argument")
        self.assertEqual(len(self.runs()), 1)
        self.assertEqual(self.artifact_snapshot(), artifacts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    options = parser.parse_args()
    SignedOnnxCliTest.cli = options.cli.resolve(strict=True)
    SignedOnnxCliTest.exporter = options.exporter.resolve(strict=True)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SignedOnnxCliTest))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
