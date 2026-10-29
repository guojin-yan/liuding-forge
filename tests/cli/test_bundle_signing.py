# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native signing CLI contracts; Python provides an independent RFC 8032 oracle only."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from test_model_bundle import ModelBundleCliTest, digest


# RFC 8032 section 7.1, TEST 1. These are public test bytes, never product keys.
SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
PUBLIC = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
EMPTY_SIGNATURE = bytes.fromhex(
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555f"
    "b8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")
KEY_ID = "ed25519-sha256-" + hashlib.sha256(PUBLIC).hexdigest()
_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P


def _base() -> tuple[int, int, int, int]:
    y = 4 * pow(5, _P - 2, _P) % _P
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * pow(2, (_P - 1) // 4, _P) % _P
    if x & 1:
        x = _P - x
    return x, y, 1, x * y % _P


def _add(first: tuple, second: tuple) -> tuple:
    x1, y1, z1, t1 = first
    x2, y2, z2, t2 = second
    a = (y1 - x1) * (y2 - x2) % _P
    b = (y1 + x1) * (y2 + x2) % _P
    c = t1 * 2 * _D * t2 % _P
    d = z1 * 2 * z2 % _P
    e, f, g, h = b - a, d - c, d + c, b + a
    return e * f % _P, g * h % _P, f * g % _P, e * h % _P


def _multiply(scalar: int) -> tuple:
    result, point = (0, 1, 1, 0), _base()
    while scalar:
        if scalar & 1:
            result = _add(result, point)
        point = _add(point, point)
        scalar >>= 1
    return result


def _encode(point: tuple) -> bytes:
    x, y, z, _ = point
    inverse = pow(z, _P - 2, _P)
    return ((y * inverse % _P) | ((x * inverse % _P & 1) << 255)).to_bytes(32, "little")


def reference_signature(message: bytes) -> bytes:
    """Test-only RFC 8032 reference, independent of OpenSSL and product code."""
    hashed = hashlib.sha512(SEED).digest()
    scalar = int.from_bytes(hashed[:32], "little") & ((1 << 254) - 8) | (1 << 254)
    public = _encode(_multiply(scalar))
    if public != PUBLIC:
        raise AssertionError("RFC 8032 oracle public key mismatch")
    nonce = int.from_bytes(hashlib.sha512(hashed[32:] + message).digest(), "little") % _L
    encoded = _encode(_multiply(nonce))
    challenge = int.from_bytes(hashlib.sha512(encoded + public + message).digest(), "little") % _L
    return encoded + ((nonce + challenge * scalar) % _L).to_bytes(32, "little")


def canonical_payload(document: dict) -> bytes:
    # Callers use an already normalized published manifest. Sorting fixture
    # arrays here would reproduce the implementation under test unnecessarily.
    unsigned = {key: value for key, value in document.items() if key != "signatures"}
    version = unsigned["schema_version"]
    return (f"JYPPX-MODEL-BUNDLE-SIGNATURE-V{version}\n".encode()
            + json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


class SigningFiles:
    """Public RFC key fixture and an independently byte-bound external policy."""

    def __init__(self, directory: Path) -> None:
        self.key = directory / "PRIVATE-PATH-MARKER 秘密.pem"
        self.policy = directory / "external trust policy.json"
        self.write_key(SEED)
        self.write_policy()

    def write_key(self, seed: bytes) -> None:
        der = bytes.fromhex("302e020100300506032b657004220420") + seed
        self.pem = (b"-----BEGIN PRIVATE KEY-----\n" + base64.b64encode(der)
                    + b"\n-----END PRIVATE KEY-----\n")
        self.key.write_bytes(self.pem)

    def write_policy(self, *, revoked: bool = False) -> None:
        document = {"format": "jyppx.liuding.bundle_trust_policy", "schema_version": 1,
                    "algorithm": "ed25519", "key_id": KEY_ID,
                    "public_key_hex": PUBLIC.hex(), "revoked": revoked}
        self.policy_bytes = (json.dumps(document, indent=2) + "\n").encode()
        self.policy.write_bytes(self.policy_bytes)

    def arguments(self) -> tuple:
        return ("--trust-policy", self.policy,
                "--trust-policy-sha256", hashlib.sha256(self.policy_bytes).hexdigest())


class BundleSigningCliTest(unittest.TestCase):
    cli: Path
    enabled: bool
    setUp = ModelBundleCliTest.setUp
    add_file = ModelBundleCliTest.add_file
    save = ModelBundleCliTest.save
    add_evidence = ModelBundleCliTest.add_evidence

    def invoke(self, *arguments: object, expected: int = 0) -> dict:
        process = subprocess.run(
            [str(self.cli), *(str(value) for value in arguments), "--format", "json"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
            timeout=30, shell=False, check=False,
        )
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        self.assertEqual(process.stderr, "")
        result = json.loads(process.stdout)
        self.assertEqual(result["schema_version"], 1)
        self.assert_redacted(process.stdout)
        return result

    def assert_redacted(self, output: str) -> None:
        for secret in ("PRIVATE-PATH-MARKER", "PRIVATE-BYTES-MARKER", SEED.hex(),
                       "-----BEGIN PRIVATE KEY-----", "-----BEGIN ENCRYPTED PRIVATE KEY-----",
                       base64.b64encode(bytes.fromhex("302e020100300506032b657004220420") + SEED).decode()):
            self.assertNotIn(secret, output)

    def require_enabled(self) -> SigningFiles:
        if not self.enabled:
            self.skipTest("This profile explicitly disables native bundle_signing")
        return SigningFiles(self.root.parent)

    def canonical_source(self) -> None:
        destination = self.root.parent / "normalized unsigned"
        self.invoke("model", "bundle", "pack", "--manifest", self.root / "manifest.json",
                    "--manifest-sha256", digest(self.encoded)["hex"], "--source", self.root,
                    "--output", destination)
        self.root = destination
        self.encoded = (self.root / "manifest.json").read_bytes()
        self.document = json.loads(self.encoded)

    def sign(self, files: SigningFiles, *, source: Path | None = None,
             output: Path | None = None, expected: int = 0,
             manifest_sha256: str | None = None, extra: tuple = ()) -> dict:
        source = source or self.root
        output = output or self.root.parent / "signed 包"
        manifest_sha256 = manifest_sha256 or hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
        return self.invoke("model", "bundle", "sign", "--source", source,
                           "--manifest-sha256", manifest_sha256, "--output", output,
                           "--private-key-file", files.key, *files.arguments(), *extra, expected=expected)

    def test_help_and_disabled_component_contract(self) -> None:
        capability = self.invoke("--help")["data"]["model_bundle_signing"]
        self.assertEqual(capability["enabled"], self.enabled)
        self.assertEqual(capability["algorithm"], "ed25519" if self.enabled else None)
        self.assertFalse(capability["interactive_prompt"])
        self.assertFalse(capability["encrypted_private_key_cli"])
        if not self.enabled:
            files = SigningFiles(self.root.parent)
            self.assertEqual(self.sign(files, expected=2)["error"]["name"], "not_implemented")
            result = self.invoke("model", "bundle", "verify", "--path", self.root,
                                 *files.arguments(), expected=2)
            self.assertEqual(result["error"]["name"], "not_implemented")
            self.assertFalse((self.root.parent / "signed 包").exists())
        self.assertFalse(self.invoke("model", "bundle", "verify", "--path", self.root)["data"]["authenticated"])

    def test_external_policy_pair_boolean_and_singletons_are_required(self) -> None:
        for options in (("--trust-policy", "not-read.json"),
                        ("--trust-policy-sha256", "a" * 64),
                        ("--require-trusted", "yes"),
                        ("--require-trusted", "false", "--require-trusted", "false"),
                        ("--trust-policy", "first", "--trust-policy", "second")):
            with self.subTest(options=options):
                result = self.invoke("model", "bundle", "verify", "--path", self.root,
                                     *options, expected=2)
                self.assertEqual(result["error"]["name"], "invalid_argument")
        result = self.invoke("model", "bundle", "verify", "--path", self.root,
                             "--require-trusted", "true", expected=2)
        self.assertEqual(result["error"]["name"], "compatibility")

    def assert_native_signature(self, evidence: bool) -> None:
        files = self.require_enabled()
        self.assertEqual(reference_signature(b""), EMPTY_SIGNATURE)
        if evidence:
            self.add_evidence()
        original = {path.relative_to(self.root): path.read_bytes()
                    for path in self.root.rglob("*") if path.is_file()}
        data = self.sign(files)["data"]
        self.assertEqual(data["operation"], "model.bundle.sign")
        self.assertEqual(data["source_manifest_sha256"], digest(self.encoded)["hex"])
        self.assertEqual(data["key_id"], KEY_ID)
        self.assertTrue(data["published"])
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["signature_count"], 1)
        self.assertFalse(data["runtime_executed"])
        published = self.root.parent / "signed 包"
        encoded = (published / "manifest.json").read_bytes()
        document = json.loads(encoded)
        self.assertEqual(data["manifest_sha256"], hashlib.sha256(encoded).hexdigest())
        self.assertEqual(document["signatures"], [{"algorithm": "ed25519", "key_id": KEY_ID,
                                                   "signature_hex": reference_signature(canonical_payload(document)).hex()}])
        for path, value in original.items():
            self.assertEqual((self.root / path).read_bytes(), value)
            if path != Path("manifest.json"):
                self.assertEqual((published / path).read_bytes(), value)
        plain = self.invoke("model", "bundle", "verify", "--path", published)["data"]
        self.assertEqual(plain["signature_status"], "present_unverified")
        self.assertFalse(plain["authenticated"])
        trusted = self.invoke("model", "bundle", "verify", "--path", published,
                              *files.arguments())["data"]
        self.assertTrue(trusted["authenticated"])
        self.assertEqual(trusted["trust_policy"]["policy_sha256"], hashlib.sha256(files.policy_bytes).hexdigest())

    def test_native_v1_signature_matches_independent_rfc_oracle(self) -> None:
        self.assert_native_signature(evidence=False)

    def test_native_v2_signature_matches_independent_rfc_oracle(self) -> None:
        self.assert_native_signature(evidence=True)

    def test_independent_signature_verifies_and_packs_without_rewriting_it(self) -> None:
        files = self.require_enabled()
        self.add_evidence()
        self.canonical_source()
        self.document["signatures"] = [{"algorithm": "ed25519", "key_id": KEY_ID,
                                         "signature_hex": reference_signature(canonical_payload(self.document)).hex()}]
        self.save()
        self.assertTrue(self.invoke("model", "bundle", "verify", "--path", self.root,
                                    *files.arguments())["data"]["authenticated"])
        packed = self.invoke("model", "bundle", "pack", "--manifest", self.root / "manifest.json",
                             "--manifest-sha256", digest(self.encoded)["hex"], "--source", self.root,
                             "--output", self.root.parent / "repacked trusted", *files.arguments())["data"]
        self.assertTrue(packed["authenticated"])
        self.assertEqual(packed["manifest"]["signatures"], self.document["signatures"])

    def test_unsigned_policy_default_and_explicit_false_do_not_bypass_all_of(self) -> None:
        files = self.require_enabled()
        result = self.invoke("model", "bundle", "verify", "--path", self.root,
                             *files.arguments(), expected=2)
        self.assertEqual(result["error"]["operation"], "signature.required")
        self.assertEqual(self.invoke("model", "bundle", "verify", "--path", self.root,
                                    *files.arguments(), "--require-trusted", "false")["data"]["signature_status"], "unsigned")
        self.sign(files)
        path = self.root.parent / "signed 包"
        original = json.loads((path / "manifest.json").read_bytes())
        for change in ("bad_signature", "unknown_key", "wrong_algorithm", "extra_foreign_signature"):
            with self.subTest(change=change):
                document = json.loads(json.dumps(original))
                signature = document["signatures"][0]
                if change == "bad_signature":
                    signature["signature_hex"] = "00" * 64
                elif change == "unknown_key":
                    signature["key_id"] = "untrusted.other"
                elif change == "wrong_algorithm":
                    signature["algorithm"] = "ed25519ph"
                else:
                    document["signatures"].append({**signature, "key_id": "untrusted.other"})
                (path / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
                result = self.invoke("model", "bundle", "verify", "--path", path,
                                     *files.arguments(), "--require-trusted", "false", expected=2)
                self.assertEqual(result["error"]["operation"], "signature.trust")
                self.assertEqual(result["error"]["name"], "serialization")

    def test_resigning_preserves_valid_signature_and_rejects_invalid_source(self) -> None:
        files = self.require_enabled()
        self.sign(files)
        source = self.root.parent / "signed 包"
        original = (source / "manifest.json").read_bytes()
        output = self.root.parent / "signed again"
        data = self.sign(files, source=source, output=output)["data"]
        self.assertEqual(data["signature_count"], 1)
        self.assertEqual((source / "manifest.json").read_bytes(), original)
        self.assertEqual(json.loads((output / "manifest.json").read_bytes())["signatures"],
                         json.loads(original)["signatures"])
        document = json.loads(original)
        document["signatures"][0]["signature_hex"] = "00" * 64
        (source / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
        rejected = self.root.parent / "invalid resign"
        result = self.sign(files, source=source, output=rejected, expected=2)
        self.assertEqual(result["error"]["operation"], "signature.trust")
        self.assertFalse(rejected.exists())

    def test_metadata_and_rehashed_content_tampering_cannot_retain_trust(self) -> None:
        files = self.require_enabled()
        self.sign(files)
        path = self.root.parent / "signed 包"
        original = json.loads((path / "manifest.json").read_bytes())
        for change in ("metadata", "rehash"):
            with self.subTest(change=change):
                document = json.loads(json.dumps(original))
                if change == "metadata":
                    document["bundle_id"] = "attacker.rewritten"
                else:
                    entry = next(item for item in document["files"] if item["id"] == "evaluation")
                    value = b'{"owner_schema":"evaluation","version":2}'
                    (path / entry["path"]).write_bytes(value)
                    entry["artifact"]["digest"] = digest(value)
                    entry["artifact"]["size_bytes"] = len(value)
                (path / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
                result = self.invoke("model", "bundle", "verify", "--path", path,
                                     *files.arguments(), expected=2)
                self.assertEqual(result["error"]["operation"], "signature.trust")

    def test_policy_original_bytes_revocation_and_wrong_private_key(self) -> None:
        files = self.require_enabled()
        self.sign(files)
        path = self.root.parent / "signed 包"
        files.policy.write_bytes(files.policy_bytes + b" ")
        result = self.invoke("model", "bundle", "verify", "--path", path,
                             *files.arguments(), expected=2)
        self.assertEqual(result["error"]["operation"], "policy.digest")
        files.write_policy(revoked=True)
        result = self.invoke("model", "bundle", "verify", "--path", path,
                             *files.arguments(), expected=2)
        self.assertEqual(result["error"]["operation"], "signature.trust")
        self.sign(files, output=self.root.parent / "revoked", expected=2)
        self.assertFalse((self.root.parent / "revoked").exists())
        files.write_policy()
        files.write_key(bytes([SEED[0] ^ 1]) + SEED[1:])
        result = self.sign(files, output=self.root.parent / "wrong key", expected=2)
        self.assertEqual(result["error"]["operation"], "publication.authorized_signer")
        self.assertFalse((self.root.parent / "wrong key").exists())

    def test_sign_publication_checks_identity_containment_and_no_replace(self) -> None:
        files = self.require_enabled()
        original = (self.root / "manifest.json").read_bytes()
        self.sign(files, manifest_sha256="0" * 64, expected=2)
        self.assertFalse((self.root.parent / "signed 包").exists())
        for output in (self.root, self.root / "descendant"):
            with self.subTest(output=output):
                self.sign(files, output=output, expected=2)
        output = self.root.parent / "occupied"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_bytes(b"do not replace")
        self.sign(files, output=output, expected=3)
        self.assertEqual(marker.read_bytes(), b"do not replace")
        self.assertEqual((self.root / "manifest.json").read_bytes(), original)
        self.assertEqual(list(self.root.parent.glob(".jyppx-bundle-stage-*")), [])

    def test_cli_never_prompts_or_accepts_secret_argv_and_redacts_key_failures(self) -> None:
        files = self.require_enabled()
        for extra in (("--password", "PRIVATE-BYTES-MARKER"),
                      ("--private-key", "PRIVATE-BYTES-MARKER"),
                      ("--private-key-file", "PRIVATE-PATH-MARKER-duplicate")):
            with self.subTest(extra=extra):
                result = self.sign(files, extra=extra, expected=2)
                self.assertEqual(result["error"]["name"], "invalid_argument")
        files.key.write_bytes(b"-----BEGIN ENCRYPTED PRIVATE KEY-----\nPRIVATE-BYTES-MARKER\n"
                              b"-----END ENCRYPTED PRIVATE KEY-----\n")
        result = self.sign(files, expected=2)
        self.assertEqual(result["error"]["operation"], "private_key.passphrase_callback")
        files.key.write_bytes(b"-----BEGIN PRIVATE KEY-----\nPRIVATE-BYTES-MARKER\n"
                              b"-----END PRIVATE KEY-----\n")
        self.assertEqual(self.sign(files, expected=3)["error"]["name"], "backend")
        files.key.unlink()
        self.assertEqual(self.sign(files, expected=3)["error"]["name"], "io")
        self.assertFalse((self.root.parent / "signed 包").exists())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--bundle-signing", choices=("on", "off"), required=True)
    options = parser.parse_args()
    BundleSigningCliTest.cli = options.cli.resolve(strict=True)
    BundleSigningCliTest.enabled = options.bundle_signing == "on"
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(BundleSigningCliTest))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
