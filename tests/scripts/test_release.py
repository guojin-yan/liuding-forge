#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Release candidate, tamper, bounded extraction, and external signature tests.

Synthetic evidence and ephemeral keys in this test never authorize a release.
No build, network connection, installed binary, or release signing key is used.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/release"))

import prepare_release as prepare  # noqa: E402
import release_common as common  # noqa: E402
import verify_release as verify  # noqa: E402


class ReleaseTest(unittest.TestCase):
    """Exercise public preparation and independent verification with real files."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="jyppx-release-test-")
        self.root = Path(self.temporary.name)
        self.prefix = self.root / "install"
        self.header = self.prefix / "include/jyppx/foundation/build_config.hpp"
        self.header.parent.mkdir(parents=True)
        self.header.write_text(
            '#define JYPPX_VERSION_MAJOR 1\n#define JYPPX_VERSION_MINOR 0\n'
            '#define JYPPX_VERSION_PATCH 0\n#define JYPPX_VERSION "1.0.0"\n',
            encoding="utf-8",
        )
        (self.prefix / "lib").mkdir()
        (self.prefix / "lib/jyppx_fixture.a").write_bytes(b"synthetic archive fixture\0\x01")
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.license = self.inputs / "LICENSE"
        self.notices = self.inputs / "THIRD_PARTY_NOTICES.md"
        shutil.copyfile(ROOT / "LICENSE", self.license)
        shutil.copyfile(ROOT / "THIRD_PARTY_NOTICES.md", self.notices)
        self.lock = self.inputs / "dependency-manifest.json"
        self.index = self.inputs / "evidence-index.json"
        self.output = self.root / "candidate"
        self.release = {
            "version": "1.0.0", "platform": "linux-x86_64-cpu",
            "source": {"commit": "1" * 40, "dirty": False},
            "source_date_epoch": 1_750_000_001,
        }
        self.dependencies = {
            "schema_version": 1, "status": "complete", "hash_algorithm": "sha256",
            "dependencies": [{
                "id": "fixture", "scope": "test", "version": "1.2.3",
                "source": {"type": "source-archive", "url": "https://example.test/fixture-1.2.3.tar.gz",
                           "cache_file": None},
                "integrity": {"status": "verified", "sha256": "2" * 64},
                "license": {"spdx": ["MIT"], "third_party_notice_audit": "recorded"},
                "platforms": ["linux-x86_64"], "linkage": "static", "support_status": "test-only",
            }],
        }
        self.evidence = self.evidence_document()
        self.write_inputs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def evidence_document(self) -> dict:
        """Synthetic evidence is deliberately labeled as such in the fixture."""
        log = self.inputs / "synthetic.log"
        log.write_bytes(b"Synthetic unit-test fixture: this is not release acceptance evidence.\n")
        record = {"path": log.name, "size": log.stat().st_size,
                  "sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
        inventory = common.canonical(common.prefix_inventory(self.prefix, common.Limits()))
        return {
            "schema_version": 1,
            "release": {key: self.release[key] for key in ("version", "platform", "source")},
            "installation_inventory_sha256": hashlib.sha256(inventory).hexdigest(),
            "dependency_lock_sha256": hashlib.sha256(common.canonical(self.dependencies)).hexdigest(),
            "checks": [{"id": identity, "status": "passed", "details": "synthetic unit-test check",
                        "artifacts": [copy.deepcopy(record)]} for identity in sorted(common.CHECK_IDS)],
            "dependency_audit": {
                "status": "passed", "audited_dependency_ids": ["fixture"],
                "blocking_licenses": [], "blocking_vulnerabilities": [],
                "artifacts": [copy.deepcopy(record)],
            },
        }

    def write_inputs(self) -> None:
        self.lock.write_bytes(common.canonical(self.dependencies))
        self.evidence["dependency_lock_sha256"] = hashlib.sha256(self.lock.read_bytes()).hexdigest()
        self.index.write_bytes(common.canonical(self.evidence))

    def candidate(self, output: Path | None = None, **overrides) -> dict:
        arguments = dict(
            prefix=self.prefix, out=output or self.output, release=self.release,
            evidence_index=self.index, dependency_lock=self.lock, license_file=self.license,
            notices_file=self.notices,
        )
        arguments.update(overrides)
        return prepare.prepare_release(**arguments)

    def inspect(self, directory: Path | None = None, **overrides) -> dict:
        arguments = dict(
            expected_version=self.release["version"],
            expected_source_commit=self.release["source"]["commit"],
            expected_source_dirty=self.release["source"]["dirty"],
            expected_platform=self.release["platform"],
        )
        arguments.update(overrides)
        return verify.verify_release(directory or self.output, **arguments)

    def load_manifest(self) -> dict:
        return json.loads((self.output / "release-manifest.json").read_bytes())

    def test_candidate_round_trip_unpacks_and_keeps_signature_pending(self) -> None:
        self.assertEqual(self.candidate()["ga_blockers"], ["signature.pending"])
        result = self.inspect()
        self.assertTrue(result["candidate_verified"])
        self.assertFalse(result["ga_ready"])
        self.assertEqual(result["unpacked_files"], 2)
        self.assertEqual(result["ga_blockers"], ["signature.pending"])

    def test_outputs_are_deterministic_and_do_not_overwrite(self) -> None:
        self.candidate()
        other = self.root / "candidate-2"
        self.candidate(other)
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (other / path.relative_to(self.output)).read_bytes())
        with self.assertRaisesRegex(common.ReleaseError, "already exists"):
            self.candidate()

    def test_checked_in_draft_dependency_manifest_never_becomes_audit_pass(self) -> None:
        self.dependencies = json.loads((ROOT / "dependency-manifest.json").read_bytes())
        self.evidence["dependency_audit"] = {
            "status": "pending", "audited_dependency_ids": [],
            "blocking_licenses": [], "blocking_vulnerabilities": [], "artifacts": [],
        }
        self.write_inputs()
        self.candidate()
        result = self.inspect()
        self.assertFalse(result["ga_ready"])
        self.assertIn("dependency-manifest.draft", result["ga_blockers"])
        self.assertIn("dependency-audit.pending", result["ga_blockers"])
        self.assertTrue(any("integrity-pending" in item for item in result["ga_blockers"]))
        sbom = json.loads((self.output / "sbom.cdx.json").read_bytes())
        self.assertNotIn("vulnerabilities", sbom)

    def test_dirty_or_prerelease_source_remains_explicit(self) -> None:
        self.release["source"]["dirty"] = True
        self.release["version"] = "1.0.0-rc.1"
        self.evidence["release"] = {key: self.release[key] for key in ("version", "platform", "source")}
        self.write_inputs()
        self.candidate()
        result = self.inspect()
        self.assertIn("source.dirty", result["ga_blockers"])
        self.assertIn("version.not-ga", result["ga_blockers"])
        with self.assertRaises(common.ReleaseError):
            self.inspect(expected_source_dirty=False)

    def test_rejects_claimed_ga_version_that_does_not_match_installed_sdk(self) -> None:
        self.header.write_text(self.header.read_text().replace("1.0.0", "0.3.0"), encoding="utf-8")
        with self.assertRaisesRegex(common.ReleaseError, "version string"):
            self.candidate()

    def test_rejects_stale_source_prefix_and_lock_evidence(self) -> None:
        for field, replacement in (
            ("installation_inventory_sha256", "3" * 64),
            ("dependency_lock_sha256", "4" * 64),
            ("release", {**self.evidence["release"], "source": {"commit": "5" * 40, "dirty": False}}),
        ):
            original = self.evidence[field]
            self.evidence[field] = replacement
            self.index.write_bytes(common.canonical(self.evidence))
            with self.assertRaises(common.ReleaseError):
                self.candidate()
            self.evidence[field] = original

    def test_missing_inventory_binding_and_checks_are_pending(self) -> None:
        self.evidence["installation_inventory_sha256"] = None
        self.evidence["checks"] = []
        self.write_inputs()
        self.candidate()
        result = self.inspect()
        self.assertIn("evidence.installation-inventory.pending", result["ga_blockers"])
        self.assertIn("evidence.windows.missing", result["ga_blockers"])
        self.assertIn("evidence.ubuntu-cuda.missing", result["ga_blockers"])

    def test_unknown_license_and_audit_findings_cannot_be_hidden_by_passed_status(self) -> None:
        self.dependencies["dependencies"][0]["license"]["spdx"] = ["Not-A-Real-License"]
        self.evidence["dependency_audit"]["blocking_licenses"] = ["unresolved redistribution obligation"]
        self.evidence["dependency_audit"]["blocking_vulnerabilities"] = ["unresolved P1 finding"]
        self.write_inputs()
        self.candidate()
        result = self.inspect()
        self.assertIn("dependency.fixture.license-review-pending", result["ga_blockers"])
        self.assertIn("dependency-audit.blocking-licenses", result["ga_blockers"])
        self.assertIn("dependency-audit.blocking-vulnerabilities", result["ga_blockers"])

    def test_passed_check_requires_real_artifact_and_rejects_log_tamper(self) -> None:
        self.evidence["checks"][0]["artifacts"] = []
        self.write_inputs()
        with self.assertRaisesRegex(common.ReleaseError, "actual hashed artifacts"):
            self.candidate()
        self.evidence = self.evidence_document()
        self.write_inputs()
        (self.inputs / "synthetic.log").write_bytes(b"substituted log")
        with self.assertRaisesRegex(common.ReleaseError, "evidence content"):
            self.candidate()

    def test_rejects_unknown_files_expected_identity_and_digest_corruption(self) -> None:
        self.candidate()
        for override in ({"expected_version": "9.0.0"}, {"expected_source_commit": "a" * 40},
                         {"expected_platform": "other-platform"}):
            with self.assertRaises(common.ReleaseError):
                self.inspect(**override)
        extra = self.output / "unexpected.bin"
        extra.write_bytes(b"not inventoried")
        with self.assertRaisesRegex(common.ReleaseError, "unknown or missing"):
            self.inspect()
        extra.unlink()
        (self.output / "evidence/synthetic.log").write_bytes(b"tampered")
        with self.assertRaisesRegex(common.ReleaseError, "size/hash"):
            self.inspect()

    def test_rejects_duplicate_unknown_noncanonical_and_deep_json(self) -> None:
        self.candidate()
        path = self.output / "release-manifest.json"
        original = path.read_bytes()
        for malformed in (
            b'{"schema_version":1,' + original[1:],
            common.canonical({**json.loads(original), "unknown": True}),
            json.dumps(json.loads(original), indent=2).encode(),
            b"[" * 25 + b"0" + b"]" * 25,
        ):
            path.write_bytes(malformed)
            with self.assertRaises(common.ReleaseError):
                self.inspect()
        path.write_bytes(original)

    def test_paths_reject_traversal_drives_devices_aliases_and_links(self) -> None:
        for name in ("../x", "a/../x", "/abs", "C:/x", "a\\x", "file:stream",
                     "con", "NUL.txt", "com1.dll", "trail.", "a//b", "white space"):
            with self.subTest(name=name), self.assertRaises(common.ReleaseError):
                common.relative_name(name)
        for names in (["A", "a"], ["a", "a/b"], ["a/b", "a"], ["x", "x"]):
            with self.assertRaises(common.ReleaseError):
                common.check_names(names, common.Limits())
        source = self.prefix / "lib/jyppx_fixture.a"
        alias = self.prefix / "lib/alias.a"
        os.link(source, alias)
        with self.assertRaisesRegex(common.ReleaseError, "unaliased"):
            self.candidate()
        alias.unlink()
        try:
            alias.symlink_to(source)
        except OSError:
            return  # Windows without symlink privileges still tested hardlink rejection.
        with self.assertRaisesRegex(common.ReleaseError, "links"):
            self.candidate()

    def test_stable_read_detects_same_size_write_during_copy(self) -> None:
        source = self.inputs / "mutable.bin"
        source.write_bytes(b"original")
        initial_time = source.stat().st_mtime_ns

        class MutatingOutput(io.BytesIO):
            def write(self, data: bytes) -> int:
                with source.open("r+b") as file:
                    file.write(b"modified")
                os.utime(source, ns=(initial_time, initial_time + 2_000_000_000))
                return super().write(data)

        with self.assertRaisesRegex(common.ReleaseError, "changed during read"):
            common.stream_file(source, 1024, MutatingOutput())

    def test_resource_bounds_reject_prefix_and_central_directory_before_extraction(self) -> None:
        limits = common.Limits(max_files=1)
        with self.assertRaises(common.ReleaseError):
            self.candidate(limits=limits)
        with self.assertRaises(common.ReleaseError):
            self.candidate(limits=common.Limits(max_file_bytes=4))
        self.candidate()
        manifest = self.load_manifest()
        archive = self.output / manifest["archive"]["path"]
        inventory = common.prefix_inventory(self.prefix, common.Limits())
        with self.assertRaisesRegex(common.ReleaseError, "central-directory"):
            verify.verify_archive(archive, inventory, self.release, limits)
        with self.assertRaises(common.ReleaseError):
            common.json_document(b"[0,0,0,0]", common.Limits(max_json_nodes=3))

    def test_independent_extractor_rejects_compression_types_traversal_and_unknown_files(self) -> None:
        inventory = common.prefix_inventory(self.prefix, common.Limits())
        archive = self.root / "hostile.zip"
        for attack in ("traversal", "symlink", "compressed", "unknown", "duplicate"):
            with self.subTest(attack=attack):
                with zipfile.ZipFile(archive, "w") as output:
                    for position, entry in enumerate(inventory["files"]):
                        name = "payload/" + entry["path"]
                        if position == 0 and attack == "traversal":
                            name = "payload/../outside"
                        info = zipfile.ZipInfo(name, prepare.zip_timestamp(self.release["source_date_epoch"]))
                        info.create_system = 3
                        info.external_attr = (stat.S_IFREG | entry["mode"]) << 16
                        if position == 0 and attack == "symlink":
                            info.external_attr = (stat.S_IFLNK | 0o777) << 16
                        if attack == "compressed":
                            info.compress_type = zipfile.ZIP_DEFLATED
                        output.writestr(info, (self.prefix / entry["path"]).read_bytes())
                    if attack in {"unknown", "duplicate"}:
                        output.writestr("payload/unknown" if attack == "unknown" else info.filename, b"x")
                with self.assertRaises(common.ReleaseError):
                    verify.verify_archive(archive, inventory, self.release, common.Limits())
                self.assertFalse((self.root / "outside").exists())

    def test_independent_extractor_checks_hashes_truncation_hidden_data_and_local_names(self) -> None:
        self.candidate()
        manifest = self.load_manifest()
        archive = self.output / manifest["archive"]["path"]
        inventory = common.prefix_inventory(self.prefix, common.Limits())
        wrong = copy.deepcopy(inventory)
        wrong["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(common.ReleaseError, "size/hash"):
            verify.verify_archive(archive, wrong, self.release, common.Limits())
        original = archive.read_bytes()
        name_size = struct.unpack_from("<H", original, 26)[0]
        changed_name = bytearray(original)
        changed_name[30 + name_size - 1] ^= 1
        for data in (original[:-1], original + b"hidden", b"hidden" + original, bytes(changed_name)):
            hostile = self.root / "hostile.zip"
            hostile.write_bytes(data)
            with self.assertRaises((common.ReleaseError, zipfile.BadZipFile)):
                verify.verify_archive(hostile, inventory, self.release, common.Limits())

    def test_cli_require_ga_exits_pending_and_writes_structured_reason(self) -> None:
        self.candidate()
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/release/verify_release.py"),
             "--directory", str(self.output), "--expected-version", "1.0.0",
             "--expected-source-commit", "1" * 40, "--expected-platform", self.release["platform"],
             "--require-ga"], capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads(result.stdout)["ga_blockers"], ["signature.pending"])

    def test_real_external_signature_requires_pinned_trust_and_never_removes_audit_blockers(self) -> None:
        openssl_name = os.environ.get("JYPPX_RELEASE_TEST_OPENSSL") or shutil.which("openssl")
        if not openssl_name:
            self.skipTest("external OpenSSL integration requires an explicit test executable")
        openssl = Path(openssl_name).resolve()
        # Ephemeral unit-test key: created only here, outside release output.
        private = self.root / "test-only-private.pem"
        public = self.root / "test-only-public.pem"
        signature = self.root / "test-only.sig"

        def invoke(*arguments: str) -> None:
            result = subprocess.run([str(openssl), *arguments], capture_output=True,
                                    timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

        invoke("genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", str(private))
        invoke("pkey", "-in", str(private), "-pubout", "-out", str(public))
        self.candidate()
        payload = self.output / "signing-payload.json"
        invoke("dgst", "-sha256", "-sign", str(private), "-out", str(signature), str(payload))
        trust = dict(signature=signature, trusted_public_key=public,
                     trusted_public_key_sha256=hashlib.sha256(public.read_bytes()).hexdigest(), openssl=openssl)
        result = self.inspect(**trust)
        self.assertTrue(result["ga_ready"])
        self.assertTrue(result["signature_verified"])
        with self.assertRaisesRegex(common.ReleaseError, "pin mismatch"):
            self.inspect(**{**trust, "trusted_public_key_sha256": "0" * 64})
        signature.write_bytes(b"invalid")
        with self.assertRaisesRegex(common.ReleaseError, "signature verification failed"):
            self.inspect(**trust)
        self.dependencies["status"] = "draft"
        self.write_inputs()
        candidate = self.root / "draft-candidate"
        self.candidate(candidate)
        invoke("dgst", "-sha256", "-sign", str(private), "-out", str(signature), str(candidate / "signing-payload.json"))
        self.assertIn("dependency-manifest.draft", self.inspect(candidate, **trust)["ga_blockers"])


if __name__ == "__main__":
    unittest.main()
