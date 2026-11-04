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
        self.header = self.prefix / "include/jyppx/liuding/foundation/build_config.hpp"
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
        self.build_metadata = self.prefix / "share/jyppx/build-provenance.json"
        self.build_metadata.parent.mkdir(parents=True)
        self.simulate_rebuild()
        self.evidence = self.evidence_document()
        self.write_inputs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def simulate_rebuild(self) -> None:
        """Explicitly model a new built prefix, never an external relabeling."""
        self.build_metadata.write_bytes(common.canonical({
            "schema_version": 1, "project_version": self.release["version"].split("-", 1)[0],
            "git_commit": self.release["source"]["commit"] + (
                "+dirty" if self.release["source"]["dirty"] else ""
            ),
            "dependency_lock_sha256": hashlib.sha256(common.canonical(self.dependencies)).hexdigest(),
            "configuration": "Release",
        }))
        if hasattr(self, "evidence"):
            self.bind_evidence_to_prefix()

    def bind_evidence_to_prefix(self) -> None:
        """Make external claims coherent without changing the installed stamp."""
        self.evidence["release"] = copy.deepcopy({
            key: self.release[key] for key in ("version", "platform", "source")
        })
        self.evidence["installation_inventory_sha256"] = hashlib.sha256(
            common.canonical(common.prefix_inventory(self.prefix, common.Limits()))
        ).hexdigest()

    def evidence_document(self) -> dict:
        """Synthetic evidence is deliberately labeled as such in the fixture."""
        log = self.inputs / "synthetic.log"
        log.write_bytes(b"Synthetic unit-test fixture: this is not release acceptance evidence.\n")
        record = {"path": log.name, "size": log.stat().st_size,
                  "sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
        inventory = common.canonical(common.prefix_inventory(self.prefix, common.Limits()))
        return {
            "schema_version": 1,
            "release": copy.deepcopy({key: self.release[key] for key in ("version", "platform", "source")}),
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

    def unchecked_candidate(self, output: Path) -> None:
        """Forge a coherent outer package without calling the validating producer.

        Only extracted build metadata can expose these fixture declarations;
        every archive, inventory, evidence and signing-payload hash is current.
        """
        limits = common.Limits()
        output.mkdir()
        inventory = common.prefix_inventory(self.prefix, limits)
        inventory_data = common.canonical(inventory)
        (output / "inventory.json").write_bytes(inventory_data)
        for source, name in ((self.lock, "dependency-manifest.json"),
                             (self.index, "evidence-index.json"),
                             (self.license, "LICENSE"), (self.notices, "THIRD_PARTY_NOTICES.md")):
            shutil.copyfile(source, output / name)
        blockers, records = common.validate_evidence(
            self.evidence, self.release, hashlib.sha256(inventory_data).hexdigest(),
            hashlib.sha256(self.lock.read_bytes()).hexdigest(),
            common.validate_dependencies(self.dependencies), limits,
        )
        artifacts = []
        for record in records:
            name = "evidence/" + record["path"]
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.inputs / record["path"], destination)
            artifacts.append(common.descriptor(destination, name, limits))
        archive_name = f"liuding-forge-{self.release['version']}-{self.release['platform']}.zip"
        prepare.write_archive(output / archive_name, self.prefix, inventory,
                              self.release["source_date_epoch"], limits)
        archive = common.descriptor(output / archive_name, archive_name, limits)
        (output / "sbom.cdx.json").write_bytes(common.canonical(
            common.sbom(self.release, self.dependencies, archive["sha256"])
        ))
        documents = {
            identity: common.descriptor(output / name, name, limits)
            for identity, name in verify.DOCUMENTS.items() if identity != "provenance"
        }
        (output / "provenance.intoto.json").write_bytes(common.canonical(
            common.provenance(self.release, archive, documents)
        ))
        documents["provenance"] = common.descriptor(
            output / "provenance.intoto.json", "provenance.intoto.json", limits
        )
        manifest = {
            "schema_version": 1, "kind": "jyppx-sdk-release", "release": self.release,
            "archive": archive, "documents": documents, "evidence_artifacts": artifacts,
            "ga": {"status": "pending", "blockers": common.ga_blockers(
                self.release, self.dependencies, blockers
            )},
        }
        (output / "release-manifest.json").write_bytes(common.canonical(manifest))
        files = [archive, *documents.values(), *artifacts, common.descriptor(
            output / "release-manifest.json", "release-manifest.json", limits
        )]
        (output / "signing-payload.json").write_bytes(common.canonical(common.signing_payload(files)))

    def test_candidate_round_trip_unpacks_and_keeps_signature_pending(self) -> None:
        self.assertEqual(self.candidate()["ga_blockers"], ["signature.pending"])
        self.assertEqual(self.load_manifest()["archive"]["path"],
                         "liuding-forge-1.0.0-linux-x86_64-cpu.zip")
        sbom = json.loads((self.output / "sbom.cdx.json").read_bytes())
        self.assertEqual(sbom["metadata"]["component"]["name"], "jyppx")
        self.assertIn({"name": "liuding-forge:display-name-en", "value": "Liuding Forge"},
                      sbom["metadata"]["properties"])
        result = self.inspect()
        self.assertTrue(result["candidate_verified"])
        self.assertFalse(result["ga_ready"])
        self.assertEqual(result["unpacked_files"], 3)
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

    def test_coherent_source_relabel_cannot_reuse_built_prefix(self) -> None:
        for index, (built_dirty, commit, dirty) in enumerate((
            (False, "2" * 40, False), (False, "1" * 40, True), (True, "1" * 40, False),
        )):
            with self.subTest(index=index):
                self.release["source"] = {"commit": "1" * 40, "dirty": built_dirty}
                self.simulate_rebuild()
                built_stamp = self.build_metadata.read_bytes()
                self.release["source"] = {"commit": commit, "dirty": dirty}
                self.bind_evidence_to_prefix()
                self.write_inputs()
                self.assertEqual(self.build_metadata.read_bytes(), built_stamp)
                with self.assertRaisesRegex(common.ReleaseError, "build provenance source"):
                    self.candidate(self.root / f"relabel-source-{index}")

    def test_coherent_lock_relabel_cannot_reuse_built_prefix(self) -> None:
        original = copy.deepcopy(self.dependencies)
        built_stamp = self.build_metadata.read_bytes()
        for raw_bytes_only in (False, True):
            with self.subTest(raw_bytes_only=raw_bytes_only):
                self.dependencies = copy.deepcopy(original)
                if not raw_bytes_only:
                    self.dependencies["dependencies"][0]["version"] = "1.2.4"
                self.bind_evidence_to_prefix()
                self.write_inputs()
                if raw_bytes_only:
                    self.lock.write_bytes(b" \n" + self.lock.read_bytes())
                    self.evidence["dependency_lock_sha256"] = hashlib.sha256(self.lock.read_bytes()).hexdigest()
                    self.index.write_bytes(common.canonical(self.evidence))
                self.assertEqual(self.build_metadata.read_bytes(), built_stamp)
                with self.assertRaisesRegex(common.ReleaseError, "build provenance dependency lock"):
                    self.candidate(self.root / f"relabel-lock-{raw_bytes_only}")

    def test_independent_forgery_fixture_has_coherent_outer_bindings(self) -> None:
        self.unchecked_candidate(self.output)
        self.assertTrue(self.inspect()["candidate_verified"])

    def test_independent_verifier_rejects_coherent_relabels_without_producer(self) -> None:
        original_release = copy.deepcopy(self.release)
        original_dependencies = copy.deepcopy(self.dependencies)
        original_stamp = self.build_metadata.read_bytes()
        for attack in ("commit", "dirty", "dirty-cleared", "lock", "lock-bytes"):
            with self.subTest(attack=attack):
                self.release = copy.deepcopy(original_release)
                self.dependencies = copy.deepcopy(original_dependencies)
                self.build_metadata.write_bytes(original_stamp)
                if attack == "dirty-cleared":
                    self.release["source"]["dirty"] = True
                    self.simulate_rebuild()
                installed_stamp = self.build_metadata.read_bytes()
                if attack == "commit":
                    self.release["source"]["commit"] = "a" * 64
                elif attack == "dirty":
                    self.release["source"]["dirty"] = True
                elif attack == "dirty-cleared":
                    self.release["source"]["dirty"] = False
                elif attack == "lock":
                    self.dependencies["dependencies"][0]["version"] = "1.2.4"
                self.bind_evidence_to_prefix()
                self.write_inputs()
                if attack == "lock-bytes":
                    self.lock.write_bytes(self.lock.read_bytes() + b" \n")
                    self.evidence["dependency_lock_sha256"] = hashlib.sha256(self.lock.read_bytes()).hexdigest()
                    self.index.write_bytes(common.canonical(self.evidence))
                self.assertEqual(self.build_metadata.read_bytes(), installed_stamp)
                output = self.root / f"forged-{attack}"
                self.unchecked_candidate(output)
                with self.assertRaisesRegex(common.ReleaseError, "build provenance (source|dependency lock)"):
                    self.inspect(output)

    def malformed_build_provenance(self) -> list[tuple[str, bytes | None]]:
        """Malformed installed metadata is hidden behind fully coherent outer hashes."""
        metadata = json.loads(self.build_metadata.read_bytes())
        values = [
            ("schema-bool", {**metadata, "schema_version": True}),
            ("schema-unsupported", {**metadata, "schema_version": 2}),
            ("schema-text", {**metadata, "schema_version": "1"}),
            ("version-mismatch", {**metadata, "project_version": "1.0.1"}),
            ("version-prerelease", {**metadata, "project_version": "1.0.0-rc.1"}),
            ("version-type", {**metadata, "project_version": ["1.0.0"]}),
            ("commit-unknown", {**metadata, "git_commit": "unknown"}),
            ("commit-short", {**metadata, "git_commit": "1" * 39}),
            ("commit-uppercase", {**metadata, "git_commit": "A" * 40}),
            ("commit-double-dirty", {**metadata, "git_commit": "1" * 40 + "+dirty+dirty"}),
            ("commit-type", {**metadata, "git_commit": False}),
            ("lock-unlocked", {**metadata, "dependency_lock_sha256": "unlocked"}),
            ("lock-uppercase", {**metadata, "dependency_lock_sha256": "A" * 64}),
            ("lock-type", {**metadata, "dependency_lock_sha256": True}),
            ("configuration-empty", {**metadata, "configuration": ""}),
            ("configuration-blank", {**metadata, "configuration": " "}),
            ("configuration-type", {**metadata, "configuration": False}),
            ("configuration-control", {**metadata, "configuration": "Release\x00"}),
            ("configuration-limit", {**metadata, "configuration": "x" * 129}),
            ("unknown-field", {**metadata, "unknown": True}),
            ("missing-field", {key: value for key, value in metadata.items() if key != "git_commit"}),
        ]
        original = common.canonical(metadata)
        return [(name, common.canonical(value)) for name, value in values] + [
            ("duplicate", b'{"schema_version":1,' + original[1:]),
            ("not-json", b"not JSON"), ("utf8", b"\xff"), ("absent", None),
        ]

    def test_missing_or_malformed_installed_provenance_is_rejected_by_producer(self) -> None:
        for name, data in self.malformed_build_provenance():
            with self.subTest(name=name):
                if data is None:
                    self.build_metadata.unlink()
                else:
                    self.build_metadata.write_bytes(data)
                self.bind_evidence_to_prefix()
                self.write_inputs()
                with self.assertRaises(common.ReleaseError):
                    self.candidate(self.root / f"invalid-{name}")

    def test_independent_verifier_rejects_missing_or_malformed_installed_provenance(self) -> None:
        for name, data in self.malformed_build_provenance():
            with self.subTest(name=name):
                if data is None:
                    self.build_metadata.unlink()
                else:
                    self.build_metadata.write_bytes(data)
                self.bind_evidence_to_prefix()
                self.write_inputs()
                output = self.root / f"forged-invalid-{name}"
                self.unchecked_candidate(output)
                with self.assertRaises(common.ReleaseError):
                    self.inspect(output)

    def test_built_identity_accepts_rc_base_version_and_full_64_digit_commits(self) -> None:
        for index, (commit, dirty, version) in enumerate((
            ("1" * 40, False, "1.0.0-rc.1"),
            ("a" * 64, False, "1.0.0"),
            ("b" * 64, True, "1.0.0-rc.1"),
        )):
            with self.subTest(index=index):
                self.release["source"] = {"commit": commit, "dirty": dirty}
                self.release["version"] = version
                self.simulate_rebuild()
                self.write_inputs()
                output = self.root / f"built-identity-{index}"
                self.candidate(output)
                self.assertTrue(self.inspect(output)["candidate_verified"])

    def test_checked_in_draft_dependency_manifest_never_becomes_audit_pass(self) -> None:
        self.dependencies = json.loads((ROOT / "dependency-manifest.json").read_bytes())
        self.evidence["dependency_audit"] = {
            "status": "pending", "audited_dependency_ids": [],
            "blocking_licenses": [], "blocking_vulnerabilities": [], "artifacts": [],
        }
        self.simulate_rebuild()
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
        self.simulate_rebuild()
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
        self.simulate_rebuild()
        self.write_inputs()
        self.candidate()
        result = self.inspect()
        self.assertIn("dependency.fixture.license-review-pending", result["ga_blockers"])
        self.assertIn("dependency-audit.blocking-licenses", result["ga_blockers"])
        self.assertIn("dependency-audit.blocking-vulnerabilities", result["ga_blockers"])

    def test_curl_license_acceptance_keeps_notice_and_security_gates(self) -> None:
        self.dependencies["dependencies"][0]["license"]["spdx"] = ["curl"]
        self.evidence["dependency_audit"]["status"] = "pending"
        self.evidence["dependency_audit"]["blocking_vulnerabilities"] = [
            "synthetic unresolved advisory: not a release audit"
        ]
        self.simulate_rebuild()
        self.write_inputs()
        self.candidate()
        result = self.inspect()
        self.assertFalse(result["ga_ready"])
        self.assertNotIn("dependency.fixture.license-review-pending", result["ga_blockers"])
        self.assertIn("dependency-audit.pending", result["ga_blockers"])
        self.assertIn("dependency-audit.blocking-vulnerabilities", result["ga_blockers"])
        self.assertIn("signature.pending", result["ga_blockers"])
        sbom = json.loads((self.output / "sbom.cdx.json").read_bytes())
        self.assertEqual(sbom["components"][0]["licenses"], [{"license": {"id": "curl"}}])

        self.dependencies["dependencies"][0]["license"]["third_party_notice_audit"] = "pending"
        self.evidence["dependency_audit"]["status"] = "passed"
        self.simulate_rebuild()
        self.write_inputs()
        other = self.root / "curl-notice-pending"
        self.candidate(other)
        pending = self.inspect(other)
        self.assertFalse(pending["ga_ready"])
        self.assertIn("dependency.fixture.notice-pending", pending["ga_blockers"])
        self.assertIn("dependency-audit.blocking-vulnerabilities", pending["ga_blockers"])

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
            verify.verify_archive(archive, inventory, self.release,
                                  hashlib.sha256(self.lock.read_bytes()).hexdigest(), limits)
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
                    verify.verify_archive(archive, inventory, self.release,
                                          hashlib.sha256(self.lock.read_bytes()).hexdigest(), common.Limits())
                self.assertFalse((self.root / "outside").exists())

    def test_independent_extractor_checks_hashes_truncation_hidden_data_and_local_names(self) -> None:
        self.candidate()
        manifest = self.load_manifest()
        archive = self.output / manifest["archive"]["path"]
        inventory = common.prefix_inventory(self.prefix, common.Limits())
        wrong = copy.deepcopy(inventory)
        wrong["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(common.ReleaseError, "size/hash"):
            verify.verify_archive(archive, wrong, self.release,
                                  hashlib.sha256(self.lock.read_bytes()).hexdigest(), common.Limits())
        original = archive.read_bytes()
        name_size = struct.unpack_from("<H", original, 26)[0]
        changed_name = bytearray(original)
        changed_name[30 + name_size - 1] ^= 1
        for data in (original[:-1], original + b"hidden", b"hidden" + original, bytes(changed_name)):
            hostile = self.root / "hostile.zip"
            hostile.write_bytes(data)
            with self.assertRaises((common.ReleaseError, zipfile.BadZipFile)):
                verify.verify_archive(hostile, inventory, self.release,
                                      hashlib.sha256(self.lock.read_bytes()).hexdigest(), common.Limits())

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
        self.simulate_rebuild()
        self.write_inputs()
        candidate = self.root / "draft-candidate"
        self.candidate(candidate)
        invoke("dgst", "-sha256", "-sign", str(private), "-out", str(signature), str(candidate / "signing-payload.json"))
        self.assertIn("dependency-manifest.draft", self.inspect(candidate, **trust)["ga_blockers"])


if __name__ == "__main__":
    unittest.main()
