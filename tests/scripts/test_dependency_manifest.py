#!/usr/bin/env python3
"""Unit tests for the strict dependency manifest validator."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "validation" / "dependency_manifest.py"
SPEC = importlib.util.spec_from_file_location("dependency_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DependencyManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads((ROOT / "dependency-manifest.json").read_text())

    def test_checked_in_manifest_passes_and_google_license_is_recorded(self) -> None:
        MODULE.validate(self.document)
        MODULE.validate_license_index(ROOT / "third_party/licenses/manifest.json", self.document)
        google = next(item for item in self.document["dependencies"] if item["id"] == "googletest")
        self.assertEqual(google["license"]["third_party_notice_audit"], "recorded")

    def test_unknown_dependency_field_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["dependencies"][0]["unexpected"] = True
        with self.assertRaises(MODULE.DependencyManifestError):
            MODULE.validate(mutated)

    def test_non_https_source_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["dependencies"][0]["source"]["url"] = "file:///dependency.zip"
        with self.assertRaises(MODULE.DependencyManifestError):
            MODULE.validate(mutated)

    def test_pending_digest_and_duplicate_platform_are_rejected(self) -> None:
        pending = copy.deepcopy(self.document)
        candidate = next(
            item for item in pending["dependencies"] if item["id"] == "libtorch-candidate-cpu"
        )
        candidate["integrity"]["sha256"] = "0" * 64
        with self.assertRaises(MODULE.DependencyManifestError):
            MODULE.validate(pending)

        duplicate = copy.deepcopy(self.document)
        duplicate["dependencies"][0]["platforms"].append("windows-x86_64")
        with self.assertRaises(MODULE.DependencyManifestError):
            MODULE.validate(duplicate)

    def test_duplicate_json_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jyppx-manifest-") as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaises(MODULE.DependencyManifestError):
                MODULE.load_document(path)

    def license_fixture(self, directory: str) -> tuple[Path, dict]:
        root = Path(directory)
        raw = b"Synthetic test copyright and license fixture.\n"
        (root / "LICENSE").write_bytes(raw)
        dependencies = {
            "dependencies": [{"id": "fixture", "license": {"third_party_notice_audit": "recorded"}}]
        }
        index = {
            "schema_version": 1, "audit_date": "2026-10-21", "hash_algorithm": "sha256",
            "scope": "synthetic test", "dependency_notice_states": {"fixture": "recorded"},
            "limitations": [],
            "files": [{
                "dependency_id": "fixture", "path": "LICENSE", "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "origin": {"kind": "upstream-document", "url": "https://example.invalid/LICENSE",
                           "artifact_sha256": hashlib.sha256(raw).hexdigest()},
            }],
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(index), encoding="utf-8")
        return path, dependencies

    def test_license_bytes_and_recorded_coverage_are_verified(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jyppx-license-") as directory:
            path, dependencies = self.license_fixture(directory)
            MODULE.validate_license_index(path, dependencies)
            license_path = path.parent / "LICENSE"
            raw = license_path.read_bytes()
            license_path.write_bytes(b"!" + raw[1:])
            with self.assertRaisesRegex(MODULE.DependencyManifestError, "digest"):
                MODULE.validate_license_index(path, dependencies)
            license_path.write_bytes(raw)
            index = json.loads(path.read_text())
            index["files"][0]["dependency_id"] = "another-dependency"
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(MODULE.DependencyManifestError, "no retained"):
                MODULE.validate_license_index(path, dependencies)

    def test_license_paths_links_and_duplicate_aliases_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jyppx-license-") as directory:
            path, dependencies = self.license_fixture(directory)
            original = json.loads(path.read_text())
            index = copy.deepcopy(original)
            index["files"][0]["path"] = "../LICENSE"
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(MODULE.DependencyManifestError, "relative"):
                MODULE.validate_license_index(path, dependencies)
            index = copy.deepcopy(original)
            index["files"].append(copy.deepcopy(index["files"][0]))
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(MODULE.DependencyManifestError, "duplicate"):
                MODULE.validate_license_index(path, dependencies)
            path.write_text(json.dumps(original))
            alias = path.parent / "hardlink-alias"
            os.link(path.parent / "LICENSE", alias)
            with self.assertRaisesRegex(MODULE.DependencyManifestError, "unaliased"):
                MODULE.validate_license_index(path, dependencies)

    def test_boolean_schema_version_is_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["schema_version"] = True
        with self.assertRaises(MODULE.DependencyManifestError):
            MODULE.validate(document)

    def test_license_origin_kind_and_credentials_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jyppx-license-") as directory:
            path, dependencies = self.license_fixture(directory)
            original = json.loads(path.read_text())
            for invalid_kind in ({}, [], True):
                index = copy.deepcopy(original)
                index["files"][0]["origin"]["kind"] = invalid_kind
                path.write_text(json.dumps(index))
                with self.assertRaises(MODULE.DependencyManifestError):
                    MODULE.validate_license_index(path, dependencies)
            index = copy.deepcopy(original)
            index["files"][0]["origin"]["url"] = "https://user:password@example.invalid/LICENSE"
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(MODULE.DependencyManifestError, "credential-free"):
                MODULE.validate_license_index(path, dependencies)


if __name__ == "__main__":
    unittest.main()
