#!/usr/bin/env python3
"""Validate the repository dependency manifest without third-party packages."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any


class DependencyManifestError(ValueError):
    """Raised when dependency-manifest.json violates its schema."""


def load_document(path: pathlib.Path) -> dict[str, Any]:
    """Load one JSON object and report file or syntax errors consistently."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DependencyManifestError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise DependencyManifestError("dependency manifest must be a JSON object")
    return document


def require_string(parent: dict[str, Any], key: str, location: str) -> str:
    """Return a required non-empty string field."""
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise DependencyManifestError(f"{location}.{key} must be a non-empty string")
    return value


def require_mapping(parent: dict[str, Any], key: str, location: str) -> dict[str, Any]:
    """Return a required object field."""
    value = parent.get(key)
    if not isinstance(value, dict):
        raise DependencyManifestError(f"{location}.{key} must be an object")
    return value


def validate(document: dict[str, Any]) -> None:
    """Validate dependency identity, provenance, integrity, and license fields."""
    if document.get("schema_version") != 1:
        raise DependencyManifestError("schema_version must be 1")
    if document.get("status") not in {"draft", "complete"}:
        raise DependencyManifestError("status must be draft or complete")
    if document.get("hash_algorithm") != "sha256":
        raise DependencyManifestError("hash_algorithm must be sha256")

    dependencies = document.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise DependencyManifestError("dependencies must be a non-empty array")

    ids: set[str] = set()
    for index, dependency in enumerate(dependencies):
        location = f"dependencies[{index}]"
        if not isinstance(dependency, dict):
            raise DependencyManifestError(f"{location} must be an object")
        dependency_id = require_string(dependency, "id", location)
        if dependency_id in ids:
            raise DependencyManifestError(f"duplicate dependency id: {dependency_id}")
        ids.add(dependency_id)
        require_string(dependency, "scope", location)
        require_string(dependency, "version", location)
        source = require_mapping(dependency, "source", location)
        require_string(source, "type", f"{location}.source")
        if not (source.get("url") or source.get("url_template")):
            raise DependencyManifestError(
                f"{location}.source must define url or url_template"
            )

        integrity = require_mapping(dependency, "integrity", location)
        integrity_status = require_string(integrity, "status", f"{location}.integrity")
        digest = integrity.get("sha256")
        if integrity_status == "verified":
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise DependencyManifestError(
                    f"{location}.integrity.sha256 must be a lowercase SHA-256 digest"
                )
        elif integrity_status == "pending":
            if digest is not None:
                raise DependencyManifestError(
                    f"{location}.integrity.sha256 must be null while pending"
                )
        else:
            raise DependencyManifestError(
                f"{location}.integrity.status must be verified or pending"
            )

        license_info = require_mapping(dependency, "license", location)
        spdx = license_info.get("spdx")
        if not isinstance(spdx, list) or not spdx or not all(
            isinstance(value, str) and value for value in spdx
        ):
            raise DependencyManifestError(f"{location}.license.spdx must be a non-empty array")
        require_string(license_info, "third_party_notice_audit", f"{location}.license")

        platforms = dependency.get("platforms")
        if not isinstance(platforms, list) or not platforms or not all(
            isinstance(value, str) and value for value in platforms
        ):
            raise DependencyManifestError(f"{location}.platforms must be a non-empty array")
        require_string(dependency, "linkage", location)
        require_string(dependency, "support_status", location)


def main() -> int:
    """Validate one manifest and print a concise result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "dependency-manifest.json",
        help="dependency manifest to inspect",
    )
    args = parser.parse_args()
    try:
        document = load_document(args.file)
        validate(document)
    except DependencyManifestError as error:
        print(f"dependency manifest validation failed: {error}", file=sys.stderr)
        return 1
    print(f"Dependency manifest passed: {args.file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
