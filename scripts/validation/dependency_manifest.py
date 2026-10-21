#!/usr/bin/env python3
"""Validate the repository dependency manifest without third-party packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import stat
import sys
from typing import Any
from urllib.parse import urlsplit


class DependencyManifestError(ValueError):
    """Raised when dependency-manifest.json violates its schema."""


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys instead of silently choosing one."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DependencyManifestError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_document(path: pathlib.Path) -> dict[str, Any]:
    """Load one JSON object and report file or syntax errors consistently."""
    try:
        with path.open("rb") as source:
            raw = source.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise DependencyManifestError("dependency document exceeds 4 MiB")
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise DependencyManifestError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise DependencyManifestError("dependency manifest must be a JSON object")
    return document


def require_string(parent: dict[str, Any], key: str, location: str) -> str:
    """Return a required non-empty string field."""
    value = parent.get(key)
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 8192:
        raise DependencyManifestError(f"{location}.{key} must be a non-empty string")
    return value


def require_mapping(parent: dict[str, Any], key: str, location: str) -> dict[str, Any]:
    """Return a required object field."""
    value = parent.get(key)
    if not isinstance(value, dict):
        raise DependencyManifestError(f"{location}.{key} must be an object")
    return value


def https_url(value: Any, location: str) -> str:
    """Require a bounded credential-free HTTPS source URL."""
    if not isinstance(value, str) or not value.startswith("https://") or len(value) > 8192:
        raise DependencyManifestError(f"{location} must be an HTTPS URL")
    try:
        parsed = urlsplit(value)
        if (not parsed.hostname or parsed.username or parsed.password or
                any(ord(character) < 0x21 or ord(character) == 0x7F for character in value)):
            raise ValueError("invalid host, credentials, or control characters")
        parsed.port
    except ValueError as error:
        raise DependencyManifestError(f"{location} must use a credential-free HTTPS URL") from error
    return value


def validate(document: dict[str, Any]) -> None:
    """Validate dependency identity, provenance, integrity, and license fields."""
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise DependencyManifestError("schema_version must be 1")
    if not isinstance(document.get("status"), str) or document["status"] not in {"draft", "complete"}:
        raise DependencyManifestError("status must be draft or complete")
    if document.get("hash_algorithm") != "sha256":
        raise DependencyManifestError("hash_algorithm must be sha256")
    if set(document) != {"schema_version", "status", "hash_algorithm", "dependencies"}:
        raise DependencyManifestError("dependency manifest has unknown or missing fields")

    dependencies = document.get("dependencies")
    if not isinstance(dependencies, list) or not 0 < len(dependencies) <= 10_000:
        raise DependencyManifestError("dependencies must be a non-empty array")

    ids: set[str] = set()
    for index, dependency in enumerate(dependencies):
        location = f"dependencies[{index}]"
        if not isinstance(dependency, dict):
            raise DependencyManifestError(f"{location} must be an object")
        expected = {
            "id", "scope", "version", "source", "integrity", "license",
            "platforms", "linkage", "support_status",
        }
        if set(dependency) != expected:
            raise DependencyManifestError(f"{location} has unknown or missing fields")
        dependency_id = require_string(dependency, "id", location)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", dependency_id):
            raise DependencyManifestError(f"{location}.id must be a portable identifier")
        if dependency_id in ids:
            raise DependencyManifestError(f"duplicate dependency id: {dependency_id}")
        ids.add(dependency_id)
        require_string(dependency, "scope", location)
        require_string(dependency, "version", location)
        source = require_mapping(dependency, "source", location)
        if set(source) != {"type", "url", "cache_file"} and set(source) != {
            "type", "url_template", "cache_file"
        }:
            raise DependencyManifestError(f"{location}.source has unknown or missing fields")
        require_string(source, "type", f"{location}.source")
        source_url = https_url(source.get("url", source.get("url_template")), f"{location}.source")
        if source.get("url_template") is not None and "{" not in source_url:
            raise DependencyManifestError(f"{location}.source.url_template must be templated")
        cache_file = source.get("cache_file")
        if cache_file is not None and (not isinstance(cache_file, str) or not cache_file):
            raise DependencyManifestError(f"{location}.source.cache_file must be null or text")
        if cache_file is not None:
            relative_path(cache_file)

        integrity = require_mapping(dependency, "integrity", location)
        if set(integrity) != {"status", "sha256"}:
            raise DependencyManifestError(f"{location}.integrity has unknown or missing fields")
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
        if set(license_info) != {"spdx", "third_party_notice_audit"}:
            raise DependencyManifestError(f"{location}.license has unknown or missing fields")
        spdx = license_info.get("spdx")
        if not isinstance(spdx, list) or not spdx or not all(
            isinstance(value, str) and value for value in spdx
        ):
            raise DependencyManifestError(f"{location}.license.spdx must be a non-empty array")
        notice_audit = require_string(
            license_info, "third_party_notice_audit", f"{location}.license"
        )
        if notice_audit not in {"recorded", "pending"}:
            raise DependencyManifestError(
                f"{location}.license.third_party_notice_audit must be recorded or pending"
            )

        platforms = dependency.get("platforms")
        if not isinstance(platforms, list) or not platforms or not all(
            isinstance(value, str) and value for value in platforms
        ):
            raise DependencyManifestError(f"{location}.platforms must be a non-empty array")
        if len(set(platforms)) != len(platforms):
            raise DependencyManifestError(f"{location}.platforms must not contain duplicates")
        require_string(dependency, "linkage", location)
        require_string(dependency, "support_status", location)


def relative_path(value: str) -> pathlib.PurePosixPath:
    """Require a portable relative evidence/cache path."""
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise DependencyManifestError("invalid relative evidence path")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or "\\" in value or ":" in value or any(
        part in {"", ".", ".."} or part.endswith((".", " "))
        for part in value.split("/")
    ) or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise DependencyManifestError("unsafe relative evidence path")
    return path


def validate_license_index(path: pathlib.Path, dependencies: dict[str, Any]) -> None:
    """Verify retained license bytes and bind recorded states to the lock."""
    index = load_document(path)
    expected = {
        "schema_version", "audit_date", "hash_algorithm", "scope",
        "dependency_notice_states", "limitations", "files",
    }
    if (set(index) != expected or type(index["schema_version"]) is not int or
            index["schema_version"] != 1 or index["hash_algorithm"] != "sha256"):
        raise DependencyManifestError("unsupported license evidence index")
    require_string(index, "audit_date", "license index")
    require_string(index, "scope", "license index")
    if not isinstance(index["limitations"], list) or not all(
        isinstance(value, str) and value for value in index["limitations"]
    ):
        raise DependencyManifestError("license index limitations must be an array of text")
    states = {
        item["id"]: item["license"]["third_party_notice_audit"]
        for item in dependencies["dependencies"]
    }
    if index["dependency_notice_states"] != states:
        raise DependencyManifestError("license evidence states differ from dependency manifest")
    records = index["files"]
    if not isinstance(records, list) or not 0 < len(records) <= 1024:
        raise DependencyManifestError("license evidence file count exceeds limit")
    seen: set[str] = set()
    covered: set[str] = set()
    total = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "dependency_id", "path", "size_bytes", "sha256", "origin"
        }:
            raise DependencyManifestError("invalid license evidence descriptor")
        if not isinstance(record["dependency_id"], str) or not record["dependency_id"]:
            raise DependencyManifestError("license evidence requires a dependency identity")
        relative = relative_path(record["path"])
        name = relative.as_posix()
        if name.casefold() in seen:
            raise DependencyManifestError("duplicate license evidence path")
        seen.add(name.casefold())
        if type(record["size_bytes"]) is not int or not 0 < record["size_bytes"] <= 2 * 1024 * 1024:
            raise DependencyManifestError("invalid license evidence size")
        if not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise DependencyManifestError("invalid license evidence digest")
        origin = record["origin"]
        if (not isinstance(origin, dict) or not isinstance(origin.get("kind"), str) or
                not isinstance(
                    origin.get("artifact_sha256"), str
                ) or not re.fullmatch(r"[0-9a-f]{64}", origin["artifact_sha256"])):
            raise DependencyManifestError("license evidence requires exact HTTPS artifact provenance")
        https_url(origin.get("url"), "license evidence origin")
        common_fields = {"kind", "url", "artifact_sha256"}
        if origin.get("kind") == "upstream-document":
            required = common_fields
        elif origin.get("kind") in {"archive-entry", "wheel-entry"}:
            required = common_fields | {"entry"}
        elif origin.get("kind") == "archive-excerpt":
            required = common_fields | {"entry", "byte_offset", "byte_length"}
            if (type(origin.get("byte_offset")) is not int or origin["byte_offset"] < 0 or
                    type(origin.get("byte_length")) is not int or
                    origin["byte_length"] != record["size_bytes"]):
                raise DependencyManifestError("invalid license evidence excerpt range")
        else:
            raise DependencyManifestError("unknown license evidence origin")
        if set(origin) != required:
            raise DependencyManifestError("license evidence origin has unknown or missing fields")
        if "entry" in origin:
            relative_path(origin["entry"])
        current = path.parent
        for part in relative.parts:
            current /= part
            named_info = current.lstat()
            if (stat.S_ISLNK(named_info.st_mode) or
                    getattr(named_info, "st_file_attributes", 0) & 0x400):
                raise DependencyManifestError("license evidence links are forbidden")
        info = current.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != record["size_bytes"]:
            raise DependencyManifestError("license evidence must be an unaliased file of declared size")
        with current.open("rb") as source:
            raw = source.read(record["size_bytes"] + 1)
        if len(raw) != record["size_bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise DependencyManifestError("license evidence digest mismatch")
        total += info.st_size
        if total > 32 * 1024 * 1024:
            raise DependencyManifestError("license evidence aggregate exceeds 32 MiB")
        covered.add(record["dependency_id"])
    if any(state == "recorded" and identity not in covered for identity, state in states.items()):
        raise DependencyManifestError("recorded notice audit has no retained license evidence")


def main() -> int:
    """Validate one manifest and print a concise result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2] / "dependency-manifest.json",
        help="dependency manifest to inspect",
    )
    parser.add_argument("--license-index", type=pathlib.Path,
                        help="optional exact license evidence index to verify")
    args = parser.parse_args()
    try:
        document = load_document(args.file)
        validate(document)
        license_index = args.license_index or args.file.parent / "third_party/licenses/manifest.json"
        if args.license_index is not None or license_index.exists():
            validate_license_index(license_index, document)
    except (DependencyManifestError, OSError, ValueError) as error:
        print(f"dependency manifest validation failed: {error}", file=sys.stderr)
        return 1
    print(f"Dependency manifest passed: {args.file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
