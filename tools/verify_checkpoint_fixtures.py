#!/usr/bin/env python3
"""Validate versioned checkpoint fixture provenance, sizes, and SHA-256 digests."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import pathlib
import re
import sys
from typing import Any


class FixtureError(ValueError):
    """Raised when a checkpoint fixture violates the repository contract."""


def require_mapping(
    parent: dict[str, Any], key: str, location: str
) -> dict[str, Any]:
    """Return a required object field with its validation location."""
    value = parent.get(key)
    if not isinstance(value, dict):
        raise FixtureError(f"{location}.{key} must be an object")
    return value


def require_string(parent: dict[str, Any], key: str, location: str) -> str:
    """Return a required non-empty string field."""
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise FixtureError(f"{location}.{key} must be a non-empty string")
    return value


def load_metadata(path: pathlib.Path) -> dict[str, Any]:
    """Load one fixture metadata document."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise FixtureError(f"{path} must contain a JSON object")
    return document


def decode_field(
    metadata_path: pathlib.Path, field: dict[str, Any], index: int
) -> bytes:
    """Read and strictly decode one fixture field without allowing path traversal."""
    location = f"fields[{index}]"
    relative_name = require_string(field, "file", location)
    relative_path = pathlib.Path(relative_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise FixtureError(f"{location}.file must stay inside the fixture directory")
    field_path = (metadata_path.parent / relative_path).resolve()
    if field_path.parent != metadata_path.parent.resolve():
        raise FixtureError(f"{location}.file must be a direct fixture child")
    if field.get("encoding") != "base64":
        raise FixtureError(f"{location}.encoding must be base64")
    try:
        encoded = b"".join(field_path.read_bytes().split())
        return base64.b64decode(encoded, validate=True)
    except (OSError, binascii.Error) as error:
        raise FixtureError(f"cannot decode {field_path}: {error}") from error


def validate_fixture(metadata_path: pathlib.Path) -> None:
    """Validate one canonical checkpoint fixture and all of its field payloads."""
    document = load_metadata(metadata_path)
    if document.get("fixture_schema") != 1:
        raise FixtureError(f"{metadata_path}: fixture_schema must be 1")
    require_string(document, "participant_id", "root")
    schema = document.get("participant_schema")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema <= 0:
        raise FixtureError(f"{metadata_path}: participant_schema must be positive")
    if document.get("restore_level") not in {"exact", "degraded", "weights_only"}:
        raise FixtureError(f"{metadata_path}: restore_level is invalid")
    require_mapping(document, "scenario", "root")

    producer = require_mapping(document, "producer", "root")
    for key in ("jyppx_version", "git_commit", "libtorch_version", "writer"):
        require_string(producer, key, "producer")
    if not re.fullmatch(r"[0-9a-f]{40}", producer["git_commit"]):
        raise FixtureError(f"{metadata_path}: producer.git_commit must be a full SHA")

    fields = document.get("fields")
    if not isinstance(fields, list) or not fields:
        raise FixtureError(f"{metadata_path}: fields must be a non-empty array")
    names: set[str] = set()
    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            raise FixtureError(f"fields[{index}] must be an object")
        name = require_string(field, "name", f"fields[{index}]")
        if name in names:
            raise FixtureError(f"{metadata_path}: duplicate field {name}")
        names.add(name)
        require_string(field, "media_type", f"fields[{index}]")
        if field.get("required") is not True:
            raise FixtureError(f"fields[{index}].required must be true")
        payload = decode_field(metadata_path, field, index)
        expected_size = field.get("decoded_size")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool):
            raise FixtureError(f"fields[{index}].decoded_size must be an integer")
        maximum_size = field.get("max_bytes")
        if not isinstance(maximum_size, int) or isinstance(maximum_size, bool):
            raise FixtureError(f"fields[{index}].max_bytes must be an integer")
        if maximum_size < expected_size:
            raise FixtureError(f"fields[{index}].max_bytes is below decoded_size")
        if len(payload) != expected_size:
            raise FixtureError(
                f"{metadata_path}: {name} size {len(payload)} != {expected_size}"
            )
        expected_digest = require_string(field, "sha256", f"fields[{index}]")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise FixtureError(f"fields[{index}].sha256 must be lowercase SHA-256")
        actual_digest = hashlib.sha256(payload).hexdigest()
        if actual_digest != expected_digest:
            raise FixtureError(
                f"{metadata_path}: {name} SHA-256 {actual_digest} != {expected_digest}"
            )


def main() -> int:
    """Validate every canonical checkpoint fixture below one root directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        metadata_files = sorted(args.root.resolve().glob("*/metadata.json"))
        if not metadata_files:
            raise FixtureError(f"no checkpoint fixtures found below {args.root}")
        for metadata_path in metadata_files:
            validate_fixture(metadata_path)
            print(f"validated checkpoint fixture: {metadata_path.parent.name}")
    except FixtureError as error:
        print(f"checkpoint fixture validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
