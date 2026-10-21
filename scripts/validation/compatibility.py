#!/usr/bin/env python3
"""Validate and query the repository's LibTorch compatibility source.

The JSON file is deliberately small and dependency-free so CI, local builds,
release tooling, and downstream audit scripts can consume the same values.
This tool does not infer support from a version range: every supported claim
must be backed by an explicit verified lane in compatibility.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import string
import sys
from typing import Any


class CompatibilityError(ValueError):
    """Raised when compatibility.json violates the repository contract."""


LANE_STATUSES = {"candidate", "development-baseline", "supported", "unsupported"}
VERIFIED_LANE_STRING_FIELDS = (
    "os",
    "architecture",
    "compiler",
    "accelerator",
    "build_type",
    "tests",
    "instrumentation",
)


def load_document(path: pathlib.Path) -> dict[str, Any]:
    """Load one JSON object and reject malformed top-level values."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompatibilityError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise CompatibilityError("compatibility source must be a JSON object")
    return document


def require_mapping(parent: dict[str, Any], key: str, location: str) -> dict[str, Any]:
    """Return a required object field with a useful validation location."""
    value = parent.get(key)
    if not isinstance(value, dict):
        raise CompatibilityError(f"{location}.{key} must be an object")
    return value


def require_string(parent: dict[str, Any], key: str, location: str) -> str:
    """Return a required non-empty string field."""
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise CompatibilityError(f"{location}.{key} must be a non-empty string")
    return value


def validate_commit(commit: str, location: str) -> None:
    """Require a seven-to-forty-character hexadecimal Git commit identity."""
    if not 7 <= len(commit) <= 40 or any(
        character not in string.hexdigits for character in commit
    ):
        raise CompatibilityError(
            f"{location}.commit must be a 7-40 character hexadecimal Git commit"
        )


def validate_quality_evidence(evidence: dict[str, Any], location: str) -> None:
    """Require auditable, non-empty string evidence for a verified lane."""
    if not evidence:
        raise CompatibilityError(f"{location}.quality_evidence must not be empty")
    for key, value in evidence.items():
        if not isinstance(key, str) or not key:
            raise CompatibilityError(
                f"{location}.quality_evidence keys must be non-empty strings"
            )
        if not isinstance(value, str) or not value:
            raise CompatibilityError(
                f"{location}.quality_evidence.{key} must be a non-empty string"
            )


def validate(document: dict[str, Any]) -> None:
    """Validate schema, lane references, and the no-implicit-support policy."""
    if document.get("schema_version") != 1:
        raise CompatibilityError("schema_version must be 1")

    project = require_mapping(document, "project", "root")
    require_string(project, "version", "project")
    if project.get("cxx_standard") != 20:
        raise CompatibilityError("project.cxx_standard must be 20")
    require_string(project, "cmake_minimum", "project")
    require_string(project, "dependency_manifest", "project")

    libtorch = require_mapping(document, "libtorch", "root")
    lane_names = ("primary", "compatibility_candidate")
    verified_lane_ids: set[str] = set()
    for lane_name in lane_names:
        lane = require_mapping(libtorch, lane_name, "libtorch")
        require_string(lane, "version", f"libtorch.{lane_name}")
        require_string(lane, "distribution", f"libtorch.{lane_name}")
        status = require_string(lane, "status", f"libtorch.{lane_name}")
        if status not in LANE_STATUSES:
            raise CompatibilityError(
                f"libtorch.{lane_name}.status must be one of "
                f"{', '.join(sorted(LANE_STATUSES))}"
            )
        verified_lanes = lane.get("verified_lanes")
        if not isinstance(verified_lanes, list):
            raise CompatibilityError(
                f"libtorch.{lane_name}.verified_lanes must be an array"
            )
        for index, verified in enumerate(verified_lanes):
            location = f"libtorch.{lane_name}.verified_lanes[{index}]"
            if not isinstance(verified, dict):
                raise CompatibilityError(f"{location} must be an object")
            lane_id = require_string(verified, "id", location)
            if lane_id in verified_lane_ids:
                raise CompatibilityError(f"duplicate verified lane id: {lane_id}")
            verified_lane_ids.add(lane_id)
            for key in VERIFIED_LANE_STRING_FIELDS:
                require_string(verified, key, location)
            validate_quality_evidence(
                require_mapping(verified, "quality_evidence", location), location
            )
            validate_commit(require_string(verified, "commit", location), location)
        if status == "supported" and not verified_lanes:
            raise CompatibilityError(
                f"libtorch.{lane_name} cannot be supported without verified lanes"
            )

    ci = require_mapping(document, "ci", "root")
    candidate_lane = require_string(ci, "candidate_lane", "ci")
    if candidate_lane not in set(lane_names):
        raise CompatibilityError(f"ci.candidate_lane references unknown lane {candidate_lane}")
    for key in ("operating_systems", "build_types", "required_evidence"):
        values = ci.get(key)
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value for value in values
        ):
            raise CompatibilityError(f"ci.{key} must be a non-empty string array")

    policy = require_mapping(document, "policy", "root")
    for key in (
        "support_requires_green_lane",
        "minor_versions_are_not_implicitly_compatible",
        "unlisted_combinations_are_unsupported",
        "cuda_requires_exact_libtorch_and_driver_evidence",
    ):
        if policy.get(key) is not True:
            raise CompatibilityError(f"policy.{key} must be true")


def main() -> int:
    """Run validation or print one exact lane value for CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2] / "compatibility.json",
        help="compatibility source to inspect",
    )
    parser.add_argument("--validate", action="store_true", help="validate the source")
    parser.add_argument(
        "--version",
        metavar="LANE",
        help="print the exact LibTorch version for a named lane",
    )
    args = parser.parse_args()
    try:
        document = load_document(args.file)
        validate(document)
        if args.version:
            lane = require_mapping(
                require_mapping(document, "libtorch", "root"),
                args.version,
                "libtorch",
            )
            print(require_string(lane, "version", f"libtorch.{args.version}"))
        elif args.validate:
            print(f"Compatibility source passed: {args.file}")
        else:
            parser.error("one of --validate or --version is required")
    except CompatibilityError as error:
        print(f"compatibility validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
