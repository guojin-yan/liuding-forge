#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Bound retained notice declarations without claiming a completed license audit."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from release_common import Limits, check_names, digest, exact, fail, relative_name, text


LICENSE_INDEX = "third_party/licenses/manifest.json"
LICENSE_ROOT = "third_party/licenses/"


def license_artifacts(value: Any, dependencies: dict[str, Any],
                      limits: Limits) -> list[dict[str, Any]]:
    """Validate the retained index and return its complete sorted file inventory.

    Additional optional dependency originals may be retained, but never count
    toward coverage of the exact lock. Pending audit states remain pending.
    Actual file bytes are checked independently by the producer and verifier.
    """
    exact(value, {"schema_version", "audit_date", "hash_algorithm", "scope",
                  "dependency_notice_states", "limitations", "files"}, "license index")
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["hash_algorithm"] != "sha256"):
        fail("unsupported license index schema or hash")
    text(value["audit_date"], "license index audit date")
    text(value["scope"], "license index scope")
    if not isinstance(value["limitations"], list):
        fail("license index limitations must be an array")
    for limitation in value["limitations"]:
        text(limitation, "license index limitation")
    states = {item["id"]: item["license"]["third_party_notice_audit"]
              for item in dependencies["dependencies"]}
    if value["dependency_notice_states"] != states:
        fail("license index states differ from dependency lock")
    records = value["files"]
    if not isinstance(records, list) or not 0 < len(records) <= min(1024, limits.max_files):
        fail("license index file count outside bound")
    artifacts = []
    covered = set()
    total = 0
    for record in records:
        exact(record, {"dependency_id", "path", "size_bytes", "sha256", "origin"},
              "license index file")
        identity = text(record["dependency_id"], "license dependency identity")
        name = LICENSE_ROOT + relative_name(record["path"])
        size = record["size_bytes"]
        if type(size) is not int or not 0 < size <= min(2 * 1024**2, limits.max_file_bytes):
            fail("license original size outside bound")
        digest(record["sha256"], "license original digest")
        origin = record["origin"]
        if not isinstance(origin, dict):
            fail("license original requires source provenance")
        fields = {"kind", "url", "artifact_sha256"}
        kind = origin.get("kind")
        if kind == "upstream-document":
            pass
        elif isinstance(kind, str) and kind in {"archive-entry", "wheel-entry", "vendored-header"}:
            fields.add("entry")
        elif kind == "archive-excerpt":
            fields.update(("entry", "byte_offset", "byte_length"))
            offset = origin.get("byte_offset")
            if (type(offset) is not int or not 0 <= offset < 2**63
                    or type(origin.get("byte_length")) is not int
                    or origin["byte_length"] != size):
                fail("invalid license original excerpt range")
        else:
            fail("unknown license original source kind")
        exact(origin, fields, "license original source")
        url = text(origin["url"], "license original source URL")
        try:
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                    or parsed.password is not None or any(character.isspace() for character in url)):
                fail("license original source requires credential-free HTTPS")
            parsed.port
        except ValueError:
            fail("invalid license original source URL")
        digest(origin["artifact_sha256"], "license original source digest")
        if "entry" in origin:
            # Source excerpt descriptions can contain spaces (for example
            # "stb_image.h license block"). They are never output paths.
            entry = text(origin["entry"], "license original source entry", 1024)
            if (entry.startswith("/") or "\\" in entry or ":" in entry
                    or any(part in {"", ".", ".."} or part.endswith((".", " "))
                           for part in entry.split("/"))):
                fail("unsafe license original source entry")
        total += size
        if total > min(32 * 1024**2, limits.max_total_bytes):
            fail("license originals aggregate byte limit exceeded")
        artifacts.append({"path": name, "size": size, "sha256": record["sha256"]})
        covered.add(identity)
    check_names([LICENSE_INDEX, *[item["path"] for item in artifacts]], limits)
    if any(state == "recorded" and identity not in covered for identity, state in states.items()):
        fail("recorded notice audit has no retained license original")
    return sorted(artifacts, key=lambda item: item["path"])
