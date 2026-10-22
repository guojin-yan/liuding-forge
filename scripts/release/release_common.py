#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared, bounded data contracts for offline SDK release preparation.

No dependency download, build, vulnerability scan, signing, or execution of
installed binaries happens here. Evidence is supplied by the publisher.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO
import uuid


class ReleaseError(ValueError):
    """An unsafe input, inconsistent declaration, or resource limit."""


@dataclasses.dataclass(frozen=True)
class Limits:
    """Finite per-operation bounds; callers may select smaller bounds."""

    max_files: int = 100_000
    max_file_bytes: int = 4 * 1024**3
    max_total_bytes: int = 32 * 1024**3
    max_metadata_bytes: int = 32 * 1024**2
    max_json_nodes: int = 2_000_000
    max_json_depth: int = 24
    max_text_bytes: int = 8192

    def validate(self) -> None:
        """Reject booleans, zero/negative bounds, and relaxed hard caps."""
        caps = Limits()
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or not 0 < value <= getattr(caps, field.name):
                fail(f"limits.{field.name} is outside its hard bound")


CHECK_IDS = frozenset(
    {
        "windows",
        "ubuntu-cuda",
        "compatibility-matrix",
        "install-consumer",
        "sanitizers",
        "fuzz",
        "static-analysis",
        "coverage-risk-review",
        "performance-memory-build-size",
        "documentation-api-examples",
        "upgrade-rollback",
        "release-blockers",
        "offline-rebuild",
    }
)
STATES = {"passed", "pending", "failed"}
# Conservative release-policy allowlist, independent of a scan's declaration.
# Extending it requires explicit license-policy review, not a manifest spelling.
RELEASE_LICENSES = frozenset({
    "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "MIT", "ISC", "Zlib",
    "BSL-1.0", "CC0-1.0", "Unlicense", "blessing",
})
SEMVER = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
)
HASH = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
COMPONENT = re.compile(r"[A-Za-z0-9_+.-]+")
RESERVED = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}
CHUNK = 1024 * 1024


def fail(message: str) -> None:
    """Raise one predictable input error without echoing payload content."""
    raise ReleaseError(message)


def exact(value: Any, keys: set[str], where: str) -> dict[str, Any]:
    """Require an exact object schema, including rejection of unknown keys."""
    if not isinstance(value, dict) or set(value) != keys:
        fail(f"{where}: missing or unknown fields")
    return value


def text(value: Any, where: str, maximum: int = 8192) -> str:
    """Require nonempty bounded UTF-8 text without control characters."""
    if not isinstance(value, str) or not value:
        fail(f"{where}: expected nonempty text")
    try:
        if len(value.encode("utf-8")) > maximum:
            fail(f"{where}: text limit exceeded")
    except UnicodeError as error:
        raise ReleaseError(f"{where}: malformed UTF-8") from error
    if any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value):
        fail(f"{where}: control character")
    return value


def number(value: Any, maximum: int, where: str) -> int:
    """Require a nonnegative integer, explicitly excluding JSON booleans."""
    if type(value) is not int or not 0 <= value <= maximum:
        fail(f"{where}: integer outside bound")
    return value


def digest(value: Any, where: str) -> str:
    """Require an exact lowercase SHA-256 digest."""
    if not isinstance(value, str) or HASH.fullmatch(value) is None:
        fail(f"{where}: expected lowercase SHA-256")
    return value


def relative_name(value: Any) -> str:
    """Reject nonportable names, traversal, drives, streams, and device names."""
    name = text(value, "path", 1024)
    parts = name.split("/")
    if any(
        not part
        or len(part) > 255
        or COMPONENT.fullmatch(part) is None
        or part in {".", ".."}
        or part.endswith(".")
        or part.split(".", 1)[0].casefold() in RESERVED
        for part in parts
    ):
        fail("path: unsafe or nonportable relative name")
    return name


def check_names(names: list[str], limits: Limits) -> None:
    """Reject duplicates, portable aliases, and file/directory conflicts."""
    if not names or len(names) > limits.max_files:
        fail("file count outside bound")
    files: set[str] = set()
    directories: set[str] = set()
    for name in names:
        normalized = relative_name(name).casefold()
        if normalized in files or normalized in directories:
            fail("duplicate, case alias, or file/directory collision")
        parts = normalized.split("/")
        for index in range(1, len(parts)):
            parent = "/".join(parts[:index])
            if parent in files:
                fail("file/directory collision")
            directories.add(parent)
        files.add(normalized)


def canonical(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes, including the signed final newline."""
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ReleaseError("cannot encode canonical metadata") from error


def json_document(data: bytes, limits: Limits, *, canonical_only: bool = False) -> Any:
    """Parse strict JSON after a string-aware, bounded-depth preflight."""
    if len(data) > limits.max_metadata_bytes:
        fail("metadata byte limit exceeded")
    depth = 0
    quoted = escaped = token = False
    preflight_nodes = 0
    for byte in data:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
                preflight_nodes += 1
        elif byte == 34:
            quoted = True
            token = False
        elif byte in (91, 123):
            depth += 1
            token = False
            preflight_nodes += 1
            if depth > limits.max_json_depth:
                fail("JSON depth limit exceeded")
        elif byte in (93, 125):
            depth -= 1
            token = False
        elif byte in (9, 10, 13, 32, 44, 58):
            token = False
        elif not token:
            token = True
            preflight_nodes += 1
        if preflight_nodes > limits.max_json_nodes:
            fail("JSON node limit exceeded before parse")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                fail("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(_: str) -> None:
        fail("nonfinite JSON number")

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=invalid_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ReleaseError("malformed or ambiguous JSON") from error
    nodes = 0
    pending = [value]
    while pending:
        node = pending.pop()
        nodes += 1
        if nodes > limits.max_json_nodes:
            fail("JSON node limit exceeded")
        if isinstance(node, str):
            try:
                if len(node.encode("utf-8")) > limits.max_text_bytes:
                    fail("JSON string limit exceeded")
            except UnicodeError as error:
                raise ReleaseError("malformed Unicode in JSON") from error
        elif isinstance(node, dict):
            pending.extend(node)
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    if canonical_only and canonical(value) != data:
        fail("metadata is not canonical JSON")
    return value


def no_links(path: Path) -> Path:
    """Check the full existing path, including Windows junction ancestors."""
    path = Path(os.path.abspath(path))
    for ancestor in [*reversed(path.parents), path]:
        try:
            info = ancestor.lstat()
        except OSError as error:
            raise ReleaseError("input path is missing or unreadable") from error
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            fail("links and reparse points are forbidden")
    return path


def regular_info(path: Path) -> os.stat_result:
    """Require an unaliased regular file before opening it."""
    no_links(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        fail("expected an unaliased regular file")
    return info


def stream_file(path: Path, maximum: int, output: BinaryIO | None = None) -> dict[str, Any]:
    """Hash/copy a bounded file and detect replacement or concurrent mutation."""
    initial = regular_info(path)
    if initial.st_size > maximum:
        fail("file byte limit exceeded")
    sha = hashlib.sha256()
    count = 0
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as source:
        opened = os.fstat(source.fileno())
        if (opened.st_dev, opened.st_ino) != (initial.st_dev, initial.st_ino):
            fail("input changed while opening")
        while True:
            chunk = source.read(min(CHUNK, maximum - count + 1))
            if not chunk:
                break
            count += len(chunk)
            if count > maximum:
                fail("file grew beyond byte limit")
            sha.update(chunk)
            if output is not None:
                output.write(chunk)
        final = os.fstat(source.fileno())
    after = regular_info(path)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    # Windows lstat and CRT fstat can expose different st_ctime meanings (birth
    # versus change/last-write time, depending on Python/CRT version). Compare
    # ctime only within the same API; identity/length/mtime/link count must still
    # agree across the named path and the actual open handle.
    cross_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_nlink")
    if count != initial.st_size or any(
        getattr(before, key) != getattr(later, key)
        for before, later, fields in ((initial, after, stable_fields),
                                      (opened, final, stable_fields),
                                      (initial, opened, cross_fields),
                                      (after, final, cross_fields))
        for key in fields
    ):
        fail("input changed during read")
    return {"size": count, "sha256": sha.hexdigest()}


def read_file(path: Path, maximum: int) -> bytes:
    """Read a bounded, stable file without following links."""
    import io

    buffer = io.BytesIO()
    stream_file(path, maximum, buffer)
    return buffer.getvalue()


def descriptor(path: Path, name: str, limits: Limits) -> dict[str, Any]:
    """Describe one release file without allowing path aliases."""
    return {"path": relative_name(name), **stream_file(path, limits.max_total_bytes)}


def validate_descriptor(value: Any, limits: Limits) -> dict[str, Any]:
    """Validate an exact size/hash inventory record."""
    exact(value, {"path", "size", "sha256"}, "file descriptor")
    relative_name(value["path"])
    number(value["size"], limits.max_total_bytes, "file.size")
    digest(value["sha256"], "file.sha256")
    return value


def walk_files(root: Path, limits: Limits) -> list[tuple[str, Path]]:
    """Enumerate regular files with bounded directory traversal."""
    root = no_links(root)
    if not root.is_dir():
        fail("expected an input directory")
    result: list[tuple[str, Path]] = []
    visited = 0
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        for name in subdirectories:
            path = no_links(Path(directory) / name)
            relative_name(path.relative_to(root).as_posix())
            visited += 1
            if visited > limits.max_files:
                fail("directory count limit exceeded")
        for name in filenames:
            path = Path(directory) / name
            relative = relative_name(path.relative_to(root).as_posix())
            regular_info(path)
            result.append((relative, path))
            if len(result) > limits.max_files:
                fail("file count limit exceeded")
    result.sort()
    check_names([name for name, _ in result], limits)
    return result


def prefix_inventory(prefix: Path, limits: Limits) -> dict[str, Any]:
    """Inventory the actual install prefix rather than a build/source tree."""
    limits.validate()
    files: list[dict[str, Any]] = []
    total = 0
    for name, path in walk_files(prefix, limits):
        info = regular_info(path)
        entry = {"path": name, **stream_file(path, limits.max_file_bytes)}
        entry["mode"] = 0o755 if (
            info.st_mode & 0o111 or path.suffix.lower() in {".exe", ".com", ".cmd", ".bat"}
        ) else 0o644
        total += entry["size"]
        if total > limits.max_total_bytes:
            fail("install prefix total byte limit exceeded")
        files.append(entry)
    return {"schema_version": 1, "files": files, "total_bytes": total}


def validate_installed_version(prefix: Path, release: dict[str, Any], limits: Limits) -> None:
    """Prevent relabeling an installed 0.x SDK as a 1.x release without executing it."""
    data = read_file(prefix / "include/jyppx/foundation/build_config.hpp", limits.max_metadata_bytes)
    try:
        header = data.decode("utf-8")
    except UnicodeError as error:
        raise ReleaseError("installed version header is not UTF-8") from error
    version = SEMVER.fullmatch(release["version"])
    assert version is not None  # validate_release is mandatory at the entry point.
    for suffix, expected in (("MAJOR", version[1]), ("MINOR", version[2]), ("PATCH", version[3])):
        found = re.findall(rf"^\s*#\s*define\s+JYPPX_VERSION_{suffix}\s+(\d+)\s*$", header, re.MULTILINE)
        if found != [expected]:
            fail("installed SDK version differs from declared release")
    found = re.findall(r'^\s*#\s*define\s+JYPPX_VERSION\s+"([^"]+)"\s*$', header, re.MULTILINE)
    if found != [".".join(version[index] for index in (1, 2, 3))]:
        fail("installed SDK version string differs from declared release")


def validate_inventory(value: Any, limits: Limits) -> dict[str, Any]:
    """Validate the canonical archive file set and aggregate byte budget."""
    exact(value, {"schema_version", "files", "total_bytes"}, "inventory")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        fail("unsupported inventory schema")
    files = value["files"]
    if not isinstance(files, list) or not files or len(files) > limits.max_files:
        fail("inventory file count outside bound")
    total = 0
    for entry in files:
        exact(entry, {"path", "size", "sha256", "mode"}, "inventory entry")
        validate_descriptor({key: entry[key] for key in ("path", "size", "sha256")}, limits)
        number(entry["size"], limits.max_file_bytes, "installed file.size")
        if type(entry["mode"]) is not int or entry["mode"] not in {0o644, 0o755}:
            fail("unsupported installed file mode")
        total += entry["size"]
        if total > limits.max_total_bytes:
            fail("inventory total byte limit exceeded")
    names = [entry["path"] for entry in files]
    check_names(names, limits)
    if names != sorted(names):
        fail("inventory entries must be sorted")
    if number(value["total_bytes"], limits.max_total_bytes, "inventory.total") != total:
        fail("inventory total disagrees with file sizes")
    return value


def validate_release(value: Any) -> dict[str, Any]:
    """Validate precise version/platform/source assertions for one artifact."""
    exact(value, {"version", "platform", "source", "source_date_epoch"}, "release")
    version = text(value["version"], "release.version", 128)
    match = SEMVER.fullmatch(version)
    if match is None or any(
        part.isdigit() and len(part) > 1 and part.startswith("0")
        for part in (match[4] or "").split(".")
    ):
        fail("release version must be exact SemVer without build metadata")
    platform = relative_name(value["platform"])
    if "/" in platform or len(platform) > 128:
        fail("platform must be one portable component")
    source = exact(value["source"], {"commit", "dirty"}, "release.source")
    if not isinstance(source["commit"], str) or COMMIT.fullmatch(source["commit"]) is None:
        fail("source commit must be a full lowercase Git object ID")
    if type(source["dirty"]) is not bool:
        fail("source dirty state must be explicit boolean")
    number(value["source_date_epoch"], 4_354_819_199, "source_date_epoch")
    return value


def string_list(value: Any, where: str, maximum: int = 10_000) -> list[str]:
    """Validate a bounded duplicate-free array of text."""
    if not isinstance(value, list) or len(value) > maximum:
        fail(f"{where}: array limit exceeded")
    for item in value:
        text(item, where)
    if len(set(value)) != len(value):
        fail(f"{where}: duplicate entry")
    return value


def validate_dependencies(value: Any) -> list[str]:
    """Preserve the checked-in draft manifest's verified/pending semantics."""
    exact(value, {"schema_version", "status", "hash_algorithm", "dependencies"},
          "dependency manifest")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        fail("unsupported dependency schema")
    if value["hash_algorithm"] != "sha256" or value["status"] not in {"draft", "complete"}:
        fail("unsupported dependency hash or status")
    entries = value["dependencies"]
    if not isinstance(entries, list) or not entries or len(entries) > 10_000:
        fail("dependency count outside bound")
    ids: list[str] = []
    for entry in entries:
        exact(entry, {"id", "scope", "version", "source", "integrity", "license",
                      "platforms", "linkage", "support_status"}, "dependency")
        for key in ("id", "scope", "version", "linkage", "support_status"):
            text(entry[key], f"dependency.{key}")
        relative_name(entry["id"])
        ids.append(entry["id"])
        if not string_list(entry["platforms"], "dependency.platforms"):
            fail("dependency platforms cannot be empty")
        source = entry["source"]
        if not isinstance(source, dict):
            fail("dependency.source: missing or unknown fields")
        source_fields = set(source)
        if source_fields in ({"type", "url", "cache_file"},
                             {"type", "url_template", "cache_file"}):
            text(source["type"], "dependency.source.type")
            url = text(source.get("url", source.get("url_template")),
                       "dependency.source.url")
            if not url.startswith("https://"):
                fail("dependency source must use HTTPS")
            if source["cache_file"] is not None:
                relative_name(source["cache_file"])
        elif source_fields == {"type", "url", "revision", "path"}:
            text(source["type"], "dependency.source.type")
            url = text(source["url"], "dependency.source.url")
            if not url.startswith("https://"):
                fail("dependency source must use HTTPS")
            text(source["revision"], "dependency.source.revision")
            relative_name(source["path"])
        else:
            fail("dependency.source: missing or unknown fields")
        integrity_fields = set(entry["integrity"])
        if integrity_fields not in ({"status", "sha256"},
                                    {"status", "sha256", "size_bytes"}):
            fail("dependency.integrity: missing or unknown fields")
        integrity = entry["integrity"]
        if integrity["status"] == "verified":
            digest(integrity["sha256"], "dependency.integrity.sha256")
        elif integrity["status"] != "pending" or integrity["sha256"] is not None:
            fail("dependency integrity must be verified or explicitly pending")
        if "size_bytes" in integrity:
            size_bytes = integrity["size_bytes"]
            if (integrity["status"] != "verified" or type(size_bytes) is not int or
                    not 0 < size_bytes <= 2 * 1024 * 1024 * 1024):
                fail("dependency.integrity.size_bytes must be a bounded verified size")
        license_info = exact(entry["license"], {"spdx", "third_party_notice_audit"},
                             "dependency.license")
        if not string_list(license_info["spdx"], "dependency.license.spdx"):
            fail("dependency license cannot be empty")
        if license_info["third_party_notice_audit"] not in {"recorded", "pending"}:
            fail("dependency notice audit must be recorded or pending")
    if len(set(ids)) != len(ids):
        fail("duplicate dependency identity")
    return ids


def validate_evidence(value: Any, release: dict[str, Any], inventory_sha: str,
                      lock_sha: str, dependency_ids: list[str],
                      limits: Limits) -> tuple[list[str], list[dict[str, Any]]]:
    """Bind actual evidence files to this source, prefix and exact dependency lock."""
    exact(value, {"schema_version", "release", "installation_inventory_sha256",
                  "dependency_lock_sha256", "checks", "dependency_audit"}, "evidence")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        fail("unsupported evidence schema")
    expected = {key: release[key] for key in ("version", "platform", "source")}
    validate_release({**exact(value["release"], {"version", "platform", "source"},
                              "evidence.release"),
                      "source_date_epoch": release["source_date_epoch"]})
    if value["release"] != expected:
        fail("evidence source/version/platform differs from release")
    if digest(value["dependency_lock_sha256"], "evidence lock") != lock_sha:
        fail("evidence dependency lock digest mismatch")
    blockers: list[str] = []
    if value["installation_inventory_sha256"] is None:
        blockers.append("evidence.installation-inventory.pending")
    elif digest(value["installation_inventory_sha256"], "evidence inventory") != inventory_sha:
        fail("evidence installation inventory digest mismatch")
    rows = value["checks"]
    if not isinstance(rows, list) or len(rows) > len(CHECK_IDS):
        fail("evidence check count outside bound")
    seen: set[str] = set()
    artifacts: dict[str, dict[str, Any]] = {}

    def records(items: Any, state: str) -> None:
        if not isinstance(items, list) or len(items) > limits.max_files:
            fail("evidence artifact count outside bound")
        if state == "passed" and not items:
            fail("passed evidence requires actual hashed artifacts")
        for item in items:
            validate_descriptor(item, limits)
            number(item["size"], limits.max_file_bytes, "evidence artifact.size")
            if item["path"] in artifacts and artifacts[item["path"]] != item:
                fail("conflicting evidence artifact descriptor")
            artifacts[item["path"]] = item

    for row in rows:
        exact(row, {"id", "status", "details", "artifacts"}, "evidence check")
        identity = text(row["id"], "check.id")
        if identity not in CHECK_IDS or identity in seen or row["status"] not in STATES:
            fail("unknown/duplicate check or invalid status")
        seen.add(identity)
        text(row["details"], "check.details")
        records(row["artifacts"], row["status"])
        if row["status"] != "passed":
            blockers.append(f"evidence.{identity}.{row['status']}")
    blockers.extend(f"evidence.{identity}.missing" for identity in CHECK_IDS - seen)
    audit = exact(value["dependency_audit"],
                  {"status", "audited_dependency_ids", "blocking_licenses",
                   "blocking_vulnerabilities", "artifacts"}, "dependency audit")
    if audit["status"] not in STATES:
        fail("invalid dependency audit status")
    audited = string_list(audit["audited_dependency_ids"], "audited dependencies")
    if not set(audited) <= set(dependency_ids):
        fail("audit includes unknown dependency")
    license_blocks = string_list(audit["blocking_licenses"], "blocking licenses")
    vulnerability_blocks = string_list(audit["blocking_vulnerabilities"], "blocking vulnerabilities")
    records(audit["artifacts"], audit["status"])
    if audit["status"] != "passed":
        blockers.append(f"dependency-audit.{audit['status']}")
    if set(audited) != set(dependency_ids):
        blockers.append("dependency-audit.inventory-incomplete")
    if license_blocks:
        blockers.append("dependency-audit.blocking-licenses")
    if vulnerability_blocks:
        blockers.append("dependency-audit.blocking-vulnerabilities")
    names = sorted(artifacts)
    if names:
        check_names(names, limits)
    total = sum(item["size"] for item in artifacts.values())
    if total > limits.max_total_bytes:
        fail("evidence artifact total byte limit exceeded")
    return sorted(blockers), [artifacts[name] for name in names]


def ga_blockers(release: dict[str, Any], dependencies: dict[str, Any],
                evidence_blockers: list[str]) -> list[str]:
    """Report incomplete gates without upgrading draft/audit declarations."""
    blockers = [*evidence_blockers, "signature.pending"]
    if release["source"]["dirty"]:
        blockers.append("source.dirty")
    version = SEMVER.fullmatch(release["version"])
    if version is None or int(version[1]) < 1 or version[4] is not None:
        blockers.append("version.not-ga")
    if dependencies["status"] != "complete":
        blockers.append("dependency-manifest.draft")
    for dependency in dependencies["dependencies"]:
        identity = dependency["id"]
        if dependency["integrity"]["status"] != "verified":
            blockers.append(f"dependency.{identity}.integrity-pending")
        if "url" not in dependency["source"]:
            blockers.append(f"dependency.{identity}.source-template")
        if dependency["license"]["third_party_notice_audit"] != "recorded":
            blockers.append(f"dependency.{identity}.notice-pending")
        if any(item not in RELEASE_LICENSES for item in dependency["license"]["spdx"]):
            blockers.append(f"dependency.{identity}.license-review-pending")
    return sorted(set(blockers))


def sbom(release: dict[str, Any], dependencies: dict[str, Any], archive_sha: str) -> dict[str, Any]:
    """Emit CycloneDX 1.5 declared dependencies; never invent CVE scan results."""
    components = []
    for dependency in sorted(dependencies["dependencies"], key=lambda item: item["id"]):
        component: dict[str, Any] = {
            "type": "library", "bom-ref": dependency["id"], "name": dependency["id"],
            "version": dependency["version"],
            "licenses": [{"license": {"id" if identity in RELEASE_LICENSES else "name": identity}}
                         for identity in dependency["license"]["spdx"]],
            "properties": [{"name": f"jyppx:{key}", "value": dependency[key]}
                           for key in ("scope", "linkage", "support_status")]
            + [{"name": "jyppx:integrity-status", "value": dependency["integrity"]["status"]},
               {"name": "jyppx:notice-audit", "value": dependency["license"]["third_party_notice_audit"]}],
        }
        if dependency["integrity"]["status"] == "verified":
            component["hashes"] = [{"alg": "SHA-256", "content": dependency["integrity"]["sha256"]}]
        if "url" in dependency["source"]:
            component["externalReferences"] = [{"type": "distribution", "url": dependency["source"]["url"]}]
        components.append(component)
    return {
        "bomFormat": "CycloneDX", "specVersion": "1.5", "version": 1,
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, 'urn:jyppx:' + archive_sha)}",
        "metadata": {
            "timestamp": timestamp(release["source_date_epoch"]),
            "component": {"type": "framework", "bom-ref": "jyppx-release", "name": "jyppx",
                          "version": release["version"],
                          "hashes": [{"alg": "SHA-256", "content": archive_sha}],
                          "licenses": [{"license": {"id": "Apache-2.0"}}]},
            "properties": [{"name": "jyppx:inventory-kind",
                            "value": "declared lock; resolved/transitive graph requires dependency audit evidence"}],
        },
        "components": components,
    }


def timestamp(epoch: int) -> str:
    """Render caller-supplied SOURCE_DATE_EPOCH independently of wall time."""
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def provenance(release: dict[str, Any], archive: dict[str, Any],
               documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Record packaging as an in-toto/SLSA statement, not a build attestation."""
    parameters = {
        "release": release,
        "installationInventorySha256": documents["inventory"]["sha256"],
        "dependencyLockSha256": documents["dependency_lock"]["sha256"],
        "validationEvidenceSha256": documents["evidence_index"]["sha256"],
        "claim": "packages a supplied install prefix; original build and tests are publisher-supplied evidence",
    }
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": archive["path"], "digest": {"sha256": archive["sha256"]}}],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "urn:jyppx:install-prefix-packaging:v1",
                "externalParameters": parameters, "internalParameters": {},
                "resolvedDependencies": [
                    {"uri": "urn:jyppx:dependency-lock", "digest": {"sha256": documents["dependency_lock"]["sha256"]}},
                    {"uri": "urn:jyppx:installation-inventory", "digest": {"sha256": documents["inventory"]["sha256"]}},
                ],
            },
            "runDetails": {
                "builder": {"id": "urn:jyppx:prepare-release:v1"},
                "metadata": {"invocationId": hashlib.sha256(canonical(parameters)).hexdigest()},
            },
        },
    }


def signing_payload(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind every published byte except this payload itself and external signature."""
    return {"schema_version": 1, "kind": "jyppx-release-signing-payload",
            "files": sorted(files, key=lambda item: item["path"])}
