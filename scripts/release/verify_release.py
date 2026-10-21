#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Independently verify and safely unpack an SDK candidate or signed GA asset.

Signatures are supplied externally. Only a caller-pinned public key and an
explicit mature OpenSSL executable can authenticate a signing payload.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
from typing import Any, BinaryIO
import zipfile

from release_common import (
    CHUNK, Limits, ReleaseError, canonical, check_names, descriptor, digest, exact,
    fail, ga_blockers, json_document, no_links, provenance, read_file,
    relative_name, sbom, signing_payload, stream_file, validate_dependencies,
    validate_descriptor, validate_evidence, validate_inventory, validate_release,
    validate_installed_version, walk_files,
)

DOCUMENTS = {
    "inventory": "inventory.json", "sbom": "sbom.cdx.json",
    "provenance": "provenance.intoto.json", "dependency_lock": "dependency-manifest.json",
    "evidence_index": "evidence-index.json", "license": "LICENSE",
    "notices": "THIRD_PARTY_NOTICES.md",
}


def directory_bounds(source: BinaryIO, size: int, limits: Limits) -> tuple[int, int]:
    """Bound the ZIP central directory before ZipFile allocates its entry list.

    Canonical assets have no archive comment, split disks, prepended data, or
    trailing bytes. The standard ZIP64 end records are supported explicitly.
    """
    if size < 22:
        fail("truncated ZIP archive")
    source.seek(size - 22)
    fields = struct.unpack("<4s4H2LH", source.read(22))
    signature, disk, central_disk, disk_count, count, length, offset, comment = fields
    if signature != b"PK\x05\x06" or disk or central_disk or disk_count != count or comment:
        fail("noncanonical ZIP end record")
    central_end = size - 22
    if size >= 42:
        source.seek(size - 42)
        locator = source.read(20)
        if locator[:4] == b"PK\x06\x07":
            _, zip_disk, position, disks = struct.unpack("<4sIQI", locator)
            if zip_disk or disks != 1 or position + 56 != size - 42:
                fail("unsupported ZIP64 layout")
            source.seek(position)
            record = source.read(56)
            if len(record) != 56:
                fail("truncated ZIP64 end record")
            (marker, record_size, _made, needed, disk, central_disk,
             disk_count, count, length, offset) = struct.unpack("<4sQHHIIQQQQ", record)
            if marker != b"PK\x06\x06" or record_size != 44 or needed > 45 or disk or central_disk or disk_count != count:
                fail("unsupported ZIP64 end record")
            central_end = position
        elif length == 0xFFFFFFFF or offset == 0xFFFFFFFF:
            fail("ZIP64 locator is missing")
    if not 0 < count <= limits.max_files or length > limits.max_metadata_bytes:
        fail("ZIP central-directory count/byte limit exceeded")
    if offset + length != central_end or offset < 0:
        fail("ZIP central directory has gaps or trailing data")
    return offset, count


def zip64_extra(data: bytes) -> bytes | None:
    """Permit only the standard, bounded ZIP64 size/offset extra field."""
    if not data:
        return None
    if len(data) < 4:
        fail("truncated ZIP extra field")
    kind, length = struct.unpack("<HH", data[:4])
    if kind != 1 or length not in {8, 16, 24} or len(data) != length + 4:
        fail("unknown or duplicate ZIP extra field")
    return data[4:]


def verify_archive(path: Path, inventory: dict[str, Any], release: dict[str, Any],
                   limits: Limits) -> None:
    """Actually extract every regular file into a fresh private directory.

    Data is streamed with exact size/hash/CRC checks. extractall(), stored path
    modes, executable code, and compressed data are never consumed implicitly.
    """
    expected_date = dt.datetime.fromtimestamp(max(release["source_date_epoch"], 315532800), dt.timezone.utc)
    expected_time = (expected_date.year, expected_date.month, expected_date.day,
                     expected_date.hour, expected_date.minute, expected_date.second // 2 * 2)
    with path.open("rb") as source:
        size = os.fstat(source.fileno()).st_size
        if size > limits.max_total_bytes:
            fail("archive byte limit exceeded")
        central_offset, count = directory_bounds(source, size, limits)
        source.seek(0)
        with zipfile.ZipFile(source, "r") as archive:
            entries = archive.infolist()
            if len(entries) != count or len(entries) != len(inventory["files"]):
                fail("ZIP file count differs from inventory")
            cursor = 0
            total = 0
            with tempfile.TemporaryDirectory(prefix="jyppx-verify-") as temporary:
                extraction = Path(temporary)
                for info, expected in zip(entries, inventory["files"]):
                    name = relative_name(info.filename)
                    if name != "payload/" + expected["path"] or info.is_dir():
                        fail("unknown, missing, or reordered ZIP file")
                    if (info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size
                            or info.file_size != expected["size"] or info.flag_bits != 0):
                        fail("encrypted, compressed, or incorrectly sized ZIP entry")
                    if (info.create_system != 3 or info.internal_attr != 0 or info.comment
                            or info.external_attr != (stat.S_IFREG | expected["mode"]) << 16
                            or info.extract_version not in {20, 45} or info.date_time != expected_time):
                        fail("unsafe ZIP file type, mode, timestamp, or version")
                    zip64_extra(info.extra)
                    if info.header_offset != cursor:
                        fail("ZIP has prepended, hidden, or overlapping local data")
                    source.seek(cursor)
                    local = source.read(30)
                    if len(local) != 30:
                        fail("truncated ZIP local header")
                    (marker, needed, flags, method, dos_time, dos_date, crc,
                     compressed, uncompressed, name_size, extra_size) = struct.unpack("<4s5H3L2H", local)
                    if (marker != b"PK\x03\x04" or needed not in {20, 45} or flags or method
                            or name_size > 1024 or extra_size > 28 or crc != info.CRC):
                        fail("noncanonical ZIP local header")
                    wanted_time = expected_time[3] << 11 | expected_time[4] << 5 | expected_time[5] // 2
                    wanted_date = (expected_time[0] - 1980) << 9 | expected_time[1] << 5 | expected_time[2]
                    if dos_time != wanted_time or dos_date != wanted_date:
                        fail("ZIP local timestamp differs from release")
                    if source.read(name_size) != name.encode("ascii"):
                        fail("ZIP local and central filenames differ")
                    extra = zip64_extra(source.read(extra_size))
                    if extra is None:
                        if compressed != info.file_size or uncompressed != info.file_size:
                            fail("ZIP local size mismatch")
                    elif (len(extra) != 16 or compressed != 0xFFFFFFFF or uncompressed != 0xFFFFFFFF
                          or struct.unpack("<QQ", extra) != (info.file_size, info.file_size)):
                        fail("ZIP64 local size mismatch")
                    cursor += 30 + name_size + extra_size + info.file_size
                    if cursor > central_offset:
                        fail("ZIP file overlaps central directory")
                    destination = extraction / expected["path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    sha = hashlib.sha256()
                    written = 0
                    with archive.open(info, "r") as incoming, destination.open("xb") as output:
                        while True:
                            chunk = incoming.read(min(CHUNK, expected["size"] - written + 1))
                            if not chunk:
                                break
                            written += len(chunk)
                            total += len(chunk)
                            if written > expected["size"] or total > limits.max_total_bytes:
                                fail("ZIP extraction byte limit exceeded")
                            sha.update(chunk)
                            output.write(chunk)
                    if written != expected["size"] or sha.hexdigest() != expected["sha256"]:
                        fail("unpacked file size/hash differs from inventory")
                if cursor != central_offset or total != inventory["total_bytes"]:
                    fail("ZIP hidden data or aggregate size mismatch")
                actual_names = [name for name, _ in walk_files(extraction, limits)]
                if actual_names != [entry["path"] for entry in inventory["files"]]:
                    fail("extracted file set differs from inventory")
                validate_installed_version(extraction, release, limits)


def verify_signature(payload: bytes, signature: Path, public_key: Path,
                     key_sha256: str, openssl: Path) -> None:
    """Use a pinned public key with an external verifier; no development key exists."""
    digest(key_sha256, "trusted public key SHA-256")
    if not public_key.is_absolute() or not openssl.is_absolute():
        fail("trusted public key and OpenSSL executable paths must be absolute")
    no_links(openssl)
    if not openssl.is_file():
        fail("OpenSSL verifier executable is missing")
    key_data = read_file(public_key, 1024 * 1024)
    signature_data = read_file(signature, 1024 * 1024)
    if hashlib.sha256(key_data).hexdigest() != key_sha256 or not signature_data:
        fail("trusted public key pin mismatch or empty signature")
    environment = {key: os.environ[key] for key in ("SystemRoot", "WINDIR", "TEMP", "TMP") if key in os.environ}
    environment["OPENSSL_CONF"] = os.devnull
    with tempfile.TemporaryDirectory(prefix="jyppx-signature-") as temporary:
        root = Path(temporary)
        for name, data in (("payload", payload), ("signature", signature_data), ("public-key.pem", key_data)):
            (root / name).write_bytes(data)
        with (root / "verifier.log").open("wb") as log:
            try:
                result = subprocess.run(
                    [str(openssl), "dgst", "-sha256", "-verify", str(root / "public-key.pem"),
                     "-signature", str(root / "signature"), str(root / "payload")],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    cwd=root, env=environment, shell=False, timeout=30, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ReleaseError("external signature verifier unavailable or timed out") from error
        if result.returncode != 0 or (root / "verifier.log").stat().st_size > 65536:
            fail("external OpenSSL signature verification failed")


def verify_release(directory: Path, *, expected_version: str,
                   expected_source_commit: str, expected_platform: str,
                   expected_source_dirty: bool = False,
                   signature: Path | None = None, trusted_public_key: Path | None = None,
                   trusted_public_key_sha256: str | None = None, openssl: Path | None = None,
                   limits: Limits = Limits()) -> dict[str, Any]:
    """Verify all declarations, bytes and extracted files against caller expectations."""
    limits.validate()
    directory = no_links(directory)
    manifest_data = read_file(directory / "release-manifest.json", limits.max_metadata_bytes)
    manifest = json_document(manifest_data, limits, canonical_only=True)
    exact(manifest, {"schema_version", "kind", "release", "archive", "documents",
                     "evidence_artifacts", "ga"}, "release manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["kind"] != "jyppx-sdk-release":
        fail("unsupported release manifest schema or kind")
    release = validate_release(manifest["release"])
    if (release["version"] != expected_version or release["platform"] != expected_platform
            or release["source"]["commit"] != expected_source_commit
            or release["source"]["dirty"] is not expected_source_dirty):
        fail("release identity differs from explicit caller expectation")
    archive = validate_descriptor(manifest["archive"], limits)
    if archive["path"] != f"jyppx-{release['version']}-{release['platform']}.zip":
        fail("archive name differs from release identity")
    documents = exact(manifest["documents"], set(DOCUMENTS), "release documents")
    for identity, name in DOCUMENTS.items():
        if validate_descriptor(documents[identity], limits)["path"] != name:
            fail("release document name differs from canonical contract")
        if documents[identity]["size"] > limits.max_metadata_bytes:
            fail("release document byte limit exceeded")
    evidence_artifacts = manifest["evidence_artifacts"]
    if not isinstance(evidence_artifacts, list) or len(evidence_artifacts) > limits.max_files:
        fail("release evidence artifact count outside bound")
    for artifact in evidence_artifacts:
        validate_descriptor(artifact, limits)
        if not artifact["path"].startswith("evidence/") or artifact["size"] > limits.max_file_bytes:
            fail("invalid evidence artifact location or size")
    files = [archive, *documents.values(), *evidence_artifacts,
             {"path": "release-manifest.json", "size": len(manifest_data),
              "sha256": hashlib.sha256(manifest_data).hexdigest()}]
    names = [entry["path"] for entry in files] + ["signing-payload.json"]
    check_names(names, limits)
    if [name for name, _ in walk_files(directory, limits)] != sorted(names):
        fail("release folder has unknown or missing files")
    for entry in files:
        if descriptor(directory / entry["path"], entry["path"], limits) != entry:
            fail("published file size/hash differs from signed inventory")
    payload_data = read_file(directory / "signing-payload.json", limits.max_metadata_bytes)
    if json_document(payload_data, limits, canonical_only=True) != signing_payload(files):
        fail("signing payload does not bind the exact complete release file set")
    supplied_signature_fields = (signature, trusted_public_key, trusted_public_key_sha256, openssl)
    signature_verified = False
    if any(value is not None for value in supplied_signature_fields):
        if any(value is None for value in supplied_signature_fields):
            fail("signature verification requires signature, trusted key, key pin and OpenSSL")
        verify_signature(payload_data, signature, trusted_public_key,
                         trusted_public_key_sha256, openssl)
        signature_verified = True

    def document(identity: str, *, generated: bool = False) -> Any:
        return json_document(read_file(directory / DOCUMENTS[identity], limits.max_metadata_bytes),
                             limits, canonical_only=generated)

    inventory = validate_inventory(document("inventory", generated=True), limits)
    dependencies = document("dependency_lock")
    dependency_ids = validate_dependencies(dependencies)
    evidence = document("evidence_index")
    evidence_blockers, records = validate_evidence(
        evidence, release, documents["inventory"]["sha256"],
        documents["dependency_lock"]["sha256"], dependency_ids, limits,
    )
    expected_artifacts = [{**item, "path": "evidence/" + item["path"]} for item in records]
    if expected_artifacts != evidence_artifacts:
        fail("release evidence files differ from the bound validation index")
    blockers = ga_blockers(release, dependencies, evidence_blockers)
    if manifest["ga"] != {"status": "pending", "blockers": blockers}:
        fail("release GA declaration differs from actual bound gate statuses")
    if document("sbom", generated=True) != sbom(release, dependencies, archive["sha256"]):
        fail("SBOM differs from release/dependency declarations")
    if document("provenance", generated=True) != provenance(release, archive, documents):
        fail("provenance differs from exact archive and evidence bindings")
    license_data = read_file(directory / "LICENSE", limits.max_metadata_bytes)
    if not all(marker in license_data for marker in (
        b"Apache License", b"Version 2.0", b"TERMS AND CONDITIONS", b"END OF TERMS AND CONDITIONS"
    )) or not read_file(directory / "THIRD_PARTY_NOTICES.md", limits.max_metadata_bytes):
        fail("project license/third-party notices missing or inconsistent")
    verify_archive(directory / archive["path"], inventory, release, limits)
    if descriptor(directory / archive["path"], archive["path"], limits) != archive:
        fail("archive changed during extraction")
    if signature_verified:
        blockers.remove("signature.pending")
    return {
        "candidate_verified": True, "signature_verified": signature_verified,
        "ga_ready": not blockers, "ga_blockers": blockers, "release": release,
        "archive_sha256": archive["sha256"],
        "signing_payload_sha256": hashlib.sha256(payload_data).hexdigest(),
        "unpacked_files": len(inventory["files"]), "unpacked_bytes": inventory["total_bytes"],
    }


def main() -> int:
    """Exit 0 for valid candidate/GA, 1 for pending required GA, and 2 for invalid assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-platform", required=True)
    parser.add_argument("--expected-source-state", choices=("clean", "dirty"), default="clean")
    parser.add_argument("--signature", type=Path)
    parser.add_argument("--trusted-public-key", type=Path)
    parser.add_argument("--trusted-public-key-sha256")
    parser.add_argument("--openssl", type=Path)
    parser.add_argument("--require-ga", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_release(
            args.directory, expected_version=args.expected_version,
            expected_source_commit=args.expected_source_commit,
            expected_platform=args.expected_platform,
            expected_source_dirty=args.expected_source_state == "dirty",
            signature=args.signature, trusted_public_key=args.trusted_public_key,
            trusted_public_key_sha256=args.trusted_public_key_sha256, openssl=args.openssl,
        )
    except (ReleaseError, OSError, zipfile.BadZipFile, EOFError, OverflowError,
            TypeError, struct.error, UnicodeError, RuntimeError) as error:
        print(f"release verification failed: {error}", file=sys.stderr)
        return 2
    print(canonical(result).decode("utf-8"), end="")
    return 1 if args.require_ga and not result["ga_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
