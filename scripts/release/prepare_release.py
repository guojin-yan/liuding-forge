#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prepare an offline, unsigned, reviewable candidate from an installed SDK.

The explicit publisher declarations and evidence are bound, never invented.
Use verify_release.py independently before external signing or distribution.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any
import zipfile

from release_common import (
    Limits, ReleaseError, canonical, descriptor, fail, ga_blockers, json_document,
    no_links, prefix_inventory, provenance, read_file, relative_name, sbom,
    signing_payload, stream_file, validate_dependencies, validate_evidence,
    validate_installed_version, validate_release,
)


def write_document(path: Path, value: Any, limits: Limits) -> None:
    """Publish only bounded canonical documents into an exclusively owned stage."""
    data = canonical(value)
    json_document(data, limits, canonical_only=True)
    with path.open("xb") as output:
        output.write(data)


def zip_timestamp(epoch: int) -> tuple[int, int, int, int, int, int]:
    """ZIP has a two-second UTC resolution and cannot represent pre-1980 dates."""
    value = dt.datetime.fromtimestamp(max(epoch, 315532800), dt.timezone.utc)
    return (value.year, value.month, value.day, value.hour, value.minute,
            value.second // 2 * 2)


def write_archive(path: Path, prefix: Path, inventory: dict[str, Any],
                  epoch: int, limits: Limits) -> None:
    """Write only sorted regular entries with deterministic portable attributes."""
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for entry in inventory["files"]:
            info = zipfile.ZipInfo("payload/" + entry["path"], zip_timestamp(epoch))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | entry["mode"]) << 16
            info.file_size = entry["size"]
            with archive.open(info, "w", force_zip64=entry["size"] >= zipfile.ZIP64_LIMIT) as output:
                actual = stream_file(prefix / entry["path"], limits.max_file_bytes, output)
            if actual != {key: entry[key] for key in ("size", "sha256")}:
                fail("install prefix changed after inventory")
    if path.stat().st_size > limits.max_total_bytes:
        fail("archive byte limit exceeded")


def prepare_release(*, prefix: Path, out: Path, release: dict[str, Any],
                    evidence_index: Path, dependency_lock: Path, license_file: Path,
                    notices_file: Path, limits: Limits = Limits()) -> dict[str, Any]:
    """Stage candidate assets and commit to a new destination without overwrites.

    The caller exclusively owns the prefix, inputs and output parent for this
    operation. A failed publication can leave a partial output; verification
    rejects it and it must not be reused as a successful release.
    """
    limits.validate()
    validate_release(release)
    prefix = no_links(prefix)
    out = Path(os.path.abspath(out))
    no_links(out.parent)
    if out.exists() or out.is_symlink():
        fail("output directory already exists")
    if prefix == out or prefix in out.parents:
        fail("release output cannot be inside install prefix")
    inventory = prefix_inventory(prefix, limits)
    validate_installed_version(prefix, release, limits)
    inventory_sha = hashlib.sha256(canonical(inventory)).hexdigest()
    lock_data = read_file(dependency_lock, limits.max_metadata_bytes)
    dependencies = json_document(lock_data, limits)
    dependency_ids = validate_dependencies(dependencies)
    evidence_data = read_file(evidence_index, limits.max_metadata_bytes)
    evidence = json_document(evidence_data, limits)
    blockers, records = validate_evidence(
        evidence, release, inventory_sha, hashlib.sha256(lock_data).hexdigest(),
        dependency_ids, limits,
    )
    blockers = ga_blockers(release, dependencies, blockers)
    license_data = read_file(license_file, limits.max_metadata_bytes)
    notices_data = read_file(notices_file, limits.max_metadata_bytes)
    if not license_data or not notices_data:
        fail("LICENSE and THIRD_PARTY_NOTICES must both be supplied")
    # A project license claim must at least carry the Apache 2.0 license text;
    # exact third-party obligations remain the explicit dependency audit gate.
    if not all(marker in license_data for marker in (
        b"Apache License", b"Version 2.0", b"TERMS AND CONDITIONS", b"END OF TERMS AND CONDITIONS"
    )):
        fail("supplied project license is not the Apache 2.0 license text")
    archive_name = relative_name(f"liuding-forge-{release['version']}-{release['platform']}.zip")
    with tempfile.TemporaryDirectory(prefix=".liuding-forge-release-", dir=out.parent) as temporary:
        stage = Path(temporary)
        write_document(stage / "inventory.json", inventory, limits)
        for name, data in (("dependency-manifest.json", lock_data),
                           ("evidence-index.json", evidence_data),
                           ("LICENSE", license_data), ("THIRD_PARTY_NOTICES.md", notices_data)):
            with (stage / name).open("xb") as output:
                output.write(data)
        evidence_artifacts = []
        for record in records:
            name = "evidence/" + record["path"]
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                actual = stream_file(evidence_index.parent / record["path"], limits.max_file_bytes, output)
            if actual != {key: record[key] for key in ("size", "sha256")}:
                fail("validation evidence content differs from indexed digest")
            evidence_artifacts.append({"path": name, **actual})
        write_archive(stage / archive_name, prefix, inventory,
                      release["source_date_epoch"], limits)
        archive = descriptor(stage / archive_name, archive_name, limits)
        write_document(stage / "sbom.cdx.json", sbom(release, dependencies, archive["sha256"]), limits)
        documents = {
            identity: descriptor(stage / path, path, limits)
            for identity, path in {
                "inventory": "inventory.json", "sbom": "sbom.cdx.json",
                "dependency_lock": "dependency-manifest.json",
                "evidence_index": "evidence-index.json", "license": "LICENSE",
                "notices": "THIRD_PARTY_NOTICES.md",
            }.items()
        }
        write_document(stage / "provenance.intoto.json", provenance(release, archive, documents), limits)
        documents["provenance"] = descriptor(stage / "provenance.intoto.json", "provenance.intoto.json", limits)
        manifest = {
            "schema_version": 1, "kind": "jyppx-sdk-release", "release": release,
            "archive": archive, "documents": documents,
            "evidence_artifacts": evidence_artifacts,
            "ga": {"status": "pending", "blockers": blockers},
        }
        write_document(stage / "release-manifest.json", manifest, limits)
        files = [archive, *documents.values(), *evidence_artifacts,
                 descriptor(stage / "release-manifest.json", "release-manifest.json", limits)]
        write_document(stage / "signing-payload.json", signing_payload(files), limits)
        # Exclusive mkdir reserves this publication. Hard links refuse existing
        # destinations on both supported OSes; unlink staging aliases afterwards.
        out.mkdir()
        names = sorted(item["path"] for item in files) + ["signing-payload.json"]
        for name in names:
            destination = out / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(stage / name, destination)
            (stage / name).unlink()
    return {"candidate_prepared": True, "ga_ready": False,
            "ga_blockers": blockers, "output": str(out),
            "signing_payload": str(out / "signing-payload.json")}


def parser() -> argparse.ArgumentParser:
    """Keep inventory generation independent of uncompleted release evidence."""
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="compute the exact prefix inventory binding")
    inventory.add_argument("--prefix", required=True, type=Path)
    inventory.add_argument("--output", required=True, type=Path)
    prepare = commands.add_parser("prepare", help="prepare an unsigned reviewable candidate")
    for name in ("prefix", "out", "evidence-index", "dependency-lock", "license", "notices"):
        prepare.add_argument("--" + name, required=True, type=Path)
    prepare.add_argument("--version", required=True)
    prepare.add_argument("--platform", required=True)
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--source-state", required=True, choices=("clean", "dirty"))
    prepare.add_argument("--source-date-epoch", required=True, type=int)
    return result


def main() -> int:
    """Exit zero for a prepared candidate; pending is never a GA assertion."""
    args = parser().parse_args()
    limits = Limits()
    try:
        if args.command == "inventory":
            output = Path(os.path.abspath(args.output))
            no_links(output.parent)
            prefix = no_links(args.prefix)
            if prefix in output.parents:
                fail("inventory output cannot be inside install prefix")
            write_document(output, prefix_inventory(prefix, limits), limits)
            result = {"inventory": str(output), "sha256": stream_file(output, limits.max_metadata_bytes)["sha256"]}
        else:
            result = prepare_release(
                prefix=args.prefix, out=args.out,
                release={"version": args.version, "platform": args.platform,
                         "source": {"commit": args.source_commit, "dirty": args.source_state == "dirty"},
                         "source_date_epoch": args.source_date_epoch},
                evidence_index=args.evidence_index, dependency_lock=args.dependency_lock,
                license_file=args.license, notices_file=args.notices,
            )
    except (ReleaseError, OSError, zipfile.BadZipFile, OverflowError, TypeError,
            UnicodeError, RuntimeError) as error:
        print(f"release preparation failed: {error}", file=sys.stderr)
        return 2
    print(canonical(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
