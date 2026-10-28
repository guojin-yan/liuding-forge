#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Convert a verified local CIFAR-10 binary archive to lossless ImageFolder PNGs.

Uses only Python's standard library, does not access the network, and never
loads Python pickle files. Training and image decoding remain in C++.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import tarfile
import tempfile
import zlib


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "recipes/image-classification/cifar10.dataset.json"
RECORD_BYTES = 1 + 3 * 32 * 32
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog",
           "frog", "horse", "ship", "truck")


def reject_links(path: Path) -> None:
    """Reject existing symlink/reparse components; caller keeps paths stable."""
    for candidate in (path, *path.parents):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(metadata.st_mode)
                or getattr(metadata, "st_file_attributes", 0) & 0x400):
            raise ValueError(f"dataset paths must not traverse links: {candidate}")


def rgb_png(planes: bytes) -> bytes:
    """Encode exact 32x32 planar uint8 RGB values without image dependencies."""
    if len(planes) != RECORD_BYTES - 1:
        raise ValueError("CIFAR-10 image must contain 3072 RGB bytes")
    rgb = bytearray(3072)
    for channel in range(3):
        rgb[channel::3] = planes[channel * 1024:(channel + 1) * 1024]
    scanlines = bytearray(32 * 97)
    for row in range(32):
        scanlines[row * 97 + 1:(row + 1) * 97] = rgb[row * 96:(row + 1) * 96]

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines, level=1)) + chunk(b"IEND", b""))


def archive_identity(archive: Path, contract: dict) -> dict:
    """Stream and verify the compressed bytes before creating any output."""
    reject_links(archive)
    if not archive.is_file() or archive.stat().st_size != contract["archive_bytes"]:
        raise ValueError("CIFAR-10 archive size mismatch")
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    count = 0
    with archive.open("rb") as source:
        while content := source.read(1024 * 1024):
            count += len(content)
            if count > contract["archive_bytes"]:
                raise ValueError("CIFAR-10 archive grew while reading")
            sha256.update(content)
            md5.update(content)
    if (count != contract["archive_bytes"] or sha256.hexdigest() != contract["sha256"]
            or md5.hexdigest() != contract["official_md5"]):
        raise ValueError("CIFAR-10 archive SHA-256/official MD5 mismatch")
    return {"bytes": count, "sha256": sha256.hexdigest(), "official_md5": md5.hexdigest()}


def convert_archive(archive: Path, staging: Path, contract: dict) -> dict:
    """Stream the allowlisted tar members and one RGB record at a time."""
    prefix = "cifar-10-batches-bin"
    entries = contract["batches"]
    expected_names = set(entries) | {"batches.meta.txt", "readme.html"}
    seen: set[str] = set()
    counts = {split: [0] * len(CLASSES) for split in ("train", "validation")}
    payload_hashes = {}
    total_decoded = 0
    for split in counts:
        for category in CLASSES:
            (staging / split / category).mkdir(parents=True)
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts
                    or "\\" in member.name or len(name.parts) not in (1, 2)
                    or name.parts[0] != prefix):
                raise ValueError(f"unexpected archive member: {member.name}")
            if len(name.parts) == 1:
                if not member.isdir() or "<root>" in seen:
                    raise ValueError("invalid/duplicate CIFAR-10 root entry")
                seen.add("<root>")
                continue
            key = name.parts[1]
            if key not in expected_names or key in seen or not member.isfile():
                raise ValueError(f"unexpected, duplicate or non-file member: {member.name}")
            seen.add(key)
            if key in entries:
                info = entries[key]
                limit = info["samples"] * RECORD_BYTES
            else:
                limit = 65536
            if member.size > limit or (key in entries and member.size != limit):
                raise ValueError(f"invalid decoded member size: {key}")
            total_decoded += member.size
            if total_decoded > contract["max_decoded_bytes"]:
                raise ValueError("CIFAR-10 decoded byte limit exceeded")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read CIFAR-10 member: {key}")
            with stream:
                if key not in entries:
                    content = stream.read(limit + 1)
                    if len(content) != member.size:
                        raise ValueError(f"truncated metadata: {key}")
                    if key == "batches.meta.txt":
                        if tuple(content.decode("ascii").splitlines()) != CLASSES:
                            raise ValueError("CIFAR-10 category mapping differs")
                    continue
                split = info["split"]
                content_hash = hashlib.sha256()
                for number in range(info["samples"]):
                    record = stream.read(RECORD_BYTES)
                    if len(record) != RECORD_BYTES or record[0] >= len(CLASSES):
                        raise ValueError(f"truncated record or invalid label: {key}/{number}")
                    content_hash.update(record)
                    label = record[0]
                    counts[split][label] += 1
                    destination = staging / split / CLASSES[label] / f"{key[:-4]}-{number:05d}.png"
                    with destination.open("xb") as image:
                        image.write(rgb_png(record[1:]))
                if stream.read(1):
                    raise ValueError(f"extra batch data: {key}")
                payload_hashes[key] = content_hash.hexdigest()
                print(f"prepared {key}: {info['samples']} images", flush=True)
    if not (set(entries) | {"batches.meta.txt"}).issubset(seen):
        raise ValueError("CIFAR-10 archive is missing required batches or metadata")
    for split, per_class in counts.items():
        if per_class != [contract["samples_per_class"][split]] * len(CLASSES):
            raise ValueError(f"CIFAR-10 class counts differ: {split}")
    return {"classes": list(CLASSES), "class_counts": counts,
            "batch_sha256": payload_hashes, "decoded_bytes": total_decoded}


def prepare(archive: Path, data_root: Path, contract: dict) -> Path:
    """Publish a new dataset only after validation; never replace prior data."""
    archive = Path(os.path.abspath(archive))
    data_root = Path(os.path.abspath(data_root))
    reject_links(data_root)
    if data_root == ROOT or ROOT in data_root.parents:
        raise ValueError("prepared datasets must be outside the code repository")
    destination = data_root / "cifar10-imagefolder-v1"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace prepared dataset: {destination}")
    identity = archive_identity(archive, contract)
    data_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".cifar10-preparing-", dir=data_root))
    try:
        result = convert_archive(archive, staging, contract)
        result.update({"schema_version": 1, "dataset": "CIFAR-10 binary version",
                       "source": contract["source"], "archive": identity,
                       "dataset_license": contract["dataset_license"],
                       "pixels": "32x32 RGB uint8; exact planar-to-PNG conversion",
                       "split_policy": "complete source train/test; test is named validation; no subset"})
        (staging / "preparation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Dataset parents are caller-owned and must stay quiescent. Publication
        # is a same-filesystem rename, not a hostile-filesystem sandbox.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"dataset appeared during preparation: {destination}")
        staging.rename(destination)
    finally:
        if staging.exists():
            assert staging.parent.resolve() == data_root.resolve()
            assert staging.name.startswith(".cifar10-preparing-")
            shutil.rmtree(staging)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True,
                        help="locally downloaded official cifar-10-binary.tar.gz")
    parser.add_argument("--data-root", type=Path,
                        help="external destination root; defaults to JYPPX_LIUDING_DATA_ROOT")
    args = parser.parse_args()
    data_root = args.data_root or (Path(os.environ["JYPPX_LIUDING_DATA_ROOT"])
                                  if os.environ.get("JYPPX_LIUDING_DATA_ROOT") else None)
    if data_root is None:
        parser.error("supply --data-root or set JYPPX_LIUDING_DATA_ROOT")
    try:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        destination = prepare(args.archive, data_root, contract)
        print(f"verified CIFAR-10 ImageFolder: {destination}", flush=True)
    except (OSError, ValueError, EOFError, tarfile.TarError) as error:
        parser.exit(1, f"CIFAR-10 preparation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
