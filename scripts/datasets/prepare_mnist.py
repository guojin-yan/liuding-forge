#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Prepare verified MNIST IDX files in an explicitly selected external cache."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "examples/mnist/dataset-manifest.json"


def atomic_write(path: Path, data: bytes) -> None:
    """Publish complete verified bytes, removing temporary files on failure."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_file(directory: Path, source: str, entry: dict, offline: bool) -> Path:
    """Verify compressed identity and bound decompression before publication."""
    archive = directory / entry["name"]
    expected_size = entry["size_bytes"]
    if archive.exists():
        with archive.open("rb") as stream:
            encoded = stream.read(expected_size + 1)
    elif offline:
        raise FileNotFoundError(f"missing MNIST archive in offline mode: {archive}")
    else:
        with urllib.request.urlopen(source + entry["name"], timeout=30) as response:
            if not response.geturl().startswith("https://"):
                raise ValueError("MNIST download redirected away from HTTPS")
            encoded = response.read(expected_size + 1)
    if len(encoded) != expected_size or hashlib.sha256(encoded).hexdigest() != entry["sha256"]:
        raise ValueError(f"MNIST archive size/SHA-256 mismatch: {archive}")
    with gzip.GzipFile(fileobj=io.BytesIO(encoded)) as stream:
        decoded = stream.read(entry["decoded_size_bytes"] + 1)
    if len(decoded) != entry["decoded_size_bytes"]:
        raise ValueError(f"MNIST decoded size mismatch: {archive}")
    # A failed fetch, checksum, CRC, or decompression never becomes a cache hit.
    if not archive.exists():
        atomic_write(archive, encoded)
    destination = archive.with_suffix("")
    if (not destination.exists() or destination.stat().st_size != len(decoded)
            or destination.read_bytes() != decoded):
        atomic_write(destination, decoded)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="external dataset root; defaults to JYPPX_DATA_ROOT")
    parser.add_argument("--offline", action="store_true", help="verify cached archives without any network access")
    args = parser.parse_args()
    data_root = args.data_root or (Path(os.environ["JYPPX_DATA_ROOT"]) if os.environ.get("JYPPX_DATA_ROOT") else None)
    if data_root is None:
        parser.error("set JYPPX_DATA_ROOT or supply --data-root")
    data_root = data_root.resolve()
    if data_root == ROOT or ROOT in data_root.parents:
        parser.error("downloaded datasets must be outside the code repository")
    directory = data_root / "mnist"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    try:
        for entry in manifest["files"]:
            destination = prepare_file(directory, manifest["source"], entry, args.offline)
            print(f"verified: {destination}", flush=True)
    except (OSError, ValueError, EOFError) as error:
        parser.exit(1, f"MNIST preparation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
