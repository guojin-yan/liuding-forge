#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Prepare verified local Fashion-MNIST IDX gzip files as RGB ImageFolder PNGs.

Uses only the standard library and never downloads data. Even subset selection
validates every source sample. Images are streamed one at a time; training and
model-specific resizing/normalization remain in C++.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import zlib


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "recipes/image-classification/fashion-mnist.dataset.json"
LICENSE = CONTRACT.with_name("fashion-mnist.LICENSE")
DATASET_ID = "jyppx.liuding.fashion-mnist/v1"
SPLITS = ("train", "validation")
SELECTIONS = ("first100-per-class", "full")
CLASS_NAMES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
               "Shirt", "Sneaker", "Bag", "Ankle boot")
CLASS_DIRS = ("00-t-shirt-top", "01-trouser", "02-pullover", "03-dress", "04-coat",
              "05-sandal", "06-shirt", "07-sneaker", "08-bag", "09-ankle-boot")
SOURCE_NAMES = {"train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
                "validation": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")}
SIDE = 28
PIXELS = SIDE * SIDE


def reject_links(path: Path) -> None:
    """Require stable caller-owned paths without symlinks or reparse points."""
    for candidate in (path, *path.parents):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(metadata.st_mode)
                or getattr(metadata, "st_file_attributes", 0) & 0x400):
            raise ValueError(f"dataset paths must not traverse links: {candidate}")


def validate_contract(contract: dict, selection: str) -> bytes:
    """Check the fixed class/path mapping, source limits and original license."""
    classes = [{"id": i, "name": name, "directory": CLASS_DIRS[i]}
               for i, name in enumerate(CLASS_NAMES)]
    if (contract["schema_version"] != 1 or contract["dataset_id"] != DATASET_ID
            or contract["classes"] != classes or selection not in SELECTIONS):
        raise ValueError("unsupported Fashion-MNIST dataset contract or selection")
    for split in SPLITS:
        info = contract["files"][split]
        count = info["samples"]
        per_class = info["samples_per_class"]
        if (type(count) is not int or type(per_class) is not int
                or not 0 < count <= 60000 or count != 10 * per_class
                or (selection == "first100-per-class" and per_class < 100)):
            raise ValueError(f"invalid Fashion-MNIST source count or subset quota: {split}")
        for kind, name in zip(("images", "labels"), SOURCE_NAMES[split]):
            source = info[kind]
            if (source["name"] != name or type(source["bytes"]) is not int
                    or not 0 < source["bytes"] <= 32 * 1024 * 1024):
                raise ValueError("invalid Fashion-MNIST source name or byte limit")
    reject_links(LICENSE)
    if LICENSE.stat().st_size > 65536:
        raise ValueError("Fashion-MNIST license exceeds its byte limit")
    license_bytes = LICENSE.read_bytes()
    if (contract["license"]["spdx"] != "MIT"
            or contract["license"]["file"] != LICENSE.name
            or hashlib.sha256(license_bytes).hexdigest() != contract["license"]["sha256"]):
        raise ValueError("Fashion-MNIST license identity mismatch")
    return license_bytes


def archive_identity(path: Path, expected: dict) -> dict:
    """Hash bounded compressed input chunks before preparing any images."""
    reject_links(path)
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Fashion-MNIST archive size mismatch: {path.name}")
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    count = 0
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            count += len(block)
            if count > expected["bytes"]:
                raise ValueError("Fashion-MNIST archive grew while reading")
            sha256.update(block)
            md5.update(block)
    if (count != expected["bytes"] or sha256.hexdigest() != expected["sha256"]
            or md5.hexdigest() != expected["official_md5"]):
        raise ValueError(f"Fashion-MNIST archive SHA-256/official MD5 mismatch: {path.name}")
    return {"file": path.name, "bytes": count, "sha256": sha256.hexdigest(),
            "official_md5": md5.hexdigest()}


def rgb_png(gray: bytes) -> bytes:
    """Repeat each original grayscale byte into all three channels, losslessly."""
    if len(gray) != PIXELS:
        raise ValueError("Fashion-MNIST image must contain 784 grayscale bytes")
    rgb = bytearray(3 * PIXELS)
    for channel in range(3):
        rgb[channel::3] = gray
    stride = SIDE * 3 + 1
    rows = bytearray(SIDE * stride)
    for y in range(SIDE):
        rows[y * stride + 1:(y + 1) * stride] = rgb[y * SIDE * 3:(y + 1) * SIDE * 3]

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", SIDE, SIDE, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, level=1)) + chunk(b"IEND", b""))


def exact_read(source, count: int, description: str) -> bytes:
    content = source.read(count)
    if len(content) != count:
        raise ValueError(f"truncated Fashion-MNIST {description}")
    return content


def convert_split(source_dir: Path, staging: Path, split: str, info: dict,
                  selection: str) -> dict:
    """Read the entire IDX split, retaining only the requested original indices."""
    count = info["samples"]
    quota = 100 if selection == "first100-per-class" else info["samples_per_class"]
    source_counts, selected_counts = [0] * 10, [0] * 10
    content_hash = hashlib.sha256(b"jyppx.liuding.fashion-mnist.split.v1\0"
                                 + split.encode("ascii") + b"\0"
                                 + selection.encode("ascii") + b"\0")
    index_hash = hashlib.sha256()
    image_bytes = index_bytes = 0
    for directory in CLASS_DIRS:
        (staging / split / directory).mkdir(parents=True)
    index_file = f"index/{split}.jsonl"
    with (gzip.open(source_dir / info["images"]["name"], "rb") as images,
          gzip.open(source_dir / info["labels"]["name"], "rb") as labels,
          (staging / index_file).open("xb") as index):
        image_header = struct.unpack(">IIII", exact_read(images, 16, "image header"))
        label_header = struct.unpack(">II", exact_read(labels, 8, "label header"))
        if image_header != (2051, count, SIDE, SIDE) or label_header != (2049, count):
            raise ValueError(f"Fashion-MNIST IDX header mismatch: {split}")
        for number in range(count):
            gray = exact_read(images, PIXELS, "image payload")
            label = exact_read(labels, 1, "label payload")[0]
            if label >= 10:
                raise ValueError(f"invalid Fashion-MNIST label: {split}/{number}")
            source_counts[label] += 1
            if selected_counts[label] >= quota:
                continue
            selected_counts[label] += 1
            relative = f"{split}/{CLASS_DIRS[label]}/{number:06d}.png"
            png = rgb_png(gray)
            with (staging / relative).open("xb") as output:
                output.write(png)
            image_bytes += len(png)
            row = {"source_index": number, "class_id": label, "path": relative,
                   "grayscale_sha256": hashlib.sha256(gray).hexdigest(),
                   "png_sha256": hashlib.sha256(png).hexdigest()}
            line = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
            index.write(line)
            index_hash.update(line)
            index_bytes += len(line)
            content_hash.update(struct.pack("<QB", number, label))
            content_hash.update(gray)
        # Read through both gzip trailers, including CRC/ISIZE validation. Even
        # first100-per-class validates every unselected image and label above.
        if images.read(1) or labels.read(1):
            raise ValueError(f"extra Fashion-MNIST IDX payload: {split}")
    if (source_counts != [info["samples_per_class"]] * 10
            or selected_counts != [quota] * 10):
        raise ValueError(f"Fashion-MNIST class counts or subset quota mismatch: {split}")
    print(f"prepared {split}: {sum(selected_counts)} of {count} images", flush=True)
    return {"source_samples": count, "source_class_counts": source_counts,
            "samples": sum(selected_counts), "class_counts": selected_counts,
            "content_sha256": content_hash.hexdigest(), "png_bytes": image_bytes,
            "index": {"file": index_file, "bytes": index_bytes, "sha256": index_hash.hexdigest()}}


def prepare(source_dir: Path, data_root: Path, selection: str, contract: dict) -> Path:
    """Verify and stage an external dataset, then publish without replacement."""
    source_dir, data_root = (Path(os.path.abspath(path)) for path in (source_dir, data_root))
    reject_links(source_dir)
    reject_links(data_root)
    if data_root == ROOT or ROOT in data_root.parents:
        raise ValueError("prepared datasets must be outside the code repository")
    license_bytes = validate_contract(contract, selection)
    destination = data_root / f"fashion-mnist-{selection}-imagefolder-v1"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace prepared dataset: {destination}")
    sources = {}
    for split in SPLITS:
        sources[split] = {kind: archive_identity(source_dir / contract["files"][split][kind]["name"],
                                                 contract["files"][split][kind])
                          for kind in ("images", "labels")}
    data_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fashion-mnist-preparing-", dir=data_root))
    try:
        (staging / "index").mkdir()
        splits = {split: convert_split(source_dir, staging, split, contract["files"][split], selection)
                  for split in SPLITS}
        identity = hashlib.sha256(b"jyppx.liuding.fashion-mnist.imagefolder.v1\0"
                                  + selection.encode("ascii") + b"\0")
        for split in SPLITS:
            identity.update(split.encode("ascii") + b"\0"
                            + bytes.fromhex(splits[split]["content_sha256"]))
        manifest = {"schema_version": 1, "dataset_id": DATASET_ID,
                    "source": contract["source"], "source_files": sources,
                    "license": contract["license"], "selection": selection,
                    "classes": contract["classes"], "splits": splits,
                    "pixels": "28x28 RGB uint8 PNG; each source grayscale byte repeated in R/G/B; no resizing",
                    "split_policy": "original train/test; test is named validation; retain source order and indices",
                    "content_sha256": identity.hexdigest(),
                    "content_hash_format": {
                        "split": "SHA256(ASCII jyppx.liuding.fashion-mnist.split.v1 NUL + split NUL + selection NUL + selected records in source order); record = uint64 little-endian source index + uint8 label + 784 original grayscale bytes",
                        "dataset": "SHA256(ASCII jyppx.liuding.fashion-mnist.imagefolder.v1 NUL + selection NUL + train NUL + raw train split SHA256 + validation NUL + raw validation split SHA256)"}}
        with (staging / LICENSE.name).open("xb") as output:
            output.write(license_bytes)
        with (staging / "preparation.json").open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        # The caller keeps parents stable. This same-filesystem publication is
        # not a sandbox against concurrent hostile filesystem mutation.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"dataset appeared during preparation: {destination}")
        staging.rename(destination)
    finally:
        if staging.exists():
            assert staging.parent.resolve() == data_root.resolve()
            assert staging.name.startswith(".fashion-mnist-preparing-")
            shutil.rmtree(staging)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="directory containing the four official IDX .gz files")
    parser.add_argument("--selection", choices=SELECTIONS, default="first100-per-class")
    parser.add_argument("--data-root", type=Path,
                        help="external output root; defaults to JYPPX_LIUDING_DATA_ROOT")
    args = parser.parse_args()
    data_root = args.data_root or (Path(os.environ["JYPPX_LIUDING_DATA_ROOT"])
                                  if os.environ.get("JYPPX_LIUDING_DATA_ROOT") else None)
    if data_root is None:
        parser.error("supply --data-root or set JYPPX_LIUDING_DATA_ROOT")
    try:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        destination = prepare(args.source_dir, data_root, args.selection, contract)
        print(f"verified Fashion-MNIST ImageFolder: {destination}", flush=True)
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        parser.exit(1, f"Fashion-MNIST preparation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
