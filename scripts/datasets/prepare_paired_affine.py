#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Prepare deterministic paired-input classification/regression JSONL data.

Uses only the Python standard library, generates one record at a time, and
publishes a new external dataset after verifying its frozen content digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "recipes/multi-io/paired-affine.dataset.json"
DATASET_ID = "jyppx.liuding.paired-affine/v1"
SEED = 20261028
SPLITS = {"train": 1024, "validation": 256}
MAX_RECORD_BYTES = 512
MAX_DATA_BYTES = 1024 * 1024


def reject_links(path: Path) -> None:
    """Require caller-owned stable paths without symlinks or reparse points."""
    for candidate in (path, *path.parents):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(metadata.st_mode)
                or getattr(metadata, "st_file_attributes", 0) & 0x400):
            raise ValueError(f"dataset paths must not traverse links: {candidate}")


def record(split: str, index: int) -> dict:
    """Derive two distinct feature inputs and two exactly dyadic targets."""
    raw = hashlib.sha256(f"{DATASET_ID}:{SEED}:{split}:{index}".encode("ascii")).digest()
    values = [(int.from_bytes(raw[offset:offset + 2], "little") % 2049 - 1024) / 1024
              for offset in range(0, 8, 2)]
    left0, left1, right0, right1 = values
    return {"id": f"{split}:{index:06d}", "left": values[:2], "right": values[2:],
            "class_id": int(left0 + 0.5 * left1 + right0 - 0.25 * right1 >= 0),
            "regression": [(4 * left0 - 2 * left1 + 3 * right0 + right1 + 0.5) / 8]}


def write_split(staging: Path, split: str, count: int) -> dict:
    """Write and hash at most one bounded JSON record in memory at a time."""
    digest = hashlib.sha256()
    total, largest = 0, 0
    classes = [0, 0]
    with (staging / f"{split}.jsonl").open("xb") as output:
        for index in range(count):
            value = record(split, index)
            content = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False) + "\n").encode("ascii")
            if len(content) > MAX_RECORD_BYTES or total + len(content) > MAX_DATA_BYTES:
                raise ValueError("paired-affine data exceeds its byte limits")
            output.write(content)
            digest.update(content)
            total += len(content)
            largest = max(largest, len(content))
            classes[value["class_id"]] += 1
    return {"file": f"{split}.jsonl", "samples": count, "bytes": total,
            "sha256": digest.hexdigest(), "class_counts": classes,
            "max_record_bytes_observed": largest}


def prepare(data_root: Path, contract: dict) -> Path:
    """Verify all generated split bytes before publishing; never replace data."""
    if (contract["schema_version"] != 1 or contract["dataset_id"] != DATASET_ID
            or contract["seed"] != SEED or contract["samples"] != SPLITS):
        raise ValueError("unsupported paired-affine dataset contract")
    data_root = Path(os.path.abspath(data_root))
    reject_links(data_root)
    if data_root == ROOT or ROOT in data_root.parents:
        raise ValueError("prepared datasets must be outside the code repository")
    destination = data_root / "paired-affine-v1"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace prepared dataset: {destination}")
    data_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".paired-affine-preparing-", dir=data_root))
    try:
        splits = {}
        total = 0
        for split, count in SPLITS.items():
            generated = write_split(staging, split, count)
            if generated != contract["files"][split]:
                raise ValueError(f"paired-affine generated content identity mismatch: {split}")
            total += generated["bytes"]
            if total > MAX_DATA_BYTES:
                raise ValueError("paired-affine total data exceeds its byte limit")
            splits[split] = generated
        manifest = {"schema_id": "jyppx.liuding.paired-affine.preparation", "schema_version": 1,
                    "dataset_id": DATASET_ID, "seed": SEED,
                    "dataset_license": contract["dataset_license"],
                    "source": "self-authored deterministic paired-affine example; no external data",
                    "files": splits, "max_record_bytes": MAX_RECORD_BYTES,
                    "max_data_bytes": MAX_DATA_BYTES, "data_bytes": total}
        with (staging / "preparation.json").open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        # The caller keeps the parent stable. This is same-filesystem
        # publication, not isolation from adversarial filesystem mutation.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"dataset appeared during preparation: {destination}")
        staging.rename(destination)
    finally:
        if staging.exists():
            assert staging.parent.resolve() == data_root.resolve()
            assert staging.name.startswith(".paired-affine-preparing-")
            shutil.rmtree(staging)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        help="external output root; defaults to JYPPX_LIUDING_DATA_ROOT")
    args = parser.parse_args()
    data_root = args.data_root or (Path(os.environ["JYPPX_LIUDING_DATA_ROOT"])
                                  if os.environ.get("JYPPX_LIUDING_DATA_ROOT") else None)
    if data_root is None:
        parser.error("supply --data-root or set JYPPX_LIUDING_DATA_ROOT")
    try:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        destination = prepare(data_root, contract)
        print(f"verified paired-affine dataset: {destination}", flush=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"paired-affine preparation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
