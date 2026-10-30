#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent synthetic fixtures and Python reference for native preparation.

No network or real Fashion-MNIST files. Python is only a test driver/oracle.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fashion_reference", ROOT / "tests/reference/fashion_mnist.py")
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
BASE = json.loads((ROOT / "recipes/image-classification/fashion-mnist.dataset.json").read_text(encoding="utf-8"))
HARNESS = None
CLI = None


def png_pixels(content):
    """Validate chunks/CRC/geometry and decode fixed filter-0 rows independently."""
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("bad PNG signature")
    chunks, offset = [], 8
    while offset < len(content):
        size = struct.unpack_from(">I", content, offset)[0]
        kind = content[offset + 4:offset + 8]
        data = content[offset + 8:offset + 8 + size]
        crc = struct.unpack_from(">I", content, offset + 8 + size)[0]
        if zlib.crc32(kind + data) & 0xffffffff != crc:
            raise AssertionError("bad PNG CRC")
        chunks.append((kind, data))
        offset += size + 12
    if offset != len(content) or [kind for kind, _ in chunks] != [b"IHDR", b"IDAT", b"IEND"]:
        raise AssertionError("unexpected PNG chunks")
    if chunks[0][1] != struct.pack(">IIBBBBB", 28, 28, 8, 2, 0, 0, 0):
        raise AssertionError("unexpected PNG geometry")
    decoder = zlib.decompressobj()
    rows = decoder.decompress(chunks[1][1]) + decoder.flush()
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or len(rows) != 2380:
        raise AssertionError("bad PNG zlib framing")
    if any(rows[y * 85] for y in range(28)) or chunks[2][1]:
        raise AssertionError("unexpected PNG filter or IEND")
    return b"".join(rows[y * 85 + 1:(y + 1) * 85] for y in range(28))


class NativeFashion(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-fashion-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "源数据"
        self.source.mkdir()
        self.output = self.root / "输出"
        self.contract_file = self.root / "contract.json"
        self.contract = copy.deepcopy(BASE)
        self.payloads = {}
        self.fixture(2)

    def fixture(self, per_class):
        for split in ("train", "validation"):
            info = self.contract["files"][split]
            count = 10 * per_class
            info.update(samples=count, samples_per_class=per_class)
            images = struct.pack(">IIII", 2051, count, 28, 28)
            images += b"".join(bytes((i * 17 + p * 13) % 256 for p in range(784)) for i in range(count))
            labels = struct.pack(">II", 2049, count) + bytes(i % 10 for i in range(count))
            for kind, content in (("images", images), ("labels", labels)):
                self.payloads[split, kind] = content
                self.store(split, kind, gzip.compress(content, mtime=0))

    def store(self, split, kind, compressed):
        entry = self.contract["files"][split][kind]
        (self.source / entry["name"]).write_bytes(compressed)
        entry.update(bytes=len(compressed), sha256=hashlib.sha256(compressed).hexdigest(),
                     official_md5=hashlib.md5(compressed).hexdigest())
        self.contract_file.write_text(json.dumps(self.contract), encoding="utf-8")

    def invoke(self, selection="full", success=True):
        result = subprocess.run([str(HARNESS), str(self.source), str(self.output),
                                 str(self.contract_file), selection], capture_output=True, text=True,
                                encoding="utf-8", timeout=120)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return Path(json.loads(result.stdout)["destination"])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        if self.output.exists():
            self.assertEqual(list(self.output.iterdir()), [], "failure left output artifacts")
        return result

    def check_reference(self, selection):
        native = self.invoke(selection)
        reference = REFERENCE.prepare(self.source, self.root / "reference", selection, self.contract)
        actual = json.loads((native / "preparation.json").read_text(encoding="utf-8"))
        expected = json.loads((reference / "preparation.json").read_text(encoding="utf-8"))
        for key in ("dataset_id", "source", "license", "classes", "selection", "pixels",
                    "split_policy", "content_sha256", "content_hash_format"):
            self.assertEqual(actual[key], expected[key], key)
        self.assertEqual((native / "fashion-mnist.LICENSE").read_bytes(), REFERENCE.LICENSE.read_bytes())
        for split in ("train", "validation"):
            a, e = actual["splits"][split], expected["splits"][split]
            for key in ("source_samples", "source_class_counts", "samples", "class_counts", "content_sha256"):
                self.assertEqual(a[key], e[key], (split, key))
            index = (native / a["index"]["file"]).read_bytes()
            self.assertEqual(len(index), a["index"]["bytes"])
            self.assertEqual(hashlib.sha256(index).hexdigest(), a["index"]["sha256"])
            # Preserve the existing byte identity as well as decoded pixels.
            # A different reference zlib version is not a reason to skip this
            # gate; any encoding difference must remain an explicit failure.
            self.assertEqual(index, (reference / e["index"]["file"]).read_bytes())
            self.assertEqual(a["index"], e["index"])
            rows = [json.loads(line) for line in index.splitlines()]
            reference_rows = [json.loads(line) for line in (reference / e["index"]["file"]).read_bytes().splitlines()]
            self.assertEqual(len(rows), a["samples"])
            total = 0
            for row, old in zip(rows, reference_rows):
                self.assertEqual({k: v for k, v in row.items() if k != "png_sha256"},
                                 {k: v for k, v in old.items() if k != "png_sha256"})
                original = self.payloads[split, "images"][16 + row["source_index"] * 784:16 + (row["source_index"] + 1) * 784]
                png = (native / row["path"]).read_bytes()
                total += len(png)
                self.assertEqual(hashlib.sha256(png).hexdigest(), row["png_sha256"])
                self.assertEqual(png_pixels(png), bytes(v for pixel in original for v in (pixel, pixel, pixel)))
                self.assertEqual(png, (reference / old["path"]).read_bytes())
            self.assertEqual(total, a["png_bytes"])
            self.assertEqual(a["png_bytes"], e["png_bytes"])
        return native

    def test_full_matches_independent_reference(self):
        self.check_reference("full")

    def test_subset_retains_original_indices_and_hashes(self):
        self.fixture(101)
        self.check_reference("first100-per-class")

    def test_sha_preflight_creates_no_output(self):
        info = self.contract["files"]["train"]["images"]
        data = bytearray((self.source / info["name"]).read_bytes())
        data[-1] ^= 1
        (self.source / info["name"]).write_bytes(data)
        self.invoke(success=False)
        self.assertFalse(self.output.exists())

    def test_compressed_size_preflight(self):
        info = self.contract["files"]["validation"]["labels"]
        with (self.source / info["name"]).open("ab") as stream:
            stream.write(b"x")
        self.invoke(success=False)
        self.assertFalse(self.output.exists())

    def test_idx_headers_lengths_labels_and_balance(self):
        original = copy.deepcopy(self.payloads)
        mutations = [("images", 0, b"\x00\x00\x00\x00"),
                     ("images", 4, struct.pack(">I", 19)),
                     ("images", 8, struct.pack(">I", 27)),
                     ("images", 12, struct.pack(">I", 29)),
                     ("labels", 0, struct.pack(">I", 2050)),
                     ("labels", 4, struct.pack(">I", 21)),
                     ("labels", 8, b"\x0a"),
                     ("labels", 8, b"\x01")]
        for kind, offset, replacement in mutations:
            with self.subTest(kind=kind, offset=offset, replacement=replacement):
                self.fixture(2)
                value = original["train", kind]
                value = value[:offset] + replacement + value[offset + len(replacement):]
                self.store("train", kind, gzip.compress(value, mtime=0))
                self.invoke(success=False)
        for kind in ("images", "labels"):
            for suffix in (b"", b"x"):
                with self.subTest(kind=kind, suffix=suffix):
                    self.fixture(2)
                    value = original["train", kind]
                    value = value[:-1] if not suffix else value + suffix
                    self.store("train", kind, gzip.compress(value, mtime=0))
                    self.invoke(success=False)

    def test_gzip_crc_isize_header_truncation_tail_and_members(self):
        for mutation in ("crc", "isize", "header", "truncated", "tail", "member"):
            with self.subTest(mutation=mutation):
                self.fixture(2)
                compressed = bytearray(gzip.compress(self.payloads["train", "images"], mtime=0))
                if mutation == "crc": compressed[-8] ^= 1
                elif mutation == "isize": compressed[-4] ^= 1
                elif mutation == "header": compressed[3] |= 0xe0
                elif mutation == "truncated": compressed = compressed[:-3]
                elif mutation == "tail": compressed += b"tail"
                elif mutation == "member": compressed += gzip.compress(b"", mtime=0)
                self.store("train", "images", compressed)
                self.invoke(success=False)

    def test_subset_validates_unselected_labels_and_image_tail(self):
        self.fixture(101)
        labels = self.payloads["train", "labels"][:-1] + b"\x0a"
        self.store("train", "labels", gzip.compress(labels, mtime=0))
        self.invoke("first100-per-class", success=False)
        self.fixture(101)
        self.store("train", "images", gzip.compress(self.payloads["train", "images"] + b"x", mtime=0))
        self.invoke("first100-per-class", success=False)

    def test_existing_output_preserved(self):
        destination = self.output / "fashion-mnist-full-imagefolder-v1"
        destination.mkdir(parents=True)
        sentinel = destination / "keep"
        sentinel.write_bytes(b"existing")
        result = subprocess.run([str(HARNESS), str(self.source), str(self.output), str(self.contract_file), "full"],
                                capture_output=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_bytes(), b"existing")
        self.assertEqual(list(self.output.iterdir()), [destination])

    def test_publication_race_preserves_even_empty_destination(self):
        result = subprocess.run([str(HARNESS), str(self.output), "--publication-collision"], capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(self.output.iterdir()), [self.output / "collision"])

    def test_repository_root_invalid_selection_and_limits(self):
        previous = self.output
        self.output = ROOT / "forbidden-fashion-test"
        try:
            self.invoke(success=False)
            self.assertFalse(self.output.exists())
        finally:
            self.output = previous
        self.invoke("unsupported", success=False)
        for value in (0, 32 * 1024 * 1024 + 1):
            self.contract["files"]["train"]["images"]["bytes"] = value
            self.contract_file.write_text(json.dumps(self.contract), encoding="utf-8")
            self.invoke(success=False)

    def test_link_ancestors_rejected_when_supported(self):
        link = self.root / "link"
        try:
            link.symlink_to(self.source, target_is_directory=True)
        except OSError:
            self.skipTest("OS account cannot create directory symlinks")
        self.source = link
        self.invoke(success=False)

    def test_product_cli_rejects_synthetic_official_identity(self):
        if CLI is None:
            self.skipTest("CLI target not enabled")
        result = subprocess.run([str(CLI), "data", "prepare", "--name", "fashion-mnist",
                                 "--source-dir", str(self.source), "--data-root", str(self.output),
                                 "--selection", "full"], capture_output=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--cli", type=Path)
    args, remaining = parser.parse_known_args()
    HARNESS, CLI = args.harness.resolve(), args.cli.resolve() if args.cli else None
    unittest.main(argv=[__file__, *remaining])
