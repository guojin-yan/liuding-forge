#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Check IDX integrity, source selection, lossless RGB and dataset publication."""

import contextlib
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prepare_fashion_mnist", ROOT / "tests/reference/fashion_mnist.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def decode_png(path):
    """Independent strict decoder for the emitted 28x28 RGB, filter-zero PNG."""
    content = path.read_bytes()
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("PNG signature")
    offset, idat, kinds = 8, bytearray(), []
    while offset < len(content):
        size, kind = struct.unpack_from(">I4s", content, offset)
        payload = content[offset + 8:offset + 8 + size]
        crc, = struct.unpack_from(">I", content, offset + 8 + size)
        if len(payload) != size or crc != zlib.crc32(kind + payload) & 0xffffffff:
            raise AssertionError("PNG length/CRC")
        kinds.append(kind)
        if kind == b"IHDR" and struct.unpack(">IIBBBBB", payload) != (28, 28, 8, 2, 0, 0, 0):
            raise AssertionError("PNG geometry/type")
        if kind == b"IDAT":
            idat.extend(payload)
        if kind == b"IEND" and payload:
            raise AssertionError("PNG IEND")
        offset += size + 12
    if offset != len(content) or kinds != [b"IHDR", b"IDAT", b"IEND"]:
        raise AssertionError("PNG chunks/tail")
    decoder = zlib.decompressobj()
    rows = decoder.decompress(idat) + decoder.flush()
    if not decoder.eof or decoder.unused_data or len(rows) != 28 * 85:
        raise AssertionError("PNG deflate length/tail")
    rgb = bytearray()
    for y in range(28):
        if rows[y * 85] != 0:
            raise AssertionError("PNG filter")
        rgb.extend(rows[y * 85 + 1:(y + 1) * 85])
    return bytes(rgb)


class PrepareFashionMnistTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="liuding-fashion-prepare-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "official-gzip"
        self.source.mkdir()
        self.output = self.root / "数据目录"
        self.contract = json.loads(MODULE.CONTRACT.read_text(encoding="utf-8"))
        self.contract["source"] = {"repository": "self-authored synthetic IDX test",
                                   "revision": "not-official-data", "raw_base": "unused"}
        self.raw = {}
        self.configure_samples(2)

    @staticmethod
    def gray(split, index):
        shift = 13 if split == "train" else 101
        return bytes((index * 17 + shift + p * 3 + p // 28) % 256 for p in range(784))

    def configure_samples(self, per_class, grouped=False):
        for split in ("train", "validation"):
            count = per_class * 10
            info = self.contract["files"][split]
            info.update(samples=count, samples_per_class=per_class)
            labels = bytes(i // per_class if grouped else (i * 7 + 3) % 10
                           for i in range(count))
            self.raw[split, "images"] = (struct.pack(">IIII", 2051, count, 28, 28)
                                         + b"".join(self.gray(split, i) for i in range(count)))
            self.raw[split, "labels"] = struct.pack(">II", 2049, count) + labels
        self.write_sources()

    def write_sources(self):
        for (split, kind), content in self.raw.items():
            self.write_compressed(split, kind, gzip.compress(content, mtime=0))

    def write_compressed(self, split, kind, content):
        info = self.contract["files"][split][kind]
        (self.source / info["name"]).write_bytes(content)
        info.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                    official_md5=hashlib.md5(content).hexdigest())

    def prepare(self, selection="full"):
        with contextlib.redirect_stdout(io.StringIO()):
            return MODULE.prepare(self.source, self.output, selection, self.contract)

    def assert_not_published(self):
        self.assertFalse((self.output / "fashion-mnist-full-imagefolder-v1").exists())
        self.assertFalse((self.output / "fashion-mnist-first100-per-class-imagefolder-v1").exists())
        if self.output.exists():
            self.assertEqual(list(self.output.iterdir()), [])

    def test_full_rgb_pixels_class_order_indices_and_provenance(self):
        destination = self.prepare()
        manifest = json.loads((destination / "preparation.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["selection"], "full")
        self.assertEqual(manifest["source"], self.contract["source"])
        self.assertEqual((destination / "fashion-mnist.LICENSE").read_bytes(), MODULE.LICENSE.read_bytes())
        dataset_hash = hashlib.sha256(b"jyppx.liuding.fashion-mnist.imagefolder.v1\0full\0")
        for split in ("train", "validation"):
            directories = sorted(p.name for p in (destination / split).iterdir())
            self.assertEqual(directories, [c["directory"] for c in self.contract["classes"]])
            index_bytes = (destination / "index" / f"{split}.jsonl").read_bytes()
            records = [json.loads(line) for line in index_bytes.splitlines()]
            self.assertEqual([r["source_index"] for r in records], list(range(20)))
            self.assertEqual(manifest["splits"][split]["source_class_counts"], [2] * 10)
            self.assertEqual(manifest["splits"][split]["class_counts"], [2] * 10)
            self.assertEqual(manifest["splits"][split]["index"]["sha256"],
                             hashlib.sha256(index_bytes).hexdigest())
            split_hash = hashlib.sha256(b"jyppx.liuding.fashion-mnist.split.v1\0"
                                        + split.encode("ascii") + b"\0full\0")
            for i, record in enumerate(records):
                label = (i * 7 + 3) % 10
                gray = self.gray(split, i)
                image = destination / record["path"]
                self.assertEqual(record["class_id"], label)
                self.assertEqual(image.parent.name, directories[label])
                self.assertEqual(image.name, f"{i:06d}.png")
                self.assertEqual(decode_png(image), b"".join(bytes([value]) * 3 for value in gray))
                self.assertEqual(record["grayscale_sha256"], hashlib.sha256(gray).hexdigest())
                self.assertEqual(record["png_sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
                split_hash.update(struct.pack("<Q", i) + bytes([label]) + gray)
            self.assertEqual(manifest["splits"][split]["content_sha256"], split_hash.hexdigest())
            dataset_hash.update(split.encode("ascii") + b"\0" + split_hash.digest())
        self.assertEqual(manifest["content_sha256"], dataset_hash.hexdigest())

    def test_subset_selects_first100_for_each_class_with_original_indices(self):
        self.configure_samples(101, grouped=True)
        destination = self.prepare("first100-per-class")
        manifest = json.loads((destination / "preparation.json").read_text(encoding="utf-8"))
        self.assertEqual(len(list(destination.rglob("*.png"))), 2000)
        expected = [label * 101 + i for label in range(10) for i in range(100)]
        for split in ("train", "validation"):
            records = [json.loads(line) for line in
                       (destination / "index" / f"{split}.jsonl").read_bytes().splitlines()]
            self.assertEqual([r["source_index"] for r in records], expected)
            self.assertEqual([r["class_id"] for r in records], [i // 100 for i in range(1000)])
            self.assertEqual(manifest["splits"][split]["source_class_counts"], [101] * 10)
            self.assertEqual(manifest["splits"][split]["class_counts"], [100] * 10)
            for record in (records[0], records[99], records[100], records[-1]):
                gray = self.gray(split, record["source_index"])
                self.assertEqual(decode_png(destination / record["path"]),
                                 b"".join(bytes([value]) * 3 for value in gray))

    def test_subset_validates_unselected_source_records_and_tail(self):
        self.configure_samples(101, grouped=True)
        valid_labels = self.raw["train", "labels"]
        # The final source image is unselected: class 9 already has 100 images.
        self.raw["train", "labels"] = valid_labels[:-1] + bytes([10])
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "invalid.*label"):
            self.prepare("first100-per-class")
        self.assert_not_published()
        self.raw["train", "labels"] = valid_labels + b"\x00"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "extra.*payload"):
            self.prepare("first100-per-class")
        self.assert_not_published()

    def test_source_size_sha256_and_md5_rejected_before_output(self):
        info = self.contract["files"]["train"]["images"]
        path = self.source / info["name"]
        original = path.read_bytes()
        for damaged in (original[:-1], original + b"x", b"x" + original[1:]):
            with self.subTest(bytes=len(damaged)):
                path.write_bytes(damaged)
                with self.assertRaisesRegex(ValueError, "size mismatch|SHA-256"):
                    self.prepare()
                self.assert_not_published()
        path.write_bytes(original)
        info["official_md5"] = "0" * 32
        with self.assertRaisesRegex(ValueError, "MD5"):
            self.prepare()
        self.assert_not_published()

    def test_idx_magic_count_dimensions_and_label_header_rejected(self):
        original = copy.deepcopy(self.raw)
        for kind, header in (("images", (2050, 20, 28, 28)),
                             ("images", (2051, 19, 28, 28)),
                             ("images", (2051, 20, 27, 28)),
                             ("images", (2051, 20, 28, 29)),
                             ("labels", (2048, 20)), ("labels", (2049, 21))):
            with self.subTest(kind=kind, header=header):
                self.raw = copy.deepcopy(original)
                size = 4 * len(header)
                self.raw["train", kind] = struct.pack(">" + "I" * len(header), *header) + original["train", kind][size:]
                self.write_sources()
                with self.assertRaisesRegex(ValueError, "IDX header"):
                    self.prepare()
                self.assert_not_published()

    def test_truncated_and_extra_idx_payloads_rejected(self):
        original = copy.deepcopy(self.raw)
        for kind in ("images", "labels"):
            for mode in ("header", "truncated", "extra"):
                with self.subTest(kind=kind, mode=mode):
                    self.raw = copy.deepcopy(original)
                    raw = original["validation", kind]
                    self.raw["validation", kind] = (raw[:3] if mode == "header"
                                                     else raw[:-1] if mode == "truncated" else raw + b"x")
                    self.write_sources()
                    with self.assertRaisesRegex(ValueError, "truncated|extra"):
                        self.prepare()
                    self.assert_not_published()

    def test_gzip_crc_and_truncated_trailers_rejected(self):
        for kind in ("images", "labels"):
            original = gzip.compress(self.raw["validation", kind], mtime=0)
            bad_crc = bytearray(original)
            bad_crc[-8] ^= 1
            for damaged in (bytes(bad_crc), original[:-3]):
                with self.subTest(kind=kind, bytes=len(damaged)):
                    self.write_compressed("validation", kind, damaged)
                    with self.assertRaises((OSError, EOFError)):
                        self.prepare()
                    self.assert_not_published()
            self.write_compressed("validation", kind, original)

    def test_invalid_labels_unbalanced_counts_and_impossible_subset_rejected(self):
        original = self.raw["train", "labels"]
        for label, message in ((10, "invalid.*label"), (4, "class counts")):
            self.raw["train", "labels"] = original[:-1] + bytes([label])
            self.write_sources()
            with self.assertRaisesRegex(ValueError, message):
                self.prepare()
            self.assert_not_published()
        self.write_compressed("train", "labels", gzip.compress(original, mtime=0))
        with self.assertRaisesRegex(ValueError, "subset quota"):
            self.prepare("first100-per-class")
        self.assert_not_published()

    def test_existing_output_and_failed_publish_preserve_other_data(self):
        self.output.mkdir()
        marker = self.output / "other-dataset.txt"
        marker.write_text("keep", encoding="utf-8")

        def fail_rename(staging, destination):
            self.assertTrue(staging.name.startswith(".fashion-mnist-preparing-"))
            destination.mkdir()
            (destination / "other-owner.txt").write_text("keep", encoding="utf-8")
            raise FileExistsError("injected concurrent publication")

        with patch.object(Path, "rename", new=fail_rename):
            with self.assertRaises(FileExistsError):
                self.prepare()
        destination = self.output / "fashion-mnist-full-imagefolder-v1"
        self.assertEqual((destination / "other-owner.txt").read_text(), "keep")
        self.assertEqual(marker.read_text(), "keep")
        self.assertFalse(list(self.output.glob(".fashion-mnist-preparing-*")))
        with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
            self.prepare()
        self.assertEqual(list(destination.iterdir()), [destination / "other-owner.txt"])

    def test_repository_output_rejected_without_creating_a_directory(self):
        output = ROOT / "out/fashion-mnist-must-not-exist"
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "outside the code repository"):
            MODULE.prepare(self.source, output, "full", self.contract)
        self.assertFalse(output.exists())

    def test_symlink_and_reparse_components_rejected(self):
        original_lstat = Path.lstat
        for location in (self.source, self.source / MODULE.SOURCE_NAMES["train"][0], self.output):
            for link_mode in ("symlink", "reparse"):
                with self.subTest(location=location.name, mode=link_mode):
                    def fake_lstat(path, *args, **kwargs):
                        if path == location:
                            return SimpleNamespace(st_mode=stat.S_IFLNK if link_mode == "symlink" else stat.S_IFDIR,
                                                   st_file_attributes=0 if link_mode == "symlink" else 0x400)
                        return original_lstat(path, *args, **kwargs)
                    with patch.object(Path, "lstat", new=fake_lstat):
                        with self.assertRaisesRegex(ValueError, "must not traverse links"):
                            self.prepare()
                    self.assert_not_published()

    def test_contract_paths_class_mapping_and_license_are_not_inferred(self):
        original = copy.deepcopy(self.contract)
        for change in ("schema", "class", "path", "count", "license"):
            with self.subTest(change=change):
                self.contract = copy.deepcopy(original)
                if change == "schema":
                    self.contract["schema_version"] = 2
                elif change == "class":
                    self.contract["classes"][0]["name"] = "T-shirt"
                elif change == "path":
                    self.contract["files"]["train"]["images"]["name"] = "../escape.gz"
                elif change == "count":
                    self.contract["files"]["train"]["samples"] = 70000
                else:
                    self.contract["license"]["sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assert_not_published()
        self.contract = original
        with self.assertRaisesRegex(ValueError, "selection"):
            self.prepare("random")

    def test_decompressed_reads_stay_bounded_to_one_image(self):
        original_read = gzip.GzipFile.read
        seen = []

        def bounded_read(stream, size=-1):
            self.assertGreater(size, 0, "unbounded gzip read")
            self.assertLessEqual(size, 784, "more than one grayscale image read")
            seen.append(size)
            return original_read(stream, size)

        with patch.object(gzip.GzipFile, "read", new=bounded_read):
            destination = self.prepare()
        self.assertEqual(len(list(destination.rglob("*.png"))), 40)
        self.assertIn(784, seen)

    def test_cli_unicode_environment_explicit_override_and_missing_root(self):
        contract = self.root / "synthetic-contract.json"
        contract.write_text(json.dumps(self.contract), encoding="utf-8")
        argv = ["prepare_fashion_mnist.py", "--source-dir", str(self.source), "--selection", "full"]
        with (patch.object(MODULE, "CONTRACT", contract), patch.object(sys, "argv", argv),
              patch.dict(os.environ, {"JYPPX_LIUDING_DATA_ROOT": str(self.output)}),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(MODULE.main(), 0)
        self.assertIn("verified Fashion-MNIST", output.getvalue())
        explicit = self.root / "另一个输出"
        with (patch.object(MODULE, "CONTRACT", contract),
              patch.object(sys, "argv", argv + ["--data-root", str(explicit)]),
              patch.dict(os.environ, {"JYPPX_LIUDING_DATA_ROOT": str(self.output)}),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(MODULE.main(), 0)
        self.assertTrue((explicit / "fashion-mnist-full-imagefolder-v1").is_dir())
        with (patch.object(sys, "argv", argv), patch.dict(os.environ, {}, clear=True),
              contextlib.redirect_stderr(io.StringIO())):
            with self.assertRaises(SystemExit) as error:
                MODULE.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
