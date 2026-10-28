#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Verify CIFAR binary conversion, data integrity and publication failures."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zlib


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("prepare_cifar10", ROOT / "scripts/datasets/prepare_cifar10.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareCifar10Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="liuding-cifar-prepare-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "source.tar.gz"
        self.output = self.root / "datasets"
        self.planes = bytes((n // 1024 * 71 + n % 251) % 256 for n in range(3072))
        record = lambda label: bytes([label]) + self.planes
        self.members = [("batches.meta.txt", ("\n".join(MODULE.CLASSES) + "\n").encode()),
                        ("data_batch_1.bin", b"".join(record(label) for label in range(10))),
                        ("test_batch.bin", b"".join(record(label) for label in range(10)))]
        self.contract = {"source": "self-authored offline test; not official data",
                         "dataset_license": "CC0-1.0; generated fixture",
                         "max_decoded_bytes": 65536,
                         "samples_per_class": {"train": 1, "validation": 1},
                         "batches": {"data_batch_1.bin": {"samples": 10, "split": "train"},
                                     "test_batch.bin": {"samples": 10, "split": "validation"}}}
        self.write_archive()

    def write_archive(self, members=None):
        with tarfile.open(self.archive, "w:gz") as archive:
            for key, content in self.members if members is None else members:
                member = tarfile.TarInfo("cifar-10-batches-bin/" + key)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        content = self.archive.read_bytes()
        self.contract.update(archive_bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                             official_md5=hashlib.md5(content).hexdigest())

    def prepare(self):
        return MODULE.prepare(self.archive, self.output, self.contract)

    def assert_not_published(self):
        self.assertFalse((self.output / "cifar10-imagefolder-v1").exists())
        if self.output.exists():
            self.assertEqual(list(self.output.iterdir()), [])

    def test_lossless_rgb_labels_and_balanced_split_publication(self):
        destination = self.prepare()
        self.assertEqual(len(list(destination.rglob("*.png"))), 20)
        image = (destination / "train/bird/data_batch_1-00002.png").read_bytes()
        self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
        offset = 8
        compressed = b""
        while offset < len(image):
            length, kind = struct.unpack(">I4s", image[offset:offset + 8])
            content = image[offset + 8:offset + 8 + length]
            crc, = struct.unpack(">I", image[offset + 8 + length:offset + 12 + length])
            self.assertEqual(crc, zlib.crc32(kind + content) & 0xffffffff)
            if kind == b"IHDR":
                self.assertEqual(struct.unpack(">IIBBBBB", content), (32, 32, 8, 2, 0, 0, 0))
            elif kind == b"IDAT":
                compressed += content
            offset += length + 12
        rows = zlib.decompress(compressed)
        self.assertEqual(len(rows), 32 * 97)
        for y in range(32):
            self.assertEqual(rows[y * 97], 0)
            for x in range(32):
                for channel in range(3):
                    self.assertEqual(rows[y * 97 + 1 + x * 3 + channel],
                                     self.planes[channel * 1024 + y * 32 + x])
        summary = json.loads((destination / "preparation.json").read_text())
        self.assertEqual(summary["class_counts"], {"train": [1] * 10, "validation": [1] * 10})
        self.assertEqual(summary["archive"]["sha256"], self.contract["sha256"])
        with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
            self.prepare()
        self.assertEqual(len(list(destination.rglob("*.png"))), 20)

    def test_damaged_archive_never_creates_outputs(self):
        before = self.archive.read_bytes()
        for content in (before[:-1], before + b"extra", b"x" + before[1:]):
            with self.subTest(size=len(content)):
                self.archive.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "size mismatch|SHA-256"):
                    self.prepare()
                self.assert_not_published()
        self.archive.write_bytes(before)

    def test_missing_duplicate_unsafe_and_unknown_members_rejected(self):
        cases = [self.members[:-1], self.members + self.members[:1],
                 self.members + [("../../escaped", b"x")],
                 self.members + [("/absolute", b"x")],
                 self.members + [("unexpected.bin", b"x")],
                 self.members + [("nested/file.bin", b"x")]]
        for members in cases:
            with self.subTest(last=members[-1][0]):
                self.write_archive(members)
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assert_not_published()
        self.assertFalse((self.root / "escaped").exists())

    def test_labels_metadata_size_and_class_counts_rejected(self):
        original = self.members
        for change in ("label", "mapping", "truncated", "counts", "oversized"):
            self.members = copy.deepcopy(original)
            if change == "label":
                key, data = self.members[1]
                self.members[1] = (key, bytes([10]) + data[1:])
            elif change == "mapping":
                self.members[0] = ("batches.meta.txt", b"wrong\n")
            else:
                key, data = self.members[1]
                self.members[1] = (key, data[:-1] if change == "truncated" else
                                  (bytes([1]) + data[1:] if change == "counts" else data + b"x"))
            self.write_archive()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare()
            self.assert_not_published()

    def test_decoded_budget_and_publication_failure_clean_only_staging(self):
        self.contract["max_decoded_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "byte limit"):
            self.prepare()
        self.assert_not_published()
        self.contract["max_decoded_bytes"] = 65536
        with patch.object(MODULE.Path, "rename", side_effect=OSError("injected publish failure")):
            with self.assertRaisesRegex(OSError, "injected publish"):
                self.prepare()
        self.assert_not_published()
        self.assertTrue(self.archive.is_file())

    def test_tar_symlink_member_cannot_redirect_output(self):
        with tarfile.open(self.archive, "w:gz") as archive:
            member = tarfile.TarInfo("cifar-10-batches-bin/data_batch_1.bin")
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
            archive.addfile(member)
        content = self.archive.read_bytes()
        self.contract.update(archive_bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                             official_md5=hashlib.md5(content).hexdigest())
        with self.assertRaisesRegex(ValueError, "non-file"):
            self.prepare()
        self.assert_not_published()

    def test_repository_output_and_invalid_pixel_payload_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the code repository"):
            MODULE.prepare(self.archive, ROOT / "out/cifar-preparation-must-not-exist", self.contract)
        with self.assertRaisesRegex(ValueError, "3072"):
            MODULE.rgb_png(bytes(3071))

    def test_cli_uses_product_data_root_and_keeps_legacy_root_untouched(self):
        contract_file = self.root / "fixture-contract.json"
        contract_file.write_text(json.dumps(self.contract), encoding="utf-8")
        product_root = self.root / "图片数据"
        legacy_root = self.root / "legacy-must-not-be-used"
        with patch.object(MODULE, "CONTRACT", contract_file), patch.dict(
                MODULE.os.environ, {"JYPPX_LIUDING_DATA_ROOT": str(product_root),
                                    "JYPPX_DATA_ROOT": str(legacy_root)}, clear=True), patch(
                "sys.argv", ["prepare_cifar10.py", "--archive", str(self.archive)]):
            self.assertEqual(MODULE.main(), 0)
        self.assertTrue((product_root / "cifar10-imagefolder-v1/preparation.json").is_file())
        self.assertFalse(legacy_root.exists())


if __name__ == "__main__":
    unittest.main()
