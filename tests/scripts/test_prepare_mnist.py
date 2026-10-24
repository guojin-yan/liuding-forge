#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Exercise dataset cache publication and failure paths without network access."""

import gzip
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("prepare_mnist", ROOT / "scripts/datasets/prepare_mnist.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Response(io.BytesIO):
    def geturl(self):
        return "https://example.invalid/mnist/fixture.gz"


class PrepareMnistTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="liuding-mnist-cache-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.decoded = b"deterministic test IDX payload" * 8
        self.encoded = gzip.compress(self.decoded, mtime=0)
        self.entry = {
            "name": "fixture.gz", "size_bytes": len(self.encoded),
            "sha256": hashlib.sha256(self.encoded).hexdigest(),
            "decoded_size_bytes": len(self.decoded),
        }

    def prepare(self, offline=False):
        return MODULE.prepare_file(self.directory, "https://example.invalid/mnist/", self.entry, offline)

    def test_verified_download_and_repeated_offline_cache(self):
        with patch.object(MODULE.urllib.request, "urlopen", return_value=Response(self.encoded)) as fetch:
            destination = self.prepare()
            fetch.assert_called_once()
        self.assertEqual(destination.read_bytes(), self.decoded)
        with patch.object(MODULE.urllib.request, "urlopen", side_effect=AssertionError("network in offline run")):
            self.assertEqual(self.prepare(offline=True), destination)
            destination.write_bytes(b"damaged raw cache")
            self.prepare(offline=True)
            self.assertEqual(destination.read_bytes(), self.decoded)

    def test_missing_offline_archive_never_fetches(self):
        with patch.object(MODULE.urllib.request, "urlopen", side_effect=AssertionError("network in offline run")):
            with self.assertRaisesRegex(FileNotFoundError, "offline"):
                self.prepare(offline=True)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_bad_download_is_not_published(self):
        for data in (b"truncated", self.encoded[:-1] + b"x", self.encoded + b"extra"):
            with self.subTest(size=len(data)):
                with patch.object(MODULE.urllib.request, "urlopen", return_value=Response(data)):
                    with self.assertRaisesRegex(ValueError, "SHA-256"):
                        self.prepare()
                self.assertEqual(list(self.directory.iterdir()), [])

    def test_corrupted_cached_archive_is_preserved_and_rejected(self):
        archive = self.directory / self.entry["name"]
        archive.write_bytes(b"damaged archive")
        with patch.object(MODULE.urllib.request, "urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                self.prepare()
        self.assertEqual(archive.read_bytes(), b"damaged archive")
        self.assertFalse(archive.with_suffix("").exists())

    def test_decode_limit_and_gzip_crc_do_not_publish(self):
        self.entry["decoded_size_bytes"] -= 1
        with patch.object(MODULE.urllib.request, "urlopen", return_value=Response(self.encoded)):
            with self.assertRaisesRegex(ValueError, "decoded size"):
                self.prepare()
        self.assertEqual(list(self.directory.iterdir()), [])
        self.entry["decoded_size_bytes"] += 1
        corrupted = self.encoded[:-8] + bytes(8)
        self.entry["sha256"] = hashlib.sha256(corrupted).hexdigest()
        with patch.object(MODULE.urllib.request, "urlopen", return_value=Response(corrupted)):
            with self.assertRaises(gzip.BadGzipFile):
                self.prepare()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_failed_publication_preserves_existing_file_and_cleans_temporary(self):
        destination = self.directory / "existing"
        destination.write_bytes(b"original")
        with patch.object(MODULE.os, "replace", side_effect=OSError("injected publication failure")):
            with self.assertRaisesRegex(OSError, "publication failure"):
                MODULE.atomic_write(destination, b"replacement")
        self.assertEqual(destination.read_bytes(), b"original")
        self.assertEqual(list(self.directory.iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
