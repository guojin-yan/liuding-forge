"""Synthetic offline-input tests; no CMake, SDK, download or instrumentation."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("coverage_dependencies", ROOT / "scripts/validation/coverage_dependencies.py")
assert SPEC and SPEC.loader
DEPENDENCIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPENDENCIES)


def synthetic_inputs(root: Path, init_cache: Path | None = None) -> dict:
    """Only test callers may substitute these synthetic archive locks."""
    root.mkdir(parents=True, exist_ok=True)
    init_cache = root / "fixture-init-cache.cmake" if init_cache is None else init_cache
    if not init_cache.exists():
        init_cache.write_text("# Synthetic fixture; never executed by CMake.\n", encoding="utf-8")
    locks = copy.deepcopy(DEPENDENCIES.PINNED_FETCHCONTENT)
    declarations = {}
    for name, lock in locks.items():
        source = root / (name + " source")
        source.mkdir()
        files = {"CMakeLists.txt": b"# Synthetic source fixture, never configured.\n",
                 "include/fixture.h": ("// synthetic " + name + "; never compiled\n").encode()}
        for relative, raw in files.items():
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        archive_path = root / (name + (".tar.xz" if lock["format"] == "tar.xz" else ".zip"))
        if lock["format"] == "tar.xz":
            with tarfile.open(archive_path, "w:xz") as archive:
                for relative, raw in files.items():
                    member = tarfile.TarInfo(lock["archive_root"] + "/" + relative)
                    member.size, member.mode = len(raw), 0o644
                    archive.addfile(member, io.BytesIO(raw))
        else:
            with zipfile.ZipFile(archive_path, "w") as archive:
                for relative, raw in files.items():
                    archive.writestr(lock["archive_root"] + "/" + relative, raw)
        lock["sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        declarations[name] = {"archive": str(archive_path), "source_root": str(source)}
    document = {"schema_version": 1, "kind": "jyppx-coverage-offline-inputs",
        "init_cache_sha256": hashlib.sha256(init_cache.read_bytes()).hexdigest(), "fetchcontent": declarations}
    inputs_file = root / "synthetic-offline-inputs.json"
    inputs_file.write_text(json.dumps(document), encoding="utf-8")
    return {"locks": locks, "document": document, "inputs_file": inputs_file, "init_cache": init_cache}


class CoverageDependenciesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-coverage-dependencies-")
        self.root = Path(temporary.name).resolve()
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        self.addCleanup(temporary.cleanup)
        self.fixture = synthetic_inputs(self.root)

    def verify(self):
        return DEPENDENCIES.verify_inputs(self.fixture["inputs_file"],
            hashlib.sha256(self.fixture["inputs_file"].read_bytes()).hexdigest(), self.fixture["init_cache"],
            hashlib.sha256(self.fixture["init_cache"].read_bytes()).hexdigest(), locks=self.fixture["locks"])

    def save_document(self):
        self.fixture["inputs_file"].write_text(json.dumps(self.fixture["document"]), encoding="utf-8")

    def cache(self, receipt):
        lines = [name + ":BOOL=ON" for name in DEPENDENCIES.OFFLINE_OPTIONS]
        lines.extend(row["cache_option"] + ":PATH=" + Path(row["source_root"]).as_posix()
                     for row in receipt["sources"].values())
        path = self.root / "synthetic-CMakeCache-not-actual.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path, lines

    def test_production_archive_pins_match_actual_cmake_contract(self):
        text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8") + (ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
        for lock in DEPENDENCIES.PINNED_FETCHCONTENT.values():
            DEPENDENCIES.checked_digest(lock["sha256"], "production pin")
            self.assertIn(lock["sha256"], text)

    def test_exact_archives_and_sources_pass_without_qualification(self):
        receipt = self.verify()
        self.assertEqual(len(receipt["sources"]), 3)
        self.assertFalse(receipt["coverage_evaluated"])
        self.assertFalse(receipt["release_qualified"])
        cache, _ = self.cache(receipt)
        DEPENDENCIES.verify_cache(cache, receipt)

    def test_reviewed_descriptor_cache_hashes_and_binding_are_required(self):
        kwargs = {"inputs_file": self.fixture["inputs_file"], "inputs_sha256": "0" * 64,
            "init_cache": self.fixture["init_cache"],
            "init_cache_sha256": self.fixture["document"]["init_cache_sha256"], "locks": self.fixture["locks"]}
        with self.assertRaisesRegex(ValueError, "descriptor SHA-256 mismatch"):
            DEPENDENCIES.verify_inputs(**kwargs)
        kwargs["inputs_sha256"] = hashlib.sha256(self.fixture["inputs_file"].read_bytes()).hexdigest()
        kwargs["init_cache_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "init-cache SHA-256 mismatch"):
            DEPENDENCIES.verify_inputs(**kwargs)
        self.fixture["document"]["init_cache_sha256"] = "0" * 64
        self.save_document()
        with self.assertRaisesRegex(ValueError, "different init-cache"):
            self.verify()

    def test_every_fetchcontent_dependency_is_mandatory(self):
        del self.fixture["document"]["fetchcontent"]["googletest"]
        self.save_document()
        with self.assertRaisesRegex(ValueError, "every locked FetchContent"):
            self.verify()

    def test_missing_archive_or_source_fails_before_configure(self):
        item = self.fixture["document"]["fetchcontent"]["googletest"]
        for field in ("archive", "source_root"):
            original = item[field]
            item[field] = str(self.root / "missing")
            self.save_document()
            with self.subTest(field=field), self.assertRaises(OSError):
                self.verify()
            item[field] = original

    def test_archive_bytes_cannot_be_replaced_with_another_digest(self):
        path = Path(self.fixture["document"]["fetchcontent"]["googletest"]["archive"])
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "locked local archive SHA-256 mismatch"):
            self.verify()

    def test_modified_missing_extra_source_files_reject(self):
        source = Path(self.fixture["document"]["fetchcontent"]["googletest"]["source_root"])
        target = source / "CMakeLists.txt"
        original = target.read_bytes()
        target.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self.verify()
        target.unlink()
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self.verify()
        target.write_bytes(original)
        extra = source / "extra.cpp"
        extra.write_bytes(b"// synthetic extra\n")
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self.verify()

    def test_archive_traversal_and_links_reject_without_extraction(self):
        path = self.root / "unsafe.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("fixture/../escape.h", b"no extraction")
        lock = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "archive_root": "fixture", "format": "zip"}
        with self.assertRaisesRegex(ValueError, "unsafe"):
            DEPENDENCIES.archive_inventory(path, lock)
        path = self.root / "link.tar.xz"
        with tarfile.open(path, "w:xz") as archive:
            member = tarfile.TarInfo("fixture/link")
            member.type, member.linkname = tarfile.SYMTYPE, "outside"
            archive.addfile(member)
        lock.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), format="tar.xz")
        with self.assertRaisesRegex(ValueError, "link or special"):
            DEPENDENCIES.archive_inventory(path, lock)

    def test_hardlinked_source_is_not_accepted(self):
        target = Path(self.fixture["document"]["fetchcontent"]["googletest"]["source_root"]) / "CMakeLists.txt"
        os.link(target, self.root / "alias")
        with self.assertRaisesRegex(ValueError, "unaliased"):
            self.verify()

    def test_offline_flags_require_exact_bool_on_and_uniqueness(self):
        receipt = self.verify()
        cache, lines = self.cache(receipt)
        for name in DEPENDENCIES.OFFLINE_OPTIONS:
            correct = name + ":BOOL=ON"
            for replacement in (None, name + ":BOOL=OFF", name + ":STRING=ON", name + ":BOOL=TRUE"):
                altered = [replacement if line == correct else line for line in lines]
                cache.write_text("\n".join(line for line in altered if line is not None) + "\n")
                with self.subTest(name=name, replacement=replacement), self.assertRaisesRegex(ValueError, "missing/type/value"):
                    DEPENDENCIES.verify_cache(cache, receipt)
        cache.write_text("\n".join(lines + [lines[0]]) + "\n")
        with self.assertRaisesRegex(ValueError, "repeats key"):
            DEPENDENCIES.verify_cache(cache, receipt)

    def test_configured_local_source_paths_are_exact_and_typed(self):
        receipt = self.verify()
        cache, lines = self.cache(receipt)
        for row in receipt["sources"].values():
            correct = row["cache_option"] + ":PATH=" + Path(row["source_root"]).as_posix()
            for replacement in (None, correct.replace(":PATH=", ":STRING="), row["cache_option"] + ":PATH=/wrong"):
                cache.write_text("\n".join(replacement if line == correct else line for line in lines
                    if line != correct or replacement is not None) + "\n")
                with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, "missing/type/value"):
                    DEPENDENCIES.verify_cache(cache, receipt)

    def test_fixed_resource_bounds_reject_oversized_member(self):
        previous = DEPENDENCIES.MAX_MEMBER_BYTES
        try:
            DEPENDENCIES.MAX_MEMBER_BYTES = 1
            with self.assertRaisesRegex(ValueError, "fixed byte bound"):
                self.verify()
        finally:
            DEPENDENCIES.MAX_MEMBER_BYTES = previous

    def test_descriptor_unknown_fields_duplicate_keys_and_boolean_version_reject(self):
        self.fixture["document"]["caller_supplied_pin"] = "untrusted"
        self.save_document()
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            self.verify()
        del self.fixture["document"]["caller_supplied_pin"]
        self.fixture["document"]["schema_version"] = True
        self.save_document()
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            self.verify()
        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate input JSON key"):
            DEPENDENCIES.read_object(duplicate)

    def test_end_recheck_detects_source_identity_changes(self):
        before = self.verify()
        self.assertEqual(before, self.verify())
        target = Path(self.fixture["document"]["fetchcontent"]["jyppx_sqlite"]["source_root"]) / "CMakeLists.txt"
        target.write_bytes(target.read_bytes() + b"// changed during fixture collection\n")
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
