#!/usr/bin/env python3
"""Verify reviewed local FetchContent inputs before an offline coverage configure.

This tool never extracts, downloads, configures, builds or runs a product.
Archive hashes are fixed here; callers supply only absolute local paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import zipfile

PINNED_FETCHCONTENT = {
    "jyppx_nlohmann_json": {
        "sha256": "42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa",
        "archive_root": "json", "format": "tar.xz",
        "cache_option": "FETCHCONTENT_SOURCE_DIR_JYPPX_NLOHMANN_JSON",
    },
    "jyppx_sqlite": {
        "sha256": "0a9c8e34352dad46006aa82836d5383bff9dc6bc9ce5a60553c05aa30e6ad1f9",
        "archive_root": "sqlite-amalgamation-3540100", "format": "zip",
        "cache_option": "FETCHCONTENT_SOURCE_DIR_JYPPX_SQLITE",
    },
    "googletest": {
        "sha256": "1f357c27ca988c3f7c6b4bf68a9395005ac6761f034046e9dde0896e3aba00e4",
        "archive_root": "googletest-1.14.0", "format": "zip",
        "cache_option": "FETCHCONTENT_SOURCE_DIR_GOOGLETEST",
    },
}
OFFLINE_OPTIONS = {
    "FETCHCONTENT_FULLY_DISCONNECTED": "ON",
    "FETCHCONTENT_UPDATES_DISCONNECTED": "ON",
}
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_MEMBERS = 4096


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while raw := stream.read(1024 * 1024):
            result.update(raw)
    return result.hexdigest()


def read_object(path: Path) -> dict:
    require(path.stat().st_size <= 2 * 1024 * 1024, "input JSON exceeds fixed byte bound")
    def unique(pairs):
        result = {}
        for name, value in pairs:
            require(name not in result, "duplicate input JSON key: " + name)
            result[name] = value
        return result
    value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    require(isinstance(value, dict), "input JSON root must be an object")
    return value


def checked_digest(value, label: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None,
            label + " must be a lowercase SHA-256")
    return value


def local_path(value, label: str, *, directory: bool = False) -> Path:
    require(isinstance(value, str) and value and not any(char in value for char in "\r\n;\x00"),
            label + " must be one absolute local path")
    path = Path(value)
    require(path.is_absolute(), label + " must be absolute")
    require(path.resolve(strict=True) == path, label + " must be canonical without aliases")
    info = path.lstat()
    require(not path.is_symlink() and not getattr(info, "st_file_attributes", 0) & 0x400,
            label + " must not be a symlink/reparse path")
    require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
            label + " has the wrong file type")
    if not directory:
        require(info.st_nlink == 1, label + " must be an unaliased regular file")
    return path


def member_relative(name: str, root: str) -> str | None:
    cleaned = name.rstrip("/")
    path = PurePosixPath(cleaned)
    require(cleaned and "\\" not in cleaned and not path.is_absolute() and
            path.as_posix() == cleaned and not any(part in {".", ".."} for part in path.parts),
            "archive member path is unsafe: " + repr(name))
    require(path.parts[0] == root, "archive member has an unexpected top-level root")
    return PurePosixPath(*path.parts[1:]).as_posix() if len(path.parts) > 1 else None


def archive_inventory(path: Path, lock: dict) -> dict[str, dict]:
    require(path.stat().st_size <= MAX_ARCHIVE_BYTES, "archive exceeds fixed byte bound")
    require(sha256_file(path) == lock["sha256"], "locked local archive SHA-256 mismatch")
    files, names, folded = {}, set(), set()
    total = 0

    def add(name, is_directory, size, stream):
        nonlocal total
        relative = member_relative(name, lock["archive_root"])
        normalized = name.rstrip("/")
        require(normalized not in names and normalized.casefold() not in folded, "duplicate/aliased archive member")
        names.add(normalized)
        folded.add(normalized.casefold())
        require(len(names) <= MAX_MEMBERS, "archive member count exceeds fixed bound")
        if is_directory:
            return
        require(relative is not None and 0 <= size <= MAX_MEMBER_BYTES, "archive member exceeds fixed byte bound")
        total += size
        require(total <= MAX_TOTAL_BYTES, "archive expanded bytes exceed fixed bound")
        raw = stream.read(MAX_MEMBER_BYTES + 1)
        require(len(raw) == size, "archive member byte length mismatch")
        files[relative] = {"bytes": size, "sha256": digest(raw)}

    if lock["format"] == "tar.xz":
        with tarfile.open(path, "r:xz") as archive:
            for entry in archive:
                require(entry.isdir() or entry.isfile(), "archive contains a link or special member")
                if entry.isdir():
                    add(entry.name, True, 0, None)
                else:
                    with archive.extractfile(entry) as stream:
                        add(entry.name, False, entry.size, stream)
    elif lock["format"] == "zip":
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                mode = entry.external_attr >> 16
                require(not entry.flag_bits & 1 and (entry.is_dir() or stat.S_IFMT(mode) in {0, stat.S_IFREG}),
                        "archive contains an encrypted, link or special member")
                if entry.is_dir():
                    add(entry.filename, True, 0, None)
                else:
                    with archive.open(entry) as stream:
                        add(entry.filename, False, entry.file_size, stream)
    else:
        raise ValueError("unsupported locked archive format")
    require(bool(files), "locked archive contains no regular files")
    require(sha256_file(path) == lock["sha256"], "archive identity changed while reading")
    return files


def source_inventory(root: Path) -> dict[str, dict]:
    result, folded = {}, set()
    nodes, total = 0, 0
    for path in sorted(root.rglob("*")):
        nodes += 1
        require(nodes <= MAX_MEMBERS * 2, "source tree node count exceeds fixed bound")
        info = path.lstat()
        require(not path.is_symlink() and not getattr(info, "st_file_attributes", 0) & 0x400,
                "source tree contains a link/reparse member")
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "source tree file is not unaliased regular data")
        relative = path.relative_to(root).as_posix()
        require(relative.casefold() not in folded, "source tree has case aliases")
        folded.add(relative.casefold())
        require(info.st_size <= MAX_MEMBER_BYTES, "source member exceeds fixed byte bound")
        total += info.st_size
        require(total <= MAX_TOTAL_BYTES and len(result) < MAX_MEMBERS,
                "source tree exceeds fixed file/byte bound")
        result[relative] = {"bytes": info.st_size, "sha256": sha256_file(path)}
    return result


def inventory_digest(files: dict[str, dict]) -> str:
    return digest("".join(name + "\0" + str(row["bytes"]) + "\0" + row["sha256"] + "\n"
                          for name, row in sorted(files.items())).encode("utf-8"))


def verify_inputs(inputs_file: Path, inputs_sha256: str, init_cache: Path, init_cache_sha256: str,
                  *, locks: dict | None = None) -> dict:
    locks = PINNED_FETCHCONTENT if locks is None else locks
    inputs_file = local_path(str(inputs_file), "reviewed input descriptor")
    init_cache = local_path(str(init_cache), "reviewed init-cache")
    inputs_sha256 = checked_digest(inputs_sha256, "reviewed input descriptor digest")
    init_cache_sha256 = checked_digest(init_cache_sha256, "reviewed init-cache digest")
    require(sha256_file(inputs_file) == inputs_sha256, "reviewed input descriptor SHA-256 mismatch")
    require(sha256_file(init_cache) == init_cache_sha256, "reviewed init-cache SHA-256 mismatch")
    document = read_object(inputs_file)
    require(set(document) == {"schema_version", "kind", "init_cache_sha256", "fetchcontent"}
            and type(document["schema_version"]) is int and document["schema_version"] == 1
            and document["kind"] == "jyppx-coverage-offline-inputs", "offline input descriptor schema mismatch")
    require(document["init_cache_sha256"] == init_cache_sha256, "descriptor is bound to a different init-cache")
    declarations = document["fetchcontent"]
    require(isinstance(declarations, dict) and set(declarations) == set(locks),
            "offline descriptor must name every locked FetchContent source exactly once")
    sources = {}
    for name, lock in locks.items():
        checked_digest(lock["sha256"], "locked archive digest")
        item = declarations[name]
        require(isinstance(item, dict) and set(item) == {"archive", "source_root"}, "invalid offline source descriptor")
        archive = local_path(item["archive"], name + " archive")
        source = local_path(item["source_root"], name + " pre-expanded source", directory=True)
        expected = archive_inventory(archive, lock)
        observed = source_inventory(source)
        require(observed == expected, name + " source tree does not exactly match its locked archive")
        sources[name] = {"archive": str(archive), "archive_sha256": lock["sha256"], "source_root": str(source),
            "source_tree_sha256": inventory_digest(observed), "file_count": len(observed),
            "total_bytes": sum(row["bytes"] for row in observed.values()), "cache_option": lock["cache_option"]}
    require(sha256_file(inputs_file) == inputs_sha256 and sha256_file(init_cache) == init_cache_sha256,
            "reviewed input identity changed during preflight")
    return {"schema_version": 1, "kind": "jyppx-coverage-offline-input-receipt",
        "inputs_file": str(inputs_file), "inputs_sha256": inputs_sha256,
        "init_cache": str(init_cache), "init_cache_sha256": init_cache_sha256,
        "sources": sources, "offline_options": dict(OFFLINE_OPTIONS),
        "coverage_evaluated": False, "release_qualified": False}


def verify_cache(cache: Path, receipt: dict) -> None:
    require(receipt.get("kind") == "jyppx-coverage-offline-input-receipt", "invalid offline input receipt")
    required = {name: ("BOOL", value) for name, value in OFFLINE_OPTIONS.items()}
    required.update({row["cache_option"]: ("PATH", Path(row["source_root"]).as_posix())
                     for row in receipt["sources"].values()})
    entries = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        name_type, value = line.split("=", 1)
        name, kind = name_type.split(":", 1)
        if name in required:
            require(name not in entries, "offline CMake cache repeats key: " + name)
            entries[name] = (kind, value)
    for name, expected in required.items():
        require(entries.get(name) == expected, "offline CMake cache option missing/type/value changed: " + name)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--inputs-file", type=Path, required=True)
    prepare.add_argument("--inputs-sha256", required=True)
    prepare.add_argument("--init-cache", type=Path, required=True)
    prepare.add_argument("--init-cache-sha256", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--cmake-args-output", type=Path, required=True)
    verify = sub.add_parser("verify-cache")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--cmake-cache", type=Path, required=True)
    recheck = sub.add_parser("recheck")
    recheck.add_argument("--receipt", type=Path, required=True)
    recheck.add_argument("--cmake-cache", type=Path, required=True)
    recheck.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        receipt = verify_inputs(args.inputs_file, args.inputs_sha256, args.init_cache, args.init_cache_sha256)
        write_json(args.output, receipt)
        args.cmake_args_output.parent.mkdir(parents=True, exist_ok=True)
        with args.cmake_args_output.open("x", encoding="utf-8", newline="\n") as stream:
            for row in receipt["sources"].values():
                stream.write("-D" + row["cache_option"] + ":PATH=" + Path(row["source_root"]).as_posix() + "\n")
    else:
        receipt = read_object(args.receipt)
        verify_cache(args.cmake_cache, receipt)
        if args.action == "recheck":
            current = verify_inputs(Path(receipt["inputs_file"]), receipt["inputs_sha256"],
                                    Path(receipt["init_cache"]), receipt["init_cache_sha256"])
            require(current == receipt, "offline dependency identity changed during collection")
            write_json(args.output, current)
    print("Offline dependency inputs verified; no coverage or release qualification granted")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from None
