#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent native producer checks. Python is never a product fallback."""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
import struct
import subprocess
import tarfile
import tempfile
import zlib

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog",
           "frog", "horse", "ship", "truck")
PREFIX = "cifar-10-batches-bin/"
DESTINATION = "cifar10-imagefolder-v1"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            result.update(block)
    return result.hexdigest()


def run(command, timeout=180, env=None):
    return subprocess.run([str(item) for item in command], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, env=env)


def archive_bytes(members, root=True):
    storage = io.BytesIO()
    with tarfile.open(fileobj=storage, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        if root:
            entry = tarfile.TarInfo(PREFIX)
            entry.type = tarfile.DIRTYPE
            tar.addfile(entry)
        for item in members:
            name, payload = item[:2]
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            if len(item) > 2:
                entry.type, entry.linkname = item[2:]
            tar.addfile(entry, io.BytesIO(payload))
    return storage.getvalue()


def write_fixture(path, contract, raw=None, compressed=None):
    content = gzip.compress(raw, compresslevel=1, mtime=0) if compressed is None else compressed
    path.write_bytes(content)
    result = copy.deepcopy(contract)
    result.update(archive_bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                  official_md5=hashlib.md5(content).hexdigest())
    return result


def native(harness, contract, archive, output, contract_path, env=None):
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return run([harness, contract_path, archive, output], env=env)


def compare_tree(actual, expected):
    actual_names = sorted(path.relative_to(actual).as_posix() for path in actual.rglob("*.png"))
    expected_names = sorted(path.relative_to(expected).as_posix() for path in expected.rglob("*.png"))
    require(actual_names == expected_names, "exact ImageFolder relative names")
    for name in actual_names:
        require((actual / name).read_bytes() == (expected / name).read_bytes(),
                "exact reference PNG bytes: " + name)
    require((actual / "preparation.json").read_bytes() ==
            (expected / "preparation.json").read_bytes(), "exact report ordering/bytes")
    return len(actual_names)


def check_pixels(path, planes):
    content = path.read_bytes()
    require(content[:8] == b"\x89PNG\r\n\x1a\n", "PNG signature")
    cursor, kinds, compressed = 8, [], b""
    while cursor < len(content):
        length, kind = struct.unpack(">I4s", content[cursor:cursor + 8])
        payload = content[cursor + 8:cursor + 8 + length]
        require(len(payload) == length, "PNG payload extent")
        crc, = struct.unpack(">I", content[cursor + 8 + length:cursor + 12 + length])
        require(crc == zlib.crc32(kind + payload) & 0xffffffff, "PNG CRC")
        if kind == b"IHDR":
            require(struct.unpack(">IIBBBBB", payload) == (32, 32, 8, 2, 0, 0, 0), "RGB8 shape")
        elif kind == b"IDAT":
            compressed += payload
        kinds.append(kind)
        cursor += length + 12
    require(cursor == len(content) and kinds == [b"IHDR", b"IDAT", b"IEND"], "single IDAT format")
    rows = zlib.decompress(compressed)
    require(len(rows) == 32 * 97, "scanline extent")
    for y in range(32):
        require(rows[y * 97] == 0, "filter zero")
        for x in range(32):
            for channel in range(3):
                require(rows[y * 97 + 1 + x * 3 + channel] ==
                        planes[channel * 1024 + y * 32 + x], "exact planar RGB byte")


def fixture_checks(args, module, root):
    archive, contract_path = root / "fixture.tar.gz", root / "contract.json"
    planes = bytes((index // 1024 * 71 + index % 251) % 256 for index in range(3072))
    records = b"".join(bytes([label]) + planes for label in range(10))
    members = [(PREFIX + "batches.meta.txt", ("\n".join(CLASSES) + "\n\n").encode()),
               (PREFIX + "data_batch_1.bin", records),
               (PREFIX + "test_batch.bin", records)]
    contract = {"source": "self-authored independent offline fixture; 独立参考",
                "dataset_license": "CC0-1.0; generated fixture",
                "max_decoded_bytes": 65536,
                "samples_per_class": {"train": 1, "validation": 1},
                "batches": {"data_batch_1.bin": {"samples": 10, "split": "train"},
                            "test_batch.bin": {"samples": 10, "split": "validation"}}}
    raw = archive_bytes(members)
    frozen = write_fixture(archive, contract, raw)
    output = root / "原生 图片"
    env = os.environ.copy()
    env["PATH"] = ""
    completed = native(args.harness, frozen, archive, output, contract_path, env)
    require(completed.returncode == 0, completed.stderr)
    actual = output / DESTINATION
    expected = module.prepare(archive, root / "independent-reference", frozen)
    count = compare_tree(actual, expected)
    require(count == 20, "all fixture samples retained")
    for split in ("train", "validation"):
        batch = "data_batch_1" if split == "train" else "test_batch"
        for label, category in enumerate(CLASSES):
            check_pixels(actual / split / category / f"{batch}-{label:05d}.png", planes)
    require(native(args.harness, frozen, archive, output, contract_path).returncode == 1,
            "repeat must refuse replacement")
    require(compare_tree(actual, expected) == 20, "repeat preserves complete dataset")

    failures = []

    def expect_failure(name, bad_raw=None, bad_compressed=None, alter=None):
        current = write_fixture(archive, contract, raw if bad_raw is None else bad_raw,
                                compressed=bad_compressed)
        if alter:
            alter(current)
        target = root / ("negative-" + name)
        target.mkdir()
        sentinel = target / "keep"
        sentinel.write_bytes(b"not owned by preparation")
        result = native(args.harness, current, archive, target, contract_path)
        require(result.returncode == 1, name + ": " + result.stdout + result.stderr)
        require(not (target / DESTINATION).exists(), name + ": no publication")
        require(sorted(path.name for path in target.iterdir()) == ["keep"] and
                sentinel.read_bytes() == b"not owned by preparation", name + ": rollback own stage only")
        failures.append(name)

    expect_failure("compressed-sha", alter=lambda value: value.update(sha256="0" * 64))
    expect_failure("official-md5", alter=lambda value: value.update(official_md5="0" * 32))
    expect_failure("compressed-size", alter=lambda value: value.update(archive_bytes=value["archive_bytes"] + 1))
    expect_failure("decoded-budget", alter=lambda value: value.update(max_decoded_bytes=50000))
    expect_failure("compressed-budget", alter=lambda value: value.update(archive_bytes=170052172))
    expect_failure("missing-batch", archive_bytes(members[:-1]))
    expect_failure("missing-metadata", archive_bytes(members[1:]))
    expect_failure("duplicate-file", archive_bytes(members + members[1:2]))
    expect_failure("duplicate-root", archive_bytes([(PREFIX, b"", tarfile.DIRTYPE, "")] + members))
    for suffix in ("../../escaped", "/absolute", "unexpected.bin", "nested/file.bin", "bad\\name"):
        expect_failure("path-" + str(len(failures)), archive_bytes(members + [(PREFIX + suffix, b"x")]))
    for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.XHDTYPE,
                 tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_SPARSE):
        expect_failure("type-" + str(len(failures)), archive_bytes(
            [(PREFIX + "data_batch_1.bin", b"", kind,
              "../../outside" if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else "")]))
    expect_failure("absolute-path", archive_bytes(members + [("/absolute", b"x")]))
    expect_failure("oversized-metadata", archive_bytes(members + [(PREFIX + "readme.html", b"x" * 65537)]))
    expect_failure("metadata-interior-blank", archive_bytes(
        [(members[0][0], members[0][1].replace(b"cat\n", b"cat\n\n"))] + members[1:]))
    expect_failure("metadata-non-ascii", archive_bytes(
        [(members[0][0], members[0][1].replace(b"cat", b"ca\xff"))] + members[1:]))
    expect_failure("invalid-label", archive_bytes(
        members[:1] + [(members[1][0], bytes([10]) + records[1:])] + members[2:]))
    expect_failure("class-counts", archive_bytes(
        members[:1] + [(members[1][0], bytes([1]) + records[1:])] + members[2:]))
    expect_failure("short-record", archive_bytes(
        members[:1] + [(members[1][0], records[:-1])] + members[2:]))
    expect_failure("long-record", archive_bytes(
        members[:1] + [(members[1][0], records + b"x")] + members[2:]))
    damaged_header = bytearray(raw)
    damaged_header[0] ^= 1
    expect_failure("tar-checksum", bytes(damaged_header))
    # The first member after the root is metadata, followed by zero padding.
    damaged_padding = bytearray(raw)
    damaged_padding[1024 + len(members[0][1])] = 1
    expect_failure("nonzero-member-padding", bytes(damaged_padding))
    # GNU/ustar archive output is rounded to a 10 KiB record. Strip all trailing
    # zero blocks, then add one: native parsing requires two terminator blocks.
    end = len(raw)
    while end >= 512 and raw[end - 512:end] == bytes(512):
        end -= 512
    expect_failure("missing-terminator", raw[:end])
    expect_failure("single-terminator", raw[:end] + bytes(512))
    expect_failure("nonzero-trailing-data", raw + b"unexpected")
    expect_failure("inflated-zero-padding-budget", raw + bytes(200000))
    compressed = gzip.compress(raw, compresslevel=1, mtime=0)
    expect_failure("gzip-truncated", bad_compressed=compressed[:-4])
    expect_failure("gzip-trailing-raw", bad_compressed=compressed + b"raw")
    expect_failure("gzip-concatenated", bad_compressed=compressed + gzip.compress(b"", mtime=0))
    damaged_crc = bytearray(compressed)
    damaged_crc[-8] ^= 1
    expect_failure("gzip-crc", bad_compressed=bytes(damaged_crc))
    damaged_size = bytearray(compressed)
    damaged_size[-4] ^= 1
    expect_failure("gzip-isize", bad_compressed=bytes(damaged_size))
    require(not (root / "escaped").exists() and not (root / "outside").exists(), "no tar extraction escape")

    frozen = write_fixture(archive, contract, raw)
    # Competing native publishers must preserve whichever complete pair wins.
    contract_path.write_text(json.dumps(frozen), encoding="utf-8")
    target = root / "concurrent"
    command = [str(args.harness), str(contract_path), str(archive), str(target)]
    processes = []
    try:
        for _ in range(2):
            processes.append(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        for process in processes:
            process.communicate(timeout=180)
        require(sorted(process.returncode for process in processes) == [0, 1], "no-clobber concurrent publish")
        require(compare_tree(target / DESTINATION, expected) == 20, "concurrent winner intact")
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()

    link_skip = None
    link = root / "linked-source.tar.gz"
    try:
        link.symlink_to(archive)
    except OSError as error:
        link_skip = str(error)
    else:
        result = native(args.harness, frozen, link, root / "linked-output", contract_path)
        require(result.returncode == 1 and not (root / "linked-output").exists(), "reject source symlink")
    repository_output = args.repository / "out" / "cifar10-native-must-not-be-created"
    require(not repository_output.exists(), "reserved repository negative path already exists")
    require(native(args.harness, frozen, archive, repository_output, contract_path).returncode == 1 and
            not repository_output.exists(), "reject output inside repository before mutation")
    if args.cli:
        for extra in ([], ["--selection", "full"], ["--source-dir", root]):
            command = [args.cli, "data", "prepare", "--name", "cifar10", "--data-root", root / "product-invalid"] + extra
            result = run(command)
            require(result.returncode != 0 and not (root / "product-invalid").exists(), "CIFAR-only CLI arguments")
    if args.example:
        require(run([args.example]).returncode == 2, "example usage")
        require(run([args.example, archive, root / "example-invalid"]).returncode == 1 and
                not (root / "example-invalid").exists(), "example uses fixed official identity")
    return {"fixture_images": count, "negative_cases": failures,
            "concurrent_publish": "passed", "source_symlink_skip": link_skip,
            "empty_path_native_execution": "passed"}


def official_checks(args, module, root):
    contract_path = args.repository / "recipes/image-classification/cifar10.dataset.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output = root / "native-official"
    if args.cli:
        command = [args.cli, "data", "prepare", "--name", "cifar10", "--archive", args.official_archive,
                   "--format", "json"]
        producer = "product-cli"
        env = os.environ.copy()
        env["JYPPX_LIUDING_DATA_ROOT"] = str(output)
        env["JYPPX_DATA_ROOT"] = str(root / "legacy-root-must-not-exist")
    else:
        command = [args.harness, contract_path, args.official_archive, output]
        producer = "private-contract-harness"
        env = None
    result = run(command, timeout=3600, env=env)
    require(result.returncode == 0, result.stderr)
    require(not (root / "legacy-root-must-not-exist").exists(), "legacy data root untouched")
    expected = module.prepare(args.official_archive, root / "reference-official", contract)
    actual = output / DESTINATION
    count = compare_tree(actual, expected)
    require(count == 60000, "complete 50,000 train and 10,000 official test images")
    report = json.loads((actual / "preparation.json").read_text(encoding="utf-8"))
    require(report["class_counts"] == {"train": [5000] * 10, "validation": [1000] * 10}, "full split counts")
    return {"images_compared_byte_for_byte": count, "producer": producer,
            "source_sha256": digest(args.official_archive),
            "preparation_sha256": digest(actual / "preparation.json"),
            "python_checker_zlib": zlib.ZLIB_RUNTIME_VERSION}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--example", type=Path)
    parser.add_argument("--official-archive", type=Path)
    parser.add_argument("--official-only", action="store_true")
    args = parser.parse_args()
    for name in ("harness", "reference", "repository", "cli", "example", "official_archive"):
        value = getattr(args, name)
        if value:
            setattr(args, name, value.resolve())
    spec = importlib.util.spec_from_file_location("independent_cifar10_reference", args.reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="liuding-native-cifar10-") as temporary:
        root = Path(temporary)
        require(not args.official_only or args.official_archive, "--official-only requires --official-archive")
        result = {} if args.official_only else {"fixture": fixture_checks(args, module, root)}
        if args.official_archive:
            result["official"] = official_checks(args, module, root)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
