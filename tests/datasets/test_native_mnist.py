#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent IDX byte/reference and process checks, never a product fallback."""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import json
import os
import struct
import subprocess
import tempfile

NAMES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
SOURCE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
CHECKSUM = ("https://github.com/tensorflow/datasets/blob/master/"
            "tensorflow_datasets/url_checksums/mnist.txt")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(command, env=None):
    return subprocess.run([str(item) for item in command], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=180, env=env)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            result.update(block)
    return result.hexdigest()


def make_fixture():
    decoded = {}
    for prefix, count in (("train", 13), ("t10k", 7)):
        images = bytes((i * 17 + i // 784) % 256 for i in range(count * 784))
        decoded[f"{prefix}-images-idx3-ubyte.gz"] = struct.pack(">IIII", 2051, count, 28, 28) + images
        decoded[f"{prefix}-labels-idx1-ubyte.gz"] = struct.pack(">II", 2049, count) + bytes(i % 10 for i in range(count))
    return decoded


def write_inputs(directory, decoded, overrides=None):
    directory.mkdir()
    entries = []
    for index, name in enumerate(NAMES):
        raw = decoded[name]
        encoded = (overrides or {}).get(name, gzip.compress(raw, mtime=0))
        (directory / name).write_bytes(encoded)
        samples = 13 if index < 2 else 7
        entries.append({"name": name, "compressed_bytes": len(encoded),
                        "compressed_sha256": hashlib.sha256(encoded).hexdigest(),
                        "decoded_bytes": 16 + samples * 784 if index % 2 == 0 else 8 + samples,
                        "decoded_sha256": hashlib.sha256(raw).hexdigest()})
    return {"train_samples": 13, "test_samples": 7, "max_output_bytes": 54950048,
            "files": entries}


def native(args, source, target, contract, contract_path, env=None):
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return run([args.harness, contract_path, source, target], env)


def report_reference(contract, counts):
    return {"schema_version": 1, "dataset_id": "jyppx.liuding.mnist.idx/v1",
            "source": SOURCE, "checksum_reference": CHECKSUM,
            "samples": {"train": contract["train_samples"], "test": contract["test_samples"]},
            "label_counts": counts, "pixels": "28x28 grayscale uint8; unchanged IDX bytes",
            "split_policy": "complete official train/test; no subset",
            "files": [{"source_file": entry["name"], "source_bytes": entry["compressed_bytes"],
                       "source_sha256": entry["compressed_sha256"],
                       "idx_file": entry["name"][:-3], "idx_bytes": entry["decoded_bytes"],
                       "idx_sha256": entry["decoded_sha256"]} for entry in contract["files"]]}


def compare_fixture(output, decoded, contract):
    expected_names = sorted([name[:-3] for name in NAMES] + ["preparation.json"])
    require(sorted(path.name for path in output.iterdir()) == expected_names, "only four IDX and report")
    for name, raw in decoded.items():
        require((output / name[:-3]).read_bytes() == raw, "independent exact IDX bytes: " + name)
    counts = {split: [0] * 10 for split in ("train", "test")}
    for split, name in (("train", NAMES[1]), ("test", NAMES[3])):
        for label in decoded[name][8:]:
            counts[split][label] += 1
    expected = (json.dumps(report_reference(contract, counts), indent=2) + "\n").encode()
    require((output / "preparation.json").read_bytes() == expected, "stable binary LF report and identities")


def fixture_checks(args, root):
    decoded = make_fixture()
    source, output = root / "local source", root / "原生 IDX output"
    contract = write_inputs(source, decoded)
    contract_path = root / "contract.json"
    original = {name: digest(source / name) for name in NAMES}
    env = os.environ.copy()
    env["PATH"] = ""
    result = native(args, source, output, contract, contract_path, env)
    require(result.returncode == 0, result.stderr)
    compare_fixture(output / "mnist", decoded, contract)
    require(native(args, source, output, contract, contract_path).returncode == 1, "repeat refuses replacement")
    compare_fixture(output / "mnist", decoded, contract)
    negative = []

    def expect_failure(name, changed=None, encoded=None, mutate=None):
        data = copy.deepcopy(decoded)
        if changed:
            changed(data)
        case_source = root / ("source-" + name)
        current = write_inputs(case_source, data, encoded)
        if mutate:
            mutate(current)
        target = root / ("output-" + name)
        target.mkdir()
        (target / "keep").write_bytes(b"not owned by preparation")
        result = native(args, case_source, target, current, contract_path)
        require(result.returncode == 1, name + ": " + result.stdout + result.stderr)
        require(sorted(path.name for path in target.iterdir()) == ["keep"], name + ": rollback own stage only")
        require((target / "keep").read_bytes() == b"not owned by preparation", name + ": sentinel intact")
        negative.append(name)

    expect_failure("source-sha", mutate=lambda c: c["files"][3].update(compressed_sha256="0" * 64))
    expect_failure("source-size", mutate=lambda c: c["files"][0].update(compressed_bytes=1))
    expect_failure("source-budget", mutate=lambda c: c["files"][0].update(compressed_bytes=9912423))
    expect_failure("decoded-sha", mutate=lambda c: c["files"][2].update(decoded_sha256="0" * 64))
    expect_failure("output-budget", mutate=lambda c: c.update(max_output_bytes=1))
    expect_failure("invalid-train-count", mutate=lambda c: c.update(train_samples=0))
    expect_failure("invalid-test-count", mutate=lambda c: c.update(test_samples=10001))
    expect_failure("contract-path-escape", mutate=lambda c: c["files"][0].update(name="../escaped"))
    expect_failure("contract-decoded-length", mutate=lambda c: c["files"][0].update(decoded_bytes=17))
    expect_failure("contract-duplicate", mutate=lambda c: c["files"][1].update(name=NAMES[0]))

    def replace(name, offset, data):
        return lambda values: values.update({name: values[name][:offset] + data + values[name][offset + len(data):]})

    expect_failure("image-magic", replace(NAMES[0], 0, struct.pack(">I", 2049)))
    expect_failure("label-magic", replace(NAMES[1], 0, struct.pack(">I", 2051)))
    expect_failure("image-count", replace(NAMES[0], 4, struct.pack(">I", 12)))
    expect_failure("label-count", replace(NAMES[1], 4, struct.pack(">I", 12)))
    expect_failure("image-height", replace(NAMES[0], 8, struct.pack(">I", 27)))
    expect_failure("image-width", replace(NAMES[0], 12, struct.pack(">I", 29)))
    expect_failure("out-of-range-label", replace(NAMES[3], 8, b"\x0a"))
    expect_failure("short-header", lambda d: d.update({NAMES[0]: d[NAMES[0]][:8]}))
    expect_failure("short-payload", lambda d: d.update({NAMES[0]: d[NAMES[0]][:-1]}))
    expect_failure("extra-payload", lambda d: d.update({NAMES[0]: d[NAMES[0]] + b"x"}))
    encoded = gzip.compress(decoded[NAMES[2]], mtime=0)
    expect_failure("gzip-truncated", encoded={NAMES[2]: encoded[:-4]})
    expect_failure("gzip-crc", encoded={NAMES[2]: encoded[:-8] + bytes([encoded[-8] ^ 1]) + encoded[-7:]})
    expect_failure("gzip-isize", encoded={NAMES[2]: encoded[:-4] + bytes([encoded[-4] ^ 1]) + encoded[-3:]})
    expect_failure("gzip-trailing", encoded={NAMES[2]: encoded + b"trailing"})
    expect_failure("gzip-concatenated", encoded={NAMES[2]: encoded + gzip.compress(b"", mtime=0)})
    expect_failure("not-gzip", encoded={NAMES[2]: b"not a gzip member"})

    missing_source = root / "missing-source-file"
    missing_contract = write_inputs(missing_source, decoded)
    (missing_source / NAMES[3]).unlink()
    missing_target = root / "missing-must-not-create-root"
    require(native(args, missing_source, missing_target, missing_contract, contract_path).returncode == 1
            and not missing_target.exists(), "preflight all sources before output creation")
    for kind in ("empty-directory", "foreign-file"):
        target = root / kind
        target.mkdir()
        if kind == "empty-directory":
            (target / "mnist").mkdir()
        else:
            (target / "mnist").write_bytes(b"foreign")
        require(native(args, source, target, contract, contract_path).returncode == 1, "existing output never replaced")
        require(sorted(path.name for path in target.iterdir()) == ["mnist"], "no new stage on existing output")
        if kind == "foreign-file":
            require((target / "mnist").read_bytes() == b"foreign", "foreign destination intact")

    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    target, processes = root / "concurrent", []
    command = [str(args.harness), str(contract_path), str(source), str(target)]
    try:
        for _ in range(2):
            processes.append(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        for process in processes:
            process.communicate(timeout=180)
        require(sorted(process.returncode for process in processes) == [0, 1], "no-clobber concurrent publication")
        compare_fixture(target / "mnist", decoded, contract)
        require(sorted(path.name for path in target.iterdir()) == ["mnist"], "loser cleans only own stage")
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    link_skip = None
    link = root / "linked-source"
    try:
        link.symlink_to(source, target_is_directory=True)
    except OSError as error:
        link_skip = str(error)
    else:
        link_output = root / "linked-must-not-create"
        require(native(args, link, link_output, contract, contract_path).returncode == 1 and
                not link_output.exists(), "reject source symlink")
    repository_output = args.repository / "out" / "mnist-native-must-not-be-created"
    require(not repository_output.exists(), "reserved repository negative path already exists")
    require(native(args, source, repository_output, contract, contract_path).returncode == 1 and
            not repository_output.exists(), "reject repository output before mutation")
    require({name: digest(source / name) for name in NAMES} == original, "source gzip unchanged")
    if args.cli:
        for extra in ([], ["--source-dir", source, "--selection", "full"],
                      ["--source-dir", source, "--archive", source / NAMES[0]]):
            target = root / "cli-invalid"
            result = run([args.cli, "data", "prepare", "--name", "mnist", "--data-root", target, *extra])
            require(result.returncode != 0 and not target.exists(), "MNIST CLI arguments and no output")
    if args.example:
        require(run([args.example]).returncode == 2, "data example usage")
        target = root / "example-fixed-identity"
        require(run([args.example, source, target]).returncode == 1 and not target.exists(),
                "data example cannot accept synthetic identity")
    return {"images": 20, "negative_cases": negative, "concurrent_publish": "passed",
            "empty_path_native_execution": "passed", "source_symlink_skip": link_skip}


def official_checks(args, root):
    require(args.official_source.is_dir(), "official local source directory")
    manifest = json.loads((args.repository / "examples/mnist/dataset-manifest.json").read_text(encoding="utf-8"))
    counts, entries = {"train": [0] * 10, "test": [0] * 10}, []
    # Independent Python gzip reads never write preparation data or invoke the old downloader.
    for index, item in enumerate(manifest["files"]):
        path = args.official_source / item["name"]
        require(path.stat().st_size == item["size_bytes"] and digest(path) == item["sha256"], "official compressed identity")
        hash_value, size = hashlib.sha256(), 0
        with gzip.open(path, "rb") as reference:
            image = index % 2 == 0
            header = reference.read(16 if image else 8)
            samples = 60000 if index < 2 else 10000
            require(header == (struct.pack(">IIII", 2051, samples, 28, 28) if image
                               else struct.pack(">II", 2049, samples)), "official independent IDX header")
            hash_value.update(header)
            size = len(header)
            for block in iter(lambda: reference.read(65536), b""):
                if index % 2:
                    for label in block:
                        require(label < 10, "official reference label")
                        counts["train" if index < 2 else "test"][label] += 1
                hash_value.update(block)
                size += len(block)
        require(size == item["decoded_size_bytes"], "official decoded reference size")
        entries.append({"name": item["name"], "compressed_bytes": item["size_bytes"],
                        "compressed_sha256": item["sha256"], "decoded_bytes": size,
                        "decoded_sha256": hash_value.hexdigest()})
    contract = {"train_samples": 60000, "test_samples": 10000,
                "max_output_bytes": 54950048, "files": entries}
    output = root / "official-native"
    if args.cli:
        env = os.environ.copy()
        env["JYPPX_DATA_ROOT"] = str(output)
        env["JYPPX_LIUDING_DATA_ROOT"] = str(root / "other-root-must-not-exist")
        result = run([args.cli, "data", "prepare", "--name", "mnist",
                      "--source-dir", args.official_source, "--format", "json"], env)
        producer = "product-cli-JYPPX_DATA_ROOT"
    else:
        result = native(args, args.official_source, output, contract, root / "official-contract.json")
        producer = "private-harness"
    require(result.returncode == 0, result.stderr)
    require(not (root / "other-root-must-not-exist").exists(), "MNIST uses its documented environment variable")
    prepared = output / "mnist"
    for entry in entries:
        actual = prepared / entry["name"][:-3]
        require(actual.stat().st_size == entry["decoded_bytes"] and digest(actual) == entry["decoded_sha256"], "official complete IDX identity")
        with gzip.open(args.official_source / entry["name"], "rb") as reference, actual.open("rb") as produced:
            while True:
                expected, observed = reference.read(65536), produced.read(65536)
                require(expected == observed, "official exact decompressed bytes")
                if not expected:
                    break
        cached = args.official_source / entry["name"][:-3]
        if cached.exists():
            require(cached.stat().st_size == entry["decoded_bytes"] and digest(cached) == entry["decoded_sha256"],
                    "existing official IDX reference matches without modification")
    expected_report = (json.dumps(report_reference(contract, counts), indent=2) + "\n").encode()
    require((prepared / "preparation.json").read_bytes() == expected_report, "official report exact independent bytes")
    return {"producer": producer, "train": 60000, "test": 10000,
            "idx_bytes_compared": sum(item["decoded_bytes"] for item in entries),
            "report_sha256": digest(prepared / "preparation.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--example", type=Path)
    parser.add_argument("--official-source", type=Path)
    parser.add_argument("--official-only", action="store_true")
    args = parser.parse_args()
    for name in ("harness", "repository", "cli", "example", "official_source"):
        value = getattr(args, name)
        if value:
            setattr(args, name, value.resolve())
    require(not args.official_only or args.official_source, "--official-only requires source")
    with tempfile.TemporaryDirectory(prefix="liuding-native-mnist-") as temporary:
        root = Path(temporary)
        result = {} if args.official_only else {"fixture": fixture_checks(args, root)}
        if args.official_source:
            result["official"] = official_checks(args, root)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
