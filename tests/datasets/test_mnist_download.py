#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent checks; simulated transport by default, real HTTPS opt-in only."""
from pathlib import Path
import argparse
import copy
import json
import os
import struct
import subprocess
import tempfile

import test_native_mnist as reference

RAW_SHA256 = (
    "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
    "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
    "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
    "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
)
require = reference.require


def run(command, env=None, timeout=180):
    return subprocess.run([str(item) for item in command], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, env=env)


def checked_success(result):
    require(result.returncode == 0,
            f"command failed with returncode {result.returncode}\n" + result.stdout + result.stderr)


def fake_checks(args, root):
    decoded = reference.make_fixture()
    source = root / "source"
    contract = reference.write_inputs(source, decoded)
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    source_hashes = {name: reference.digest(source / name) for name in reference.NAMES}

    def invoke(output, mode, fixture=contract, env=None):
        contract_path.write_text(json.dumps(fixture), encoding="utf-8")
        return run([args.harness, contract_path, source, output, mode], env)

    for mode in ("fragmented", "chunked", "large-block", "body-before-head",
                 "late-chunked"):
        output = root / mode
        environment = dict(os.environ)
        environment["PATH"] = ""
        result = invoke(output, mode, env=environment)
        checked_success(result)
        reference.compare_fixture(output / "mnist", decoded, contract)
        require(sorted(item.name for item in output.iterdir()) == ["mnist"],
                "compressed stage removed after success")
        expected_requests = [
            f"GET {reference.SOURCE}{entry['name']} {entry['compressed_bytes']}"
            for entry in contract["files"]]
        require(result.stdout.splitlines() == expected_requests,
                "exactly four pinned HTTPS requests, in frozen manifest order")

    failures = ("redirect", "notfound", "length-large", "length-small",
                "missing-headers", "missing-metadata-after-body", "repeated-head",
                "empty-body", "short-body", "extra-body", "tampered",
                "transport", "throw-after-body", "fail-third",
                "oversized-callback", "late-redirect", "late-notfound",
                "late-length-large", "late-length-small", "late-repeated-head",
                "late-throw-after-body")
    for mode in failures:
        output = root / mode
        output.mkdir()
        (output / "keep").write_bytes(b"unrelated")
        result = invoke(output, mode)
        require(result.returncode != 0, "negative transport accepted: " + mode)
        require(sorted(item.name for item in output.iterdir()) == ["keep"],
                "failed stage/output leak: " + mode)
        require((output / "keep").read_bytes() == b"unrelated",
                "cleanup touched unrelated file")

    output = root / "late-destination"
    result = invoke(output, "late-destination")
    require(result.returncode != 0, "late destination must win")
    require(sorted(item.name for item in output.iterdir()) == ["mnist"],
            "only competing destination remains")
    require((output / "mnist" / "winner").read_bytes() in
            (b"preserved\n", b"preserved\r\n"), "concurrent winner preserved")

    for directory in (False, True):
        output = root / ("existing-directory" if directory else "existing-file")
        output.mkdir()
        destination = output / "mnist"
        if directory:
            destination.mkdir()
        else:
            destination.write_bytes(b"existing")
        result = invoke(output, "fragmented")
        require(result.returncode != 0 and not result.stdout, "refuse before fetch")
        require(destination.is_dir() if directory else
                destination.read_bytes() == b"existing", "preserve existing output")

    invalid = copy.deepcopy(contract)
    invalid["files"][0]["name"] = "../unexpected.gz"
    output = root / "invalid-contract"
    result = invoke(output, "fragmented", invalid)
    require(result.returncode != 0 and not result.stdout and not output.exists(),
            "invalid contract rejected before filesystem/transport")
    output = args.repository / "mnist-network-must-not-create"
    require(not output.exists(), "repository rejection sentinel already exists")
    result = invoke(output, "fragmented")
    require(result.returncode != 0 and not result.stdout and not output.exists(),
            "repository output rejected before transport")
    require({name: reference.digest(source / name) for name in reference.NAMES}
            == source_hashes, "simulated sources must remain unchanged")


def cli_checks(args, root):
    if not args.cli:
        return
    prefix = [args.cli, "data", "prepare", "--format", "json"]
    bad = (
        ["--name", "mnist", "--download", "false"],
        ["--name", "mnist", "--download", "1"],
        ["--name", "mnist", "--download", "true", "--source-dir", str(root)],
        ["--name", "mnist", "--download", "true", "--selection", "full"],
        ["--name", "mnist", "--download", "true", "--archive", "unused.gz"],
        ["--name", "mnist", "--download", "true", "--download", "true"],
        ["--name", "paired-affine", "--download", "true"],
        ["--name", "mnist", "--url", "https://evil.example/"],
        ["--name", "mnist"],
    )
    for index, arguments in enumerate(bad):
        output = root / f"bad-cli-{index}"
        result = run(prefix + arguments + ["--data-root", output])
        require(result.returncode != 0 and not output.exists(),
                "invalid CLI must fail before creating/downloading data")
        require("invalid_argument" in (result.stdout + result.stderr).lower(),
                "CLI argument failure must retain foundation error code")
    if not args.network_enabled:
        output = root / "disabled"
        result = run(prefix + ["--name", "mnist", "--download", "true",
                               "--data-root", output])
        require(result.returncode != 0 and not output.exists(),
                "disabled network must not create a stage or silently fall back")
        require("not_implemented" in (result.stdout + result.stderr).lower(),
                "explicit disabled transport error")
        if args.example:
            result = run([args.example, "--download", root / "disabled-example"])
            require(result.returncode != 0 and
                    "disabled" in (result.stdout + result.stderr).lower(),
                    "teaching entry reports disabled native HTTPS")


def official_network(args, root):
    require(args.network_enabled and args.cli, "explicit enabled CLI required")
    manifest = json.loads((args.repository / "examples/mnist/dataset-manifest.json")
                          .read_text(encoding="utf-8"))
    require(manifest["source"] == reference.SOURCE, "frozen HTTPS origin")
    require(tuple(entry["name"] for entry in manifest["files"]) == reference.NAMES,
            "frozen official source order")
    output = root / "download"
    environment = dict(os.environ)
    environment["JYPPX_DATA_ROOT"] = str(output)
    environment["JYPPX_LIUDING_DATA_ROOT"] = str(root / "wrong-environment-root")
    result = run([args.cli, "data", "prepare", "--name", "mnist",
                  "--download", "true", "--format", "json"],
                 env=environment, timeout=600)
    checked_success(result)
    require(not (root / "wrong-environment-root").exists(), "MNIST data-root env")
    require(sorted(item.name for item in output.iterdir()) == ["mnist"],
            "no persistent compressed archive or download stage")
    destination = output / "mnist"
    require(sorted(item.name for item in destination.iterdir()) ==
            sorted([name[:-3] for name in reference.NAMES] + ["preparation.json"]),
            "complete four original IDX and report only")
    contract = {"train_samples": 60000, "test_samples": 10000,
                "max_output_bytes": 54950048, "files": []}
    counts = {"train": [0] * 10, "test": [0] * 10}
    for index, entry in enumerate(manifest["files"]):
        path = destination / entry["name"][:-3]
        count = 60000 if index < 2 else 10000
        require(path.stat().st_size == entry["decoded_size_bytes"], "IDX size")
        require(reference.digest(path) == RAW_SHA256[index], "independent IDX SHA")
        with path.open("rb") as stream:
            if index % 2 == 0:
                require(struct.unpack(">IIII", stream.read(16)) ==
                        (2051, count, 28, 28), "official image header")
            else:
                require(struct.unpack(">II", stream.read(8)) ==
                        (2049, count), "official label header")
                labels = stream.read(count + 1)
                require(len(labels) == count, "complete labels")
                split = "train" if index < 2 else "test"
                for label in labels:
                    require(0 <= label <= 9, "official label range")
                    counts[split][label] += 1
        contract["files"].append({
            "name": entry["name"], "compressed_bytes": entry["size_bytes"],
            "compressed_sha256": entry["sha256"],
            "decoded_bytes": entry["decoded_size_bytes"],
            "decoded_sha256": RAW_SHA256[index]})
    expected_report = (json.dumps(reference.report_reference(contract, counts),
                                  indent=2) + "\n").encode()
    require((destination / "preparation.json").read_bytes() == expected_report,
            "complete exact report independently reconstructed")
    snapshot = {item.name: reference.digest(item) for item in destination.iterdir()}
    result = run([args.cli, "data", "prepare", "--name", "mnist",
                  "--download", "true", "--data-root", output])
    require(result.returncode != 0, "existing destination must refuse")
    require({item.name: reference.digest(item) for item in destination.iterdir()}
            == snapshot, "no replacement of successful official download")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--harness", type=Path)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--example", type=Path)
    parser.add_argument("--network-enabled", action="store_true")
    parser.add_argument("--network-only", action="store_true",
                        help="explicitly allow real product HTTPS downloads")
    parser.add_argument("--work-root", type=Path)
    args = parser.parse_args()
    args.repository = args.repository.resolve()
    if args.network_only and not (args.network_enabled and args.cli):
        parser.error("--network-only needs --network-enabled and --cli")
    if not args.network_only and not (args.harness or args.cli):
        parser.error("need an enabled harness or a CLI switch check")
    with tempfile.TemporaryDirectory(prefix="liuding-mnist-download-",
                                     dir=args.work_root) as directory:
        root = Path(directory)
        if args.network_only:
            official_network(args, root)
        else:
            if args.harness:
                fake_checks(args, root)
            cli_checks(args, root)
    print("MNIST HTTPS opt-in contract passed; no model training repeated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
