# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Independent maintenance-only TensorBoard/CLI interoperability gate.

Runs already-built C++ products. The existing official Python writer is a
reference oracle; it is never invoked by the native product. No missing
dependency or unsupported platform is silently skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

REFERENCE_SIDECAR_SHA256 = "c86ce5978c6753a81f3d5d0795746e192813073604b905083f47cbc56fc3ef44"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load maintenance reference: {path.name}")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def compact(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def masked_crc32c(data: bytes) -> int:
    value = 0xFFFFFFFF
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
    value ^= 0xFFFFFFFF
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def raw_records(path: Path) -> list[bytes]:
    # Independently reject partial framing, length overrun, and CRC corruption.
    records = []
    with path.open("rb") as source:
        while True:
            length = source.read(8)
            if not length:
                break
            require(len(length) == 8, "Partial TFRecord length")
            checksum = source.read(4)
            require(len(checksum) == 4, "Partial length CRC")
            require(struct.unpack("<I", checksum)[0] == masked_crc32c(length),
                    "Length CRC32C mismatch")
            count = struct.unpack("<Q", length)[0]
            require(count <= 4 * 1024 * 1024, "Event exceeds the fixture record bound")
            body = source.read(count)
            checksum = source.read(4)
            require(len(body) == count and len(checksum) == 4, "Partial event record")
            require(struct.unpack("<I", checksum)[0] == masked_crc32c(body),
                    "Event CRC32C mismatch")
            records.append(body)
    return records


def event_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*tfevents*"))
    require(len(files) == 1, "Expected one exclusive TensorBoard event file")
    return files


def events(directory: Path):
    from tensorboard.compat.proto import event_pb2
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    path = event_files(directory)[0]
    framed = raw_records(path)
    # The independent official reader must consume the same complete records.
    require(list(RawEventFileLoader(str(path)).Load()) == framed,
            "Official TFRecord reader disagrees with independently checked framing")
    decoded = [event_pb2.Event.FromString(record) for record in framed]
    require(decoded and decoded[0].file_version == "brain.Event:2", "Missing file version")
    require(decoded[0].source_metadata.writer != "", "Missing writer provenance")
    require(all(event.WhichOneof("what") == "summary" for event in decoded[1:]),
            "Unexpected record outside the supported Event/Summary contract")
    return decoded[1:]


def normalized(event):
    from tensorboard.compat.proto import event_pb2, types_pb2
    result = event_pb2.Event()
    result.CopyFrom(event)
    for value in result.summary.value:
        description = json.loads(value.metadata.summary_description)
        value.metadata.summary_description = compact(description).decode("utf-8")
        if value.tensor.dtype == types_pb2.DT_STRING:
            require(len(value.tensor.string_val) == 1, "Text is not a scalar string")
            value.tensor.string_val[0] = compact(json.loads(value.tensor.string_val[0]))
        elif value.tensor.dtype == types_pb2.DT_DOUBLE:
            require(len(value.tensor.double_val) == 1, "Scalar is not a binary64 value")
            number = value.tensor.double_val[0]
            if math.isnan(number):
                value.tensor.double_val[0] = float("nan")  # NaN payload bits have no meaning.
        else:
            raise AssertionError("Unexpected tensor dtype")
    return result.SerializeToString(deterministic=True)


def official_consumer(directory: Path, expected_events) -> None:
    from tensorboard.backend.event_processing import event_accumulator
    accumulator = event_accumulator.EventAccumulator(
        str(directory), size_guidance={event_accumulator.TENSORS: 0}).Reload()
    tags: dict[str, list] = {}
    for event in expected_events:
        for value in event.summary.value:
            tags.setdefault(value.tag, []).append((event, value))
    require(set(accumulator.Tags()[event_accumulator.TENSORS]) == set(tags),
            "Official EventAccumulator lost or invented a series")
    for tag, expected in tags.items():
        actual = accumulator.Tensors(tag)
        require(len(actual) == len(expected), "EventAccumulator changed the event count")
        for item, (event, value) in zip(actual, expected):
            require(item.step == event.step and item.wall_time == event.wall_time,
                    "EventAccumulator changed the clock or step")
            require(item.tensor_proto.dtype == value.tensor.dtype,
                    "EventAccumulator changed the tensor dtype")


def compare(directory: Path, original: list[dict], sidecar, reference: Path) -> dict:
    backend = sidecar.TensorBoardBackend(reference)
    accepted = unsupported = 0
    try:
        for envelope in original:
            prepared = backend.prepare(envelope)
            if prepared is None:
                unsupported += 1
            else:
                backend.export(prepared)
                accepted += 1
    finally:
        backend.close()
    actual, expected = events(directory), events(reference)
    require(len(actual) == len(expected), "Native/official event count differs")
    for index, (native_event, official_event) in enumerate(zip(actual, expected)):
        require(normalized(native_event) == normalized(official_event),
                f"Native/official Event differs at record {index}")
    official_consumer(directory, expected)
    return {"accepted": accepted, "unsupported": unsupported,
            "event_records": len(actual), "event_sha256": hashlib.sha256(
                event_files(directory)[0].read_bytes()).hexdigest()}


def signed_zero_contract(original: list[dict], decoded) -> dict:
    from tensorboard.compat.proto import types_pb2
    cases = (("精度.二进制64", 0, 1.0), ("精度.二进制64", 1, -1.0),
             ("zero.factory", 1, 1.0), ("zero.canonical_round_trip", 1, 1.0))
    source_points = [point for envelope in original
                     if envelope["identity"]["stream_id"] == "example.conformance"
                     for point in envelope["payload"].get("points", [])]
    native_points = [(event.step, value,
                      json.loads(value.metadata.summary_description))
                     for event in decoded for value in event.summary.value
                     if value.tensor.dtype == types_pb2.DT_DOUBLE]
    for metric_id, step, sign in cases:
        source = [point for point in source_points
                  if point["definition"]["metric_id"] == metric_id
                  and point["step"]["value"] == step]
        require(len(source) == 1 and source[0]["status"] == 1
                and source[0]["value"] == 0.0
                and math.copysign(1.0, source[0]["value"]) == sign,
                f"Source fixture lacks the declared signed-zero case: {metric_id}/{step}")
        native = [value for actual_step, value, description in native_points
                  if description.get("original_metric_id") == metric_id and actual_step == step]
        require(len(native) == 1 and len(native[0].tensor.double_val) == 1
                and native[0].tensor.double_val[0] == 0.0
                and math.copysign(1.0, native[0].tensor.double_val[0]) == sign,
                f"Official protobuf reader changed signed zero: {metric_id}/{step}")
    return {"live_positive_zero": True, "live_negative_zero": True,
            "factory_positive_zero": True, "canonical_recovery_positive_zero": True}


def example_gate(executable: Path, sidecar, root: Path) -> list[dict]:
    results = []
    for conformance in (False, True):
        output = root / ("边界 cases" if conformance else "学习 example")
        command = [str(executable), str(output)] + (["--conformance"] if conformance else [])
        process = subprocess.run(command, capture_output=True, timeout=60)
        require(process.returncode == 0, "Native example failed: " +
                process.stderr.decode("utf-8", errors="replace"))
        originals = []
        with (output / "source-events.jsonl").open("rb") as source:
            while line := source.readline(4 * 1024 * 1024 + 1):
                require(len(line) <= 4 * 1024 * 1024 and line.endswith(b"\n"),
                        "Unbounded/incomplete canonical JSONL fixture")
                originals.append(json.loads(line)["event"])
        result = compare(output, originals, sidecar, root / f"official-{conformance}")
        if conformance:
            result["signed_zero_contract"] = signed_zero_contract(originals, events(output))
        expected = (f"{result['accepted']} accepted, {result['unsupported']} unsupported")
        require(expected in process.stdout.decode("utf-8"), "C++ counters differ from oracle")
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        repeated = subprocess.run(command, capture_output=True, timeout=30)
        require(repeated.returncode != 0, "Example reused an owned directory")
        require(before == {path.name: path.read_bytes() for path in output.iterdir()},
                "Directory refusal modified existing output")
        result["case"] = "conformance" if conformance else "learning"
        results.append(result)
    return results


def cli_gate(executable: Path, fixture_path: Path, sidecar, root: Path) -> list[dict]:
    fixture = module(fixture_path, "liuding_native_tensorboard_cli_fixture")
    case = fixture.RunTelemetryContract("runTest")
    case.cli = executable
    results = []
    try:
        case.setUp()
        outputs = []
        for page_size in (1, 128):
            output = root / f"CLI 原生 {page_size}"
            data = case.query("export", "--backend", "tensorboard-native", "--output", output,
                              "--page-size", page_size, expected=1)["data"]
            require(data["scan_completed"] and not data["fully_represented"], "CLI omission policy")
            require(data["read_events"] == 4 and data["accepted_events"] == 2
                    and data["unsupported_events"] == 2 and data["transport"] is None,
                    "CLI receipts/counters or native-process isolation differ")
            originals = [fixture.projection(value) for value in case.events]
            result = compare(output, originals, sidecar, root / f"CLI official {page_size}")
            require(data["destination_counters"]["exported_events"] == result["accepted"]
                    and data["destination_counters"]["unsupported_events"] == result["unsupported"],
                    "CLI final counters do not match the official mapping")
            outputs.append([normalized(event) for event in events(output)])
            result["case"] = f"cli-page-{page_size}"
            results.append(result)
        require(outputs[0] == outputs[1], "CLI page size changed the output projection")
        subset = root / "CLI subset"
        data = case.query("export", "--backend", "tensorboard-native", "--output", subset,
                          "--after", 2, "--snapshot", 4)["data"]
        require(data["read_events"] == 2 and data["accepted_events"] == 2,
                "CLI snapshot cursor changed")
        compare(subset, [fixture.projection(value) for value in case.events[2:]],
                sidecar, root / "subset official")
        for extra in (("--resume", "true"), ("--resume", "false"),
                      ("--sidecar-config", root / "must-not-open.json")):
            refused = root / ("refused-" + hashlib.sha256(str(extra).encode()).hexdigest()[:12])
            case.query("export", "--backend", "tensorboard-native", "--output", refused,
                       *extra, expected=2)
            require(not refused.exists(), "Refused CLI configuration created output")
        internal = case.store / "must-not-create"
        case.query("export", "--backend", "tensorboard-native", "--output", internal, expected=2)
        require(not internal.exists(), "CLI wrote inside managed SQLite/CAS storage")
        bounded = root / "over-snapshot-capacity"
        case.query("export", "--backend", "tensorboard-native", "--output", bounded,
                   "--max-events", 1, expected=3)
        require(not bounded.exists(), "CLI capacity refusal created output")
        before = event_files(subset)[0].read_bytes()
        case.query("export", "--backend", "tensorboard-native", "--output", subset, expected=3)
        require(event_files(subset)[0].read_bytes() == before, "CLI reused existing output")
    finally:
        case.doCleanups()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--cli-fixture", type=Path)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    require(args.example is not None or args.cli is not None, "Select example and/or CLI gate")
    require(args.cli is None or args.cli_fixture is not None, "CLI requires its canonical fixture")
    require(sys.version_info[:2] == (3, 11), "Use the locked CPython 3.11 reference runtime")
    for package, version in (("tensorboard", "2.20.0"), ("protobuf", "6.33.5")):
        require(importlib.metadata.version(package) == version, "Reference dependency version mismatch")
    require(masked_crc32c(b"123456789") == 0xC78AB0E5, "CRC32C reference self-check failed")
    require(hashlib.sha256(args.sidecar.read_bytes()).hexdigest() == REFERENCE_SIDECAR_SHA256,
            "Official reference source drifted; review the mapping contract before updating its pin")
    sidecar = module(args.sidecar.resolve(), "liuding_official_telemetry_reference")
    with tempfile.TemporaryDirectory(prefix="liuding-native-tensorboard-") as temporary:
        root = Path(temporary)
        results = []
        if args.example:
            results += example_gate(args.example.resolve(), sidecar, root)
        if args.cli:
            results += cli_gate(args.cli.resolve(), args.cli_fixture.resolve(), sidecar, root)
    report = {"schema": "jyppx.native-tensorboard-validation/v1", "passed": True,
              "scope": "native TensorBoard event observer; OTLP is not qualified here",
              "cases": results}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8") as output:
            output.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
