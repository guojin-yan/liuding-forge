# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent official OTLP codec readback; never a product runtime entry."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

SIDECAR_SHA256 = "c86ce5978c6753a81f3d5d0795746e192813073604b905083f47cbc56fc3ef44"
JSON_ATTRIBUTES = {"jyppx.identity", "jyppx.correlation", "jyppx.attributes",
                   "jyppx.envelope", "jyppx.metric_definition",
                   "jyppx.metric_context", "jyppx.clock_mapping"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))


def reference(path: Path):
    require(hashlib.sha256(path.read_bytes()).hexdigest() == SIDECAR_SHA256,
            "Official reference mapping drifted")
    spec = importlib.util.spec_from_file_location("liuding_otlp_codec_reference", path)
    require(spec is not None and spec.loader is not None, "Cannot load independent mapping oracle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize(message):
    from google.protobuf.descriptor import FieldDescriptor
    result = type(message)()
    result.CopyFrom(message)

    def walk(item) -> None:
        full_name = item.DESCRIPTOR.full_name
        if full_name == "opentelemetry.proto.common.v1.KeyValue" and item.key in JSON_ATTRIBUTES:
            require(item.value.WhichOneof("value") == "string_value", "JSON attribute type changed")
            item.value.string_value = compact(json.loads(item.value.string_value))
        if full_name == "opentelemetry.proto.logs.v1.LogRecord":
            require(item.body.WhichOneof("value") == "string_value", "Log body type changed")
            item.body.string_value = compact(json.loads(item.body.string_value))
        for field, value in item.ListFields():
            if field.type in (FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.TYPE_FLOAT):
                if not field.is_repeated and math.isnan(value):
                    setattr(item, field.name, float("nan"))
            if field.type == FieldDescriptor.TYPE_MESSAGE:
                if field.is_repeated:
                    for child in value:
                        walk(child)
                else:
                    walk(value)

    walk(result)
    return result.SerializeToString(deterministic=True)


def validated_live_envelope(sidecar, original):
    validated = sidecar.envelope(copy.deepcopy(original))
    # The frozen sidecar validator canonicalizes zeros for persisted replay.
    # This fixture is a typed live projection: restore only validated zeros.
    field = {1: "metrics", 2: "points", 5: "measurements"}.get(original["payload"]["kind"])
    if field is not None:
        source = original["payload"][field]
        target = validated["payload"][field]
        require(len(source) == len(target), "Validator changed the point count")
        for before, after in zip(source, target):
            if before["status"] == 1 and before["value"] == 0.0:
                require(after["status"] == 1 and after["value"] == 0.0,
                        "Validator changed a finite zero observation")
                after["value"] = before["value"]
    require(compact(validated) == compact(original),
            "Live projection differs from its fully validated source")
    return validated


def projection_gate(executable: Path, sidecar, root: Path) -> dict:
    output = root / "离线 OTLP 示例"
    process = subprocess.run([str(executable), str(output)], capture_output=True, timeout=60)
    require(process.returncode == 0, "C++ OTLP example failed: " +
            process.stderr.decode("utf-8", errors="replace"))
    index = json.loads((output / "projection-index.json").read_text(encoding="utf-8"))
    require(index["schema_version"] == 1, "Projection example index schema")
    original = [json.loads(line)["event"] for line in
                (output / "source-events.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(original) == len(index["events"]), "Example source/index event count")
    # prepare() only: no socket, send(), process transport or external collector.
    oracle = sidecar.OtlpHttpBackend("http://127.0.0.1:4318", "127.0.0.1", 5.0,
                                    index["resource_instance_id"], index["clock_mappings"])
    classes = {"metrics": oracle._schemas["metrics"].ExportMetricsServiceRequest,
               "logs": oracle._schemas["logs"].ExportLogsServiceRequest,
               "traces": oracle._schemas["traces"].ExportTraceServiceRequest}
    counts = {name: 0 for name in classes}
    unsupported = 0
    proof = []
    zero_paths = {}
    for envelope, entry in zip(original, index["events"]):
        require(entry["sequence"] == envelope["identity"]["sequence"], "Source order changed")
        prepared = oracle.prepare(validated_live_envelope(sidecar, envelope))
        require(entry["unsupported"] == (prepared is None), "Unsupported projection policy differs")
        expected = [] if prepared is None else prepared
        require(len(entry["requests"]) == len(expected), "Metrics+audit pairing changed")
        if prepared is None:
            unsupported += 1
        for actual, (signal, message) in zip(entry["requests"], expected):
            require(actual["signal"] == signal, "OTLP request order/signal changed")
            name = actual["file"]
            require(Path(name).name == name, "Projection file is not local")
            path = output / name
            require(path.is_file() and path.stat().st_size <= 16 * 1024 * 1024,
                    "Unbounded or missing native protobuf")
            body = path.read_bytes()
            require(len(body) == actual["bytes"], "Encoded length mismatch")
            decoded = classes[signal].FromString(body)
            require(normalize(decoded) == normalize(message),
                    f"Native/official {signal} request differs at source {entry['sequence']}")
            zero_path = envelope["attributes"].get("zero_path")
            if signal == "metrics" and zero_path:
                point = decoded.resource_metrics[0].scope_metrics[0].metrics[0].gauge.data_points[0]
                require(point.WhichOneof("value") == "as_double" and point.as_double == 0.0,
                        "Signed-zero case lost its present double")
                source_value = envelope["payload"]["points"][0]["value"]
                require(math.copysign(1.0, source_value) == math.copysign(1.0, point.as_double),
                        "Source/wire zero signs differ")
                zero_paths[zero_path] = math.copysign(1.0, point.as_double)
            counts[signal] += 1
            proof.append({"file": name, "sha256": hashlib.sha256(body).hexdigest()})
    require(all(counts.values()) and unsupported >= 2, "Example missed a signal/omission case")
    require(zero_paths == {"typed-live": -1.0, "canonical-recovered": 1.0},
            "Live/canonical signed-zero boundary not covered")
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}
    repeated = subprocess.run([str(executable), str(output)], capture_output=True, timeout=30)
    require(repeated.returncode != 0, "Example accepted an existing output directory")
    require(before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in output.iterdir()}, "Refused reuse changed output")
    return {"events": len(original), "requests": counts, "unsupported": unsupported,
            "files": proof, "signed_zero_paths": zero_paths}


def reply_gate(executable: Path, root: Path) -> dict:
    from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2
    from opentelemetry.proto.collector.logs.v1 import logs_service_pb2
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
    directory = root / "official-replies"
    directory.mkdir()
    manifest = {"schema_version": 1, "cases": []}
    count_names = {"metrics": "rejected_data_points", "logs": "rejected_log_records",
                   "traces": "rejected_spans"}
    classes = {"metrics": metrics_service_pb2.ExportMetricsServiceResponse,
               "logs": logs_service_pb2.ExportLogsServiceResponse,
               "traces": trace_service_pb2.ExportTraceServiceResponse}
    for signal, response_class in classes.items():
        for mode in ("empty", "present-empty", "rejected", "negative-rejected", "message",
                     "malformed", "invalid-utf8", "oversized"):
            reply = response_class()
            if mode == "present-empty":
                reply.partial_success.SetInParent()
            if mode in ("rejected", "negative-rejected"):
                setattr(reply.partial_success, count_names[signal],
                        1 if mode == "rejected" else -1)
            if mode in ("message", "malformed", "invalid-utf8"):
                reply.partial_success.error_message = "消息"
            body = reply.SerializeToString(deterministic=True)
            if mode == "malformed":
                body = body[:-1]  # Truncate official bytes; no handwritten wire encoder.
            if mode == "invalid-utf8":
                body = body.replace("消息".encode("utf-8"), bytes([255]) * 6)
            if mode == "oversized":
                body = body + bytes(1024 * 1024 + 1)
            try:
                decoded = response_class.FromString(body)
                accepted = (not getattr(decoded.partial_success, count_names[signal])
                            and not decoded.partial_success.error_message)
            except Exception:
                accepted = False
            accepted = accepted and len(body) <= 1024 * 1024
            name = f"{signal}-{mode}.pb"
            (directory / name).write_bytes(body)
            manifest["cases"].append({"file": name, "signal": signal, "accepted": accepted})
    (directory / "replies.json").write_text(compact(manifest), encoding="utf-8")
    process = subprocess.run([str(executable), "--verify-replies", str(directory)],
                             capture_output=True, timeout=60)
    require(process.returncode == 0, "Native/official response checks differ: " +
            process.stderr.decode("utf-8", errors="replace"))
    return {"cases": len(manifest["cases"]), "official_generated_fixtures": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    require(sys.version_info[:2] == (3, 11), "Use pinned CPython 3.11 maintenance environment")
    for package, version in (("protobuf", "6.33.5"), ("opentelemetry-proto", "1.37.0")):
        require(importlib.metadata.version(package) == version, "Official reader version mismatch")
    oracle = reference(args.sidecar.resolve())
    with tempfile.TemporaryDirectory(prefix="liuding-otlp-projection-") as temporary:
        root = Path(temporary)
        report = {"schema": "jyppx.native-otlp-codec-validation/v1", "passed": True,
                  "scope": "Offline C++ protobuf/preflight only; native HTTP is not qualified",
                  "projection": projection_gate(args.example.resolve(), oracle, root),
                  "replies": reply_gate(args.example.resolve(), root)}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8") as output:
            output.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
