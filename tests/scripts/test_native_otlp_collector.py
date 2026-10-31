# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent official Collector/readback gate; never a product runtime."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def invoke(executable, *arguments, input=None):
    process = subprocess.run([str(executable), *map(str, arguments)], input=input,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    require(process.returncode == 0, (arguments, process.returncode, process.stdout, process.stderr))
    require(len(process.stdout) + len(process.stderr) <= 2 * 1024 * 1024, "Unbounded product output")
    return process.stdout


def official_ids(value):
    result = copy.deepcopy(value)
    pending = [result]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key, content in item.items():
                if key in ("traceId", "spanId", "parentSpanId") and content:
                    item[key] = base64.b64encode(bytes.fromhex(content)).decode("ascii")
                else:
                    pending.append(content)
        elif isinstance(item, list):
            pending.extend(item)
    return result


def attributes(items):
    result = {}
    for item in items:
        require(item.key not in result, "Duplicate OTLP attribute")
        result[item.key] = getattr(item.value, item.value.WhichOneof("value"))
    return result


def opaque_id(domain, value, length):
    encoded = hashlib.sha256(domain.encode() + b"\0" + value.encode()).digest()[:length]
    return encoded if any(encoded) else encoded[:-1] + b"\1"


def readback(output, events, instance, clock_mappings):
    from google.protobuf.json_format import ParseDict
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    raw = output.read_bytes() if output.exists() else b""
    require(len(raw) <= 2 * 1024 * 1024, "Collector output bound")
    require(not raw or raw.endswith(b"\n"), "Incomplete Collector output line")
    records = [json.loads(line) for line in raw.splitlines() if line]
    signals = {"resourceMetrics": [], "resourceLogs": [], "resourceSpans": []}
    for record in records:
        require(len(record) == 1 and next(iter(record)) in signals, "Unexpected Collector file record")
        for key, content in record.items():
            signals[key].extend(content)
    classes = {"resourceMetrics": ExportMetricsServiceRequest,
               "resourceLogs": ExportLogsServiceRequest, "resourceSpans": ExportTraceServiceRequest}
    decoded = {key: ParseDict(official_ids({key: rows}), classes[key]()) for key, rows in signals.items()}
    actual_metrics, actual_logs, actual_spans = [], [], []
    for key, root_name, scope_name, item_name, destination in (
        ("resourceMetrics", "resource_metrics", "scope_metrics", "metrics", actual_metrics),
        ("resourceLogs", "resource_logs", "scope_logs", "log_records", actual_logs),
        ("resourceSpans", "resource_spans", "scope_spans", "spans", actual_spans),
    ):
        for resource in getattr(decoded[key], root_name):
            fields = attributes(resource.resource.attributes)
            require(fields.get("service.name") == "liuding-forge" and
                    fields.get("service.instance.id") == instance, "Resource identity changed")
            for scope in getattr(resource, scope_name):
                require(scope.scope.name == "jyppx.telemetry" and scope.scope.version == "1", "Scope changed")
                destination.extend((fields, item) for item in getattr(scope, item_name))
    expected_logs = [event for event in events if event["payload"]["kind"] != 4]
    expected_metrics = [(event, point) for event in events for point in
                        event["payload"].get("points", event["payload"].get("metrics", event["payload"].get("measurements", [])))]
    expected_spans = [event for event in events if event["payload"]["kind"] == 4]
    require((len(actual_logs), len(actual_metrics), len(actual_spans)) ==
            (len(expected_logs), len(expected_metrics), len(expected_spans)), "Signal count changed or duplicate replay exported")
    by_event = {json.loads(attributes(log.attributes)["jyppx.identity"])["event_id"]: (resource, log)
                for resource, log in actual_logs}
    require(len(by_event) == len(actual_logs), "Duplicate event audit record")
    for event in expected_logs:
        resource, log = by_event[event["identity"]["event_id"]]
        fields = attributes(log.attributes)
        require(json.loads(fields["jyppx.envelope"]) == event, "Lossless source envelope changed")
        require(log.time_unix_nano == event["time"]["wall_time_unix_ns"], "Log timestamp changed")
        correlation = event["correlation"]
        for key in ("project_id", "run_id", "attempt_id"):
            require(resource["jyppx." + key] == correlation[key], "Log correlation changed")
        require(log.trace_id == opaque_id("jyppx.trace/v1", correlation["trace_id"], 16), "Log trace ID changed")
        require(log.span_id == opaque_id("jyppx.span/v1", correlation["span_id"], 8), "Log span ID changed")
        if event["payload"]["kind"] == 3:
            require(json.loads(log.body.string_value) == event["payload"], "Log body changed")
            require(log.severity_number == (0, 5, 9, 13, 17, 21)[event["payload"]["severity"]], "Log severity changed")
    metric_signatures = []
    for resource, metric in actual_metrics:
        require(set(resource) == {"service.name", "service.instance.id"}, "Metric resource cardinality changed")
        require(metric.WhichOneof("data") == "gauge" and len(metric.gauge.data_points) == 1, "Metric type/cardinality changed")
        point = metric.gauge.data_points[0]
        fields = attributes(point.attributes)
        definition, context = json.loads(fields["jyppx.metric_definition"]), json.loads(fields["jyppx.metric_context"])
        require(metric.name == "jyppx.scalar." + definition["definition_digest"], "Definition name changed")
        require(metric.description == definition["metric_id"], "Metric description changed")
        require(point.WhichOneof("value") == "as_double" and point.flags == 0, "Finite value encoding changed")
        metric_signatures.append((definition, context, point.time_unix_nano, point.as_double,
                                  fields["jyppx.step_kind"], fields["jyppx.resource_id"]))
    for event, point in expected_metrics:
        expected = (point["definition"], point["context"], event["time"]["wall_time_unix_ns"], point["value"],
                    point["step"]["kind"], event["payload"].get("resource_id", ""))
        require(expected in metric_signatures, "Metric fields/value/timestamp changed")
        metric_signatures.remove(expected)
    require(not metric_signatures, "Unexpected metric point")
    for event in expected_spans:
        matches = [(resource, span) for resource, span in actual_spans if span.name == event["payload"]["name"]]
        require(len(matches) == 1, "Missing or duplicate span")
        resource, span = matches[0]
        correlation, payload = event["correlation"], event["payload"]
        for key in ("project_id", "run_id", "attempt_id"):
            require(resource["jyppx." + key] == correlation[key], "Span correlation changed")
        require(json.loads(attributes(span.attributes)["jyppx.identity"]) == event["identity"], "Span identity changed")
        for key, domain, length in (("trace_id", "jyppx.trace/v1", 16), ("span_id", "jyppx.span/v1", 8)):
            require(getattr(span, key) == opaque_id(domain, correlation[key], length), "Span ID mapping changed")
        anchor = clock_mappings[event["time"]["clock_id"]]
        require(span.start_time_unix_nano == anchor["unix_ns"] + payload["start_monotonic_ns"] - anchor["monotonic_ns"], "Span start clock changed")
        require(span.end_time_unix_nano == anchor["unix_ns"] + payload["end_monotonic_ns"] - anchor["monotonic_ns"], "Span end clock changed")
        require(span.status.code == 1, "Span success status changed")
    return {"metrics": len(actual_metrics), "logs": len(actual_logs), "spans": len(actual_spans),
            "collector_records": records, "collector_file_sha256": hashlib.sha256(raw).hexdigest()}


class Collector:
    def __init__(self, executable, config, root):
        self.root, self.executable, self.config = root, executable, config
        self.output, self.log = root / "collector.jsonl", root / "collector.log"
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]

    def __enter__(self):
        environment = dict(os.environ, JYPPX_OTELCOL_ENDPOINT=f"127.0.0.1:{self.port}",
                           JYPPX_OTELCOL_FILE=str(self.output))
        self.stream = self.log.open("xb")
        self.process = subprocess.Popen([str(self.executable), "--config", str(self.config)], cwd=self.root,
                                        env=environment, stdin=subprocess.DEVNULL, stdout=self.stream,
                                        stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                require(self.process.poll() is None, "Real Collector stopped at startup")
                try:
                    with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                        return self
                except OSError:
                    time.sleep(0.025)
            raise AssertionError("Real Collector startup deadline")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.stream.close()
        require(self.log.stat().st_size < 1024 * 1024, "Collector diagnostic bound")

    def wait_readback(self, events, instance, anchors):
        deadline = time.monotonic() + 6
        last = None
        while time.monotonic() < deadline:
            require(self.process.poll() is None, "Real Collector stopped before readback")
            try:
                last = readback(self.output, events, instance, anchors)
                break
            except (AssertionError, json.JSONDecodeError):
                time.sleep(0.025)
        require(last is not None, "Real Collector did not export all expected observations")
        time.sleep(0.25)
        return readback(self.output, events, instance, anchors)


def canonical_events(cli, store):
    reply = json.loads(invoke(cli, "run", "events", "--root", store, "--id", "run.otlp-example", "--format", "json"))
    require(reply["status"] == "ok", "Canonical event query failed")
    data = reply["data"]
    require(data["snapshot_through"] == 3 and data["next_after"] is None, "Canonical snapshot changed")
    events = []
    for record in data["events"]:
        require(hashlib.sha256(bytes.fromhex(record["canonical_hex"])).hexdigest() == record["content_digest"], "Canonical source digest changed")
        events.append(record["event"])
    require([e["payload"]["kind"] for e in events] == [2, 3, 4], "Example payload coverage changed")
    require(events[0]["payload"]["points"][0]["value"] == 0.125, "Example calculated value changed")
    require(events[1]["payload"]["body"] == "Computed MAE from four samples", "Example log changed")
    require(events[2]["payload"]["name"] == "compute.mae", "Example span changed")
    anchors = {"example.clock": {"monotonic_ns": events[2]["payload"]["start_monotonic_ns"],
                                  "unix_ns": events[2]["time"]["wall_time_unix_ns"]}}
    return events, anchors


def load_checker(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cli", "example", "collector", "artifacts", "config"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    for key in ("cli", "example", "collector", "artifacts", "config"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    require(sys.version_info[:2] == (3, 11), "Use the locked CPython 3.11 checker")
    for package, expected in {"opentelemetry-proto": "1.37.0", "protobuf": "6.33.5"}.items():
        require(importlib.metadata.version(package) == expected, "Independent reader version changed: " + package)
    metadata = json.loads(args.artifacts.read_text(encoding="utf8"))
    platform = "windows" if sys.platform == "win32" else "linux"
    pinned = [r for r in metadata["archives"] if f"_{platform}_amd64" in r["filename"]]
    require(len(pinned) == 1 and sha(args.collector) == pinned[0]["executable_sha256"], "Pinned Collector binary changed")
    require(invoke(args.collector, "--version").decode().strip() == "otelcol-contrib version 0.135.0", "Collector version changed")
    cases = []
    with tempfile.TemporaryDirectory(prefix="liuding-native-collector-") as directory:
        root = Path(directory)
        store = root / "native-run"
        for name in ("public-api", "cli-replay", "worker-six-payloads"):
            case = root / name
            case.mkdir()
            with Collector(args.collector, args.config, case) as collector:
                instance = "native.real." + name
                if name == "public-api":
                    stdout = invoke(args.example, "127.0.0.1", collector.port, "http", instance, store)
                    require(b"Computed validation MAE from 4 samples: 0.125" in stdout and
                            b"exact replay was deduplicated" in stdout, "Example receipts changed")
                    events, anchors = canonical_events(args.cli, store)
                else:
                    worker = case / "worker"
                    worker.mkdir()
                    if name == "cli-replay":
                        events, anchors = canonical_events(args.cli, store)
                    else:
                        cli_tests = Path(__file__).resolve().parents[1] / "cli"
                        neutral = load_checker("test_run_telemetry", cli_tests / "test_run_telemetry.py")
                        protocol = load_checker("native_collector_protocol_checker", cli_tests / "test_native_otlp_worker.py")
                        payloads = [{"kind": 1, "name": "training.finished", "terminal": False,
                                     "metrics": [neutral.metric_point()], "decisions": []},
                                    {"kind": 2, "points": [neutral.metric_point(0.25)]},
                                    {"kind": 3, "severity": 3, "body": "真实 Collector 日志"},
                                    {"kind": 4, "name": "train.batch", "start_monotonic_ns": 400,
                                     "end_monotonic_ns": 500, "status": 2, "reason": ""},
                                    {"kind": 5, "resource_id": "device.cpu", "measurements": [neutral.metric_point(0.5)]},
                                    {"kind": 6, "role": "checkpoint.best", "artifact": {
                                        "algorithm": "sha256", "digest": "a" * 64, "size_bytes": 123,
                                        "logical_name": "best.model", "media_type": "application/octet-stream", "producer": "fixture"}}]
                        events = [neutral.projection(neutral.envelope(i, payload)) for i, payload in enumerate(payloads, 1)]
                        anchors = {"clock.fixture": {"monotonic_ns": 500, "unix_ns": 1_700_000_000_123_456_789}}
                    config = case / "native.json"
                    config.write_text(json.dumps({"schema_version": 1, "endpoint": {
                        "host": "127.0.0.1", "port": collector.port, "tls": False},
                        "resource_instance_id": instance, "clock_mappings": anchors,
                        "working_directory": str(worker), "inherit_environment": True}), encoding="utf8")
                    if name == "cli-replay":
                        reply = json.loads(invoke(args.cli, "run", "export", "--root", store,
                                                 "--id", "run.otlp-example", "--backend", "opentelemetry-native",
                                                 "--native-config", config, "--page-size", 1, "--format", "json"))
                        require(reply["status"] == "ok", "CLI export failed")
                        result = reply["data"]
                        require(result["accepted_events"] == result["read_events"] == result["snapshot_through"] == 3 and
                                result["fully_represented"] and result["scan_completed"] and
                                result["unsupported_events"] == result["sampled_out_events"] == 0, "CLI snapshot receipts changed")
                    else:
                        request = protocol.emit_request(events)
                        frames = (json.dumps(request, ensure_ascii=False) + "\n") * 2
                        raw = invoke(args.cli, "__liuding_otlp_worker", "--config", config,
                                     "--config-sha256", sha(config), input=frames.encode())
                        replies = [json.loads(row) for row in raw.splitlines()]
                        require(len(replies) == 2 and replies[0]["receipts"] == [1] * 6 and
                                replies[1]["receipts"] == [2] * 6 and
                                replies[1]["counters"]["exported_events"] == 6, "Worker replay receipts changed")
                report = collector.wait_readback(events, instance, anchors)
                cases.append({"name": name, "events": len(events), "source_events": events,
                              "clock_mappings": anchors, **report})
    print(json.dumps({"cases": cases, "failures": [], "skipped": 0,
                      "collector_sha256": pinned[0]["executable_sha256"], "collector_version": "0.135.0",
                      "product_paths": "native C++ public API, SQLite producer, CLI and worker",
                      "tls_qualified": False, "durable_ack_qualified": False}, ensure_ascii=True))


if __name__ == "__main__":
    main()
