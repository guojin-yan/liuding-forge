#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Real bounded Collector overflow, native prefix acknowledgement and retry.

Python only supervises fixtures. The recording hop forwards the native wire
bytes and actual Collector status without synthesizing acknowledgements.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback

_spec = importlib.util.spec_from_file_location(
    "durable_queue_fixture", Path(__file__).with_name("test_native_otlp_durable_queue.py"))
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)
require, save, identity = fixture.require, fixture.save, fixture.identity


def verify_event(raw, path, signal):
    """Decode independently observed JSON, including every source identity."""
    value = json.loads(raw)
    resource_key, scope_key, item_key = {
        "/v1/logs": ("resourceLogs", "scopeLogs", "logRecords"),
        "/v1/metrics": ("resourceMetrics", "scopeMetrics", "metrics"),
        "/v1/traces": ("resourceSpans", "scopeSpans", "spans"),
    }[path]
    require(len(value[resource_key]) == 1, "Unexpected resource count")
    resource = value[resource_key][0]
    fields = fixture.attributes(resource["resource"]["attributes"])
    require(fields["service.name"] == {"stringValue": "liuding-forge"} and
            fields["service.instance.id"] == {"stringValue": "http.integration.instance"},
            "Native resource identity changed")
    scopes = resource[scope_key]
    require(len(scopes) == 1 and scopes[0]["scope"] == {"name": "jyppx.telemetry", "version": "1"},
            "Native scope changed")
    require(len(scopes[0][item_key]) == 1, "Missing or extra native event")
    item = scopes[0][item_key][0]
    if path == "/v1/metrics":
        require(signal == "metrics" and item["description"] == "http.integration.loss",
                "Unexpected metric definition")
        points = item["gauge"]["dataPoints"]
        require(len(points) == 1, "Unexpected metric point count")
        point = points[0]
        sequence = int(point["asDouble"] * 8)
        require(point["asDouble"] == sequence / 8 and
                str(point["timeUnixNano"]) == "1700000000000000000", "Metric value/time changed")
        definition = json.loads(fixture.attributes(point["attributes"])
                                ["jyppx.metric_definition"]["stringValue"])
        require(definition["metric_id"] == "http.integration.loss" and
                item["name"] == "jyppx.scalar." + definition["definition_digest"],
                "Metric definition identity changed")
        source = None
    else:
        fields = fixture.attributes(item["attributes"])
        event_identity = json.loads(fields["jyppx.identity"]["stringValue"])
        sequence = event_identity["sequence"]
        require(event_identity == {"stream_id": "http.integration", "event_id": str(sequence),
                                   "idempotency_key": str(sequence), "sequence": sequence},
                "Native event identity changed")
        definition, source = None, None
        if path == "/v1/logs":
            require(str(item["timeUnixNano"]) == "1700000000000000000" and
                    item["severityNumber"] == 9, "Log timestamp/severity changed")
            source = json.loads(fields["jyppx.envelope"]["stringValue"])
            require(source["identity"] == event_identity and
                    json.loads(item["body"]["stringValue"]) == source["payload"],
                    "Complete source audit changed")
            if signal == "logs":
                require(source["payload"]["kind"] == 3 and
                        source["payload"]["body"] == "transport lesson", "Log payload changed")
            else:
                require(signal == "metrics" and source["payload"]["kind"] == 2,
                        "Unexpected metric audit")
                points = source["payload"]["points"]
                require(len(points) == 1 and points[0]["value"] == sequence / 8,
                        "Audited metric value changed")
                definition = points[0]["definition"]
                require(definition["metric_id"] == "http.integration.loss", "Audited definition changed")
        else:
            require(signal == "traces" and item["name"] == "transport-span" and
                    item["status"]["code"] == 1, "Span name/status changed")
            require(str(item["startTimeUnixNano"]) == "1700000000000000000" and
                    str(item["endTimeUnixNano"]) == "1700000000000000010", "Span time changed")
            for key, domain, opaque, size in (("traceId", b"jyppx.trace/v1", "opaque-trace-", 16),
                                             ("spanId", b"jyppx.span/v1", "opaque-span-", 8)):
                expected = hashlib.sha256(domain + b"\x00" + (opaque + str(sequence)).encode()).digest()[:size].hex()
                require(item[key].lower() == expected, "Trace/span identity changed")
            require(fields["jyppx.clock_id"] == {"stringValue": "producer"} and
                    json.loads(fields["jyppx.clock_mapping"]["stringValue"]) ==
                    {"monotonic_ns": 10, "unix_ns": 1700000000000000000}, "Source clock changed")
    require(type(sequence) is int and 1 <= sequence <= 16, "Sequence outside bounded native batch")
    return {"sequence": sequence, "definition": definition, "source": source, "decoded": value}


@contextmanager
def http_fixture(handler, role):
    server = HTTPServer(("127.0.0.1", 0), handler)
    server.role = role
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        require(not server_thread.is_alive(), "HTTP fixture failed to stop")


def run_case(args, root, signal):
    root.mkdir()
    (root / "queue").mkdir()
    config = args.config.read_bytes()
    config_path = root / "collector.yaml"
    config_path.write_bytes(config)
    require(b"      queue_size: 4\n" in config and b"      num_consumers: 1\n" in config and
            b"      storage: file_storage\n" in config and b"    fsync: true\n" in config,
            "Selected bounded persistent queue changed")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        collector_port = probe.getsockname()[1]
    paths = ["/v1/metrics", "/v1/logs"] if signal == "metrics" else ["/v1/" + signal]
    state = {"released": False, "phase": "initial", "downstream": [], "ingress": [], "errors": []}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            connection = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                require(self.path in paths and 0 < length <= 65536, "Unexpected request path/size")
                require(self.headers.get("Content-Encoding", "identity") in ("", "identity"),
                        "Unexpected request compression")
                raw = self.rfile.read(length)
                require(len(raw) == length, "Incomplete request")
                if self.server.role == "downstream":
                    require(self.headers.get("Content-Type", "").split(";")[0] == "application/json",
                            "Collector JSON encoding changed")
                    row = verify_event(raw, self.path, signal)
                    with lock:
                        # Keep metric delivery live but block its audit queue;
                        # the failed event then has a real partial side effect.
                        status = 200 if state["released"] or self.path == "/v1/metrics" else 503
                    reply, content_type = b"{}", "application/json"
                    category = "downstream"
                else:
                    require(self.headers.get("Content-Type") == "application/x-protobuf",
                            "Native protobuf media type changed")
                    connection = http.client.HTTPConnection("127.0.0.1", collector_port, timeout=3)
                    connection.request("POST", self.path, raw, {"Content-Type": "application/x-protobuf"})
                    response = connection.getresponse()
                    status, reply = response.status, response.read(65537)
                    require(len(reply) <= 65536, "Collector response bound")
                    content_type = response.getheader("Content-Type", "application/x-protobuf")
                    row, category = {"phase": state["phase"]}, "ingress"
                with lock:
                    index = len(state[category]) + 1
                    require(index <= 512, "Request observation bound")
                    filename = f"{category}-{index:04d}.bin"
                    (root / filename).write_bytes(raw)
                    row.update({"path": self.path, "status": status, "body": filename,
                                "body_sha256": hashlib.sha256(raw).hexdigest()})
                    state[category].append(row)
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(reply)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(reply)
                self.wfile.flush()
            except BaseException:
                with lock:
                    state["errors"].append(traceback.format_exc())
            finally:
                if connection is not None:
                    connection.close()
                self.close_connection = True

    native, collector = None, None
    lifecycle = {}
    report = {"passed": False, "signal": signal, "batch_events": 16}

    def successful():
        with lock:
            require(not state["errors"], state["errors"])
            return Counter((row["path"], row["sequence"]) for row in state["downstream"] if row["status"] == 200)

    try:
        with http_fixture(Handler, "downstream") as downstream_port, http_fixture(Handler, "ingress") as ingress_port:
            collector = fixture.Collector(args.collector, config_path, root, collector_port, downstream_port)
            collector.start()
            with (root / "native.stdout").open("xb") as stdout, (root / "native.stderr").open("xb") as stderr:
                native = subprocess.Popen([str(args.harness), str(ingress_port), "queue-full-" + signal],
                                          cwd=root, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                lifecycle["pid"] = native.pid

                def overflowed():
                    require(native.poll() is None, "Native process stopped before overflow synchronization")
                    require(not state["errors"], state["errors"])
                    return b"native-http-queue-full " in (root / "native.stdout").read_bytes()

                fixture.wait_for(overflowed, 15, "Native overflow deadline")
                marker = (root / "native.stdout").read_text().strip().split()
                require(len(marker) == 4 and marker[0] == "native-http-queue-full", "Unexpected native marker")
                prefix, count, status = map(int, marker[1:])
                require(0 < prefix < count == 16 and status in (429, 503), "Invalid overflow result")
                initial = list(state["ingress"])
                require(len(initial) == prefix * len(paths) + len(paths), "Native acknowledged prefix changed")
                require(all(row["status"] == 200 for row in initial[:-1]) and initial[-1]["status"] == status,
                        "Refusal was not an actual Collector failure after the accepted prefix")
                for index, row in enumerate(initial):
                    row["sequence"] = index // len(paths) + 1
                    require(row["path"] == paths[index % len(paths)], "Native signal order changed")
                pending = Counter((path, sequence) for sequence in range(1, prefix + 1) for path in paths)
                if signal == "metrics":
                    require(initial[-1]["path"] == "/v1/logs", "Expected partial metric/audit overflow")
                    pending[("/v1/metrics", prefix + 1)] += 1
                with lock:
                    state["released"] = True
                fixture.wait_for(lambda: successful() == pending, 10, "Accepted queue did not drain")
                with lock:
                    state["phase"] = "retry"
                native.stdin.write(b"retry\n")
                native.stdin.flush()
                code = native.wait(timeout=15)
                lifecycle.update({"returncode": code, "reaped": True})
                native.stdin.close()
                require(code == 0 and b"native-http-contract-ok" in (root / "native.stdout").read_bytes(),
                        "Native retry/disposition/flush contract failed")
                expected = Counter((path, sequence) for sequence in range(1, 17) for path in paths)
                if signal == "metrics":
                    expected[("/v1/metrics", prefix + 1)] += 1
                fixture.wait_for(lambda: successful() == expected, 10, "Retried batch did not arrive completely")
                time.sleep(0.25)
                require(successful() == expected, "Unexpected downstream duplicate or missing delivery")
                retry = state["ingress"][len(initial):]
                require(len(retry) == (16 - prefix) * len(paths) and
                        all(row["status"] == 200 and row["phase"] == "retry" for row in retry),
                        "Native prefix replayed or suffix did not complete")
                for index, row in enumerate(retry):
                    row["sequence"] = prefix + index // len(paths) + 1
                    require(row["path"] == paths[index % len(paths)], "Retried signal order changed")
                for before, after in zip(initial[-len(paths):], retry[:len(paths)]):
                    require((root / before["body"]).read_bytes() == (root / after["body"]).read_bytes(),
                            "Retry changed the original native protobuf bytes")
                grouped = {}
                for row in state["downstream"]:
                    key = (row["path"], row["sequence"])
                    require(key not in grouped or grouped[key]["decoded"] == row["decoded"],
                            "Collector retry changed complete event content")
                    grouped[key] = row
                if signal == "metrics":
                    require(all(grouped[("/v1/metrics", seq)]["definition"] ==
                                grouped[("/v1/logs", seq)]["definition"] for seq in range(1, 17)),
                            "Metric/source audit definitions differ")
                collector.stop()
                report.update({"passed": True, "accepted_prefix_events": prefix, "collector_failure_status": status,
                               "prefix_not_resent": True, "retry_wire_bytes_preserved": True,
                               "native_all_duplicate_check": True, "final_exported_events": 16,
                               "partial_metric_repeated_sequence": prefix + 1 if signal == "metrics" else None,
                               "exactly_once_qualified": False})
    except BaseException:
        report["failure"] = traceback.format_exc()
        raise
    finally:
        if native is not None and native.poll() is None:
            native.kill()
            lifecycle.update({"returncode": native.wait(timeout=5), "reaped": True, "forced_cleanup": True})
        if native is not None and native.stdin is not None:
            native.stdin.close()
        if collector is not None:
            collector.stop()
        report.update({"native_process": lifecycle, "collector_stages": collector.stages if collector else [],
                       "observations": state})
        save(root / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("harness", "collector", "artifacts", "config", "work-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    for key in ("harness", "collector", "artifacts", "config"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="queue-full-", dir=args.work_dir)).resolve()
    sources = {key: identity(getattr(args, key)) for key in ("harness", "collector", "artifacts", "config")}
    metadata = json.loads(args.artifacts.read_bytes())
    platform = "windows" if sys.platform == "win32" else "linux"
    pinned = [row for row in metadata["archives"] if f"_{platform}_amd64" in row["filename"]]
    require(len(pinned) == 1 and sources["collector"]["sha256"] == pinned[0]["executable_sha256"],
            "Pinned official Collector binary changed")
    version = subprocess.run([str(args.collector), "--version"], capture_output=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    require(version.returncode == 0 and version.stdout.decode().strip() == "otelcol-contrib version 0.135.0",
            "Pinned Collector version changed")
    report = {"passed": False, "inputs": sources, "work_directory": str(root)}
    try:
        cases = [run_case(args, root / signal, signal) for signal in ("logs", "metrics", "traces")]
        report.update({"passed": True, "cases": cases, "native_invocations": 3, "collector_invocations": 3,
                       "qualification": "selected Collector overflow preserves native accepted prefixes and permits byte-identical retry after drain; partial metric delivery can repeat"})
    except BaseException:
        report["failure"] = traceback.format_exc()
        raise
    finally:
        report["inputs_preserved"] = all(identity(getattr(args, key)) == row for key, row in sources.items())
        save(root / "report.json", report)
    require(report["inputs_preserved"], "Validation inputs changed")
    print(json.dumps({"passed": report["passed"], "cases": len(cases), "report": str(root / "report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
