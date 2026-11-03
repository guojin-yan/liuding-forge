#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Real Collector process-crash boundary with a native producer and volatile control.

Python supervises independent fixtures only. The C++ transport owns the OTLP
request, response validation and same-process replay. This selected Collector
configuration adds a durable enqueue boundary for logs, metrics with their
audit log, and traces; generic HTTP 200 does not.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def wait_for(predicate, timeout, message):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.025)
    raise AssertionError(message)


def attributes(values):
    result = {}
    for value in values:
        require(value["key"] not in result, "Duplicate OTLP attribute")
        result[value["key"]] = value["value"]
    return result


def verify_request(raw, path, signal):
    value = json.loads(raw)
    resource_key, scope_key = {"/v1/logs": ("resourceLogs", "scopeLogs"),
                               "/v1/metrics": ("resourceMetrics", "scopeMetrics"),
                               "/v1/traces": ("resourceSpans", "scopeSpans")}[path]
    resources = value[resource_key]
    require(len(resources) == 1, "Unexpected resource count")
    resource = resources[0]
    fields = attributes(resource["resource"]["attributes"])
    require(fields["service.name"] == {"stringValue": "liuding-forge"} and
            fields["service.instance.id"] == {"stringValue": "http.integration.instance"},
            "Native resource identity changed")
    scopes = resource[scope_key]
    require(len(scopes) == 1 and scopes[0]["scope"]["name"] == "jyppx.telemetry" and
            scopes[0]["scope"]["version"] == "1", "Native scope changed")
    if path == "/v1/metrics":
        metrics = scopes[0]["metrics"]
        require(signal == "metrics" and len(metrics) == 1, "Unexpected native metric count")
        metric = metrics[0]
        require(metric["description"] == "http.integration.loss" and "gauge" in metric,
                "Native metric definition/type changed")
        points = metric["gauge"]["dataPoints"]
        require(len(points) == 1 and points[0]["asDouble"] == 0.125 and
                str(points[0]["timeUnixNano"]) == "1700000000000000000",
                "Native metric point/value/time changed")
        fields = attributes(points[0]["attributes"])
        definition = json.loads(fields["jyppx.metric_definition"]["stringValue"])
        require(definition["metric_id"] == "http.integration.loss" and
                metric["name"] == "jyppx.scalar." + definition["definition_digest"],
                "Native metric definition identity changed")
        return value
    if path == "/v1/traces":
        spans = scopes[0]["spans"]
        require(signal == "traces" and len(spans) == 1, "Unexpected native span count")
        span = spans[0]
        require(span["name"] == "transport-span" and span["status"]["code"] == 1,
                "Native span name/status changed")
        require(str(span["startTimeUnixNano"]) == "1700000000000000000" and
                str(span["endTimeUnixNano"]) == "1700000000000000010",
                "Native span clock mapping changed")
        for key, domain, opaque, size in (("traceId", b"jyppx.trace/v1", b"opaque-trace", 16),
                                           ("spanId", b"jyppx.span/v1", b"opaque-span", 8)):
            expected = hashlib.sha256(domain + b"\x00" + opaque).digest()[:size].hex()
            require(span[key].lower() == expected, "Native trace/span identity changed")
        fields = attributes(span["attributes"])
        require(json.loads(fields["jyppx.identity"]["stringValue"]) ==
                {"stream_id": "http.integration", "event_id": "1",
                 "idempotency_key": "1", "sequence": 1}, "Native span event identity changed")
        require(fields["jyppx.clock_id"] == {"stringValue": "producer"} and
                json.loads(fields["jyppx.clock_mapping"]["stringValue"]) ==
                {"monotonic_ns": 10, "unix_ns": 1700000000000000000},
                "Native source clock metadata changed")
        return value
    records = scopes[0]["logRecords"]
    require(len(records) == 1, "Missing or extra native event")
    record = records[0]
    require(str(record["timeUnixNano"]) == "1700000000000000000" and
            record["severityNumber"] == 9, "Native log timestamp/severity changed")
    fields = attributes(record["attributes"])
    source = json.loads(fields["jyppx.envelope"]["stringValue"])
    require(source["identity"] == {"stream_id": "http.integration", "event_id": "1",
                                   "idempotency_key": "1", "sequence": 1},
            "Native event identity changed")
    if signal == "logs":
        require(source["payload"]["kind"] == 3 and source["payload"]["body"] == "transport lesson",
                "Native log payload changed")
    else:
        require(signal == "metrics" and source["payload"]["kind"] == 2,
                "Native metric audit payload changed")
        points = source["payload"]["points"]
        require(len(points) == 1 and points[0]["definition"]["metric_id"] == "http.integration.loss" and
                points[0]["value"] == 0.125, "Native audited metric changed")
    require(json.loads(record["body"]["stringValue"]) == source["payload"],
            "Collector lost native log fields")
    require(json.loads(fields["jyppx.identity"]["stringValue"]) == source["identity"],
            "Collector lost source identity")
    return source


@contextmanager
def backend(signal):
    state = {"accept": False, "requests": [], "errors": []}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                paths = ["/v1/metrics", "/v1/logs"] if signal == "metrics" else ["/v1/" + signal]
                require(self.path in paths and 0 < length <= 65536,
                        "Unexpected downstream request path/size")
                require(self.headers.get("Content-Type", "").split(";")[0] == "application/json",
                        "Collector JSON encoding changed")
                require(self.headers.get("Content-Encoding", "identity") in ("", "identity"),
                        "Unexpected downstream compression")
                raw = self.rfile.read(length)
                require(len(raw) == length, "Incomplete downstream request")
                source = verify_request(raw, self.path, signal)
                with lock:
                    status = 200 if state["accept"] else 503
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b"{}")
                self.wfile.flush()
                with lock:
                    require(len(state["requests"]) < 128, "Downstream request count bound")
                    state["requests"].append({"status": status, "source": source, "path": self.path,
                                              "body_sha256": hashlib.sha256(raw).hexdigest()})
            except BaseException:
                with lock:
                    state["errors"].append(traceback.format_exc())
            finally:
                self.close_connection = True

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 0.2
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, state, lock
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        require(not thread.is_alive(), "Downstream fixture failed to stop")


class Collector:
    def __init__(self, executable, config, root, port, backend_port):
        self.root, self.executable, self.config = root, executable, config
        self.port, self.backend_port = port, backend_port
        self.process = None
        self.stages = []

    def start(self):
        require(self.process is None, "Collector already active")
        stage = len(self.stages) + 1
        self.stream = (self.root / f"collector-{stage}.log").open("xb")
        environment = dict(os.environ, JYPPX_OTELCOL_ENDPOINT=f"127.0.0.1:{self.port}",
                           JYPPX_OTELCOL_BACKEND=f"http://127.0.0.1:{self.backend_port}",
                           JYPPX_OTELCOL_QUEUE_DIRECTORY=str(self.root / "queue"))
        self.process = subprocess.Popen([str(self.executable), "--config", str(self.config)],
                                        cwd=self.root, env=environment, stdin=subprocess.DEVNULL,
                                        stdout=self.stream, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.stages.append({"pid": self.process.pid, "stage": stage})

        def ready():
            require(self.process.poll() is None, "Collector stopped at startup; inspect its saved log")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    return True
            except OSError:
                return False
        wait_for(ready, 10, "Collector startup deadline")

    def stop(self, crash=False):
        if self.process is None:
            return
        if self.process.poll() is None:
            if crash:
                self.process.kill()  # SIGKILL / TerminateProcess, never a drain signal.
            else:
                self.process.terminate()
        try:
            code = self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            code = self.process.wait(timeout=5)
        self.stages[-1].update({"returncode": code, "forced_crash": crash, "reaped": True})
        self.stream.close()
        require((self.root / f"collector-{len(self.stages)}.log").stat().st_size < 1048576,
                "Collector diagnostic bound")
        self.process = None


def run_case(args, root, persistent, signal):
    root.mkdir()
    (root / "queue").mkdir()
    config = args.config.read_text(encoding="utf-8")
    marker = "      storage: file_storage\n"
    require(config.count(marker) == 1, "Persistent storage configuration changed")
    if not persistent:
        config = config.replace(marker, "")
    config_path = root / "collector.yaml"
    config_path.write_text(config, encoding="utf-8")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    expected_paths = ["/v1/metrics", "/v1/logs"] if signal == "metrics" else ["/v1/" + signal]
    with backend(signal) as (downstream_port, state, lock):
        collector = Collector(args.collector, config_path, root, port, downstream_port)
        try:
            collector.start()
            result = subprocess.run([str(args.harness), str(port), signal], cwd=root,
                                    stdin=subprocess.DEVNULL, capture_output=True, timeout=12,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            (root / "native.stdout").write_bytes(result.stdout)
            (root / "native.stderr").write_bytes(result.stderr)
            require(result.returncode == 0 and b"native-http-contract-ok" in result.stdout,
                    "Native acceptance/replay contract failed")
            wait_for(lambda: set(expected_paths) <= {item["path"] for item in state["requests"]} or bool(state["errors"]), 5,
                     "Collector did not reach the blocked downstream")
            require(not state["errors"], state["errors"])
            require(all(item["status"] == 503 for item in state["requests"]),
                    "Downstream accepted before the Collector crash")
            collector.stop(crash=True)
            before = len(state["requests"])
            files = {str(path.relative_to(root)): identity(path)
                     for path in sorted((root / "queue").rglob("*")) if path.is_file()}
            if persistent:
                require(bool(files), "No persistent queue file after acknowledged acceptance")
            with lock:
                state["accept"] = True
            collector.start()  # Same queue, no producer replay after the crash.
            if persistent:
                wait_for(lambda: len(state["requests"]) >= before + len(expected_paths) or bool(state["errors"]), 8,
                         "Acknowledged event was not recovered from the persistent queue")
                time.sleep(0.5)
                recovered = state["requests"][before:]
                require(sorted(item["path"] for item in recovered) == sorted(expected_paths) and
                        all(item["status"] == 200 for item in recovered),
                        "Recovered delivery was missing or duplicated in this observation")
                for item in recovered:
                    original = next(prior for prior in state["requests"][:before] if prior["path"] == item["path"])
                    require(item["source"] == original["source"], "Recovered native content changed")
                if signal == "metrics":
                    metric = next(item["source"] for item in recovered if item["path"] == "/v1/metrics")
                    audit = next(item["source"] for item in recovered if item["path"] == "/v1/logs")
                    definition = json.loads(attributes(metric["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]
                                                      ["gauge"]["dataPoints"][0]["attributes"])
                                            ["jyppx.metric_definition"]["stringValue"])
                    require(definition == audit["payload"]["points"][0]["definition"],
                            "Recovered metric and source audit are not paired")
            else:
                time.sleep(2)
                require(len(state["requests"]) == before,
                        "Volatile control unexpectedly recovered an acknowledged event")
            require(not state["errors"], state["errors"])
            require(collector.process.poll() is None, "Restarted Collector stopped unexpectedly")
            collector.stop()
            if persistent:
                collector.start()  # Confirm completed queue items stay removed after another restart.
                time.sleep(2)
                require(collector.process.poll() is None and len(state["requests"]) == before + len(expected_paths),
                        "Completed delivery replayed after queue acknowledgement")
                collector.stop()
            report = {"persistent": persistent, "signal": signal, "native_invocations": 1,
                      "expected_paths": expected_paths,
                      "native_protocol_accepted_and_process_local_duplicate": True,
                      "producer_replay_after_crash": 0, "queue_files_after_crash": files,
                      "collector_stages": collector.stages, "downstream_requests": state["requests"],
                      "downstream_accepted_after_restart": len(state["requests"]) - before,
                      "passed": True}
            save(root / "report.json", report)
            return report
        finally:
            collector.stop()
            save(root / "observed.json", {"requests": state["requests"], "errors": state["errors"],
                                          "collector_stages": collector.stages})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("harness", "collector", "artifacts", "config", "work-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    for key in ("harness", "collector", "artifacts", "config"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="durable-queue-", dir=args.work_dir)).resolve()
    sources = {key: identity(getattr(args, key)) for key in ("harness", "collector", "artifacts", "config")}
    metadata = json.loads(args.artifacts.read_text(encoding="utf-8"))
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
        cases = [run_case(args, root / (name if signal == "logs" else signal + "-" + name), persistent, signal)
                 for signal in ("logs", "metrics", "traces")
                 for name, persistent in (("persistent-crash-recovery", True), ("volatile-loss-control", False))]
        report.update({"passed": True, "cases": cases, "native_invocations": 6,
                       "collector_invocations": 15,
                       "qualification": "pinned Collector fsync queues recover logs, metrics with audit, and traces after process kill; volatile controls lose acknowledged events",
                       "power_loss_disk_failure_exactly_once_or_generic_http200_qualified": False})
    except BaseException:
        report["failure"] = traceback.format_exc()
        raise
    finally:
        report["inputs_preserved"] = all(identity(getattr(args, key)) == row for key, row in sources.items())
        save(root / "report.json", report)
    require(report["inputs_preserved"], "Validation inputs changed")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
