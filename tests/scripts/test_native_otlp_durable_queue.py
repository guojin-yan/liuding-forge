#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Real Collector process-crash boundary with a native producer and volatile control.

Python supervises independent fixtures only. The C++ transport owns the OTLP
request, response validation and same-process replay. This selected Collector
configuration adds a durable enqueue boundary; generic HTTP 200 does not.
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


def verify_request(raw):
    value = json.loads(raw)
    resources = value["resourceLogs"]
    require(len(resources) == 1, "Unexpected resource count")
    resource = resources[0]
    fields = attributes(resource["resource"]["attributes"])
    require(fields["service.name"] == {"stringValue": "liuding-forge"} and
            fields["service.instance.id"] == {"stringValue": "http.integration.instance"},
            "Native resource identity changed")
    scopes = resource["scopeLogs"]
    require(len(scopes) == 1 and scopes[0]["scope"]["name"] == "jyppx.telemetry" and
            scopes[0]["scope"]["version"] == "1", "Native scope changed")
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
    require(source["payload"]["kind"] == 3 and source["payload"]["body"] == "transport lesson",
            "Native log payload changed")
    require(json.loads(record["body"]["stringValue"]) == source["payload"],
            "Collector lost native log fields")
    require(json.loads(fields["jyppx.identity"]["stringValue"]) == source["identity"],
            "Collector lost source identity")
    return source


@contextmanager
def backend():
    state = {"accept": False, "requests": [], "errors": []}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                require(self.path == "/v1/logs" and 0 < length <= 65536,
                        "Unexpected downstream request path/size")
                require(self.headers.get("Content-Type", "").split(";")[0] == "application/json",
                        "Collector JSON encoding changed")
                require(self.headers.get("Content-Encoding", "identity") in ("", "identity"),
                        "Unexpected downstream compression")
                raw = self.rfile.read(length)
                require(len(raw) == length, "Incomplete downstream request")
                source = verify_request(raw)
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
                    state["requests"].append({"status": status, "source": source,
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


def run_case(args, root, persistent):
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
    with backend() as (downstream_port, state, lock):
        collector = Collector(args.collector, config_path, root, port, downstream_port)
        try:
            collector.start()
            result = subprocess.run([str(args.harness), str(port), "logs"], cwd=root,
                                    stdin=subprocess.DEVNULL, capture_output=True, timeout=12,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            (root / "native.stdout").write_bytes(result.stdout)
            (root / "native.stderr").write_bytes(result.stderr)
            require(result.returncode == 0 and b"native-http-contract-ok" in result.stdout,
                    "Native acceptance/replay contract failed")
            wait_for(lambda: bool(state["requests"]) or bool(state["errors"]), 5,
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
                wait_for(lambda: len(state["requests"]) > before or bool(state["errors"]), 8,
                         "Acknowledged event was not recovered from the persistent queue")
                time.sleep(0.5)
                require(len(state["requests"]) == before + 1 and state["requests"][-1]["status"] == 200,
                        "Recovered delivery was missing or duplicated in this observation")
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
                require(collector.process.poll() is None and len(state["requests"]) == before + 1,
                        "Completed delivery replayed after queue acknowledgement")
                collector.stop()
            report = {"persistent": persistent, "native_invocations": 1,
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
        cases = [run_case(args, root / name, persistent) for name, persistent in
                 (("persistent-crash-recovery", True), ("volatile-loss-control", False))]
        report.update({"passed": True, "cases": cases, "native_invocations": 2,
                       "collector_invocations": 5,
                       "qualification": "pinned Collector fsync queue survives process kill; volatile ACK loses event",
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
