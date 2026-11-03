#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""A real Collector accepts a native request, but its response never arrives.

Only the test hop discards one response. The C++ sink owns acknowledgement,
prefix preservation, retry and process-local duplicate handling.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler
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
    "queue_full_fixture", Path(__file__).with_name("test_native_otlp_queue_full.py"))
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)
require, save, identity = fixture.require, fixture.save, fixture.identity


def run_case(args, root, signal, lost_path):
    root.mkdir()
    (root / "queue").mkdir()
    config_path = root / "collector.yaml"
    config_path.write_bytes(args.config.read_bytes())
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        collector_port = probe.getsockname()[1]
    paths = ["/v1/metrics", "/v1/logs"] if signal == "metrics" else ["/v1/" + signal]
    lost_index = len(paths) + paths.index(lost_path) + 1
    state = {"released": False, "phase": "initial", "ingress": [], "downstream": [], "errors": []}
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
                lost = False
                if self.server.role == "downstream":
                    require(self.headers.get("Content-Type", "").split(";")[0] == "application/json",
                            "Collector JSON encoding changed")
                    row = fixture.verify_event(raw, self.path, signal)
                    require(row["sequence"] <= 3, "Sequence outside the native three-event batch")
                    with lock:
                        status = 200 if state["released"] else 503
                    reply, content_type, category = b"{}", "application/json", "downstream"
                else:
                    require(self.headers.get("Content-Type") == "application/x-protobuf",
                            "Native protobuf media type changed")
                    connection = http.client.HTTPConnection("127.0.0.1", collector_port, timeout=3)
                    connection.request("POST", self.path, raw, {"Content-Type": "application/x-protobuf"})
                    response = connection.getresponse()
                    status, reply = response.status, response.read(65537)
                    require(status == 200 and len(reply) <= 65536, "Collector did not accept the original request")
                    content_type = response.getheader("Content-Type", "application/x-protobuf")
                    with lock:
                        lost = state["phase"] == "initial" and len(state["ingress"]) + 1 == lost_index
                    require(not lost or self.path == lost_path, "Wrong response selected for loss")
                    row = {"phase": state["phase"], "response_forwarded": not lost,
                           "collector_reply_sha256": hashlib.sha256(reply).hexdigest()}
                    category = "ingress"
                with lock:
                    index = len(state[category]) + 1
                    require(index <= 256, "Request observation bound")
                    filename = f"{category}-{index:04d}.bin"
                    (root / filename).write_bytes(raw)
                    row.update({"path": self.path, "status": status, "body": filename,
                                "body_sha256": hashlib.sha256(raw).hexdigest()})
                    state[category].append(row)
                    if category == "ingress":
                        (root / f"collector-reply-{index:04d}.bin").write_bytes(reply)
                if lost:
                    # Collector's complete 200 was received and saved above.
                    # Close before any status/header/body reaches the native sink.
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
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
    report = {"passed": False, "signal": signal, "lost_response_path": lost_path, "batch_events": 3}

    def successful():
        with lock:
            require(not state["errors"], state["errors"])
            return Counter((row["path"], row["sequence"]) for row in state["downstream"] if row["status"] == 200)

    try:
        with fixture.http_fixture(Handler, "downstream") as downstream_port, fixture.http_fixture(Handler, "ingress") as ingress_port:
            collector = fixture.fixture.Collector(args.collector, config_path, root, collector_port, downstream_port)
            collector.start()
            with (root / "native.stdout").open("xb") as stdout, (root / "native.stderr").open("xb") as stderr:
                native = subprocess.Popen([str(args.harness), str(ingress_port), "lost-reply-" + signal],
                                          cwd=root, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                lifecycle["pid"] = native.pid

                def response_lost():
                    require(native.poll() is None, "Native process stopped before lost-response synchronization")
                    require(not state["errors"], state["errors"])
                    return b"native-http-lost-reply " in (root / "native.stdout").read_bytes()

                fixture.fixture.wait_for(response_lost, 15, "Native lost-response deadline")
                require((root / "native.stdout").read_text().strip() == "native-http-lost-reply 1 3 lost",
                        "Unknown delivery incorrectly acknowledged an event or lost the accepted prefix")
                initial = list(state["ingress"])
                require(len(initial) == lost_index and
                        sum(not row["response_forwarded"] for row in initial) == 1 and
                        not initial[-1]["response_forwarded"], "Unexpected hidden resend or loss boundary")
                pending = Counter()
                for index, row in enumerate(initial):
                    row["sequence"] = index // len(paths) + 1
                    require(row["path"] == paths[index % len(paths)] and row["status"] == 200,
                            "Original request order or Collector acceptance changed")
                    pending[(row["path"], row["sequence"])] += 1
                require(not successful(), "Downstream accepted before the release barrier")
                with lock:
                    state["released"] = True
                fixture.fixture.wait_for(lambda: successful() == pending, 10,
                                         "Collector did not deliver the requests with lost responses")
                with lock:
                    state["phase"] = "retry"
                native.stdin.write(b"retry\n")
                native.stdin.flush()
                code = native.wait(timeout=15)
                lifecycle.update({"returncode": code, "reaped": True})
                native.stdin.close()
                require(code == 0 and b"native-http-contract-ok" in (root / "native.stdout").read_bytes(),
                        "Native retry/disposition/final acknowledgement contract failed")
                expected = Counter((path, seq) for seq in range(1, 4) for path in paths)
                repeated_paths = [row["path"] for row in initial if row["sequence"] == 2]
                expected.update((path, 2) for path in repeated_paths)
                fixture.fixture.wait_for(lambda: successful() == expected, 10,
                                         "Retried unknown-delivery batch did not arrive completely")
                time.sleep(0.25)
                require(successful() == expected, "Unexpected duplicate or missing downstream delivery")
                retry = state["ingress"][len(initial):]
                require(len(retry) == 2 * len(paths) and all(row["response_forwarded"] for row in retry),
                        "Previously confirmed prefix resent or unconfirmed suffix missing")
                for index, row in enumerate(retry):
                    row["sequence"] = 2 + index // len(paths)
                    require(row["phase"] == "retry" and row["path"] == paths[index % len(paths)] and
                            row["status"] == 200, "Retried request order or result changed")
                    before = next((prior for prior in initial if prior["sequence"] == row["sequence"] and
                                   prior["path"] == row["path"]), None)
                    if before is not None:
                        require((root / before["body"]).read_bytes() == (root / row["body"]).read_bytes(),
                                "Unknown delivery changed original native protobuf bytes on retry")
                grouped = {}
                for row in state["downstream"]:
                    key = (row["path"], row["sequence"])
                    require(key not in grouped or grouped[key]["decoded"] == row["decoded"],
                            "Lost-response retry changed complete downstream content")
                    grouped[key] = row
                if signal == "metrics":
                    require(all(grouped[("/v1/metrics", seq)]["definition"] ==
                                grouped[("/v1/logs", seq)]["definition"] for seq in range(1, 4)),
                            "Metric/source audit definitions differ")
                collector.stop()
                report.update({"passed": True, "accepted_prefix_events": 1, "final_exported_events": 3,
                               "collector_accepted_but_native_unacknowledged": True,
                               "prefix_not_resent": True, "retry_wire_bytes_preserved": True,
                               "native_all_duplicate_check": True, "duplicate_event_sequence": 2,
                               "duplicated_downstream_paths": repeated_paths, "exactly_once_qualified": False})
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
    root = Path(tempfile.mkdtemp(prefix="lost-reply-", dir=args.work_dir)).resolve()
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
        scenarios = [("logs", "logs", "/v1/logs"), ("metrics-primary", "metrics", "/v1/metrics"),
                     ("metrics-audit", "metrics", "/v1/logs"), ("traces", "traces", "/v1/traces")]
        cases = [run_case(args, root / name, signal, path) for name, signal, path in scenarios]
        report.update({"passed": True, "cases": cases, "native_invocations": 4, "collector_invocations": 4,
                       "qualification": "selected Collector accepted but native response lost; prefix preserved and original bytes retried; externally accepted unknown events repeat"})
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
