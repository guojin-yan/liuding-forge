# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent stalled-peer cancellation/cleanup checker; never a product path."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
from socketserver import BaseRequestHandler, ThreadingTCPServer
import subprocess
import threading
import time


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Fixture:
    def __init__(self, phase):
        self.phase = phase
        self.requests, self.errors, self.closed = [], [], []
        self.stalled = queue.Queue()
        self.lock = threading.Lock()
        owner = self

        def await_close(connection, index):
            connection.settimeout(5)
            total = 0
            while True:
                try:
                    data = connection.recv(4096)
                except ConnectionResetError:
                    data = b""
                if not data:
                    owner.closed.append(index)
                    return
                total += len(data)
                require(total <= 65536, "Stalled connection received unbounded extra data")

        class HttpHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def do_POST(self):
                try:
                    size = int(self.headers.get("Content-Length", "-1"))
                    require(0 < size <= 16 * 1024 * 1024, "Invalid request bound")
                    require(self.path == "/v1/logs", "Unexpected OTLP path")
                    require(self.headers.get("Content-Type") == "application/x-protobuf", "Invalid media type")
                    payload = self.rfile.read(size)
                    require(len(payload) == size, "Request did not finish uploading")
                    with owner.lock:
                        index = len(owner.requests)
                        owner.requests.append({"index": index, "body_sha256": hashlib.sha256(payload).hexdigest(),
                                               "body_bytes": len(payload), "path": self.path})
                    if index % 2 == 0:
                        if phase == "body":
                            self.send_response(200)
                            self.send_header("Content-Type", "application/x-protobuf")
                            self.send_header("Content-Length", "2")
                            self.end_headers()
                            self.wfile.write(b"\x0a")  # A deliberately incomplete official response.
                            self.wfile.flush()
                        owner.stalled.put(index)
                        await_close(self.connection, index)
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/x-protobuf")
                        self.send_header("Content-Length", "0")
                        self.send_header("Connection", "close")
                        self.end_headers()
                    self.close_connection = True
                except BaseException as error:
                    owner.errors.append(repr(error))

        class TlsHandler(BaseRequestHandler):
            def handle(self):
                try:
                    self.request.settimeout(5)
                    hello = self.request.recv(4096)
                    require(hello and hello[0] == 0x16, "No native TLS ClientHello reached the peer")
                    with owner.lock:
                        index = len(owner.requests)
                        owner.requests.append({"index": index, "client_hello_bytes": len(hello)})
                    owner.stalled.put(index)
                    await_close(self.request, index)
                except BaseException as error:
                    owner.errors.append(repr(error))

        class TlsServer(ThreadingTCPServer):
            allow_reuse_address = True

        self.server = (TlsServer if phase == "tls" else ThreadingHTTPServer)(
            ("127.0.0.1", 0), TlsHandler if phase == "tls" else HttpHandler)
        self.server.daemon_threads = False
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02})

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        require(not self.thread.is_alive(), "Stalled-peer fixture did not stop")


def run(executable, mode):
    phase = mode.split("-", 1)[1]
    stop = mode.startswith("stop-")
    measured_rounds = 8 if stop else 2
    max_rounds = 32 + measured_rounds
    rows = []
    with Fixture(phase) as fixture:
        process = subprocess.Popen([str(executable), str(fixture.server.server_address[1]), mode],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        native_lines = []
        native_rows = queue.Queue()

        def read_native():
            for line in iter(process.stdout.readline, b""):
                native_lines.append(line)
                native_rows.put(line)

        reader = threading.Thread(target=read_native)
        reader.start()
        collected = None

        def collect(timeout):
            nonlocal collected
            if collected is not None:
                return collected
            if process.stdin is not None:
                process.stdin.close()
                process.stdin = None
            process.wait(timeout=timeout)
            reader.join(5)
            require(not reader.is_alive(), "Native output reader did not stop")
            collected = b"".join(native_lines), process.stderr.read(65537)
            return collected

        try:
            index = 0
            last_progress = started = time.monotonic()
            summary = None
            while summary is None:
                while not native_rows.empty():
                    row = json.loads(native_rows.get_nowait().decode("utf8"))
                    rows.append(row)
                    last_progress = time.monotonic()
                    if "passed" in row:
                        summary = row
                if summary is not None:
                    break
                if process.poll() is not None and not reader.is_alive():
                    break
                require(time.monotonic() - started < 150, "Cancellation checker exceeded its total bound")
                require(time.monotonic() - last_progress < 8, "Native stalled-stage/round progress stopped")
                try:
                    observed = fixture.stalled.get(timeout=0.05)
                except queue.Empty:
                    continue
                require(index < max_rounds, "Native warmup exceeded its declared bound")
                require(observed == (index if phase == "tls" else index * 2), "Hidden request/retry occurred")
                last_progress = time.monotonic()
                if stop:
                    # Let the native receive/read/handshake wait become idle;
                    # cancellation is causally after this server-side stage.
                    time.sleep(0.05)
                    process.stdin.write(b"x")
                    process.stdin.flush()
                index += 1
            output, error = collect(timeout=10)
            require(process.returncode == 0, (mode, process.returncode, output, error))
            require(not error and len(output) < 65536, "Unexpected native diagnostic output")
            rows = [json.loads(line) for line in output.decode("utf8").splitlines()]
            require(rows and rows[-1].get("passed"), "Missing native observations")
            warmup = rows[-1]["warmup_rounds"]
            rounds = warmup + measured_rounds
            require(2 <= warmup <= 32 and index == rounds and len(rows) == rounds + 1, "Bounded warmup/round count drift")
            require(rows[-1]["rounds"] == rounds and rows[-1]["measured_rounds"] == measured_rounds, "Measured round count drift")
            require(all(row["warmup"] == (i < warmup) for i, row in enumerate(rows[:-1])), "Warmup/measured phase drift")
            if os.name == "nt":
                require(warmup >= 3 and all(row["handles"] == rows[-1]["baseline_handles"]
                    for row in rows[warmup - 3:warmup]), "Windows runtime warmup did not stabilize")
            require(all(row["handles"] == rows[-1]["baseline_handles"] for row in rows[warmup:-1]), "Strict handle/fd restoration failed")
            require(all(row["elapsed_ms"] < (1000 if stop else 2500) for row in rows[:-1]), "Cancellation/deadline wait bound failed")
        except BaseException as failure:
            # Preserve startup/native diagnostics instead of replacing the
            # useful failure with an empty stalled-stage queue timeout.
            status_before_cleanup = process.poll()
            if status_before_cleanup is None:
                # EOF also releases a canceller blocked in the harness stdin
                # gate if validation failed before its first network request.
                try:
                    output, error = collect(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    output, error = collect(timeout=10)
            else:
                output, error = collect(timeout=10)
            raise AssertionError({"mode": mode, "cause": repr(failure),
                "returncode_before_cleanup": status_before_cleanup,
                "returncode": process.returncode,
                "stdout": output[:65536].decode("utf8", errors="replace"),
                "stderr": error[:65536].decode("utf8", errors="replace"),
                "requests": fixture.requests, "fixture_errors": fixture.errors}) from failure
        finally:
            if process.poll() is None:
                process.kill()
                collect(timeout=10)
    require(not fixture.errors, (mode, fixture.errors))
    require(len(fixture.requests) == (rounds if phase == "tls" else 2 * rounds), "Unexpected request count")
    require(sorted(fixture.closed) == list(range(rounds) if phase == "tls" else range(0, 2 * rounds, 2)), "Native stalled connections were not closed")
    if phase != "tls":
        for index in range(rounds):
            left, right = fixture.requests[index * 2:index * 2 + 2]
            require(left["body_sha256"] == right["body_sha256"], "Retry changed the immutable event")
    return {"mode": mode, "passed": True, "rounds": rows,
            "requests": fixture.requests, "stalled_connections_closed": fixture.closed,
            "native_request_timeout_ms": 30000, "cancel_after_stalled_peer": stop,
            "tls_trust_qualified": False, "process_tree_termination_qualified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", required=True, type=Path)
    args = parser.parse_args()
    evidence = {"passed": False, "cases": [], "failures": []}
    for mode in ("stop-headers", "stop-body", "stop-tls", "deadline-headers", "deadline-body", "deadline-tls"):
        try:
            evidence["cases"].append(run(args.harness.resolve(strict=True), mode))
        except BaseException as error:
            evidence["failures"].append({"mode": mode, "error": repr(error)})
    evidence["passed"] = not evidence["failures"] and len(evidence["cases"]) == 6
    target = os.environ.get("JYPPX_LIUDING_HTTP_CANCELLATION_EVIDENCE_DIR")
    if target:
        path = Path(target)
        path.mkdir(parents=True, exist_ok=True)
        (path / "cancellation.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf8")
    print(json.dumps(evidence, sort_keys=True))
    require(evidence["passed"], "Native stalled-peer cancellation failed")


if __name__ == "__main__":
    main()
