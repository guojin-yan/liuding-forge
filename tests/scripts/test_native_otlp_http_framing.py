# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent HTTP/1.1 chunk and Trailer-declaration acceptance fixtures."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading


def require(condition, message):
    if not condition:
        raise AssertionError(message)


@contextmanager
def collector(replies):
    observed = {"requests": [], "errors": []}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                require(0 < length <= 16 << 20, "Unexpected request length")
                require(self.path == "/v1/logs", "Unexpected signal")
                require(self.headers.get("Content-Type") == "application/x-protobuf",
                        "Unexpected request media type")
                body = self.rfile.read(length)
                require(len(body) == length, "Truncated native request")
                index = len(observed["requests"])
                observed["requests"].append(body)
                require(index < len(replies), "Unexpected application retry")
                self.wfile.write(replies[index])
                self.wfile.flush()
                self.close_connection = True
            except (BrokenPipeError, ConnectionResetError):
                pass  # Rejected metadata may close before the fixture finishes.
            except BaseException as error:
                observed["errors"].append(str(error))
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    worker = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": .02})
    worker.start()
    try:
        yield server.server_port, observed
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        require(not worker.is_alive(), "Fixture did not stop")


def response(trailers=b"", declaration=b"", chunk=b"0\r\n"):
    return (b"HTTP/1.1 200 OK\r\nContent-Type: application/x-protobuf\r\n"
            b"Transfer-Encoding: chunked\r\nConnection: close\r\n" +
            declaration + b"\r\n" + chunk + trailers + b"\r\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    harness = args.harness.resolve(strict=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="http-framing-", dir=args.work_dir))
    good = response()
    cases = [
        ("chunked-empty", "logs", good),
        ("chunk-extension-and-ordinary-trailer", "logs",
         response(b"X-Checksum: fixture\r\n", b"Trailer: X-Checksum\r\n",
                  b"0;fixture=ok\r\n")),
        ("split-protobuf-chunks", "logs",
         response(chunk=b"1\r\n\x0a\r\n1\r\n\x00\r\n0\r\n")),
        ("ordinary-trailer-name-list", "logs",
         response(b"X-Checksum: fixture\r\n", b"Trailer: \tX-Checksum, X-Digest \t\r\n")),
        ("content-coding-trailer", "reply-retry",
         response(b"Content-Encoding: gzip\r\n", b"Trailer: Content-Encoding\r\n")),
        ("content-type-trailer", "reply-retry",
         response(b"Content-Type: text/html\r\n", b"Trailer: Content-Type\r\n")),
        ("mixed-case-framing-declaration", "reply-retry",
         response(declaration=b"Trailer: X-Checksum, tRaNsFeR-EnCoDiNg\r\n")),
        ("content-length-declaration", "reply-retry",
         response(declaration=b"Trailer: Content-Length\r\n")),
        ("empty-trailer-declaration", "reply-retry",
         response(declaration=b"Trailer: \r\n")),
        ("malformed-trailer-name", "reply-retry",
         response(declaration=b"Trailer: X Checksum\r\n")),
        ("trailing-empty-trailer-name", "reply-retry",
         response(declaration=b"Trailer: X-Checksum,\r\n")),
        ("second-trailer-declaration", "reply-retry",
         response(declaration=b"Trailer: X-Checksum\r\nTrailer: Content-Encoding\r\n")),
    ]
    reports, failures = [], []
    for name, mode, reply in cases:
        (directory / (name + ".response.bin")).write_bytes(reply)
        result, observed = None, {}
        try:
            with collector([reply, good] if mode == "reply-retry" else [reply]) as (port, observed):
                result = subprocess.run([str(harness), str(port), mode],
                                        capture_output=True, timeout=15,
                                        env=dict(os.environ))
            (directory / (name + ".stdout")).write_bytes(result.stdout)
            (directory / (name + ".stderr")).write_bytes(result.stderr)
            require(result.returncode == 0, result.stderr.decode("utf-8", errors="replace"))
            require(b"native-http-contract-ok" in result.stdout, "Missing native assertion result")
            require(not observed["errors"], str(observed["errors"]))
            requests = observed["requests"]
            require(len(requests) == (2 if mode == "reply-retry" else 1),
                    "Wrong native request or duplicate count")
            if mode == "reply-retry":
                require(requests[0] == requests[1], "Rejected reply consumed or changed the event identity")
        except BaseException as error:
            failures.append({"case": name, "error": str(error)})
        reports.append({"case": name, "mode": mode,
                        "native_exit": None if result is None else result.returncode,
                        "requests": len(observed.get("requests", [])),
                        "fixture_errors": observed.get("errors", [])})
    report = {"passed": not failures, "cases": reports, "failures": failures,
              "native_processes": len(cases), "work_directory": str(directory),
              "scope": "Chunk framing and Trailer declarations; no qualification for all undeclared trailers",
              "training_runs": 0}
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
