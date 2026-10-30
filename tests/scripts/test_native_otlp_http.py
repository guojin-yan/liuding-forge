# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent loopback wire checker; never used by the product implementation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def require(condition, message):
    if not condition:
        raise AssertionError(message)


@contextmanager
def collector(replies):
    requests, errors = [], []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                require(0 <= length <= 16 * 1024 * 1024, "Unbounded/missing request length")
                require(self.headers.get("Content-Type") == "application/x-protobuf", "Wire media type")
                require(not any(self.headers.get(key) for key in ("Authorization", "Proxy-Authorization", "Cookie")), "Ambient authentication or cookie leaked")
                body = self.rfile.read(length)
                require(len(body) == length, "Truncated request body")
                index = len(requests)
                requests.append((self.path, body))
                require(index < len(replies), "Hidden application retry/redirect occurred")
                status, response, extra, delay = replies[index]
                if delay:
                    time.sleep(delay)
                self.send_response(status)
                if not any(key.lower() == "content-type" for key, _ in extra):
                    self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", str(len(response)))
                self.send_header("Connection", "close")
                for key, value in extra:
                    if value is not None:  # None explicitly omits a fixture field.
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Native timeout/cancellation intentionally closes the connection.
            except BaseException as error:
                errors.append(str(error))
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        yield server.server_port, requests, errors
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        require(not thread.is_alive(), "Independent loopback fixture failed to stop")


def run(executable, port, mode, environment):
    result = subprocess.run([str(executable), str(port), mode], env=environment,
                            capture_output=True, timeout=15)
    require(result.returncode == 0, f"{mode}: " + result.stderr.decode("utf-8", errors="replace"))
    require(b"native-http-contract-ok" in result.stdout, "Missing native assertion result")


def decode_request(path, body):
    from opentelemetry.proto.collector.logs.v1 import logs_service_pb2
    from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
    classes = {"/v1/logs": logs_service_pb2.ExportLogsServiceRequest,
               "/v1/metrics": metrics_service_pb2.ExportMetricsServiceRequest,
               "/v1/traces": trace_service_pb2.ExportTraceServiceRequest}
    require(path in classes, "Destination path was not a fixed official OTLP path")
    decoded = classes[path].FromString(body)
    if path == "/v1/logs":
        record = decoded.resource_logs[0].scope_logs[0].log_records[0]
        attrs = {item.key: item.value for item in record.attributes}
        source = json.loads(attrs["jyppx.envelope"].string_value)
        require(source["identity"]["stream_id"] == "http.integration", "Audit identity changed")
    elif path == "/v1/metrics":
        metric = decoded.resource_metrics[0].scope_metrics[0].metrics[0]
        require(metric.description == "http.integration.loss" and metric.WhichOneof("data") == "gauge", "Metric definition changed")
        points = metric.gauge.data_points
        require(len(points) == 1 and points[0].WhichOneof("value") == "as_double", "Metric gauge cardinality/type changed")
        attrs = {item.key: item.value for item in points[0].attributes}
        definition = json.loads(attrs["jyppx.metric_definition"].string_value)
        require(definition["metric_id"] == "http.integration.loss" and
                metric.name == "jyppx.scalar." + definition["definition_digest"], "Metric definition identity changed")
        require(points[0].as_double == 0.125, "Metric value changed")
    else:
        span = decoded.resource_spans[0].scope_spans[0].spans[0]
        require(span.name == "transport-span" and len(span.trace_id) == 16 and len(span.span_id) == 8, "Native span IDs/name changed")
        require(span.end_time_unix_nano - span.start_time_unix_nano == 10, "Span clock mapping changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    args = parser.parse_args()
    for package, expected in {"opentelemetry-proto": "1.37.0", "protobuf": "6.33.5"}.items():
        require(importlib.metadata.version(package) == expected, f"Use the locked independent {package} checker")
    good = (200, b"", [], 0)
    cases = [
        ("logs", [good], ["/v1/logs"]),
        ("logs", [(200, b"", [("Content-Type", 'Application/X-Protobuf; charset="utf-8"'),
                              ("Content-Encoding", "IdEnTiTy")], 0)], ["/v1/logs"]),
        ("traces", [good], ["/v1/traces"]),
        ("pair-retry", [good, (503, b"", [], 0), good, good], ["/v1/metrics", "/v1/logs", "/v1/metrics", "/v1/logs"]),
        ("prefix-retry", [good, (503, b"", [], 0), good], ["/v1/logs"] * 3),
        ("reply-retry", [(200, b"\x0a\x02\x08\x01", [], 0), good], ["/v1/logs"] * 2),
        ("reply-retry", [(200, b"\x0a\x80", [], 0), good], ["/v1/logs"] * 2),
        ("reply-retry", [(200, b"x" * 65, [], 0), good], ["/v1/logs"] * 2),
        ("reply-retry", [(200, b"", [("Content-Type", "text/html")], 0), good], ["/v1/logs"] * 2),
        ("reply-retry", [(200, b"", [("Content-Type", None)], 0), good], ["/v1/logs"] * 2),
        ("reply-retry", [(200, b"", [("Content-Encoding", "gzip")], 0), good], ["/v1/logs"] * 2),
        ("reply-retry", [(204, b"", [], 0), good], ["/v1/logs"] * 2),
        ("reply-retry", [(307, b"", [("Location", "http://127.0.0.1:1/not-allowed")], 0), good], ["/v1/logs"] * 2),
        ("reply-retry", [(401, b"", [("WWW-Authenticate", "Basic realm=do-not-authenticate"), ("Set-Cookie", "do-not-replay=1")], 0), good], ["/v1/logs"] * 2),
        ("reply-retry", [(200, b"", [("X-Large", "x" * 70000)], 0), good], ["/v1/logs"] * 2),
        ("invalid-batch", [], []), ("max-seen", [], []),
        ("cancelled", [], []), ("deadline", [], []),
        ("slow-response", [(200, b"", [], 0.4)], ["/v1/logs"]),
        ("stop-during-send", [(200, b"", [], 0.4)], ["/v1/logs"]),
        ("tls-reject", [], []),
    ]
    evidence, failures = [], []
    with collector([]) as (proxy_port, proxy_requests, proxy_errors):
        environment = os.environ.copy()
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            environment[name] = f"http://127.0.0.1:{proxy_port}"
        environment["NO_PROXY"] = environment["no_proxy"] = ""
        for index, (mode, replies, expected) in enumerate(cases, 1):
            requests = []
            try:
                with collector(replies) as (port, requests, errors):
                    run(args.harness.resolve(), port, mode, environment)
                require(not errors, f"{mode}: fixture errors {errors}")
                require([path for path, _ in requests] == expected, f"{mode}: request count/order changed")
                for path, body in requests:
                    decode_request(path, body)
                if mode == "pair-retry":
                    require(requests[0] == requests[2] and requests[1] == requests[3], "Partial pair retry changed immutable bytes")
                if mode == "prefix-retry":
                    require(requests[1] == requests[2] and requests[0] != requests[2], "Accepted prefix was resent or failed event changed")
            except Exception as error:
                failures.append({"case": index, "mode": mode, "error": str(error)})
            evidence.append({"mode": mode, "requests": len(requests)})
    require(not proxy_requests and not proxy_errors, "Environment proxy was contacted")
    print(json.dumps({"cases": evidence, "failures": failures,
                      "https_certificate_qualification": "not covered by plaintext-handshake rejection"}, sort_keys=True))
    require(not failures, "Independent wire scenarios failed: " + json.dumps(failures, sort_keys=True))


if __name__ == "__main__":
    main()
