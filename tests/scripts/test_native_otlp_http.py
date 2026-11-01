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
import socket
import ssl
import subprocess
import threading
import time


def require(condition, message):
    if not condition:
        raise AssertionError(message)


@contextmanager
def collector(replies, host="127.0.0.1"):
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

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    server = Server((host, 0), Handler)
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
    return result.stdout.decode("utf-8", errors="replace").splitlines()


@contextmanager
def certificate_collector(certificate, key):
    """A valid TLS peer whose issuing certificate is absent from system trust."""
    observed = {"connections": 0, "handshakes": 0, "rejected_handshakes": [],
                "closed_connections": 0, "http_requests": [], "errors": []}
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certificate, key)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def handle(self):
            try:
                super().handle()
                if getattr(self, "raw_requestline", None) == b"":
                    observed["closed_connections"] += 1
                else:
                    observed["errors"].append("TLS peer did not close without an HTTP request")
            except (ConnectionResetError, ssl.SSLEOFError):
                observed["closed_connections"] += 1

        def do_POST(self):
            # A disabled certificate check would send here and fail the test.
            observed["http_requests"].append(self.path)
            length = int(self.headers.get("Content-Length", "-1"))
            if 0 <= length <= 16 * 1024 * 1024:
                self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", "0")
            self.end_headers()

    class Server(ThreadingHTTPServer):
        def get_request(self):
            connection, address = self.socket.accept()
            observed["connections"] += 1
            connection.settimeout(3)
            try:
                record_type = connection.recv(1, socket.MSG_PEEK)
                require(record_type == b"\x16",
                        "The native request did not start a TLS handshake")
                # On handshake failure wrap_socket closes its owned socket.
                secured = context.wrap_socket(connection, server_side=True)
                observed["handshakes"] += 1
                return secured, address
            except (ssl.SSLError, ConnectionResetError) as error:
                observed["rejected_handshakes"].append(type(error).__name__ + ": " + str(error))
                observed["closed_connections"] += 1
                connection.close()
                raise
            except BaseException as error:
                observed["errors"].append(str(error))
                connection.close()
                raise OSError("Invalid independent TLS fixture connection") from error

    server = Server(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        yield server.server_port, observed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        require(not thread.is_alive(), "Independent TLS fixture failed to stop")


def verify_tls_fixture(port, trust_certificate):
    # Trust is confined to this independent client's context. No certificate
    # store or environment setting used by the native product is modified.
    context = ssl.create_default_context(cafile=trust_certificate)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    with socket.create_connection(("127.0.0.1", port), timeout=3) as connection:
        with context.wrap_socket(connection, server_hostname="127.0.0.1") as secured:
            require(secured.version() == "TLSv1.2", "TLS fixture control protocol")
            require(secured.getpeercert(), "Independent TLS control did not verify the certificate")


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
        ("ipv6-logs", [good], ["/v1/logs"]),
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
                with collector(replies, "::1" if mode == "ipv6-logs" else "127.0.0.1") as (port, requests, errors):
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
        fixture_root = Path(__file__).resolve().parents[1] / "integration" / "telemetry_http" / "fixtures"
        for certificate, trust in (("self-signed.pem", "self-signed.pem"),
                                   ("unknown-ca-chain.pem", "unknown-ca.pem")):
            observed, diagnostic = {}, []
            try:
                with certificate_collector(fixture_root / certificate, fixture_root / "server-key.pem") as (port, observed):
                    verify_tls_fixture(port, fixture_root / trust)
                    diagnostic = run(args.harness.resolve(), port, "tls-certificate-reject", environment)
                require(not observed["errors"], f"{certificate}: {observed['errors']}")
                require(observed["connections"] == 3 and observed["closed_connections"] == 3,
                        f"{certificate}: expected one control and two closed native TLS connections")
                # WinHTTP can complete TLS before rejecting the peer identity;
                # OpenSSL commonly aborts within the handshake. Neither phase
                # proves trust without the native certificate/zero-ACK result.
                require(1 <= observed["handshakes"] <= 3 and
                        observed["handshakes"] + len(observed["rejected_handshakes"]) == 3,
                        f"{certificate}: incomplete TLS phase observations")
                require(sum(line.startswith("native-http-certificate-rejected: ") for line in diagnostic) == 2,
                        f"{certificate}: missing native certificate rejection for each retry")
                require(not observed["http_requests"], f"{certificate}: HTTP sent to an untrusted TLS peer")
            except Exception as error:
                failures.append({"mode": "tls-certificate-reject", "certificate": certificate,
                                 "error": str(error)})
            evidence.append({"mode": "tls-certificate-reject", "certificate": certificate,
                             "observed": observed, "native_diagnostic": diagnostic})
    require(not proxy_requests and not proxy_errors, "Environment proxy was contacted")
    print(json.dumps({"cases": evidence, "failures": failures,
                      "https_certificate_qualification": "self-signed/unknown-CA rejection only; native trusted success, expiry, hostname and revocation not covered"}, sort_keys=True))
    require(not failures, "Independent wire scenarios failed: " + json.dumps(failures, sort_keys=True))


if __name__ == "__main__":
    main()
