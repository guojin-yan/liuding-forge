# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent checker for an already-built native CLI, never a product path.

The stdlib collector acknowledges empty official protobuf response messages.
Full protobuf/real Collector/TLS qualification remains in the dedicated gates.
Timing checks below do not prove OS process-tree termination or reaping.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import struct
import subprocess
import tempfile
import threading
import time
import unittest

import test_run_telemetry as reference


def canonical(value):
    """Separate neutral binary encoder, including span and artifact fixtures."""
    t, u8, u32, u64 = reference.text, reference.u8, reference.u32, reference.u64
    optional, attributes = reference.optional, reference.attributes
    identity, clocks, correlation = value["identity"], value["time"], value["correlation"]
    encoded = (
        t(reference.CONTENT_DOMAIN) + u32(value["schema_version"])
        + b"".join(t(identity[key]) for key in ("stream_id", "event_id", "idempotency_key"))
        + u64(identity["sequence"])
        + optional(clocks["wall_time_unix_ns"], lambda n: struct.pack("<q", n))
        + optional(clocks["monotonic_time_ns"], u64) + t(clocks["clock_id"])
        + b"".join(t(correlation[key]) for key in (
            "project_id", "run_id", "attempt_id", "trace_id", "span_id", "parent_span_id"))
        + u8(value["reliability"]) + attributes(value["attributes"])
    )
    payload = value["payload"]
    kind = payload["kind"]
    encoded += u8(kind)
    if kind == 1:
        encoded += t(payload["name"]) + u8(payload["terminal"])
        encoded += reference.points_bytes(payload["metrics"]) + u64(len(payload["decisions"]))
        for decision in payload["decisions"]:
            encoded += b"".join(t(decision[key]) for key in ("actor_id", "action", "reason"))
            encoded += u8(decision["effective"])
    elif kind == 2:
        encoded += reference.points_bytes(payload["points"])
    elif kind == 3:
        encoded += u8(payload["severity"]) + t(payload["body"])
    elif kind == 4:
        encoded += t(payload["name"]) + u64(payload["start_monotonic_ns"])
        encoded += u64(payload["end_monotonic_ns"]) + u8(payload["status"]) + t(payload["reason"])
    elif kind == 5:
        encoded += t(payload["resource_id"]) + reference.points_bytes(payload["measurements"])
    elif kind == 6:
        artifact = payload["artifact"]
        encoded += t(payload["role"]) + u8(1) + t(artifact["digest"]) + u64(artifact["size_bytes"])
        encoded += t(artifact["logical_name"]) + t(artifact["media_type"]) + t(artifact["producer"])
    else:
        raise AssertionError("Unknown independent fixture kind")
    return encoded


def emit_request(events):
    digests = [hashlib.sha256(canonical(event)).hexdigest() for event in events]
    identity = "jyppx.telemetry.sidecar-batch/v1" + "".join(digests)
    return {"protocol_version": 1, "backend": "opentelemetry", "operation": "emit",
            "request_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "events": [{"content_digest": digest, "envelope": reference.projection(event)}
                       for digest, event in zip(digests, events)]}


class Collector:
    def __init__(self, replies=()):
        self.requests, self.errors = [], []
        self.replies = list(replies)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                    assert 0 <= length <= 16 * 1024 * 1024
                    assert self.headers.get("Content-Type") == "application/x-protobuf"
                    assert not any(self.headers.get(name) for name in (
                        "Authorization", "Proxy-Authorization", "Cookie"))
                    body = self.rfile.read(length)
                    assert len(body) == length
                    index = len(owner.requests)
                    owner.requests.append((self.path, body))
                    status, delay = owner.replies[index] if index < len(owner.replies) else (200, 0)
                    if delay:
                        time.sleep(delay)
                    self.send_response(status)
                    self.send_header("Content-Type", "application/x-protobuf")
                    self.send_header("Content-Length", "0")
                    self.send_header("Connection", "close")
                    self.end_headers()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except BaseException as error:
                    owner.errors.append(repr(error))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = False
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.02})

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        assert not self.thread.is_alive(), "Checker collector did not stop"
        assert not self.errors, self.errors


class Worker:
    def __init__(self, cli, config):
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        self.process = subprocess.Popen(
            [str(cli), "__liuding_otlp_worker", "--config", str(config), "--config-sha256", digest],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        # Independent checker only. Product cleanup is the C++ supervisor.
        if self.process.poll() is None:
            self.process.kill()
        self.process.communicate(timeout=10)

    def exchange(self, request):
        line = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.process.stdin.write(line + b"\n")
        self.process.stdin.flush()
        reply = queue.Queue(maxsize=1)

        def read():
            try:
                reply.put(self.process.stdout.readline(128 * 1024 + 2))
            except BaseException as error:
                reply.put(error)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        result = reply.get(timeout=15)
        reader.join(1)
        assert not isinstance(result, BaseException), result
        assert result.endswith(b"\n") and len(result) <= 128 * 1024 + 1, result
        document = json.loads(result)
        assert document["protocol_version"] == 1
        assert document["request_id"] == request["request_id"]
        return document


class NativeOtlpWorkerContract(unittest.TestCase):
    cli: Path
    disabled = False

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="liuding-native-otlp-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cwd = self.root / "worker"
        self.cwd.mkdir()

    def configuration(self, collector, **extra):
        value = {"schema_version": 1,
                 "endpoint": {"host": "127.0.0.1", "port": collector.server.server_port, "tls": False},
                 "resource_instance_id": "native-worker.contract",
                 "working_directory": str(self.cwd), "inherit_environment": True,
                 "clock_mappings": {"clock.fixture": {
                     "monotonic_ns": 500, "unix_ns": 1_700_000_000_123_456_789}},
                 **extra}
        path = self.root / "native.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def invoke(self, *arguments, expected=0):
        result = subprocess.run([str(self.cli), *(str(v) for v in arguments), "--format", "json"],
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, expected, (result.stdout, result.stderr))
        self.assertEqual(result.stderr, b"")
        return json.loads(result.stdout)

    def store(self):
        store = self.root / "store"
        metadata = self.root / "metadata.json"
        metadata.write_text(json.dumps({
            "schema_version": 1, "run_id": reference.RUN_ID, "project_id": "project.fixture",
            "experiment_id": "experiment.fixture", "display_name": "Native export fixture",
            "status": "running", "created_at_unix_ms": 100, "updated_at_unix_ms": 100,
        }), encoding="utf-8")
        self.invoke("run", "start", "--root", store, "--metadata", metadata)
        values = [reference.envelope(1, {"kind": 3, "severity": 2, "body": "log fixture"}),
                  reference.envelope(2, {"kind": 2, "points": [reference.metric_point()]})]
        reference.seed_database(store, values)
        return store

    def test_cli_canonical_export_or_explicit_disabled_build(self):
        store = self.store()
        with Collector() as collector:
            config = self.configuration(collector)
            result = self.invoke("run", "export", "--root", store, "--id", reference.RUN_ID,
                                 "--backend", "opentelemetry-native", "--native-config", config,
                                 expected=2 if self.disabled else 0)
            if self.disabled:
                self.assertEqual(result["error"]["name"], "not_implemented")
                self.assertEqual(collector.requests, [])
            else:
                data = result["data"]
                self.assertEqual(data["backend"], "opentelemetry-native")
                self.assertEqual(data["accepted_events"], 2)
                self.assertEqual(data["read_events"], 2)
                self.assertTrue(data["scan_completed"] and data["fully_represented"])
                self.assertEqual(data["transport"]["completed_requests"], 3)
                self.assertEqual([path for path, _ in collector.requests],
                                 ["/v1/logs", "/v1/metrics", "/v1/logs"])

    def test_worker_six_payloads_from_independent_canonical_fixture(self):
        with Collector() as collector:
            config = self.configuration(collector)
            payloads = [
                {"kind": 1, "name": "training.finished", "terminal": False,
                 "metrics": [reference.metric_point()],
                 "decisions": [{"actor_id": "callback", "action": "continue",
                                "reason": "fixed decision", "effective": True}]},
                {"kind": 2, "points": [reference.metric_point(-0.0)]},
                {"kind": 3, "severity": 2, "body": "fixed log"},
                {"kind": 4, "name": "train.batch", "start_monotonic_ns": 400,
                 "end_monotonic_ns": 500, "status": 2, "reason": ""},
                {"kind": 5, "resource_id": "device.cpu", "measurements": [reference.metric_point()]},
                {"kind": 6, "role": "checkpoint.best", "artifact": {
                    "algorithm": "sha256", "digest": "a" * 64, "size_bytes": 123,
                    "logical_name": "best.model", "media_type": "application/octet-stream",
                    "producer": "fixture"}},
            ]
            events = [reference.envelope(i, payload) for i, payload in enumerate(payloads, 1)]
            request = emit_request(events)
            with Worker(self.cli, config) as worker:
                first = worker.exchange(request)
                self.assertNotIn("error", first)
                self.assertEqual(first["receipts"], [1] * 6)
                replay = worker.exchange(request)
                self.assertEqual(replay["receipts"], [2] * 6)
                self.assertEqual(replay["counters"]["exported_events"], 6)
            self.assertEqual([path for path, _ in collector.requests], [
                "/v1/metrics", "/v1/logs", "/v1/metrics", "/v1/logs", "/v1/logs",
                "/v1/traces", "/v1/metrics", "/v1/logs", "/v1/logs"])

    def test_partial_pair_error_keeps_prefix_without_fabricating_receipts(self):
        with Collector([(200, 0), (200, 0), (500, 0)]) as collector:
            config = self.configuration(collector)
            values = [reference.envelope(1, {"kind": 3, "severity": 2, "body": "first"}),
                      reference.envelope(2, {"kind": 2, "points": [reference.metric_point(-0.0)]})]
            request = emit_request(values)
            with Worker(self.cli, config) as worker:
                failed = worker.exchange(request)
                self.assertIn("error", failed)
                self.assertNotIn("receipts", failed)
                self.assertNotIn("counters", failed)
                retried = worker.exchange(request)
                self.assertEqual(retried["receipts"], [2, 1])
                self.assertEqual(retried["counters"]["exported_events"], 2)
                changed = copy.deepcopy(values)
                changed[1]["payload"]["points"][0]["value"] = 0.0
                conflicting = emit_request(changed)
                self.assertEqual(conflicting["request_id"], request["request_id"])
                self.assertIn("error", worker.exchange(conflicting))
            self.assertEqual([path for path, _ in collector.requests],
                             ["/v1/logs", "/v1/metrics", "/v1/logs", "/v1/metrics", "/v1/logs"])
            self.assertEqual(collector.requests[1][1], collector.requests[3][1])
            self.assertEqual(collector.requests[2][1], collector.requests[4][1])

    def test_digest_drift_rejected_before_any_network(self):
        with Collector() as collector:
            config = self.configuration(collector)
            values = [reference.envelope(1, {"kind": 3, "severity": 2, "body": "original"})]
            request = emit_request(values)
            request["events"][0]["envelope"]["payload"]["body"] = "tampered"
            with Worker(self.cli, config) as worker:
                self.assertIn("error", worker.exchange(request))
            self.assertEqual(collector.requests, [])

    def test_config_hash_mismatch_has_no_protocol_stdout(self):
        with Collector() as collector:
            config = self.configuration(collector)
            result = subprocess.run(
                [str(self.cli), "__liuding_otlp_worker", "--config", str(config),
                 "--config-sha256", "0" * 64], input=b"", capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 3)
            self.assertEqual(result.stdout, b"")
            self.assertEqual(result.stderr.replace(b"\r\n", b"\n"), b"Native OTLP worker failed.\n")
            self.assertEqual(collector.requests, [])

    def test_parent_timeout_returns_while_collector_is_blocked(self):
        store = self.store()
        with Collector([(200, 12)]) as collector:
            config = self.configuration(collector, operation_timeout_ms=6000,
                                        cleanup_timeout_ms=200)
            started = time.monotonic()
            result = subprocess.run(
                [str(self.cli), "run", "export", "--root", str(store), "--id", reference.RUN_ID,
                 "--backend", "opentelemetry-native", "--native-config", str(config),
                 "--timeout-ms", "20000", "--format", "json"], capture_output=True, timeout=18)
            elapsed = time.monotonic() - started
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(result.stdout)["status"], "error")
            self.assertEqual(len(collector.requests), 1, "The delayed native network call was not reached")
            self.assertLess(elapsed, 11.0, "Parent waited for the blocking native call")
            # This observes response latency only. OS liveness, process-tree
            # kill, handle cleanup and exact cutoff remain separate gates.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--disabled", action="store_true")
    args = parser.parse_args()
    NativeOtlpWorkerContract.cli = args.cli.resolve(strict=True)
    NativeOtlpWorkerContract.disabled = args.disabled
    if args.disabled:
        suite = unittest.TestSuite([NativeOtlpWorkerContract(
            "test_cli_canonical_export_or_explicit_disabled_build")])
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeOtlpWorkerContract)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
