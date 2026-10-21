"""Draft real collector-to-file consumer gate; never silently substitute a mock.

JYPPX_TEST_OTELCOL must name the unpacked official otelcol-contrib 0.135.0
executable obtained from a verified collector-artifacts.json archive. This gate
starts that executable with argv/no shell in a unique temporary directory and
checks its exported OTLP JSON using official generated-message consumers.
Missing configuration is an explicit skip, never formal-release evidence.
"""

from __future__ import annotations

import base64
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest

from test_protocol import observation, point
import jyppx_telemetry_sidecar as sidecar


def protobuf_json_ids(value):
    """OTLP JSON uses hex IDs; protobuf JSON's bytes parser expects base64."""
    value = copy.deepcopy(value)
    pending = [value]
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
    return value


class RealCollectorTest(unittest.TestCase):
    def test_official_collector_file_consumer_preserves_scalars_logs_and_span(self):
        configured = os.environ.get("JYPPX_TEST_OTELCOL")
        if not configured:
            self.skipTest("required real collector executable has not been configured")
        executable = Path(configured).resolve(strict=True)
        self.assertTrue(executable.is_file())
        version = subprocess.run([str(executable), "--version"], shell=False,
                                 capture_output=True, timeout=10, check=True)
        self.assertLess(len(version.stdout) + len(version.stderr), 4096)
        version_text = (version.stdout + version.stderr).decode("utf-8", errors="strict")
        self.assertRegex(version_text, r"(?m)^otelcol-contrib version 0\.135\.0\s*$")
        from google.protobuf.json_format import ParseDict
        from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
        from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        config = Path(__file__).with_name("collector.yaml").resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix="jyppx-real-otelcol-") as root:
            root_path = Path(root)
            output = root_path / "collector.jsonl"
            log = root_path / "collector.log"
            environment = os.environ.copy()
            environment["JYPPX_OTELCOL_ENDPOINT"] = f"127.0.0.1:{port}"
            environment["JYPPX_OTELCOL_FILE"] = str(output)
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with log.open("wb") as diagnostics:
                process = subprocess.Popen([str(executable), "--config", str(config)],
                                           cwd=root_path, env=environment, shell=False,
                                           stdin=subprocess.DEVNULL, stdout=diagnostics,
                                           stderr=subprocess.STDOUT, creationflags=creation_flags)
                try:
                    startup_deadline = time.monotonic() + 10
                    while True:
                        if process.poll() is not None:
                            self.fail("pinned collector failed before accepting input; inspect isolated diagnostics")
                        try:
                            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                                break
                        except OSError:
                            if time.monotonic() >= startup_deadline:
                                self.fail("pinned collector startup deadline")
                            time.sleep(0.05)
                    backend = sidecar.OtlpHttpBackend(f"http://127.0.0.1:{port}", "127.0.0.1", 2.0,
                        "official.real.collector.fixture",
                        {"clock.test": {"monotonic_ns": 100, "unix_ns": 1_700_000_000_000_000_000}})
                    backend.export(backend.prepare(observation()))
                    backend.export(backend.prepare(observation(2, {"kind": 3, "severity": 3,
                                                                   "body": "official collector fixture"})))
                    span = observation(3, {"kind": 4, "name": "fixture.forward", "start_monotonic_ns": 105,
                                           "end_monotonic_ns": 110, "status": 2, "reason": ""})
                    span["correlation"]["span_id"] = "span.fixture"
                    backend.export(backend.prepare(span))
                    backend.close()
                    deadline = time.monotonic() + 8
                    records = []
                    while time.monotonic() < deadline:
                        self.assertIsNone(process.poll(), "collector stopped before consumer readback")
                        if log.exists():
                            self.assertLess(log.stat().st_size, 1024 * 1024)
                        if output.exists():
                            self.assertLess(output.stat().st_size, 2 * 1024 * 1024)
                            lines = output.read_bytes().split(b"\n")[:-1]
                            records = [json.loads(line) for line in lines if line]
                            keys = {key for record in records for key in record}
                            if {"resourceMetrics", "resourceLogs", "resourceSpans"} <= keys:
                                break
                        time.sleep(0.05)
                    by_signal = {}
                    for record in records:
                        for key in ("resourceMetrics", "resourceLogs", "resourceSpans"):
                            if key in record:
                                by_signal.setdefault(key, []).extend(record[key])
                    self.assertEqual(set(by_signal), {"resourceMetrics", "resourceLogs", "resourceSpans"})
                    metrics = ParseDict({"resourceMetrics": by_signal["resourceMetrics"]},
                                        ExportMetricsServiceRequest())
                    logs = ParseDict(protobuf_json_ids({"resourceLogs": by_signal["resourceLogs"]}),
                                     ExportLogsServiceRequest())
                    spans = ParseDict(protobuf_json_ids({"resourceSpans": by_signal["resourceSpans"]}),
                                      ExportTraceServiceRequest())
                    actual = metrics.resource_metrics[0].scope_metrics[0].metrics[0].gauge.data_points[0]
                    self.assertEqual(actual.as_double, point()["value"])
                    self.assertEqual(actual.time_unix_nano, observation()["time"]["wall_time_unix_ns"])
                    attrs = {item.key: item.value.string_value for item in actual.attributes}
                    self.assertIn("validation.loss", attrs["jyppx.metric_definition"])
                    self.assertIn("mean", attrs["jyppx.metric_definition"])
                    log_records = [record for resource in logs.resource_logs for scope in resource.scope_logs
                                   for record in scope.log_records]
                    record = next(record for record in log_records if "official collector fixture" in record.body.string_value)
                    self.assertIn("official collector fixture", record.body.string_value)
                    self.assertEqual(record.severity_number, 13)
                    actual_span = spans.resource_spans[0].scope_spans[0].spans[0]
                    self.assertEqual(actual_span.name, "fixture.forward")
                    self.assertEqual(actual_span.end_time_unix_nano - actual_span.start_time_unix_nano, 5)
                    self.assertEqual(len(actual_span.trace_id), 16)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
