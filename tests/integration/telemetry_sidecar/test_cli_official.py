# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Actual C++ training/store/CLI/process transport to official backend consumers.

This gate uses the production offline workflow, not a protocol peer or an
independently seeded SQLite fixture. Missing prerequisites fail the formal gate.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from test_real_collector import protobuf_json_ids


class CliOfficialBackendTest(unittest.TestCase):
    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        result = subprocess.run(
            [str(cls.cli), *(str(value) for value in arguments), "--format", "json"],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
            shell=False, check=False,
        )
        if result.returncode != expected:
            raise AssertionError(
                f"CLI exit {result.returncode}, expected {expected}\n"
                f"stdout={result.stdout}\nstderr={result.stderr}"
            )
        envelope = json.loads(result.stdout)
        if envelope["schema_version"] != 1 or envelope["status"] != "ok":
            raise AssertionError("Production CLI did not return a successful schema-1 envelope")
        return envelope["data"]

    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = Path(os.environ["JYPPX_LIUDING_TEST_CLI"]).resolve(strict=True)
        cls.collector = Path(os.environ["JYPPX_LIUDING_TEST_OTELCOL"]).resolve(strict=True)
        cls.sidecar = (Path(__file__).resolve().parents[3] / "tools" /
                       "telemetry_sidecar" / "jyppx_telemetry_sidecar.py")
        cls.temporary = tempfile.TemporaryDirectory(prefix="jyppx-cli-official-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.workflow = cls.root / "actual training 数据"
        summary = cls.invoke("offline", "run", "--root", cls.workflow, "--device", "cpu")
        if not summary["valid"] or not summary["recovery"]["metrics_exact"]:
            raise AssertionError("Actual offline training/recovery did not complete")
        cls.run_id = "run.offline.baseline"
        page = cls.invoke("run", "events", "--root", cls.workflow / "store",
                          "--id", cls.run_id, "--limit", 1024)
        if page["next_after"] is not None:
            raise AssertionError("The complete bounded training fixture must fit one evidence page")
        cls.events = [entry["event"] for entry in page["events"]]
        if not cls.events or any(event["payload"]["kind"] != 1 for event in cls.events):
            raise AssertionError("Expected real neutral training lifecycle events")

    def configuration(self, backend: str, *backend_arguments: str) -> Path:
        # Supply only necessary OS locations; the product process transport
        # must start the actual locked interpreter without inheriting secrets.
        environment = {key: os.environ[key] for key in
                       ("SystemRoot", "WINDIR", "TEMP", "TMP") if key in os.environ}
        config = {
            "schema_version": 1, "backend": backend,
            "executable": sys.executable,
            "arguments": ["-I", str(self.sidecar), "--backend", backend, *backend_arguments],
            "working_directory": str(self.root), "environment": environment,
            "inherit_environment": False,
            "operation_timeout_ms": 30000, "startup_timeout_ms": 30000,
        }
        destination = self.root / f"{backend}-launch.json"
        destination.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        return destination

    def export(self, backend: str, config: Path) -> None:
        result = self.invoke(
            "run", "export", "--root", self.workflow / "store", "--id", self.run_id,
            "--backend", backend, "--sidecar-config", config,
            "--page-size", 7, "--timeout-ms", 90000,
        )
        self.assertTrue(result["scan_completed"])
        self.assertTrue(result["fully_represented"])
        self.assertEqual(result["read_events"], len(self.events))
        self.assertEqual(result["accepted_events"], len(self.events))
        self.assertEqual(result["unsupported_events"], 0)
        self.assertEqual(result["sampled_out_events"], 0)
        self.assertGreater(result["transport"]["completed_requests"], 2)
        self.assertEqual(result["delivery_guarantee"], "at_least_once")

    def test_actual_training_to_official_tensorboard_scalars_and_lifecycle(self) -> None:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.util.tensor_util import make_ndarray

        output = self.root / "tensorboard events"
        config = self.configuration("tensorboard", "--tensorboard-output", str(output))
        self.export("tensorboard", config)
        accumulator = EventAccumulator(str(output), size_guidance={"tensors": 0},
                                       purge_orphaned_data=False)
        accumulator.Reload()
        scalars, lifecycles = [], []
        for tag in accumulator.Tags()["tensors"]:
            metadata = accumulator.SummaryMetadata(tag)
            for event in accumulator.Tensors(tag):
                value = make_ndarray(event.tensor_proto)
                if metadata.plugin_data.plugin_name == "scalars":
                    description = json.loads(metadata.summary_description)
                    metric_id = description["series"]["definition"]["metric_id"]
                    scalars.append((metric_id, event.step, value.item()))
                elif metadata.plugin_data.plugin_name == "text":
                    text = value.reshape(-1)[0]
                    lifecycles.append(json.loads(text.decode("utf-8") if isinstance(text, bytes) else text))
        expected = [(point["definition"]["metric_id"], point["step"]["value"], point["value"])
                    for event in self.events for point in event["payload"]["metrics"]]
        self.assertGreater(len(expected), 0)
        self.assertEqual(sorted(scalars), sorted(expected))
        by_id = lambda event: event["identity"]["event_id"]
        self.assertEqual(sorted(lifecycles, key=by_id), sorted(self.events, key=by_id))

    @contextmanager
    def real_collector(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        output = self.root / "cli-collector.jsonl"
        environment = os.environ.copy()
        environment["JYPPX_OTELCOL_ENDPOINT"] = f"127.0.0.1:{port}"
        environment["JYPPX_OTELCOL_FILE"] = str(output)
        with (self.root / "cli-collector.log").open("wb") as diagnostics:
            process = subprocess.Popen(
                [str(self.collector), "--config", str(Path(__file__).with_name("collector.yaml"))],
                cwd=self.root, env=environment, stdin=subprocess.DEVNULL,
                stdout=diagnostics, stderr=subprocess.STDOUT, shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                deadline = time.monotonic() + 10
                while True:
                    self.assertIsNone(process.poll(), "Real collector stopped before startup")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            self.fail("Real collector startup deadline")
                        time.sleep(0.05)
                yield port, output, process
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)

    def test_actual_training_to_real_collector_metrics_and_complete_audit(self) -> None:
        from google.protobuf.json_format import ParseDict
        from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
        from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest

        expected_points = [point for event in self.events for point in event["payload"]["metrics"]]
        with self.real_collector() as (port, output, process):
            config = self.configuration(
                "opentelemetry", "--otlp-base-url", f"http://127.0.0.1:{port}",
                "--allowed-host", "127.0.0.1", "--resource-instance-id", "actual.offline.training",
            )
            self.export("opentelemetry", config)
            metrics, logs = [], []
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                self.assertIsNone(process.poll(), "Collector stopped before readback")
                if output.exists():
                    self.assertLess(output.stat().st_size, 16 * 1024 * 1024)
                    records = [json.loads(line) for line in output.read_bytes().split(b"\n")[:-1] if line]
                    metrics = [entry for record in records for entry in record.get("resourceMetrics", [])]
                    logs = [entry for record in records for entry in record.get("resourceLogs", [])]
                    metric_count = sum(len(metric.get("gauge", {}).get("dataPoints", []))
                                       for resource in metrics for scope in resource.get("scopeMetrics", [])
                                       for metric in scope.get("metrics", []))
                    log_count = sum(len(scope.get("logRecords", [])) for resource in logs
                                    for scope in resource.get("scopeLogs", []))
                    if metric_count >= len(expected_points) and log_count >= len(self.events):
                        break
                time.sleep(0.05)
            decoded_metrics = ParseDict({"resourceMetrics": metrics}, ExportMetricsServiceRequest())
            decoded_logs = ParseDict(protobuf_json_ids({"resourceLogs": logs}), ExportLogsServiceRequest())
            actual_points = [(metric.description, point.as_double)
                             for resource in decoded_metrics.resource_metrics
                             for scope in resource.scope_metrics for metric in scope.metrics
                             for point in metric.gauge.data_points]
            self.assertGreater(len(expected_points), 0)
            self.assertEqual(sorted(actual_points), sorted(
                (point["definition"]["metric_id"], point["value"]) for point in expected_points))
            actual_logs = [record for resource in decoded_logs.resource_logs
                           for scope in resource.scope_logs for record in scope.log_records]
            self.assertEqual(len(actual_logs), len(self.events))
            expected_payloads = sorted(json.dumps(event["payload"], sort_keys=True) for event in self.events)
            self.assertEqual(sorted(json.dumps(json.loads(record.body.string_value), sort_keys=True)
                                    for record in actual_logs), expected_payloads)
            self.assertEqual(sorted(record.time_unix_nano for record in actual_logs),
                             sorted(event["time"]["wall_time_unix_ns"] for event in self.events))


if __name__ == "__main__":
    unittest.main()
