"""Official-library consumer gates for the optional telemetry adapter.

Install sidecar/requirements.lock in a dedicated Python 3.11 environment before
running this file. An enabled gate must fail if its official dependency is
absent. test_real_collector.py separately requires the actual pinned collector
executable and its file-consumer readback. Execution belongs to the unified
chapter gate.
"""

from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest

from test_protocol import lifecycle, observation, point, request, span
import jyppx_telemetry_sidecar as sidecar


class OfficialConsumerTest(unittest.TestCase):
    def test_tensorboard_lifecycle_text_is_complete_and_scalar_metrics_keep_binary64(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.compat.proto import types_pb2
        from tensorboard.plugins.text.metadata import parse_plugin_metadata
        from tensorboard.util.tensor_util import make_ndarray
        with tempfile.TemporaryDirectory(prefix="jyppx-official-tb-lifecycle-") as root:
            output = Path(root) / "events"
            backend = sidecar.TensorBoardBackend(output)
            value = lifecycle(11, [point(status) for status in (1, 2, 3, 4, 5)])
            for index, metric in enumerate(value["payload"]["metrics"]):
                metric["definition"]["metric_id"] = f"training.loss.{index}"
                metric["definition"]["definition_digest"] = f"{index + 1:x}" * 64
            value["attributes"].update({"training.phase": "validation", "source": "训练/原始"})
            value["payload"]["name"] = "training.run_finished"
            value["payload"]["terminal"] = True
            value["payload"]["decisions"].append({"actor_id": "callback.audit",
                                                   "action": "stop_training", "reason": "not selected",
                                                   "effective": False})
            empty = lifecycle(12, [])
            engine = sidecar.Engine("tensorboard", backend)
            batch = request(value, empty, copy.deepcopy(value))
            result = engine.handle(batch)
            self.assertEqual(result["receipts"], [1, 1, 2])
            self.assertEqual(result["counters"], {"exported_events": 2, "sampled_out_events": 0,
                                                   "unsupported_events": 0})
            backend.close()
            accumulator = EventAccumulator(str(output), size_guidance={"tensors": 0},
                                           purge_orphaned_data=False)
            accumulator.Reload()
            lifecycles, scalars = [], {}
            for tag in accumulator.Tags()["tensors"]:
                metadata = accumulator.SummaryMetadata(tag)
                for event in accumulator.Tensors(tag):
                    decoded = make_ndarray(event.tensor_proto).item()
                    if metadata.plugin_data.plugin_name == "text":
                        self.assertEqual(parse_plugin_metadata(metadata.plugin_data.content).version, 0)
                        self.assertEqual(event.tensor_proto.dtype, types_pb2.DT_STRING)
                        original = json.loads(decoded.decode("utf-8"))
                        self.assertEqual(event.step, original["identity"]["sequence"])
                        self.assertEqual(event.wall_time, original["time"]["wall_time_unix_ns"] / 1e9)
                        self.assertEqual(json.loads(metadata.summary_description)["step_domain"],
                                         "telemetry.sequence")
                        lifecycles.append(original)
                    else:
                        self.assertEqual(metadata.plugin_data.plugin_name, "scalars")
                        self.assertEqual(event.tensor_proto.dtype, types_pb2.DT_DOUBLE)
                        self.assertEqual(event.step, point()["step"]["value"])
                        metric_id = json.loads(metadata.summary_description)["original_metric_id"]
                        scalars[metric_id] = decoded
            self.assertEqual(sorted(lifecycles, key=lambda item: item["identity"]["sequence"]),
                             [value, empty])
            self.assertEqual(len(scalars), 4)
            self.assertEqual(scalars["training.loss.0"], point()["value"])
            self.assertTrue(math.isnan(scalars["training.loss.1"]))
            self.assertEqual(scalars["training.loss.2"], float("inf"))
            self.assertEqual(scalars["training.loss.3"], -float("inf"))
            self.assertNotIn("training.loss.4", scalars)

    def test_tensorboard_whole_lifecycle_is_unsupported_before_any_partial_write(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory(prefix="jyppx-official-tb-unsupported-") as root:
            output = Path(root) / "events"
            backend = sidecar.TensorBoardBackend(output)
            no_wall = lifecycle(1)
            no_wall["time"]["wall_time_unix_ns"] = None
            large_sequence = lifecycle(1 << 63)
            large_step = lifecycle(3)
            large_step["payload"]["metrics"].append(point())
            large_step["payload"]["metrics"][-1]["step"]["value"] = 1 << 63
            engine = sidecar.Engine("tensorboard", backend)
            result = engine.handle(request(no_wall, large_sequence, large_step, span(4)))
            self.assertEqual(result["receipts"], [4, 4, 4, 4])
            self.assertEqual(result["counters"]["exported_events"], 0)
            self.assertEqual(result["counters"]["unsupported_events"], 4)
            backend.close()
            accumulator = EventAccumulator(str(output), size_guidance={"tensors": 0})
            accumulator.Reload()
            self.assertEqual(accumulator.Tags()["tensors"], [])

    def test_tensorboard_official_accumulator_reads_binary64_steps_and_series(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.util.tensor_util import make_ndarray
        with tempfile.TemporaryDirectory(prefix="jyppx-official-tb-") as root:
            output = Path(root) / "events"
            backend = sidecar.TensorBoardBackend(output)
            value = observation()
            second = copy.deepcopy(point())
            second["context"]["split"] = "test"
            value["payload"]["points"].append(second)
            backend.export(backend.prepare(value))
            backend.close()
            accumulator = EventAccumulator(str(output), size_guidance={"tensors": 0})
            accumulator.Reload()
            tags = accumulator.Tags()["tensors"]
            self.assertEqual(len(tags), 2)
            for tag in tags:
                event = accumulator.Tensors(tag)[0]
                self.assertEqual(event.step, 7)
                self.assertEqual(make_ndarray(event.tensor_proto).item(), point()["value"])
                metadata = accumulator.SummaryMetadata(tag)
                self.assertEqual(metadata.plugin_data.plugin_name, "scalars")
                self.assertIn("definition_digest", metadata.summary_description)

    def test_tensorboard_nonfinite_values_are_official_protos_missing_is_explicit(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.util.tensor_util import make_ndarray
        with tempfile.TemporaryDirectory(prefix="jyppx-official-tb-nan-") as root:
            backend = sidecar.TensorBoardBackend(Path(root) / "events")
            for status in (2, 3, 4):
                value = observation(status)
                value["payload"]["points"] = [point(status)]
                backend.export(backend.prepare(value))
            missing = observation(5)
            missing["payload"]["points"] = [point(5)]
            self.assertIsNone(backend.prepare(missing))
            historical = observation()
            historical["time"]["wall_time_unix_ns"] = None
            self.assertIsNone(backend.prepare(historical))
            backend.close()
            accumulator = EventAccumulator(str(Path(root) / "events"),
                                             size_guidance={"tensors": 0})
            accumulator.Reload()
            values = [make_ndarray(event.tensor_proto).item()
                      for tag in accumulator.Tags()["tensors"] for event in accumulator.Tensors(tag)]
            self.assertEqual(len(values), 3)
            self.assertTrue(any(math.isnan(value) for value in values))
            self.assertIn(float("inf"), values)
            self.assertIn(-float("inf"), values)

    def test_otlp_generated_types_retain_definition_flags_native_ids_and_clock_mapping(self):
        from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
        backend = sidecar.OtlpHttpBackend("http://127.0.0.1:4318", "127.0.0.1", 1.0, "official.fixture",
                                           {"clock.test": {"monotonic_ns": 100,
                                                            "unix_ns": 1_700_000_000_000_000_000}})
        value = observation()
        value["payload"]["points"].append(point(5))
        prepared = backend.prepare(value)
        signal, request = prepared[0]
        self.assertEqual(prepared[1][0], "logs")
        self.assertEqual(signal, "metrics")
        restored = ExportMetricsServiceRequest.FromString(request.SerializeToString(deterministic=True))
        metrics = restored.resource_metrics[0].scope_metrics[0].metrics
        self.assertEqual(metrics[0].WhichOneof("data"), "gauge")
        self.assertEqual(metrics[0].gauge.data_points[0].as_double, point()["value"])
        self.assertEqual(metrics[1].gauge.data_points[0].flags, 1)
        self.assertFalse(metrics[1].gauge.data_points[0].HasField("as_double"))
        attrs = {item.key: item.value for item in metrics[0].gauge.data_points[0].attributes}
        self.assertIn("validation.loss", attrs["jyppx.metric_definition"].string_value)
        for high_cardinality in ("jyppx.identity", "jyppx.correlation", "jyppx.step_value",
                                 "jyppx.wall_time_unix_ns", "jyppx.sample_count"):
            self.assertNotIn(high_cardinality, attrs)
        span = observation(2, {"kind": 4, "name": "evaluation.forward", "start_monotonic_ns": 105,
                               "end_monotonic_ns": 110, "status": 2, "reason": ""})
        span["correlation"]["span_id"] = "span.opaque"
        signal, request = backend.prepare(span)[0]
        self.assertEqual(signal, "traces")
        restored_span = ExportTraceServiceRequest.FromString(request.SerializeToString()).resource_spans[0].scope_spans[0].spans[0]
        self.assertEqual(restored_span.start_time_unix_nano, 1_700_000_000_000_000_005)
        self.assertEqual(restored_span.end_time_unix_nano, 1_700_000_000_000_000_010)
        self.assertEqual(len(restored_span.trace_id), 16)
        self.assertEqual(len(restored_span.span_id), 8)
        without_mapping = sidecar.OtlpHttpBackend("http://127.0.0.1:4318", "127.0.0.1", 1.0, "official.fixture")
        self.assertIsNone(without_mapping.prepare(span))

    def test_otlp_lifecycle_metrics_include_gauges_and_complete_neutral_audit_log(self):
        from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
        from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
        backend = sidecar.OtlpHttpBackend("http://127.0.0.1:4318", "127.0.0.1", 1.0, "official.fixture")
        value = lifecycle(2, [point(), point(5)])
        prepared = backend.prepare(value)
        self.assertEqual([signal for signal, _ in prepared], ["metrics", "logs"])
        metrics = ExportMetricsServiceRequest.FromString(prepared[0][1].SerializeToString())
        outputs = metrics.resource_metrics[0].scope_metrics[0].metrics
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].gauge.data_points[0].as_double, point()["value"])
        self.assertEqual(outputs[0].gauge.data_points[0].time_unix_nano,
                         value["time"]["wall_time_unix_ns"])
        self.assertEqual(outputs[1].gauge.data_points[0].flags, 1)
        self.assertFalse(outputs[1].gauge.data_points[0].HasField("as_double"))
        audit = ExportLogsServiceRequest.FromString(prepared[1][1].SerializeToString())
        record = audit.resource_logs[0].scope_logs[0].log_records[0]
        self.assertEqual(json.loads(record.body.string_value), value["payload"])
        attrs = {item.key: item.value for item in record.attributes}
        self.assertEqual(json.loads(attrs["jyppx.envelope"].string_value), value)
        self.assertFalse(attrs["jyppx.timestamp_missing"].bool_value)
        self.assertEqual(record.time_unix_nano, value["time"]["wall_time_unix_ns"])

    def test_otlp_lifecycle_with_metrics_and_unavailable_wall_is_wholly_unsupported(self):
        backend = sidecar.OtlpHttpBackend("http://127.0.0.1:4318", "127.0.0.1", 1.0, "official.fixture")
        for wall in (None, -1):
            value = lifecycle()
            value["time"]["wall_time_unix_ns"] = wall
            engine = sidecar.Engine("opentelemetry", backend)
            result = engine.handle(request(value, backend="opentelemetry"))
            self.assertEqual(result["receipts"], [4])
            self.assertEqual(result["counters"], {"exported_events": 0, "sampled_out_events": 0,
                                                   "unsupported_events": 1})

    def test_otlp_logs_keep_missing_time_and_empty_lifecycle_payload(self):
        backend = sidecar.OtlpHttpBackend("http://127.0.0.1:4318", "127.0.0.1", 1.0, "official.fixture")
        value = observation(2, {"kind": 1, "name": "evaluation.finished", "terminal": True,
                                "metrics": [], "decisions": []})
        value["time"]["wall_time_unix_ns"] = None
        signal, message = backend.prepare(value)[0]
        self.assertEqual(signal, "logs")
        record = message.resource_logs[0].scope_logs[0].log_records[0]
        self.assertEqual(record.time_unix_nano, 0)
        attrs = {item.key: item.value for item in record.attributes}
        self.assertTrue(attrs["jyppx.timestamp_missing"].bool_value)
        self.assertIn("evaluation.finished", record.body.string_value)
        self.assertEqual(json.loads(attrs["jyppx.envelope"].string_value), value)
        self.assertEqual(record.observed_time_unix_nano, 0)

    def test_otlp_http_uses_official_request_and_rejects_partial_success_or_redirect(self):
        from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
            ExportMetricsServiceRequest, ExportMetricsServiceResponse)

        class Collector(BaseHTTPRequestHandler):
            requests = []
            mode = "ok"

            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == "/v1/logs":
                    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
                        ExportLogsServiceRequest, ExportLogsServiceResponse)
                    ExportLogsServiceRequest.FromString(body)
                    if self.mode == "log_error":
                        self.send_response(503)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    data = ExportLogsServiceResponse().SerializeToString()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.requests.append((self.path, self.headers["Content-Type"],
                                      ExportMetricsServiceRequest.FromString(body)))
                if self.mode == "redirect":
                    self.send_response(307)
                    self.send_header("Location", "http://127.0.0.1:1/never-follow")
                    self.end_headers()
                    return
                response = ExportMetricsServiceResponse()
                if self.mode == "partial":
                    response.partial_success.rejected_data_points = 1
                    response.partial_success.error_message = "partial"
                data = response.SerializeToString()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = HTTPServer(("127.0.0.1", 0), Collector)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend = sidecar.OtlpHttpBackend(f"http://127.0.0.1:{server.server_port}", "127.0.0.1", 1.0,
                                               "official.fixture")
            prepared = backend.prepare(observation())
            backend.export(prepared)
            path, content_type, message = Collector.requests[0]
            self.assertEqual((path, content_type), ("/v1/metrics", "application/x-protobuf"))
            self.assertEqual(message.resource_metrics[0].scope_metrics[0].metrics[0].gauge.data_points[0].as_double,
                             point()["value"])
            for mode in ("partial", "redirect"):
                Collector.mode = mode
                with self.subTest(mode=mode), self.assertRaises(sidecar.ExportError):
                    backend.export(prepared)
            # A gauge may already have reached the receiver when the companion
            # audit fails. The whole lifecycle is still unaccepted; an immutable
            # retry can repeat the gauge under the declared at-least-once policy.
            engine = sidecar.Engine("opentelemetry", backend)
            batch = request(lifecycle(30), backend="opentelemetry")
            before = len(Collector.requests)
            Collector.mode = "log_error"
            with self.assertRaises(sidecar.ExportError):
                engine.handle(batch)
            self.assertEqual(engine.counters["exported_events"], 0)
            self.assertEqual(engine.seen, {})
            Collector.mode = "ok"
            completed = engine.handle(batch)
            self.assertEqual(completed["receipts"], [1])
            self.assertEqual(completed["counters"]["exported_events"], 1)
            self.assertEqual(len(Collector.requests), before + 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

if __name__ == "__main__":
    unittest.main()
