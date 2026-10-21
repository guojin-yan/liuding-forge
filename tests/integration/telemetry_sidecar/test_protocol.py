"""Bounded protocol and immutable replay tests for the chapter gate."""

from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "telemetry_sidecar"))
import jyppx_telemetry_sidecar as sidecar


def point(status=1, value=0.12345678901234568):
    return {
        "definition": {"schema_version": 1, "metric_id": "validation.loss", "version": 1,
                       "unit": 1, "aggregation": 3, "direction": 2,
                       "parameters": {"reduction": "mean"}, "context_keys": ["split"],
                       "definition_digest": "a" * 64},
        "step": {"kind": 2, "value": 7}, "status": status,
        "value": value if status == 1 else None,
        "reason": "no samples" if status == 5 else "",
        "context": {"split": "validation"}, "sample_count": None,
    }


def observation(sequence=1, payload=None):
    return {
        "schema_version": 1,
        "identity": {"stream_id": "run.test/attempt.1", "event_id": f"event.{sequence}",
                     "idempotency_key": f"operation.{sequence}", "sequence": sequence},
        "time": {"wall_time_unix_ns": 1_700_000_000_123_456_789,
                 "monotonic_time_ns": 120, "clock_id": "clock.test"},
        "correlation": {"project_id": "project.test", "run_id": "run.test",
                        "attempt_id": "attempt.1", "trace_id": "trace.test",
                        "span_id": "", "parent_span_id": ""},
        "reliability": 1,
        "attributes": {"producer": "evaluation"},
        "payload": payload if payload is not None else {"kind": 2, "points": [point()]},
    }


def lifecycle(sequence=1, metrics=None):
    return observation(sequence, {"kind": 1, "name": "training.batch_finished",
                                  "terminal": False,
                                  "metrics": [point()] if metrics is None else metrics,
                                  "decisions": [{"actor_id": "callback.monitor",
                                                 "action": "continue_training",
                                                 "reason": "继续训练: validation is finite",
                                                 "effective": True}]})


def span(sequence=2):
    value = observation(sequence, {"kind": 4, "name": "training.forward",
                                   "start_monotonic_ns": 100, "end_monotonic_ns": 110,
                                   "status": 1, "reason": ""})
    value["correlation"]["span_id"] = "span.test"
    return value


def request(*values, backend="tensorboard"):
    events = [{"content_digest": hashlib.sha256(sidecar.compact(value).encode()).hexdigest(),
               "envelope": value} for value in values]
    request_id = hashlib.sha256(("jyppx.telemetry.sidecar-batch/v1" + "".join(
        event["content_digest"] for event in events)).encode()).hexdigest()
    return {"protocol_version": 1, "backend": backend, "request_id": request_id,
            "operation": "emit", "events": events}


def control(operation):
    return {"protocol_version": 1, "backend": "tensorboard", "request_id": operation,
            "operation": operation}


class RecordingBackend:
    def __init__(self):
        self.exported = []
        self.flushes = 0
        self.closes = 0
        self.fail_after = None

    def prepare(self, value):
        return value if value["payload"]["kind"] in (1, 2) else None

    def export(self, prepared):
        if self.fail_after is not None and len(self.exported) >= self.fail_after:
            raise sidecar.ExportError("injected")
        self.exported.append(copy.deepcopy(prepared))

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closes += 1


class ProtocolTest(unittest.TestCase):
    def test_strict_json_bounds_duplicates_nonfinite_and_utf8(self):
        invalid = [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'"\xff"',
                   b'"\\ud800"', b"[" * 33 + b"0" + b"]" * 33,
                   b'{"x":"\\u0000"}']
        for raw in invalid:
            with self.subTest(raw=raw[:80]), self.assertRaises(sidecar.ProtocolError):
                sidecar.strict_json(raw)
        with self.assertRaises(sidecar.ProtocolError):
            sidecar.strict_json(b'"abcdef"', maximum=3)
        self.assertEqual(sidecar.strict_json(b'{"x":"[still a string]"}'),
                         {"x": "[still a string]"})

    def test_complete_receipts_replay_and_unsupported_counter(self):
        backend = RecordingBackend()
        engine = sidecar.Engine("tensorboard", backend)
        training = lifecycle()
        batch = request(training, span(), copy.deepcopy(training))
        result = engine.handle(batch)
        self.assertEqual(result["receipts"], [1, 4, 2])
        self.assertEqual(result["counters"], {"exported_events": 1, "sampled_out_events": 0,
                                               "unsupported_events": 1})
        self.assertEqual(engine.handle(batch)["receipts"], [2, 4, 2])
        self.assertEqual(len(backend.exported), 1)
        self.assertEqual(backend.exported[0], training)

    def test_invalid_later_lifecycle_decision_prevents_any_projection(self):
        backend = RecordingBackend()
        engine = sidecar.Engine("tensorboard", backend)
        bad = lifecycle(2)
        bad["payload"]["decisions"].append({"actor_id": "callback.other",
                                             "action": "stop_training", "reason": "conflict",
                                             "effective": True})
        with self.assertRaises(sidecar.ProtocolError):
            engine.handle(request(lifecycle(), bad))
        self.assertEqual(backend.exported, [])
        self.assertEqual(engine.counters["exported_events"], 0)

    def test_conflicting_existing_or_in_batch_keys_preflight_without_effects(self):
        for existing in (False, True):
            backend = RecordingBackend()
            engine = sidecar.Engine("tensorboard", backend)
            original = observation()
            changed = copy.deepcopy(original)
            changed["payload"]["points"][0]["value"] = 9.0
            if existing:
                engine.handle(request(original))
            before = len(backend.exported)
            with self.subTest(existing=existing), self.assertRaises(sidecar.ProtocolError):
                engine.handle(request(changed) if existing else request(original, changed))
            self.assertEqual(len(backend.exported), before)

    def test_content_digest_cannot_mask_changed_projection(self):
        backend = RecordingBackend()
        engine = sidecar.Engine("tensorboard", backend)
        original = request(observation())
        engine.handle(original)
        forged = copy.deepcopy(original)
        forged["events"][0]["envelope"]["attributes"]["producer"] = "changed"
        with self.assertRaises(sidecar.ProtocolError):
            engine.handle(forged)
        self.assertEqual(len(backend.exported), 1)

    def test_malformed_later_event_prevents_any_effects(self):
        backend = RecordingBackend()
        engine = sidecar.Engine("tensorboard", backend)
        bad = observation(2)
        bad["payload"]["points"][0]["context"] = {"undeclared": "value"}
        with self.assertRaises(sidecar.ProtocolError):
            engine.handle(request(observation(), bad))
        self.assertEqual(backend.exported, [])

    def test_request_identity_protocol_backend_and_replay_capacity(self):
        backend = RecordingBackend()
        engine = sidecar.Engine("tensorboard", backend, max_seen=1)
        for field, replacement in (("protocol_version", True), ("protocol_version", 2),
                                   ("backend", "opentelemetry"), ("request_id", "0" * 64)):
            bad = request(observation())
            bad[field] = replacement
            with self.subTest(field=field), self.assertRaises(sidecar.ProtocolError):
                engine.handle(bad)
        with self.assertRaises(sidecar.ProtocolError):
            engine.handle(request(observation(), observation(2)))
        self.assertEqual(backend.exported, [])

    def test_partial_external_failure_retry_preserves_committed_prefix(self):
        backend = RecordingBackend()
        backend.fail_after = 1
        engine = sidecar.Engine("tensorboard", backend)
        batch = request(observation(), observation(2))
        with self.assertRaises(sidecar.ExportError):
            engine.handle(batch)
        backend.fail_after = None
        result = engine.handle(batch)
        self.assertEqual(result["receipts"], [2, 1])
        self.assertEqual(len(backend.exported), 2)
        self.assertEqual(result["counters"]["exported_events"], 2)

    def test_explicit_close_is_idempotent_and_rejects_later_emit(self):
        backend = RecordingBackend()
        engine = sidecar.Engine("tensorboard", backend)
        engine.handle(control("flush"))
        engine.handle(control("close"))
        engine.handle(control("close"))
        self.assertEqual((backend.flushes, backend.closes), (1, 1))
        with self.assertRaises(sidecar.ProtocolError):
            engine.handle(request(observation()))

    def test_framing_reply_is_redacted_and_partial_frame_is_fatal(self):
        backend = RecordingBackend()
        engine = sidecar.Engine("tensorboard", backend)
        source = io.BytesIO(b'{"secret":"do-not-echo"}\npartial')
        output = io.BytesIO()
        self.assertEqual(sidecar.serve(engine, source, output), 2)
        self.assertNotIn(b"do-not-echo", output.getvalue())
        replies = [sidecar.strict_json(line) for line in output.getvalue().splitlines()]
        self.assertEqual([reply["error"] for reply in replies], ["invalid_request", "invalid_request"])


if __name__ == "__main__":
    unittest.main()
