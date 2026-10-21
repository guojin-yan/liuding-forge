# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Real CLI/SQLite and process-boundary contracts for run telemetry commands.

Requires an already built product CLI. The stdlib canonical fixture is inserted
into a database created by the product, then decoded by production readers. The
controlled subprocess verifies launch/protocol behavior; official TensorBoard
and collector interoperability remain separate mandatory integration gates.
No tests are skipped when this entry point is run.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import unittest


RUN_ID = "run.telemetry"
STREAM_ID = "stream.fixture"
CONTENT_DOMAIN = "jyppx.telemetry.content/v1"
DEFINITION_DOMAIN = "jyppx.telemetry.metric-definition/v1"


def u8(value: int) -> bytes:
    return struct.pack("<B", value)


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return u64(len(encoded)) + encoded


def attributes(value: dict[str, str]) -> bytes:
    return u64(len(value)) + b"".join(
        text(key) + text(item) for key, item in sorted(value.items())
    )


def optional(value, encode) -> bytes:
    return u8(value is not None) + (encode(value) if value is not None else b"")


def definition_bytes(value: dict) -> bytes:
    return (
        u32(value["schema_version"]) + text(value["metric_id"]) + u32(value["version"])
        + u8(value["unit"]) + u8(value["aggregation"]) + u8(value["direction"])
        + attributes(value["parameters"]) + u64(len(value["context_keys"]))
        + b"".join(text(key) for key in sorted(value["context_keys"]))
    )


def metric_point(value: float | None = 0.125, *, status: int = 1, step: int = 9,
                 step_kind: int = 2, version: int = 1) -> dict:
    return {
        "definition": {
            "schema_version": 1, "metric_id": "validation.accuracy", "version": version,
            "unit": 5 if version == 1 else 6, "aggregation": 3, "direction": 3,
            "parameters": {"average": "macro", "threshold": "0.5"},
            "context_keys": ["dataset", "split"],
        },
        "step": {"kind": step_kind, "value": step}, "status": status,
        "value": value if status == 1 else None,
        "reason": "not measured" if status == 5 else "",
        "context": {"dataset": "fixture.数据", "split": "validation"},
        "sample_count": 7 if status == 1 else None,
    }


def point_bytes(value: dict) -> bytes:
    return (
        definition_bytes(value["definition"]) + u8(value["step"]["kind"])
        + u64(value["step"]["value"]) + u8(value["status"])
        + optional(value["value"], lambda number: struct.pack("<d", 0.0 if number == 0 else number))
        + text(value["reason"]) + attributes(value["context"])
        + optional(value["sample_count"], u64)
    )


def points_bytes(values: list[dict]) -> bytes:
    return u64(len(values)) + b"".join(point_bytes(value) for value in values)


def envelope(sequence: int, payload: dict, *, wall=1_700_000_000_123_456_789,
             monotonic: int | None = 500) -> dict:
    return {
        "schema_version": 1,
        "identity": {"stream_id": STREAM_ID, "event_id": f"event.{sequence}",
                     "idempotency_key": f"key.{sequence}", "sequence": sequence},
        "time": {"wall_time_unix_ns": wall, "monotonic_time_ns": monotonic,
                 "clock_id": "clock.fixture" if monotonic is not None else ""},
        "correlation": {"project_id": "project.fixture", "run_id": RUN_ID,
                        "attempt_id": "attempt.fixture", "trace_id": "opaque.trace",
                        "span_id": f"span.{sequence}", "parent_span_id": "parent.fixture"},
        "reliability": 1,
        "attributes": {"origin": "immutable.fixture", "path": "$(never-execute); & literal"},
        "payload": payload,
    }


def canonical(value: dict) -> bytes:
    identity, clocks, correlation = value["identity"], value["time"], value["correlation"]
    encoded = (
        text(CONTENT_DOMAIN) + u32(value["schema_version"])
        + b"".join(text(identity[key]) for key in ("stream_id", "event_id", "idempotency_key"))
        + u64(identity["sequence"])
        + optional(clocks["wall_time_unix_ns"], lambda number: struct.pack("<q", number))
        + optional(clocks["monotonic_time_ns"], u64) + text(clocks["clock_id"])
        + b"".join(text(correlation[key]) for key in (
            "project_id", "run_id", "attempt_id", "trace_id", "span_id", "parent_span_id"))
        + u8(value["reliability"]) + attributes(value["attributes"])
    )
    payload = value["payload"]
    encoded += u8(payload["kind"])
    if payload["kind"] == 1:
        encoded += text(payload["name"]) + u8(payload["terminal"]) + points_bytes(payload["metrics"])
        encoded += u64(len(payload["decisions"]))
        for decision in payload["decisions"]:
            encoded += b"".join(text(decision[key]) for key in ("actor_id", "action", "reason"))
            encoded += u8(decision["effective"])
    elif payload["kind"] == 2:
        encoded += points_bytes(payload["points"])
    elif payload["kind"] == 3:
        encoded += u8(payload["severity"]) + text(payload["body"])
    elif payload["kind"] == 5:
        encoded += text(payload["resource_id"]) + points_bytes(payload["measurements"])
    else:
        raise AssertionError("Fixture supports only lifecycle, metrics, log and resources")
    return encoded


def source_points(value: dict) -> list[dict]:
    payload = value["payload"]
    return payload.get("points", payload.get("metrics", payload.get("measurements", [])))


def projection(value: dict) -> dict:
    result = copy.deepcopy(value)
    for point in source_points(result):
        definition = point["definition"]
        definition["definition_digest"] = hashlib.sha256(
            text(DEFINITION_DOMAIN) + definition_bytes(definition)
        ).hexdigest()
    return result


def seed_database(root: Path, events: list[dict]) -> None:
    """Insert independent canonical fixtures; product APIs create the schema."""
    with closing(sqlite3.connect(root / "runs.sqlite3")) as database, database:
        database.execute("PRAGMA foreign_keys=ON")
        for value in events:
            identity = value["identity"]
            sequence = identity["sequence"]
            encoded = canonical(value)
            terminal = int(value["payload"].get("terminal", False))
            database.execute(
                "INSERT INTO jyppx_event_streams(stream_id,run_id,last_sequence,terminal) "
                "VALUES(?,?,?,?) ON CONFLICT(stream_id) DO UPDATE SET "
                "last_sequence=excluded.last_sequence,terminal=excluded.terminal",
                (STREAM_ID, RUN_ID, sequence, terminal),
            )
            database.execute(
                "INSERT INTO jyppx_events(run_id,run_sequence,stream_id,producer_sequence,event_id,"
                "idempotency_key,payload_kind,content_digest,canonical) VALUES(?,?,?,?,?,?,?,?,?)",
                (RUN_ID, sequence, STREAM_ID, sequence, identity["event_id"],
                 identity["idempotency_key"], value["payload"]["kind"],
                 hashlib.sha256(encoded).hexdigest(), encoded),
            )
            for index, point in enumerate(source_points(value)):
                definition = point["definition"]
                definition_encoding = text(DEFINITION_DOMAIN) + definition_bytes(definition)
                database.execute(
                    "INSERT OR IGNORE INTO jyppx_metric_definitions(metric_id,definition_version,"
                    "definition_digest,canonical) VALUES(?,?,?,?)",
                    (definition["metric_id"], definition["version"],
                     hashlib.sha256(definition_encoding).hexdigest(), definition_encoding),
                )
                database.execute(
                    "INSERT INTO jyppx_metric_points(run_id,run_sequence,point_index,metric_id,"
                    "definition_version,step_kind,step_key,value_status) VALUES(?,?,?,?,?,?,?,?)",
                    (RUN_ID, sequence, index, definition["metric_id"], definition["version"],
                     point["step"]["kind"], struct.pack(">Q", point["step"]["value"]), point["status"]),
                )
            batch_key = f"fixture.batch.{sequence}"
            batch_digest = hashlib.sha256(
                b"jyppx.run.event-batch/v1" + struct.pack(">Q", 1)
                + struct.pack(">Q", len(encoded)) + encoded
            ).hexdigest()
            database.execute(
                "INSERT INTO jyppx_event_batches(run_id,batch_key,request_digest,input_count,"
                "committed_through) VALUES(?,?,?,?,?)", (RUN_ID, batch_key, batch_digest, 1, sequence),
            )
            database.execute(
                "INSERT INTO jyppx_event_batch_items(run_id,batch_key,ordinal,run_sequence) "
                "VALUES(?,?,?,?)", (RUN_ID, batch_key, 0, sequence),
            )
        database.execute(
            "INSERT INTO jyppx_run_event_state(run_id,last_sequence,metric_count,content_bytes,"
            "batch_count,stream_count) SELECT ?,COUNT(*),(SELECT COUNT(*) FROM jyppx_metric_points "
            "WHERE run_id=?),COALESCE(SUM(length(canonical)),0),COUNT(*),1 FROM jyppx_events "
            "WHERE run_id=? ON CONFLICT(run_id) DO UPDATE SET last_sequence=excluded.last_sequence,"
            "metric_count=excluded.metric_count,content_bytes=excluded.content_bytes,"
            "batch_count=excluded.batch_count,stream_count=excluded.stream_count", (RUN_ID, RUN_ID, RUN_ID),
        )


def fixture_sidecar() -> int:
    """Controlled protocol peer, intentionally independent of official SDKs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-sidecar", action="store_true")
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--mode", default="accept")
    parser.add_argument("--literal", required=True)
    parser.add_argument("--append-root", type=Path)
    args = parser.parse_args()
    counters = {"exported_events": 0, "sampled_out_events": 0, "unsupported_events": 0}
    args.capture.write_text(json.dumps({"launch": {
        "cwd": os.getcwd(), "literal": args.literal,
        "child_value": os.environ.get("JYPPX_CLI_CHILD_VALUE"),
        "parent_secret_present": "JYPPX_CLI_PARENT_SECRET" in os.environ,
    }}, ensure_ascii=False) + "\n", encoding="utf-8")
    for line in sys.stdin.buffer:
        request = json.loads(line)
        if args.append_root is not None and request["operation"] == "flush":
            seed_database(args.append_root, [envelope(5, {"kind": 3, "severity": 2,
                                                         "body": "concurrent export append"})])
            args.append_root = None
        if args.mode == "hang":
            time.sleep(60)
            continue
        if args.mode == "bad-frame":
            sys.stdout.buffer.write(b"{}\n{}\n")
            sys.stdout.buffer.flush()
            continue
        response = {"protocol_version": 1, "request_id": request["request_id"]}
        if request["operation"] == "emit":
            receipts = []
            for item in request["events"]:
                if args.mode == "unsupported":
                    receipt, field = 4, "unsupported_events"
                elif args.mode == "sampled":
                    receipt, field = 3, "sampled_out_events"
                else:
                    receipt, field = 1, "exported_events"
                receipts.append(receipt)
                counters[field] += 1
            response["receipts"] = receipts
        with args.capture.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"request": request}, ensure_ascii=False) + "\n")
        response["counters"] = counters.copy()
        if args.mode == "bad-counter" and request["operation"] == "close":
            response["counters"]["exported_events"] += 1
        sys.stdout.buffer.write((json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()
    return 0


class RunTelemetryContract(unittest.TestCase):
    cli: Path

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="jyppx-cli-telemetry-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "遥测 目录"
        self.root.mkdir()
        self.store = self.root / "本地 store"
        metadata = self.root / "运行 metadata.json"
        metadata.write_text(json.dumps({
            "schema_version": 1, "run_id": RUN_ID, "project_id": "project.fixture",
            "experiment_id": "experiment.fixture", "display_name": "Telemetry fixture",
            "status": "running", "created_at_unix_ms": 100, "updated_at_unix_ms": 100,
        }), encoding="utf-8")
        self.invoke("run", "start", "--root", self.store, "--metadata", metadata)
        self.events = [
            envelope(1, {"kind": 3, "severity": 2, "body": "immutable log"}, wall=None, monotonic=None),
            envelope(2, {"kind": 2, "points": [
                metric_point(), metric_point(status=2, step=3), metric_point(status=3, step=3),
                metric_point(status=4, step=2), metric_point(status=5, step=0),
                metric_point(0.75, step=2**63 + 1, step_kind=1),
            ]}),
            envelope(3, {"kind": 5, "resource_id": "device.fixture", "measurements": [
                metric_point(12.5, step=0, step_kind=3, version=2)]}, wall=-17),
            envelope(4, {"kind": 1, "name": "validation.completed", "terminal": False,
                         "metrics": [metric_point(0.875, step=0, step_kind=4)],
                         "decisions": [{"actor_id": "fixture", "action": "continue",
                                        "reason": "fixed evidence", "effective": True}]}),
        ]
        seed_database(self.store, self.events)

    def invoke(self, *values, expected: int = 0, env: dict | None = None) -> dict:
        process = subprocess.run(
            [str(self.cli), *(str(value) for value in values), "--format", "json"],
            capture_output=True, text=True, encoding="utf-8", timeout=30, env=env,
        )
        self.assertEqual(process.returncode, expected, (process.stdout, process.stderr))
        self.assertEqual(process.stderr, "")
        document = json.loads(process.stdout, parse_constant=lambda value: self.fail(value))
        self.assertEqual(document["schema_version"], 1)
        if expected in (0, 1):
            self.assertEqual(document["status"], "ok")
        else:
            self.assertEqual(document["status"], "error")
        return document

    def query(self, action: str, *options, **keywords) -> dict:
        return self.invoke("run", action, "--root", self.store, "--id", RUN_ID, *options, **keywords)

    def sidecar_config(self, backend: str, mode: str = "accept", *, append_source: bool = False,
                       **overrides) -> tuple[Path, Path, Path]:
        cwd = self.root / f"隔离 cwd {backend} {mode}"
        cwd.mkdir()
        capture = self.root / f"capture-{backend}-{mode}.jsonl"
        literal = 'literal " ; & $(not-a-command) 尾\\'
        config = {
            "schema_version": 1, "backend": backend, "executable": str(Path(sys.executable).resolve()),
            "arguments": ["-I", "-u", str(Path(__file__).resolve()), "--fixture-sidecar",
                          "--capture", str(capture), "--mode", mode, "--literal", literal],
            "working_directory": str(cwd), "inherit_environment": False,
            "environment": {"JYPPX_CLI_CHILD_VALUE": "child.值"},
        }
        if append_source:
            config["arguments"].extend(["--append-root", str(self.store)])
        config.update(overrides)
        path = self.root / f"config-{backend}-{mode}.json"
        path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        return path, capture, cwd

    def test_events_keep_canonical_bytes_and_fixed_snapshot(self) -> None:
        self.assertTrue(self.invoke("run", "verify", "--root", self.store)["data"]["valid"])
        first = self.query("events", "--limit", 2)["data"]
        self.assertEqual(first["snapshot_through"], 4)
        self.assertEqual(first["next_after"], 2)
        late = envelope(5, {"kind": 3, "severity": 3, "body": "late append"})
        seed_database(self.store, [late])
        second = self.query("events", "--limit", 2, "--after", first["next_after"],
                            "--snapshot", first["snapshot_through"])["data"]
        self.assertIsNone(second["next_after"])
        records = first["events"] + second["events"]
        self.assertEqual([record["run_sequence"] for record in records], [1, 2, 3, 4])
        for record, original in zip(records, self.events):
            encoded = canonical(original)
            self.assertEqual(record["canonical_hex"], encoded.hex())
            self.assertEqual(record["content_digest"], hashlib.sha256(encoded).hexdigest())
            self.assertEqual(record["event"], projection(original))
        self.assertEqual(records[0]["event"]["time"], {
            "wall_time_unix_ns": None, "monotonic_time_ns": None, "clock_id": ""})
        fresh = self.query("events")["data"]
        self.assertEqual(fresh["snapshot_through"], 5)
        logs = self.query("events", "--kind", "log", "--snapshot", 4)["data"]
        self.assertEqual([record["run_sequence"] for record in logs["events"]], [1])

    def test_metrics_preserve_status_definitions_and_source_provenance(self) -> None:
        records, sources = [], {}
        options: list = []
        while True:
            page = self.query("metrics", "--metric", "validation.accuracy", "--limit", 2, *options)["data"]
            records.extend(page["points"])
            sources.update((source["run_sequence"], source) for source in page["source_events"])
            cursor = page["next_after"]
            if cursor is None:
                break
            options = ["--snapshot", page["snapshot_through"], "--after-sequence",
                       cursor["run_sequence"], "--after-index", cursor["point_index"]]
        self.assertEqual(len(records), 8)
        self.assertEqual([record["status_name"] for record in records[:5]], [
            "finite", "not_a_number", "positive_infinity", "negative_infinity", "missing"])
        self.assertEqual(records[5]["point"]["step"], {"kind": 1, "value": 2**63 + 1})
        self.assertEqual(records[6]["resource_id"], "device.fixture")
        self.assertEqual(records[6]["time"]["wall_time_unix_ns"], -17)
        self.assertEqual(records[6]["point"]["definition"]["unit"], 6)
        for record in records:
            sequence, index = record["cursor"]["run_sequence"], record["cursor"]["point_index"]
            original = self.events[sequence - 1]
            self.assertEqual(record["point"], source_points(projection(original))[index])
            self.assertEqual(record["source_content_digest"], hashlib.sha256(canonical(original)).hexdigest())
            self.assertEqual(record["event_identity"], original["identity"])
            self.assertEqual(record["time"], original["time"])
            self.assertEqual(record["correlation"], original["correlation"])
            self.assertEqual(record["attributes"], original["attributes"])
            self.assertEqual(sources[sequence]["canonical_hex"], canonical(original).hex())
        filtered = self.query("metrics", "--metric", "validation.accuracy", "--definition-version", 2,
                              "--step-kind", "epoch")["data"]["points"]
        self.assertEqual([record["cursor"] for record in filtered], [{"run_sequence": 3, "point_index": 0}])

    def test_text_escapes_terminal_controls_and_json_preserves_evidence(self) -> None:
        body = "日志\x1b]52;c;dW50cnVzdGVk\x07\r\n\x1b[2J\x7f\u009b31m"
        value = envelope(5, {"kind": 3, "severity": 2, "body": body})
        value["attributes"] = {"dynamic\u009bkey": "控制\u0085text"}
        seed_database(self.store, [value])
        records = self.query("events")["data"]["events"]
        self.assertEqual(records[-1]["event"]["payload"]["body"], body)
        self.assertEqual(records[-1]["event"]["attributes"], value["attributes"])
        result = subprocess.run(
            [str(self.cli), "run", "events", "--root", str(self.store), "--id", RUN_ID,
             "--format", "text"], capture_output=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = result.stdout.decode("utf-8")
        self.assertIn("日志\\u001b]52", output)
        self.assertIn("dynamic\\u009bkey", output)
        self.assertIn("控制\\u0085text", output)
        for control in ("\x1b", "\x07", "\x7f", "\u0085", "\u009b"):
            self.assertNotIn(control, output)

    def test_metric_pages_exclude_concurrent_append(self) -> None:
        first = self.query("metrics", "--metric", "validation.accuracy", "--limit", 1)["data"]
        late = envelope(5, {"kind": 2, "points": [metric_point(0.5, step=0)]})
        seed_database(self.store, [late])
        cursor = first["next_after"]
        second = self.query("metrics", "--metric", "validation.accuracy", "--snapshot", first["snapshot_through"],
                            "--after-sequence", cursor["run_sequence"], "--after-index", cursor["point_index"])["data"]
        self.assertEqual(len(second["points"]), 7)
        self.assertTrue(all(record["cursor"]["run_sequence"] <= 4 for record in second["points"]))

    def test_jsonl_export_resumes_identical_evidence_and_recovers_tail(self) -> None:
        output = self.root / "输出 events.jsonl"
        first = self.query("export", "--backend", "jsonl", "--output", output, "--page-size", 1)["data"]
        self.assertTrue(first["fully_represented"])
        self.assertEqual(first["read_events"], 4)
        self.assertEqual(first["accepted_events"], 4)
        self.assertEqual(first["delivery_guarantee"], "at_least_once")
        original = output.read_bytes()
        for record, value in zip(map(json.loads, original.splitlines()), self.events):
            self.assertEqual(record["canonical_hex"], canonical(value).hex())
            self.assertEqual(record["event"], projection(value))
        with output.open("ab") as stream:
            stream.write(b'{"interrupted":')
        resumed = self.query("export", "--backend", "jsonl", "--output", output, "--resume", "true")["data"]
        self.assertEqual(resumed["accepted_events"], 0)
        self.assertEqual(resumed["duplicate_events"], 4)
        self.assertEqual(resumed["jsonl_recovery"]["truncated_tail_bytes"], len(b'{"interrupted":'))
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(self.query("export", "--backend", "jsonl", "--output", output,
                                   expected=2)["error"]["name"], "invalid_argument")

    def test_export_uses_authoritative_snapshot_and_capacity_preflight(self) -> None:
        output = self.root / "snapshot.jsonl"
        data = self.query("export", "--backend", "jsonl", "--output", output, "--snapshot", 3,
                          "--after", 1, "--page-size", 1)["data"]
        self.assertEqual(data["snapshot_through"], 3)
        self.assertEqual(data["read_events"], 2)
        self.assertEqual([json.loads(line)["event"]["identity"]["sequence"]
                          for line in output.read_text(encoding="utf-8").splitlines()], [2, 3])
        refused = self.root / "refused.jsonl"
        self.query("export", "--backend", "jsonl", "--output", refused, "--max-events", 1, expected=3)
        self.assertFalse(refused.exists())
        self.query("export", "--backend", "jsonl", "--output", self.store / "runs.sqlite3",
                   "--resume", "true", expected=2)
        unrelated = self.root / "unrelated.txt"
        unrelated.write_bytes(b"keep original bytes")
        self.query("export", "--backend", "jsonl", "--output", unrelated, "--resume", "true", expected=2)
        self.assertEqual(unrelated.read_bytes(), b"keep original bytes")

    def test_official_backends_use_fixed_argv_child_environment_and_final_close(self) -> None:
        for backend in ("tensorboard", "opentelemetry"):
            with self.subTest(backend=backend):
                config, capture, cwd = self.sidecar_config(backend)
                parent = os.environ.copy()
                parent["JYPPX_CLI_PARENT_SECRET"] = "do-not-inherit"
                data = self.query("export", "--backend", backend, "--sidecar-config", config,
                                  "--page-size", 1, env=parent)["data"]
                self.assertTrue(data["scan_completed"])
                self.assertTrue(data["fully_represented"])
                self.assertEqual(data["accepted_events"], 4)
                self.assertEqual(data["transport"]["completed_requests"], 6)
                items = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(Path(items[0]["launch"]["cwd"]), cwd)
                self.assertEqual(items[0]["launch"]["child_value"], "child.值")
                self.assertFalse(items[0]["launch"]["parent_secret_present"])
                self.assertEqual(items[0]["launch"]["literal"], 'literal " ; & $(not-a-command) 尾\\')
                requests = [item["request"] for item in items[1:]]
                self.assertEqual(requests[0]["operation"], "flush")
                self.assertEqual(requests[-1]["operation"], "close")
                emitted = [item for request in requests if request["operation"] == "emit" for item in request["events"]]
                self.assertEqual([item["envelope"] for item in emitted], list(map(projection, self.events)))
                self.assertEqual([item["content_digest"] for item in emitted],
                                 [hashlib.sha256(canonical(value)).hexdigest() for value in self.events])

    def test_unsupported_and_sampling_have_explicit_counts_and_verdict(self) -> None:
        for mode in ("unsupported", "sampled"):
            with self.subTest(mode=mode):
                config, _, _ = self.sidecar_config("tensorboard", mode)
                result = self.query("export", "--backend", "tensorboard", "--sidecar-config", config, expected=1)
                data = result["data"]
                self.assertTrue(data["scan_completed"])
                self.assertFalse(data["fully_represented"])
                self.assertEqual(data["accepted_events"], 0)
                self.assertEqual(data["unsupported_events" if mode == "unsupported" else "sampled_out_events"], 4)
                self.assertEqual(self.query("events")["data"]["events"][0]["event"], projection(self.events[0]))

    def test_process_export_excludes_an_append_after_initial_snapshot(self) -> None:
        config, capture, _ = self.sidecar_config("tensorboard", append_source=True)
        data = self.query("export", "--backend", "tensorboard", "--sidecar-config", config,
                          "--page-size", 1)["data"]
        self.assertEqual(data["snapshot_through"], 4)
        self.assertEqual(data["read_events"], 4)
        requests = [item["request"] for item in map(json.loads, capture.read_text(encoding="utf-8").splitlines())
                    if "request" in item]
        self.assertEqual([item["envelope"]["identity"]["sequence"] for request in requests
                          for item in request.get("events", [])], [1, 2, 3, 4])
        self.assertEqual(self.query("events")["data"]["snapshot_through"], 5)

    def test_corrupt_canonical_source_is_rejected_without_creating_output(self) -> None:
        with closing(sqlite3.connect(self.store / "runs.sqlite3")) as database, database:
            database.execute("UPDATE jyppx_events SET content_digest=? WHERE run_sequence=2", ("0" * 64,))
        self.query("events", expected=2)
        self.query("metrics", "--metric", "validation.accuracy", expected=2)
        output = self.root / "must-not-exist.jsonl"
        self.query("export", "--backend", "jsonl", "--output", output, expected=2)
        self.assertFalse(output.exists())

    def test_process_deadline_and_protocol_failures_do_not_change_evidence(self) -> None:
        config, _, _ = self.sidecar_config("tensorboard", "hang", operation_timeout_ms=100)
        self.assertEqual(self.query("export", "--backend", "tensorboard", "--sidecar-config", config,
                                    "--timeout-ms", 1000, expected=124)["error"]["name"], "deadline_exceeded")
        config, _, _ = self.sidecar_config("tensorboard", "bad-frame")
        self.query("export", "--backend", "tensorboard", "--sidecar-config", config, expected=2)
        config, _, _ = self.sidecar_config("tensorboard", "bad-counter")
        self.query("export", "--backend", "tensorboard", "--sidecar-config", config, expected=2)
        self.assertEqual(len(self.query("events")["data"]["events"]), 4)

    def test_invalid_cursor_limits_and_launch_configuration_fail_explicitly(self) -> None:
        for action, options in (
            ("events", ["--after", 1]), ("events", ["--snapshot", 99]),
            ("events", ["--kind", "unknown"]), ("events", ["--limit", 0]),
            ("events", ["--limit", 1025]), ("events", ["--page-bytes", 1]),
            ("metrics", ["--metric", "validation.accuracy", "--after-sequence", 1]),
            ("metrics", ["--metric", "validation.accuracy", "--after-index", 1]),
            ("metrics", ["--metric", "validation.accuracy", "--definition-version", 0]),
            ("metrics", ["--metric", "validation.accuracy", "--step-kind", "unknown"]),
        ):
            with self.subTest(action=action, options=options):
                self.query(action, *options, expected=2)
        config, _, _ = self.sidecar_config("tensorboard", executable="python")
        self.query("export", "--backend", "tensorboard", "--sidecar-config", config, expected=2)
        self.query("export", "--backend", "opentelemetry", "--sidecar-config", config, expected=2)
        missing = self.root / "missing" / "new-root"
        self.invoke("run", "events", "--root", missing, "--id", RUN_ID, expected=3)
        self.assertFalse(missing.parent.exists())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    options = parser.parse_args()
    RunTelemetryContract.cli = options.cli.resolve(strict=True)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(RunTelemetryContract)
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(fixture_sidecar() if "--fixture-sidecar" in sys.argv else main())
