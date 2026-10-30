# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Maintenance-only contract checker for an already-built C++ CLI.

No Python module is used by the product. Standard-library checks inspect the
native wire and independently check TFRecord CRCs. The explicit --official
gate additionally requires the pinned official reader and reference writer.
This suite does not qualify OS tree cleanup, TLS, crash recovery or fsync.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import test_native_otlp_worker as wire
import test_run_telemetry as reference

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import test_native_tensorboard as tensorboard_reference


def compact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def request(values):
    value = wire.emit_request(values)
    value["backend"] = "tensorboard"
    return value


def control(operation):
    return {"protocol_version": 1, "backend": "tensorboard",
            "operation": operation, "request_id": operation}


def fixture_events():
    payloads = [
        {"kind": 1, "name": "training.finished", "terminal": False,
         "metrics": [reference.metric_point()], "decisions": []},
        {"kind": 2, "points": [reference.metric_point(-0.0)]},
        {"kind": 3, "severity": 2, "body": "not represented in TensorBoard"},
        {"kind": 4, "name": "train.batch", "start_monotonic_ns": 400,
         "end_monotonic_ns": 500, "status": 2, "reason": ""},
        {"kind": 5, "resource_id": "device.cpu", "measurements": [reference.metric_point()]},
        {"kind": 6, "role": "checkpoint.best", "artifact": {
            "algorithm": "sha256", "digest": "a" * 64, "size_bytes": 123,
            "logical_name": "best.model", "media_type": "application/octet-stream",
            "producer": "fixture"}},
    ]
    return [reference.envelope(index, payload) for index, payload in enumerate(payloads, 1)]


class NativeTensorBoardWorkerContract(unittest.TestCase):
    cli: Path
    disabled = False
    sidecar = None

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="liuding-native-tb-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cwd, self.managed = self.root / "worker", self.root / "managed"
        self.cwd.mkdir()
        self.managed.mkdir()
        self.config_index = 0

    def configuration(self, **extra):
        self.config_index += 1
        output = self.root / f"events-{self.config_index}"
        value = {"schema_version": 1, "directory": str(output),
                 "working_directory": str(self.cwd), **extra}
        config = self.root / f"native-{self.config_index}.json"
        config.write_bytes(compact(value))
        return config, Path(value["directory"])

    def invoke(self, *arguments, expected=0):
        result = subprocess.run([str(self.cli), *(str(v) for v in arguments),
                                 "--format", "json"], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, expected, (result.stdout, result.stderr))
        self.assertEqual(result.stderr, b"")
        return json.loads(result.stdout)

    def store(self):
        store = self.root / "store"
        metadata = self.root / "metadata.json"
        metadata.write_bytes(compact({
            "schema_version": 1, "run_id": reference.RUN_ID, "project_id": "project.fixture",
            "experiment_id": "experiment.fixture", "display_name": "Native export fixture",
            "status": "running", "created_at_unix_ms": 100, "updated_at_unix_ms": 100,
        }))
        self.invoke("run", "start", "--root", store, "--metadata", metadata)
        values = fixture_events()[:3]
        reference.seed_database(store, values)
        return store, values

    def hidden(self, config, data=b"", *, digest=None, extra=()):
        result = subprocess.run([
            str(self.cli), "__liuding_tensorboard_worker", "--config", str(config),
            "--config-sha256", digest or hashlib.sha256(config.read_bytes()).hexdigest(),
            "--managed-root", str(self.managed), *(str(v) for v in extra)],
            input=data, capture_output=True, timeout=30)
        return result

    def exchange(self, config, requests):
        result = self.hidden(config, b"".join(compact(value) + b"\n" for value in requests))
        self.assertEqual(result.returncode, 0, (result.stdout, result.stderr))
        self.assertEqual(result.stderr, b"")
        self.assertLessEqual(len(result.stdout), len(requests) * (128 * 1024 + 1))
        self.assertTrue(result.stdout.endswith(b"\n"))
        replies = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(replies), len(requests))
        for source, reply in zip(requests, replies):
            self.assertEqual(reply["protocol_version"], 1)
            self.assertEqual(reply["request_id"], source["request_id"])
        return replies

    def failed(self, result):
        self.assertEqual(result.returncode, 3, (result.stdout, result.stderr))
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr.replace(b"\r\n", b"\n"),
                         b"Native TensorBoard worker failed.\n")

    def records(self, output):
        files = tensorboard_reference.event_files(output)
        return tensorboard_reference.raw_records(files[0])

    def test_cli_canonical_export_or_explicit_disabled_build(self):
        store, _ = self.store()
        config, output = self.configuration()
        result = self.invoke("run", "export", "--root", store, "--id", reference.RUN_ID,
                             "--backend", "tensorboard-native", "--native-config", config,
                             "--page-size", 1, expected=2 if self.disabled else 1)
        if self.disabled:
            self.assertEqual(result["error"]["name"], "not_implemented")
            self.assertFalse(output.exists())
            return
        data = result["data"]
        self.assertEqual(data["source"], "sqlite-canonical")
        self.assertEqual(data["backend"], "tensorboard-native")
        self.assertTrue(data["scan_completed"])
        self.assertFalse(data["fully_represented"])
        self.assertEqual((data["read_events"], data["accepted_events"],
                          data["unsupported_events"]), (3, 2, 1))
        self.assertGreater(data["transport"]["completed_requests"], 2)
        self.assertEqual(data["destination_counters"], {
            "exported_events": 2, "unsupported_events": 1, "sampled_out_events": 0})
        direct = self.root / "direct"
        original = self.invoke("run", "export", "--root", store, "--id", reference.RUN_ID,
                               "--backend", "tensorboard-native", "--output", direct,
                               expected=1)["data"]
        self.assertIsNone(original["transport"])
        self.assertEqual(original["destination_counters"], data["destination_counters"])
        # Initial headers contain independent creation times; summaries must agree.
        self.assertEqual(self.records(output)[1:], self.records(direct)[1:])

    def test_hidden_disabled_or_hash_drift_has_no_stdout_or_output(self):
        config, output = self.configuration()
        original_digest = hashlib.sha256(config.read_bytes()).hexdigest()
        config.write_bytes(config.read_bytes() + b"\n")
        result = self.hidden(config, digest=original_digest)
        if self.disabled:
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, b"")
            self.assertEqual(result.stderr.replace(b"\r\n", b"\n"),
                             b"Native TensorBoard is disabled in this build.\n")
        else:
            self.failed(result)
        self.assertFalse(output.exists())

    def test_all_payload_replay_preserves_unsupported_and_close(self):
        config, output = self.configuration()
        emit = request(fixture_events())
        replies = self.exchange(config, [emit, emit, control("flush"), control("close"),
                                          control("close"), emit])
        self.assertEqual(replies[0]["receipts"], [1, 1, 4, 4, 1, 4])
        self.assertEqual(replies[1]["receipts"], [2, 2, 4, 4, 2, 4])
        self.assertEqual(replies[2]["counters"], {
            "exported_events": 3, "unsupported_events": 3, "sampled_out_events": 0})
        self.assertEqual(replies[3], replies[4])
        self.assertEqual(replies[5]["error"], "native_tensorboard.operation_rejected")
        self.assertNotIn("receipts", replies[5])
        self.assertNotIn("counters", replies[5])
        self.assertEqual(len(self.records(output)), 5)

    def test_wrong_backend_status_and_digest_drift_do_not_write(self):
        config, output = self.configuration()
        valid = request([fixture_events()[1]])
        wrong = copy.deepcopy(valid)
        wrong["backend"] = "opentelemetry"
        drift = copy.deepcopy(valid)
        drift["events"][0]["envelope"]["attributes"]["origin"] = "changed"
        status = copy.deepcopy(valid)
        status["events"][0]["envelope"]["payload"]["points"][0]["status"] = 6
        replies = self.exchange(config, [wrong, drift, status, valid, control("close")])
        for rejected in replies[:3]:
            self.assertEqual(rejected["error"], "native_tensorboard.operation_rejected")
            self.assertNotIn("receipts", rejected)
            self.assertNotIn("counters", rejected)
        self.assertEqual(replies[3]["receipts"], [1])
        self.assertEqual(len(self.records(output)), 2)

    def test_whole_batch_capacity_failure_does_not_acknowledge_a_prefix(self):
        config, output = self.configuration(max_seen=1)
        events = fixture_events()
        replies = self.exchange(config, [request(events[:2]), control("flush"),
                                          request(events[:1]), control("close")])
        self.assertIn("error", replies[0])
        self.assertNotIn("receipts", replies[0])
        self.assertNotIn("counters", replies[0])
        self.assertEqual(replies[1]["counters"]["exported_events"], 0)
        self.assertEqual(replies[2]["receipts"], [1])
        self.assertEqual(len(self.records(output)), 3)

    def test_seen_byte_capacity_failure_writes_only_header(self):
        config, output = self.configuration(max_seen_bytes=1)
        replies = self.exchange(config, [request(fixture_events()[:1]), control("close")])
        self.assertIn("error", replies[0])
        self.assertNotIn("receipts", replies[0])
        self.assertEqual(replies[1]["counters"]["exported_events"], 0)
        self.assertEqual(len(self.records(output)), 1)

    def test_invalid_frames_are_fatal_without_protocol_reply(self):
        for frame in (b"\n", b"{}\r\n", b"{}\x00\n", b"{}",
                      b'{"request_id":"flush","request_id":"flush"}\n',
                      b" " * (16 * 1024 * 1024 + 1) + b"\n"):
            with self.subTest(frame_bytes=len(frame)):
                config, output = self.configuration()
                self.failed(self.hidden(config, frame))
                self.assertEqual(len(self.records(output)), 1)

    def test_unknown_version_fields_and_invalid_limits_fail_before_output(self):
        changes = [dict(schema_version=2), dict(schema_version=True), dict(executable="ignored"),
                   dict(directory="relative-output"), dict(working_directory="relative-worker"),
                   dict(inherit_environment=1), dict(environment={"bad=key": "value"}),
                   dict(max_seen=0), dict(max_seen=100001), dict(max_seen_bytes=0),
                   dict(max_file_bytes=0), dict(max_record_bytes=0),
                   dict(max_batch_encoded_bytes=15), dict(operation_timeout_ms=600001),
                   dict(startup_timeout_ms=0), dict(cleanup_timeout_ms=10001)]
        for change in changes:
            with self.subTest(change=change):
                config, output = self.configuration(**change)
                self.failed(self.hidden(config))
                self.assertFalse(output.exists())

    def test_duplicate_or_oversized_config_fail_before_output(self):
        for duplicate in (True, False):
            config, output = self.configuration()
            original = config.read_bytes()
            config.write_bytes(original[:-1] + b',"schema_version":1}' if duplicate else
                               original + b" " * (64 * 1024))
            self.failed(self.hidden(config))
            self.assertFalse(output.exists())

    def test_existing_directory_is_never_reused(self):
        config, output = self.configuration()
        output.mkdir()
        marker = output / "marker.txt"
        marker.write_bytes(b"owned by another run")
        self.failed(self.hidden(config))
        self.assertEqual(list(output.iterdir()), [marker])
        self.assertEqual(marker.read_bytes(), b"owned by another run")

    def test_child_and_parent_refuse_managed_store_outputs(self):
        config, output = self.configuration(directory=str(self.managed / "must-not-create"))
        self.failed(self.hidden(config))
        self.assertFalse(output.exists())
        store, _ = self.store()
        config, output = self.configuration(directory=str(store / "must-not-create"))
        self.invoke("run", "export", "--root", store, "--id", reference.RUN_ID,
                    "--backend", "tensorboard-native", "--native-config", config, expected=2)
        self.assertFalse(output.exists())

    def test_explicit_modes_are_mutually_exclusive(self):
        store, _ = self.store()
        config, output = self.configuration()
        for extra in (("--output", self.root / "other"), ("--resume", "false"),
                      ("--sidecar-config", self.root / "absent.json")):
            self.invoke("run", "export", "--root", store, "--id", reference.RUN_ID,
                        "--backend", "tensorboard-native", "--native-config", config,
                        *extra, expected=2)
            self.assertFalse(output.exists())

    def test_snapshot_seen_capacity_is_rejected_before_launch(self):
        store, _ = self.store()
        config, output = self.configuration(max_seen=1)
        self.invoke("run", "export", "--root", store, "--id", reference.RUN_ID,
                    "--backend", "tensorboard-native", "--native-config", config, expected=3)
        self.assertFalse(output.exists())

    def test_hidden_format_is_rejected_before_output(self):
        config, output = self.configuration()
        self.failed(self.hidden(config, extra=("--format", "json")))
        self.assertFalse(output.exists())

    def test_official_reader_and_writer_agree_with_real_worker(self):
        # Selected only by --official; absence of pinned dependencies is fatal.
        values = fixture_events()
        config, output = self.configuration()
        emit = request(values)
        replies = self.exchange(config, [emit, emit, control("close")])
        checked = tensorboard_reference.compare(
            output, [reference.projection(value) for value in values], self.sidecar,
            self.root / "official-reference")
        self.assertEqual(checked["accepted"], 3)
        self.assertEqual(checked["unsupported"], 3)
        self.assertEqual(replies[-1]["counters"]["exported_events"], checked["accepted"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--sidecar", type=Path)
    args = parser.parse_args()
    case = NativeTensorBoardWorkerContract
    case.cli, case.disabled = args.cli.resolve(strict=True), args.disabled
    official_test = "test_official_reader_and_writer_agree_with_real_worker"
    names = unittest.defaultTestLoader.getTestCaseNames(case)
    if args.official:
        if args.disabled or args.sidecar is None:
            parser.error("--official requires --sidecar and an enabled native build")
        if sys.version_info[:2] != (3, 11):
            raise AssertionError("Use the locked CPython 3.11 official reference runtime")
        for package, version in (("tensorboard", "2.20.0"), ("protobuf", "6.33.5")):
            if importlib.metadata.version(package) != version:
                raise AssertionError("Official reference dependency version mismatch")
        if hashlib.sha256(args.sidecar.read_bytes()).hexdigest() != \
                tensorboard_reference.REFERENCE_SIDECAR_SHA256:
            raise AssertionError("Pinned independent reference writer source drifted")
        case.sidecar = tensorboard_reference.module(
            args.sidecar.resolve(strict=True), "liuding_tb_worker_reference")
        names = [official_test]
    elif args.disabled:
        names = ["test_cli_canonical_export_or_explicit_disabled_build",
                 "test_hidden_disabled_or_hash_drift_has_no_stdout_or_output"]
    else:
        names.remove(official_test)
    suite = unittest.TestSuite(case(name) for name in names)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
