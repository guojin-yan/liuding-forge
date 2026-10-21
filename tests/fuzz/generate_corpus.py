# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regenerate small data-only parser seeds without importing framework binaries.

These fixtures encode the documented wire schemas. Each C++ harness also
generates and validates seeds through the real production codecs at startup;
the files give libFuzzer valid raw inputs as well as its structured edit path.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "fuzz"
BUILD = {
    "jyppx_version": "1.0.0", "git_commit": "fuzz-seed",
    "compiler": "clang-fuzz", "cxx_standard": "202002",
    "os_arch": "linux-x86_64", "build_type": "Debug",
    "libtorch_version": "2.5.1", "libtorch_build": "cpu",
    "accelerator_runtime": "cpu", "dependency_lock_digest": "unlocked",
}


def encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def write(family: str, name: str, value: bytes) -> None:
    path = ROOT / family / "corpus" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def dictionary(family: str, tokens: list[str]) -> None:
    directory = ROOT / family
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"# {family} bounded parser mutation tokens"]
    for index, token in enumerate(tokens):
        # libFuzzer dictionaries use C-style byte escapes, not JSON \u escapes.
        escaped = "".join(f"\\x{byte:02x}" for byte in token.encode("utf-8"))
        lines.append(f'field{index}="{escaped}"')
    (directory / f"{family}.dict").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8", newline="\n")


def digest(character: str) -> dict:
    return {"algorithm": "sha256", "hex": character * 64}


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return u64(len(raw)) + raw


def attributes(value: dict[str, str]) -> bytes:
    return u64(len(value)) + b"".join(
        string(key) + string(item) for key, item in sorted(value.items())
    )


def binary_definition() -> bytes:
    return (struct.pack("<I", 1) + string("classification.accuracy") +
            struct.pack("<I", 1) + bytes([5, 3, 3]) +
            attributes({"average": "micro"}) + u64(1) + string("split"))


def binary_point() -> bytes:
    return (binary_definition() + bytes([4]) + u64(10) + bytes([1, 1]) +
            struct.pack("<d", 0.9) + string("") +
            attributes({"split": "validation"}) + bytes([1]) + u64(10))


def binary_envelope(kind: int) -> bytes:
    prefix = (
        string("jyppx.telemetry.content/v1") + struct.pack("<I", 1) +
        string("fuzz.stream") + string("event.1") + string("operation.1") + u64(1) +
        bytes([1]) + struct.pack("<q", 1700000000123456789) + bytes([1]) + u64(100) +
        string("fuzz.clock") + string("project.fuzz") + string("run.fuzz") +
        string("attempt.1") + string("trace.fuzz") + string("span.fuzz") + string("") +
        bytes([1]) + attributes({"source": "fuzzer"}) + bytes([kind + 1])
    )
    point = binary_point()
    payloads = [
        string("training.epoch_finished") + bytes([0]) + u64(1) + point + u64(1) +
        string("early_stop") + string("continue") + string("budget remains") + bytes([1]),
        u64(1) + point,
        bytes([2]) + string("redacted log"),
        string("training.step") + u64(1) + u64(100) + bytes([2]) + string(""),
        string("cpu.0") + u64(1) + point,
        string("checkpoint.last") + bytes([1]) + string("a" * 64) + u64(16) +
        string("checkpoint") + string("application/octet-stream") + string("fuzz.fixture"),
    ]
    return prefix + payloads[kind]


def telemetry() -> None:
    for kind, name in enumerate(("lifecycle", "metric", "log", "span", "resource", "artifact")):
        write("telemetry_codec", name, binary_envelope(kind))
    write("telemetry_codec", "definition",
          string("jyppx.telemetry.metric-definition/v1") + binary_definition())
    write("telemetry_codec", "truncated-length", b"\xff" * 8)
    dictionary("telemetry_codec", ["jyppx.telemetry.content/v1",
                                  "jyppx.telemetry.metric-definition/v1",
                                  "classification.accuracy", "split", "validation"])


def checkpoints() -> None:
    manifest = {
        "schema_version": 1, "build": BUILD,
        "participants": [{"state": {
            "participant_id": "fuzz.participant", "schema_version": 1,
            "restore_level": "exact", "degradation_reasons": [],
            "fields": [
                {"name": "optional", "media_type": "application/octet-stream",
                 "max_bytes": 32, "required": False},
                {"name": "state", "media_type": "application/octet-stream",
                 "max_bytes": 64, "required": True},
            ],
        }, "blobs": [{"field_name": "state", "relative_path": "blobs/state.bin",
                       "size_bytes": 8, "digest": digest("a")}]}],
    }
    for level in ("exact", "bounded", "best_effort"):
        state = manifest["participants"][0]["state"]
        state["restore_level"] = level
        state["degradation_reasons"] = [] if level == "exact" else ["topology may differ"]
        write("checkpoint_manifest", level, encoded(manifest))
    dictionary("checkpoint_manifest", ["schema_version", "participants", "restore_level",
                                       "relative_path", "blobs", "sha256", "../state", "max_bytes"])


def safe_file(header: dict, payload: bytes = b"") -> bytes:
    raw = encoded(header)
    raw += b" " * (-len(raw) % 8)
    return u64(len(raw)) + raw + payload


def safetensors() -> bytes:
    tensor = safe_file({
        "layer.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "empty": {"dtype": "U8", "shape": [0, 3], "data_offsets": [8, 8]},
        "__metadata__": {"producer": "fuzz.fixture"},
    }, bytes.fromhex("0000803f00000080"))
    write("safetensors", "weights", tensor)
    write("safetensors", "empty", safe_file({}))
    write("safetensors", "scalar", safe_file({
        "scalar": {"dtype": "U64", "shape": [], "data_offsets": [0, 8]},
    }, bytes.fromhex("ffffffffffffffff")))
    write("safetensors", "declared-header-overflow", b"\xff" * 8)
    dictionary("safetensors", ["__metadata__", "dtype", "data_offsets", "shape",
                                "F32", "U64", "BOOL", "18446744073709551615"])
    return tensor


def bundles(tensor_file: bytes) -> None:
    manifest = {
        "format": "jyppx.model_bundle", "schema_version": 1, "bundle_id": "fuzz-bundle",
        "build": BUILD, "compression": "none", "encryption": "none", "files": [],
        "models": [{"id": "primary", "format": "safetensors", "file_ids": ["weights"],
                    "tensors": [
                        {"name": "empty", "dtype": "U8", "shape": [0, 3],
                         "device_intent": "cpu", "shard_file_id": "weights"},
                        {"name": "layer.weight", "dtype": "F32", "shape": [2],
                         "device_intent": "cpu", "shard_file_id": "weights"},
                    ]}],
        "runtime": {"backend_id": "safetensors.adapter", "backend_version": "1.0.0",
                    "environment_lock": digest("b"), "dependencies": [
                        {"name": "safetensors", "version": "0.6.2"}],
                    "devices": ["cpu", "cuda"], "dtypes": ["float32"],
                    "os_arch": ["linux-x86_64", "windows-x86_64"], "dynamic_shapes": False},
        "signatures": [{"algorithm": "ed25519", "key_id": "fuzz-key", "signature_hex": "0011aabb"}],
    }
    for field, file_id, role in (
        ("input_schema", "input", 3), ("output_schema", "output", 3),
        ("preprocessing", "pre", 4), ("postprocessing", "post", 4),
        ("labels", "labels", 5), ("evaluation", "quality", 6),
        ("sbom", "sbom", 7), ("provenance", "provenance", 8),
    ):
        manifest[field] = {"schema_id": f"fuzz.{file_id}", "schema_version": 1, "file_id": file_id}
        manifest["files"].append({
            "id": file_id, "path": f"metadata/{file_id}.json", "role": role,
            "artifact": {"digest": {"algorithm": "sha256", "hex": hashlib.sha256(b"{}").hexdigest()},
                         "size_bytes": 2, "logical_name": file_id,
                         "media_type": "application/json", "producer": "fuzz.fixture"},
        })
    manifest["weight_import_reports"] = []
    manifest["files"].append({
        "id": "weights", "path": "weights/primary.safetensors", "role": 1,
        "artifact": {"digest": {"algorithm": "sha256", "hex": hashlib.sha256(tensor_file).hexdigest()},
                     "size_bytes": len(tensor_file), "logical_name": "weights",
                     "media_type": "application/safetensors", "producer": "fuzz.fixture"},
    })
    write("model_bundle_manifest", "signed", encoded(manifest))
    manifest["labels"] = None
    manifest["signatures"] = []
    manifest["files"] = [file for file in manifest["files"] if file["role"] != 5]
    write("model_bundle_manifest", "unsigned", encoded(manifest))
    dictionary("model_bundle_manifest", ["jyppx.model_bundle", "schema_version", "environment_lock",
                                          "file_ids", "device_intent", "signatures", "signature_hex",
                                          "manifest.json", "../weights", "__metadata__"])


def evaluations() -> None:
    definition = {"schema_version": 1, "metric_id": "classification.accuracy", "version": 1,
                  "unit": 5, "aggregation": 3, "direction": 3,
                  "parameters": {"average": "micro"}, "context_keys": ["split"]}
    rule = {"rule_id": "validation.accuracy", "definition": definition,
            "context": {"split": "validation"}, "minimum_samples": 10, "minimum": 0.8,
            "maximum": 1.0, "max_absolute_regression": 0.01, "max_relative_regression": 0.02}
    suite = {"schema_version": 1, "suite_id": "classification.fuzz", "version": 1,
             "task": "classification", "dataset_digest": digest("a"),
             "environment_id": "cpu-f64-deterministic", "seed": 71, "rules": [rule]}
    identity = {"evaluation_id": "eval.fuzz", "run_id": "run.fuzz", "model_digest": digest("b"),
                "build_fingerprint": "fuzz-build", "adapter_id": "classification.local",
                "adapter_version": 1, "config_digest": digest("c")}
    point = {"definition": definition, "step": {"kind": 4, "value": 10},
             "observation": {"status": 1, "value": 0.9, "reason": ""},
             "context": {"split": "validation"}, "sample_count": 10}
    write("evaluation_json", "suite", encoded(suite))
    write("evaluation_json", "measured-input", encoded({
        "schema_version": 1, "identity": identity, "observations": [point]}))
    report = {"schema_version": 1, "suite": suite, "identity": identity, "verdict": 1,
              "results": [{"rule_id": "validation.accuracy", "verdict": 1,
                           "reason": "", "observation": point}]}
    write("evaluation_json", "passing-report", encoded(report))
    missing = copy.deepcopy(report)
    missing["verdict"] = 4
    missing["results"] = [{"rule_id": "validation.accuracy", "verdict": 4,
                           "reason": "required observation missing", "observation": None}]
    write("evaluation_json", "incomplete-report", encoded(missing))
    dictionary("evaluation_json", ["schema_version", "suite_id", "observations", "results",
                                   "verdict", "minimum_samples", "max_relative_regression",
                                   "18446744073709551615", "NaN", "<script>"])


def protocols() -> None:
    def response(request: str, count: int = 1) -> dict:
        return {"protocol_version": 1, "request_id": request,
                "counters": {"exported_events": count, "sampled_out_events": 0,
                             "unsupported_events": 0}}

    content = hashlib.sha256(binary_envelope(2)).hexdigest()
    request = hashlib.sha256(("jyppx.telemetry.sidecar-batch/v1" + content).encode()).hexdigest()
    receipt = response(request)
    receipt["receipts"] = [1]
    write("backend_protocol", "emit-accepted", encoded(receipt))
    write("backend_protocol", "flush", encoded(response("flush")))
    write("backend_protocol", "close", encoded(response("close")))
    write("backend_protocol", "counter-regression",
          encoded(response("flush", 9)) + b"\n" + encoded(response("flush", 8)))
    write("backend_protocol", "counter-increase",
          encoded(response("flush", 8)) + b"\n" + encoded(response("flush", 9)))
    write("backend_protocol", "backend-error", encoded({
        "protocol_version": 1, "request_id": "flush", "error": "backend failure"}))
    write("backend_protocol", "duplicate-key", b'{"protocol_version":1,"protocol_version":1}')
    dictionary("backend_protocol", ["protocol_version", "request_id", "counters", "receipts",
                                    "exported_events", "sampled_out_events", "unsupported_events",
                                    "flush", "close", "error", "18446744073709551615"])


def main() -> None:
    telemetry()
    checkpoints()
    bundles(safetensors())
    evaluations()
    protocols()


if __name__ == "__main__":
    main()
