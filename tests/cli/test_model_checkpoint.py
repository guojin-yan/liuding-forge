# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Process contracts for the shipped CLI; requires an already built executable.

The fixture encoders write inert, documented binary formats with stdlib only.
Every assertion exercises the real C++ CLI and its production artifact readers.
Run with --device cuda in the Ubuntu/CUDA validation lane.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


def safetensors(header: dict | str, payload: bytes) -> bytes:
    encoded = (header if isinstance(header, str) else json.dumps(header)).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def manifest(build: dict, payload: bytes) -> dict:
    return {
        "schema_version": 1,
        "build": copy.deepcopy(build),
        "participants": [
            {
                "state": {
                    "participant_id": "test.state",
                    "schema_version": 7,
                    "restore_level": "bounded",
                    "degradation_reasons": ["fixed participant tolerance"],
                    "fields": [
                        {
                            "name": "maybe",
                            "media_type": "application/octet-stream",
                            "max_bytes": 32,
                            "required": False,
                        },
                        {
                            "name": "payload",
                            "media_type": "application/octet-stream",
                            "max_bytes": 32,
                            "required": True,
                        },
                    ],
                },
                "blobs": [
                    {
                        "field_name": "payload",
                        "relative_path": "blobs/test.state/payload.bin",
                        "size_bytes": len(payload),
                        "digest": {
                            "algorithm": "sha256",
                            "hex": hashlib.sha256(payload).hexdigest(),
                        },
                    }
                ],
            }
        ],
    }


def checkpoint(document: dict, payload: bytes) -> bytes:
    """Encode the one-participant opaque fixture, without object serialization."""
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    body = b"JYPPXCK1" + struct.pack("<IQ", 1, len(encoded)) + encoded + payload
    return body + hashlib.sha256(body).digest()


class ModelCheckpointContract(unittest.TestCase):
    cli: Path
    device: str

    @classmethod
    def setUpClass(cls) -> None:
        process = subprocess.run(
            [str(cls.cli), "--version", "--format", "json"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"CLI --version failed: {process.stderr} {process.stdout}")
        version = json.loads(process.stdout)["data"]
        if version["display_name"] != "六丁炼算" or version["display_name_en"] != "Liuding Forge":
            raise RuntimeError(f"CLI brand metadata failed: {version}")
        cls.build = version["build"]

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="jyppx-cli-contract-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.payload = struct.pack("<2f", 1.0, -2.5)
        self.model = self.root / "weights space.safetensors"
        self.model_bytes = safetensors(
            {
                "__metadata__": {"purpose": "cli contract"},
                "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            },
            self.payload,
        )
        self.model.write_bytes(self.model_bytes)
        self.state = self.root / "state space.jyck"
        self.state_payload = b"opaque-state-001"
        self.document = manifest(self.build, self.state_payload)
        self.state_bytes = checkpoint(self.document, self.state_payload)
        self.state.write_bytes(self.state_bytes)

    def invoke(self, *arguments: object, expected: int = 0, format_: str = "json") -> dict:
        process = subprocess.run(
            [str(self.cli), *(str(value) for value in arguments), "--format", format_],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
            check=False,
        )
        self.assertEqual(
            process.returncode, expected,
            f"argv={arguments!r}\nstdout={process.stdout}\nstderr={process.stderr}",
        )
        result = json.loads(process.stdout)
        self.assertEqual(result["schema_version"], 1)
        return result

    def test_model_metadata_and_actual_content_hashes(self) -> None:
        for action in ("inspect", "verify"):
            with self.subTest(action=action):
                result = self.invoke("model", action, "--path", self.model)
                self.assertEqual(result["status"], "ok")
                data = result["data"]
                self.assertTrue(data["valid"])
                self.assertEqual(data["metadata"], {"purpose": "cli contract"})
                self.assertEqual(data["sha256"], hashlib.sha256(self.model_bytes).hexdigest())
                tensor = data["tensors"][0]
                self.assertEqual(tensor["dtype"], "F32")
                self.assertEqual(tensor["shape"], [2])
                self.assertEqual(tensor["data_offsets"], [0, 8])
                self.assertEqual(tensor["sha256"], hashlib.sha256(self.payload).hexdigest())
                self.assertFalse(data["verification"]["embedded_checksum"])
        formatted = subprocess.run([str(self.cli), "model", "inspect", "--path", str(self.model),
                                    "--format", "text"], capture_output=True, text=True,
                                   encoding="utf-8", timeout=30)
        self.assertEqual(formatted.returncode, 0, formatted.stdout + formatted.stderr)
        self.assertIn("tensor_count: 1", formatted.stdout)
        self.assertFalse(formatted.stdout.lstrip().startswith("{"))

    def test_model_hash_is_identity_without_embedded_integrity_claim(self) -> None:
        changed = self.model_bytes[:-1] + bytes([self.model_bytes[-1] ^ 1])
        self.model.write_bytes(changed)
        data = self.invoke("model", "verify", "--path", self.model)["data"]
        self.assertTrue(data["valid"])
        self.assertEqual(data["sha256"], hashlib.sha256(changed).hexdigest())

    def test_model_rejects_malformed_bounds_duplicates_and_unknown_dtype(self) -> None:
        fixtures = [
            (self.model_bytes[:-1], "serialization"),
            (self.model_bytes + b"tail", "serialization"),
            (safetensors({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, b"1234"), "serialization"),
            (safetensors({"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}, "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]}}, b"123"), "serialization"),
            (safetensors('{"w":{"dtype":"U8","dtype":"I8","shape":[1],"data_offsets":[0,1]}}', b"x"), "serialization"),
            (safetensors({"w": {"dtype": "F8_UNKNOWN", "shape": [1], "data_offsets": [0, 1]}}, b"x"), "compatibility"),
        ]
        for encoded, name in fixtures:
            with self.subTest(name=name, size=len(encoded)):
                self.model.write_bytes(encoded)
                result = self.invoke("model", "verify", "--path", self.model, expected=2)
                self.assertEqual(result["error"]["name"], name)

    def test_checkpoint_full_verification_and_opaque_participant(self) -> None:
        for action in ("inspect", "verify"):
            data = self.invoke("checkpoint", action, "--path", self.state)["data"]
            self.assertEqual(data["manifest"], self.document)
            self.assertEqual(data["blob_bytes"], len(self.state_payload))
            self.assertEqual(data["blob_count"], 1)
            self.assertEqual(data["verification"]["artifact_digest"], "sha256_verified")
            self.assertFalse(data["verification"]["participant_deserialization"])

    def test_checkpoint_inspection_preserves_stale_temporary_and_input(self) -> None:
        temporary = Path(str(self.state) + ".tmp")
        lock = Path(str(self.state) + ".lock")
        temporary.write_bytes(b"leave this unpublished state intact")
        self.invoke("checkpoint", "inspect", "--path", self.state)
        self.assertEqual(temporary.read_bytes(), b"leave this unpublished state intact")
        self.assertFalse(lock.exists())
        self.assertEqual(self.state.read_bytes(), self.state_bytes)

    def test_checkpoint_rejects_artifact_and_individual_blob_corruption(self) -> None:
        changed = bytearray(self.state_bytes)
        changed[-33] ^= 1
        for repair_outer in (False, True):
            with self.subTest(repair_outer=repair_outer):
                if repair_outer:
                    changed[-32:] = hashlib.sha256(changed[:-32]).digest()
                self.state.write_bytes(changed)
                result = self.invoke("checkpoint", "verify", "--path", self.state, expected=2)
                self.assertEqual(result["error"]["name"], "serialization")

    def test_checkpoint_diff_compares_build_state_field_and_blob_hash(self) -> None:
        same = self.invoke("checkpoint", "diff", "--baseline", self.state, "--candidate", self.state)
        self.assertTrue(same["data"]["equal"])
        self.assertEqual(same["data"]["changes"], [])
        candidate = self.root / "candidate.jyck"
        new_payload = b"opaque-state-002"
        document = manifest(self.build, new_payload)
        document["build"]["git_commit"] = "contract-candidate"
        state = document["participants"][0]["state"]
        state["schema_version"] = 8
        state["fields"][1]["max_bytes"] = 31
        state["degradation_reasons"].append("additional bounded tolerance")
        candidate.write_bytes(checkpoint(document, new_payload))
        data = self.invoke("checkpoint", "diff", "--baseline", self.state, "--candidate", candidate, expected=1)["data"]
        self.assertFalse(data["equal"])
        paths = {item["path"] for item in data["changes"]}
        self.assertIn("/build/git_commit", paths)
        self.assertIn("/participants/test.state/state/schema_version", paths)
        self.assertIn("/participants/test.state/state/fields/payload/max_bytes", paths)
        self.assertIn("/participants/test.state/state/degradation_reasons", paths)
        self.assertIn("/participants/test.state/blobs/payload/digest/hex", paths)

    def test_checkpoint_diff_aligns_fields_by_name(self) -> None:
        document = copy.deepcopy(self.document)
        document["participants"][0]["state"]["fields"].reverse()
        candidate = self.root / "reordered.jyck"
        candidate.write_bytes(checkpoint(document, self.state_payload))
        data = self.invoke("checkpoint", "diff", "--baseline", self.state, "--candidate", candidate)["data"]
        self.assertTrue(data["equal"])

    def test_arguments_rejected_before_filesystem_or_device_work(self) -> None:
        missing = self.root / "absent-parent" / "input.bin"
        commands = [
            ("model", "inspect", "--path", missing, "--unexpected", "x"),
            ("model", "inspect", "--path", missing, "--path", self.model),
            ("checkpoint", "verify", "--path", missing, "--unexpected", "x"),
            ("checkpoint", "diff", "--baseline", missing, "--candidate", missing, "--candidate", self.state),
            ("benchmark", "tensor", "--device", "cuda", "--size", 0),
            ("benchmark", "tensor", "--device", "cuda", "--samples", 0),
            ("benchmark", "tensor", "--size", 16777216, "--samples", 1000),
            ("benchmark", "tensor", "--warmup", 101),
            ("benchmark", "tensor", "--size", "-1"),
            ("benchmark", "tensor", "--device", "cuda:1"),
        ]
        for command in commands:
            with self.subTest(command=command):
                result = self.invoke(*command, expected=2)
                self.assertEqual(result["error"]["name"], "invalid_argument")
        self.assertFalse(missing.parent.exists())

    def test_artifact_input_requires_existing_regular_file(self) -> None:
        for command in ("model", "checkpoint"):
            for path in (self.root, self.root / "missing.bin"):
                with self.subTest(command=command, path=path):
                    result = self.invoke(command, "inspect", "--path", path, expected=3)
                    self.assertEqual(result["error"]["name"], "io")

    def test_benchmark_statistics_provenance_and_device(self) -> None:
        data = self.invoke("benchmark", "tensor", "--device", self.device,
                           "--size", 64, "--samples", 4, "--warmup", 1)["data"]
        self.assertTrue(data["output_verified"])
        self.assertEqual(data["shape"], [64])
        self.assertEqual(data["samples"], 4)
        self.assertEqual(data["warmup"], 1)
        self.assertEqual(data["device"], "cuda:0" if self.device == "cuda" else "cpu")
        self.assertEqual(data["timing_unit"], "milliseconds")
        self.assertEqual(data["percentile_method"], "nearest_rank")
        self.assertLessEqual(data["min"], data["median"])
        self.assertLessEqual(data["median"], data["p95"])
        self.assertLessEqual(data["p95"], data["max"])
        self.assertTrue(all(math.isfinite(data[key]) for key in ("min", "median", "p95", "max")))
        self.assertEqual(data["build"], self.build)
        if self.device == "cuda":
            self.assertEqual(data["sample_completion"], "cuda_device_synchronize_each_sample")
        single = self.invoke("benchmark", "tensor", "--device", self.device,
                             "--size", 1, "--samples", 1, "--warmup", 0)["data"]
        self.assertEqual(single["median"], single["p95"])
        self.assertEqual(single["min"], single["max"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    options = parser.parse_args()
    ModelCheckpointContract.cli = options.cli.resolve(strict=True)
    ModelCheckpointContract.device = options.device
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ModelCheckpointContract)
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
