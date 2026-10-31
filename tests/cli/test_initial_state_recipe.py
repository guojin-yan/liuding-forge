# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Independent stdlib acceptance of native complete zero-update initialization.

Fixtures only mutate existing C++-exported safetensors for negative tests.
No product tensor initialization, export, import or training occurs in Python.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location(
    "_liuding_initial_fixture", Path(__file__).with_name("test_update_recipe.py"))
assert _spec is not None and _spec.loader is not None
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)
EVIDENCE: Path | None = None
COMMAND = 0


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    assert 0 < length <= 16 * 1024 * 1024 and length <= len(data) - 8
    return json.loads(data[8:8 + length]), data[8 + length:]


def mutated(path: Path, header: dict, payload: bytes) -> None:
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class InitialState(unittest.TestCase):
    cli: Path
    device: str

    @classmethod
    def invoke(cls, *args: object, expected: int = 0) -> dict:
        global COMMAND
        COMMAND += 1
        result = subprocess.run([str(cls.cli), *(str(arg) for arg in args), "--format", "json"],
            capture_output=True, text=True, encoding="utf-8", timeout=180,
            env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"), check=False)
        if EVIDENCE:
            (EVIDENCE / f"command-{COMMAND:03d}.json").write_text(json.dumps({
                "arguments": [str(arg) for arg in args], "exit": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}, ensure_ascii=False, indent=2)
                + "\n", encoding="utf-8")
        assert result.returncode == expected, (args, result.returncode, result.stdout, result.stderr)
        return json.loads(result.stdout)

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="liuding-complete-initial-state-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name) / "完整初始化"
        cls.base.mkdir()
        cls.config = fixture.configuration(cls.base, cls.device)
        cls.file = cls.base / "source.json"
        cls.file.write_text(json.dumps(cls.config), encoding="utf-8")
        cls.root = cls.base / "zero-step"
        cls.report = cls.invoke("recipe", "initialize", "--file", cls.file, "--root", cls.root)["data"]
        cls.state = cls.root / "initial-state.safetensors"
        cls.header, cls.payload = decode(cls.state)
        cls.import_config = cls.config | {"initial_state_file": str(cls.state),
                                         "initial_state_sha256": sha(cls.state)}

    def candidate(self, name: str, config: dict) -> Path:
        path = self.base / (name + ".json")
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_01_complete_inventory_and_zero_updates(self) -> None:
        report = self.report
        self.assertEqual(report["status"], "initialized")
        self.assertEqual(report["global_step"], 0)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertFalse((self.root / "run-store").exists())
        self.assertFalse(any(self.root.glob("*.jyck")))
        self.assertEqual({path.name for path in self.root.iterdir()},
                         {".recipe.lock", "initial-state.safetensors", "initialization.json"})
        self.assertEqual(report["artifact"]["sha256"], sha(self.state))
        self.assertEqual(report["artifact"]["size_bytes"], self.state.stat().st_size)
        entries = {key: value for key, value in self.header.items() if key != "__metadata__"}
        self.assertEqual(len(entries), 122)  # 62 ResNet18 parameters + 60 BN buffers.
        self.assertIn("fc.weight", entries)
        self.assertIn("bn1.running_mean", entries)
        self.assertIn("bn1.num_batches_tracked", entries)
        self.assertFalse(any(key.startswith(("parameter.", "buffer.")) for key in entries))
        self.assertEqual(self.header["__metadata__"], {
            "format": "jyppx.liuding.recipe.initial-state/v1",
            "recipe": self.config["recipe"], "architecture": "resnet18",
            "classes": '["green","red"]'})
        self.assertEqual(report["initialization"]["model"]["tensors"], len(entries))
        self.assertEqual(report["initialization"]["model"]["tensor_bytes"], len(self.payload))

    def test_02_import_reexport_preserves_every_byte_under_changed_seed(self) -> None:
        config = self.import_config | {"seed": self.config["seed"] + 123}
        path = self.candidate("changed-seed", config)
        checked = self.invoke("recipe", "check", "--file", path)["data"]
        self.assertEqual(checked["initial_state"]["sha256"], sha(self.state))
        root = self.base / "reexport"
        report = self.invoke("recipe", "initialize", "--file", path, "--root", root)["data"]
        self.assertEqual(sha(root / "initial-state.safetensors"), sha(self.state))
        self.assertEqual(report["initialization"]["model"], self.report["initialization"]["model"])
        self.assertEqual(report["initialization"]["applied_tensors"], 122)
        before = sha(root / "initialization.json")
        self.invoke("recipe", "initialize", "--file", path, "--root", root, expected=2)
        self.assertEqual(sha(root / "initialization.json"), before)

    def test_03_paired_fields_digest_and_metadata_rejected_before_output(self) -> None:
        for transfer in (False, True):
            args = ("--name", "transfer") if transfer else ()
            fields = self.invoke("recipe", "schema", *args)["data"]["root"]["fields"]
            for field in ("initial_state_file", "initial_state_sha256"):
                self.assertEqual(fields[field], {"kind": "string", "required": False})
        invalid = [self.config | {"initial_state_file": str(self.state)},
                   self.config | {"initial_state_sha256": sha(self.state)}]
        invalid += [self.import_config | patch for patch in (
            {"initial_state_file": ""}, {"initial_state_file": "absent.safetensors"},
            {"initial_state_sha256": "0" * 64}, {"initial_state_sha256": "A" * 64},
            {"initial_state_sha256": "abc"}, {"initial_state_sha256": None})]
        for index, config in enumerate(invalid):
            with self.subTest(case=index):
                path = self.candidate(f"invalid-config-{index}", config)
                root = self.base / f"invalid-output-{index}"
                for action in ("initialize", "run"):
                    self.invoke("recipe", action, "--file", path, "--root", root, expected=2)
                    self.assertFalse(root.exists())
        for name, value in (("recipe", "wrong"), ("architecture", "resnet50"),
                            ("classes", '["red","green"]'), ("format", "unknown")):
            with self.subTest(metadata=name):
                header = copy.deepcopy(self.header)
                header["__metadata__"][name] = value
                state = self.base / f"bad-{name}.safetensors"
                mutated(state, header, self.payload)
                path = self.candidate(f"bad-{name}", self.import_config | {
                    "initial_state_file": str(state), "initial_state_sha256": sha(state)})
                root = self.base / f"rejected-{name}"
                self.invoke("recipe", "initialize", "--file", path, "--root", root, expected=2)
                self.assertFalse(root.exists())
                state.unlink()

    def test_04_strict_tensor_mapping_and_malformed_bytes_rejected(self) -> None:
        # Missing/extra fixtures retain a valid packed tensor directory, so
        # rejection requires strict target mapping rather than invalid offsets.
        for name in ("missing", "unexpected", "shape", "dtype", "truncated"):
            header = copy.deepcopy(self.header)
            if name in ("missing", "unexpected"):
                if name == "missing":
                    header.pop("fc.bias")
                else:
                    header["not_a_model_parameter"] = copy.deepcopy(header["fc.bias"])
            elif name == "shape":
                header["fc.weight"]["shape"] = list(reversed(header["fc.weight"]["shape"]))
            elif name == "dtype":
                header["fc.weight"]["dtype"] = "I32"
            state = self.base / f"bad-tensor-{name}.safetensors"
            payload = bytearray()
            for key, tensor in header.items():
                if key == "__metadata__":
                    continue
                start, end = tensor["data_offsets"]
                tensor["data_offsets"] = [len(payload), len(payload) + end - start]
                payload.extend(self.payload[start:end])
            mutated(state, header, payload)
            if name == "truncated":
                with state.open("r+b") as stream:
                    stream.truncate(state.stat().st_size - 1)
            path = self.candidate(f"bad-tensor-{name}", self.import_config | {
                "initial_state_file": str(state), "initial_state_sha256": sha(state)})
            root = self.base / f"rejected-tensor-{name}"
            for action in ("initialize", "run"):
                with self.subTest(tensor=name, action=action):
                    self.invoke("recipe", action, "--file", path, "--root", root, expected=2)
                    self.assertFalse(root.exists())
            state.unlink()

    def test_05_resume_binds_initial_file_and_complete_state(self) -> None:
        state = self.base / "resume-input.safetensors"
        state.write_bytes(self.state.read_bytes())
        config = self.import_config | {"initial_state_file": str(state)}
        path = self.candidate("resume-input", config)
        root = self.base / "paused"
        self.invoke("recipe", "run", "--file", path, "--root", root, "--pause-after", 1)
        before = fixture.file_sha256(root / "latest.jyck")
        with state.open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        self.invoke("recipe", "resume", "--file", path, "--root", root, expected=2)
        self.assertEqual(fixture.file_sha256(root / "latest.jyck"), before)
        path = self.candidate("changed-initial-digest", config | {"initial_state_sha256": sha(state)})
        self.invoke("recipe", "resume", "--file", path, "--root", root, expected=2)
        self.assertEqual(fixture.file_sha256(root / "latest.jyck"), before)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args, rest = parser.parse_known_args()
    InitialState.cli, InitialState.device = args.cli.resolve(), args.device
    if output := os.environ.get("JYPPX_LIUDING_INITIAL_STATE_EVIDENCE_DIR"):
        EVIDENCE = Path(output).resolve() / args.device
        EVIDENCE.mkdir(parents=True, exist_ok=False)
    unittest.main(argv=[__file__, *rest])
