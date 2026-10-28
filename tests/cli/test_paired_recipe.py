# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

"""Frozen two-input/two-output task and complete independent-process recovery.

The positive task always uses the prepared 1024/256 samples, seed 20261028,
40 epochs, 1280 updates, accuracy >= .95 and MSE <= .005. Python uses stdlib
only and invokes the production C++ readers; it does not train a second model.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
PARTICIPANTS = {
    "callbacks.dispatcher", "model.main", "recipe.lr", "recipe.sgd",
    "recipe.trainer", "recipe.training", "recipe.validation", "rng.global",
    "strategy.single", "training.diagnostics",
}
RESULT_FIELDS = (
    "global_step", "completed_epochs", "optimizer_steps", "scheduler_steps",
    "model_sha256", "training_metrics", "validation_metrics", "accuracy",
    "mse", "confusion", "validation_samples", "next_rng_sha256",
    "next_cpu_rng_sha256", "numerics",
)


@contextlib.contextmanager
def exclusive_lock(path: Path):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.HANDLE)
        api.CreateFileW.restype = wintypes.HANDLE
        api.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = api.CreateFileW(str(path), 0xc0000000, 0, None, 4, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            api.CloseHandle(handle)
    else:
        import fcntl
        with path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class PairedRecipe(unittest.TestCase):
    cli: Path
    device: str

    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        command = [str(cls.cli), *(str(value) for value in arguments), "--format", "json"]
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", timeout=180,
                                env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"),
                                check=False)
        if result.returncode != expected:
            raise AssertionError(f"args={arguments!r} exit={result.returncode} expected={expected}"
                                 f"\nstdout={result.stdout}\nstderr={result.stderr}")
        envelope = json.loads(result.stdout)
        if envelope["schema_version"] != 1:
            raise AssertionError(envelope)
        return envelope

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="liuding-paired-recipe-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name) / "双输入 recipe"
        cls.base.mkdir()
        spec = importlib.util.spec_from_file_location(
            "paired_data_preparation", ROOT / "scripts/datasets/prepare_paired_affine.py")
        preparation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preparation)
        contract = json.loads((ROOT / "recipes/multi-io/paired-affine.dataset.json").read_text(encoding="utf-8"))
        cls.data = preparation.prepare(cls.base / "datasets", contract)
        cls.config = json.loads((ROOT / "recipes/multi-io/paired-affine.json").read_text(encoding="utf-8"))
        cls.config.update(train_file=str(cls.data / "train.jsonl"),
                          validation_file=str(cls.data / "validation.jsonl"), device=cls.device)
        cls.config_file = cls.base / "config.json"
        cls.config_file.write_text(json.dumps(cls.config), encoding="utf-8")
        cls.schema = cls.invoke("recipe", "schema", "--name", "paired-affine")["data"]
        cls.preflight = cls.invoke("recipe", "check", "--file", cls.config_file)["data"]
        cls.roots = {name: cls.base / name for name in ("continuous", "paused", "crashed")}
        cls.results = {}
        cls.results["continuous"] = cls.invoke("recipe", "run", "--file", cls.config_file,
                                              "--root", cls.roots["continuous"])["data"]
        cls.pause = cls.invoke("recipe", "run", "--file", cls.config_file,
                               "--root", cls.roots["paused"], "--pause-after", 259)["data"]
        cls.results["paused"] = cls.invoke("recipe", "resume", "--file", cls.config_file,
                                          "--root", cls.roots["paused"])["data"]
        process = subprocess.Popen(
            [str(cls.cli), "recipe", "run", "--file", str(cls.config_file),
             "--root", str(cls.roots["crashed"]), "--format", "json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"))
        try:
            deadline = time.monotonic() + 120
            while not (cls.roots["crashed"] / "latest.jyck").exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    output, error = process.communicate(timeout=10)
                    raise AssertionError(f"No first durable checkpoint: {output!r} {error!r}")
                time.sleep(.002)
            if process.poll() is not None:
                raise AssertionError("Training ended before forced process termination")
            process.kill()
            process.communicate(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=30)
        cls.killed_checkpoint = cls.invoke("checkpoint", "verify", "--path",
                                          cls.roots["crashed"] / "latest.jyck")["data"]
        cls.results["crashed"] = cls.invoke("recipe", "resume", "--file", cls.config_file,
                                           "--root", cls.roots["crashed"])["data"]
        cls.manifests = {name: cls.invoke("checkpoint", "inspect", "--path", root / "final.jyck")["data"]["manifest"]
                         for name, root in cls.roots.items()}
        if output := os.environ.get("JYPPX_LIUDING_RECIPE_EVIDENCE_DIR"):
            cls.evidence = Path(output) / cls.device
            cls.evidence.mkdir(parents=True, exist_ok=False)
            for name, value in (("config", cls.config), ("preflight", cls.preflight),
                                ("paused-boundary", cls.pause), ("killed-checkpoint", cls.killed_checkpoint),
                                *cls.results.items()):
                (cls.evidence / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            for name, value in cls.manifests.items():
                (cls.evidence / f"{name}-checkpoint.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            for name in ("lineage.json", "dataset.json", "evaluation.json", "evaluation-suite.json", "diagnostics.json"):
                shutil.copy2(cls.roots["paused"] / name, cls.evidence / name)
            shutil.copy2(cls.roots["paused"] / "bundle/manifest.json", cls.evidence / "bundle-manifest.json")
            for name, root in cls.roots.items():
                shutil.copytree(root, cls.evidence / ("run-" + name))

    def test_01_frozen_full_task_quality_and_complete_recovery(self) -> None:
        self.assertEqual(self.preflight["total_steps"], 1280)
        self.assertEqual(self.preflight["dataset"]["train"]["samples"], 1024)
        self.assertEqual(self.preflight["dataset"]["validation"]["samples"], 256)
        self.assertEqual(self.pause["status"], "paused")
        self.assertEqual(self.pause["global_step"], 259)
        baseline = self.results["continuous"]
        for name, result in self.results.items():
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["global_step"], 1280)
            self.assertEqual(result["optimizer_steps"], 1280)
            self.assertEqual(result["scheduler_steps"], 40)
            self.assertEqual(result["completed_epochs"], 40)
            self.assertEqual(result["validation_samples"], 256)
            self.assertGreaterEqual(result["accuracy"], .95)
            self.assertLessEqual(result["mse"], .005)
            self.assertTrue(result["quality_passed"])
            for field in RESULT_FIELDS:
                self.assertEqual(result[field], baseline[field], (name, field))
            self.assertEqual((self.roots[name] / "model.safetensors").read_bytes(),
                             (self.roots["continuous"] / "model.safetensors").read_bytes())
        participants = {}
        for name, manifest in self.manifests.items():
            participants[name] = {entry["state"]["participant_id"]: entry for entry in manifest["participants"]}
            self.assertEqual(set(participants[name]), PARTICIPANTS)
        self.assertEqual(participants["continuous"], participants["paused"])
        self.assertEqual(participants["continuous"], participants["crashed"])

    def test_02_readers_bundle_schemas_and_idempotent_completion(self) -> None:
        root = self.roots["paused"]
        for command, path in ((('model', 'verify'), root / 'model.safetensors'),
                              (('model', 'bundle', 'verify'), root / 'bundle'),
                              (('checkpoint', 'verify'), root / 'final.jyck')):
            self.invoke(*command, "--path", path)
        self.invoke("run", "verify", "--root", root / "run-store")
        self.assertEqual(self.invoke("recipe", "resume", "--file", self.config_file, "--root", root)["data"],
                         self.results["paused"])
        self.invoke("recipe", "run", "--file", self.config_file, "--root", root, expected=2)
        manifest = json.loads((root / "bundle/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["signatures"], [])
        for entry in manifest["files"]:
            self.assertEqual(hashlib.sha256((root / "bundle" / entry["path"]).read_bytes()).hexdigest(),
                             entry["artifact"]["digest"]["hex"])
        input_schema = json.loads((root / "bundle/metadata/input.json").read_text(encoding="utf-8"))
        output_schema = json.loads((root / "bundle/metadata/output.json").read_text(encoding="utf-8"))
        self.assertEqual([entry["name"] for entry in input_schema["positional"]], ["left", "right"])
        self.assertEqual([entry["shape"] for entry in output_schema["tuple"]], [[None, 2], [None, 1]])
        self.assertFalse(self.results["paused"]["bundle"]["transitive_sbom_complete"])

    def test_03_closed_schema_and_preallocation_limits(self) -> None:
        schema = self.base / "schema.json"
        schema.write_text(json.dumps(self.schema), encoding="utf-8")
        self.invoke("config", "lint", "--schema", schema, "--file", self.config_file)
        bad = self.base / "bad.json"
        for patch in ({"surprise": True}, {"epochs": "40"}, {"workers": 9}, {"device": "auto"},
                      {"minimum_accuracy": 0.0}, {"maximum_mse": 0.0}, {"batch_size": 0},
                      {"train_file": self.config["validation_file"]}, {"max_samples": 10},
                      {"max_record_bytes": 128}, {"max_data_bytes": 1024}, {"max_batch_bytes": 895}):
            with self.subTest(patch=patch):
                bad.write_text(json.dumps(self.config | patch), encoding="utf-8")
                self.invoke("recipe", "check", "--file", bad, expected=2)
        bad.write_text(json.dumps(self.config)[:-1] + ', "seed": 7}', encoding="utf-8")
        self.invoke("recipe", "check", "--file", bad, expected=2)
        for arguments in (("recipe",), ("recipe", "unknown"), ("recipe", "schema", "--name", "unknown"),
                          ("recipe", "check", "--name", "paired-affine", "--file", self.config_file)):
            self.invoke(*arguments, expected=2)

    def test_04_bounded_data_parser_shape_type_label_and_target_range(self) -> None:
        path = self.data / "train.jsonl"
        before = path.read_bytes()
        row = json.loads(before.splitlines()[0])
        try:
            for patch in ({"left": [0]}, {"right": [True, 0]}, {"class_id": 2},
                          {"left": [1.25, 0]}, {"regression": [1.5]}, {"regression": [0.1]},
                          {"id": "train:000007"}, {"unexpected": 1}):
                with self.subTest(patch=patch):
                    changed = json.dumps(row | patch).encode() + b"\n" + before.split(b"\n", 1)[1]
                    path.write_bytes(changed)
                    self.invoke("recipe", "check", "--file", self.config_file, expected=2)
            path.write_bytes(before[:-1])
            self.invoke("recipe", "check", "--file", self.config_file, expected=2)
            for endpoint in (-19/16, 21/16):
                changed = json.dumps(row | {"regression": [endpoint]}).encode() + b"\n" + before.split(b"\n", 1)[1]
                path.write_bytes(changed)
                self.invoke("recipe", "check", "--file", self.config_file)
        finally:
            path.write_bytes(before)

    def test_05_changed_content_configuration_and_build_are_rejected(self) -> None:
        changed = self.base / "changed.json"
        changed.write_text(json.dumps(self.config | {"seed": 7}), encoding="utf-8")
        self.invoke("recipe", "resume", "--file", changed, "--root", self.roots["paused"], expected=2)
        path = self.data / "train.jsonl"
        before = path.read_bytes()
        try:
            row = json.loads(before.splitlines()[0])
            row["class_id"] = 1 - row["class_id"]
            path.write_bytes(json.dumps(row).encode() + b"\n" + before.split(b"\n", 1)[1])
            self.invoke("recipe", "resume", "--file", self.config_file, "--root", self.roots["paused"], expected=2)
        finally:
            path.write_bytes(before)
        lineage = self.roots["paused"] / "lineage.json"
        before = lineage.read_bytes()
        try:
            document = json.loads(before)
            document["build"]["libtorch_version"] = "0.0.0"
            lineage.write_text(json.dumps(document), encoding="utf-8")
            self.invoke("recipe", "resume", "--file", self.config_file, "--root", self.roots["paused"], expected=2)
        finally:
            lineage.write_bytes(before)

    def test_06_directory_lock_and_diagnostic_identity(self) -> None:
        busy = self.base / "busy"
        busy.mkdir()
        with exclusive_lock(busy / ".recipe.lock"):
            self.invoke("recipe", "resume", "--file", self.config_file, "--root", busy, expected=3)
        diagnostic = json.loads((self.roots["paused"] / "diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(diagnostic["identity"]["configuration"]["hex"], self.results["paused"]["lineage"]["config_sha256"])
        self.assertEqual(diagnostic["identity"]["dataset"]["hex"], self.preflight["dataset_sha256"])

    def test_07_failed_quality_completes_and_replays_with_exit_four(self) -> None:
        data = self.base / "contradictory.jsonl"
        rows = [{"id": f"validation:{index:06}", "left": [0, 0], "right": [0, 0],
                 "class_id": index, "regression": [0]} for index in range(2)]
        data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        config = self.base / "quality-failure.json"
        config.write_text(json.dumps(self.config | {"validation_file": str(data), "epochs": 1,
                                                   "minimum_accuracy": 1.0}), encoding="utf-8")
        root = self.base / "quality-failure"
        result = self.invoke("recipe", "run", "--file", config, "--root", root, expected=4)["data"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["global_step"], 32)
        self.assertEqual(result["accuracy"], .5)
        self.assertFalse(result["quality_passed"])
        self.assertEqual(self.invoke("recipe", "resume", "--file", config, "--root", root, expected=4)["data"], result)
        if hasattr(self, "evidence"):
            (self.evidence / "quality-failed.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments, remaining = parser.parse_known_args()
    PairedRecipe.cli = arguments.cli.resolve()
    PairedRecipe.device = arguments.device
    unittest.main(argv=[__file__, *remaining])
