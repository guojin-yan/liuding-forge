# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Real ImageFolder/ResNet18 training and independent-process recovery.

The self-authored PNG task freezes 16 training images, 8 validation images,
12 epochs (48 updates), seed 20261028 and accuracy >= 0.875 before execution.
It proves this small workflow, not CIFAR10/ImageNet quality or performance.
Python uses only its standard library; no Python Torch or image package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import time
import unittest
import zlib


def png(label: int, example: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
            ">I", zlib.crc32(kind + payload) & 0xffffffff)

    rows = bytearray()
    for y in range(32):
        rows.append(0)
        for x in range(32):
            color = [20 + ((x * 3 + y * 5 + example * 7) % 23)] * 3
            color[label] = 190 + ((x + y + example) % 50)
            rows.extend(color)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows))) + chunk(b"IEND", b""))


class ImageRecipe(unittest.TestCase):
    cli: Path
    device: str

    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
        result = subprocess.run(
            [str(cls.cli), *(str(x) for x in arguments), "--format", "json"],
            capture_output=True, text=True, encoding="utf-8", timeout=300,
            env=env, check=False,
        )
        if result.returncode != expected:
            raise AssertionError(f"args={arguments!r} exit={result.returncode} expected={expected}"
                                 f"\nstdout={result.stdout}\nstderr={result.stderr}")
        envelope = json.loads(result.stdout)
        if envelope["schema_version"] != 1:
            raise AssertionError(envelope)
        return envelope

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="liuding-image-recipe-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name) / "分类 recipe"
        cls.base.mkdir()
        for split, count in (("train", 8), ("validation", 4)):
            for label, name in enumerate(("green", "red")):
                folder = cls.base / "images" / split / name
                folder.mkdir(parents=True)
                for number in range(count):
                    (folder / f"{number}.png").write_bytes(png(label, number + (20 if split == "validation" else 0)))
        cls.config = {
            "schema_version": 1, "recipe": "jyppx.liuding.imagefolder.resnet18/v1",
            "train_root": "images/train", "validation_root": "images/validation",
            "data_license": "CC0-1.0; self-authored RGB fixture", "device": cls.device,
            "seed": 20261028, "epochs": 12, "batch_size": 4, "image_size": 32,
            "workers": 2, "prefetch_batches": 2, "learning_rate": 0.01,
            "momentum": 0.9, "weight_decay": 0.0001, "lr_gamma": 0.9,
            "checkpoint_every": 16, "minimum_accuracy": 0.875,
            "max_image_bytes": 1048576, "max_data_bytes": 10485760,
            "max_samples": 1024, "image_budget_bytes": 67108864,
        }
        cls.config_file = cls.base / "config.json"
        cls.config_file.write_text(json.dumps(cls.config), encoding="utf-8")
        cls.schema = cls.invoke("recipe", "schema")["data"]
        cls.preflight = cls.invoke("recipe", "check", "--file", cls.config_file)["data"]
        cls.baseline = cls.base / "baseline"
        cls.recovered = cls.base / "recovered"
        cls.continuous = cls.invoke("recipe", "run", "--file", cls.config_file, "--root", cls.baseline)["data"]
        cls.pause = cls.invoke("recipe", "run", "--file", cls.config_file,
                               "--root", cls.recovered, "--pause-after", 19)["data"]
        cls.resumed = cls.invoke("recipe", "resume", "--file", cls.config_file, "--root", cls.recovered)["data"]
        cls.crashed = cls.base / "crashed"
        process = subprocess.Popen(
            [str(cls.cli), "recipe", "run", "--file", str(cls.config_file),
             "--root", str(cls.crashed), "--format", "json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"),
        )
        try:
            deadline = time.monotonic() + 120
            while not (cls.crashed / "latest.jyck").exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise AssertionError("Training exited or timed out before its first durable checkpoint")
                time.sleep(0.02)
            cls.invoke("recipe", "resume", "--file", cls.config_file, "--root", cls.crashed, expected=3)
            process.kill()
            process.communicate(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=30)
        cls.after_crash = cls.invoke("recipe", "resume", "--file", cls.config_file, "--root", cls.crashed)["data"]
        cls.checkpoints = {}
        for name, root in (("baseline", cls.baseline), ("resumed", cls.recovered),
                           ("after-crash", cls.crashed)):
            cls.checkpoints[name] = cls.invoke(
                "checkpoint", "inspect", "--path", root / "final.jyck")["data"]["manifest"]
        if output := os.environ.get("JYPPX_LIUDING_RECIPE_EVIDENCE_DIR"):
            evidence = Path(output) / cls.device
            evidence.mkdir(parents=True, exist_ok=False)
            for name, value in (("preflight", cls.preflight), ("baseline", cls.continuous),
                                ("paused", cls.pause), ("resumed", cls.resumed), ("after-crash", cls.after_crash)):
                (evidence / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            for name in ("lineage.json", "dataset.json", "evaluation.json", "evaluation-suite.json", "diagnostics.json"):
                shutil.copy2(cls.recovered / name, evidence / name)
            shutil.copy2(cls.recovered / "bundle/manifest.json", evidence / "bundle-manifest.json")
            for name, manifest in cls.checkpoints.items():
                (evidence / f"{name}-checkpoint.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def test_01_real_training_quality_and_cross_process_exact_recovery(self) -> None:
        self.assertEqual(self.preflight["total_steps"], 48)
        self.assertEqual(self.pause["status"], "paused")
        self.assertEqual(self.pause["global_step"], 19)
        for result in (self.continuous, self.resumed):
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["global_step"], 48)
            self.assertEqual(result["optimizer_steps"], 48)
            self.assertEqual(result["scheduler_steps"], 12)
            self.assertEqual(result["completed_epochs"], 12)
            self.assertEqual(result["validation_samples"], 8)
            self.assertGreaterEqual(result["accuracy"], 0.875)
            self.assertTrue(result["quality_passed"])
            self.assertEqual(result["numerics"], {
                "deterministic_algorithms": True, "cudnn_benchmark": False,
                "tf32_cudnn": False, "tf32_cublas": False,
                "float32_matmul_precision": "highest",
                "intraop_threads": 1, "interop_threads": 1,
            })
        for name in ("model_sha256", "next_rng_sha256", "next_cpu_rng_sha256",
                     "training_metrics", "validation_metrics", "accuracy", "confusion"):
            self.assertEqual(self.continuous[name], self.resumed[name], name)
        for name in ("model.safetensors",):
            self.assertEqual((self.baseline / name).read_bytes(), (self.recovered / name).read_bytes())
        self.assertEqual(self.resumed["model_sha256"], hashlib.sha256(
            (self.recovered / "model.safetensors").read_bytes()).hexdigest())
        for name in ("model_sha256", "next_rng_sha256", "next_cpu_rng_sha256",
                     "training_metrics", "validation_metrics", "accuracy", "confusion",
                     "global_step", "optimizer_steps", "scheduler_steps", "completed_epochs"):
            self.assertEqual(self.continuous[name], self.after_crash[name], "process kill: " + name)
        snapshots = {}
        for name, manifest in self.checkpoints.items():
            participants = {p["state"]["participant_id"]: p for p in manifest["participants"]}
            snapshots[name] = {identity: participants[identity]
                               for identity in ("recipe.sgd", "recipe.lr")}
        self.assertEqual(snapshots["baseline"], snapshots["resumed"])
        self.assertEqual(snapshots["baseline"], snapshots["after-crash"])

    def test_02_real_readers_run_cas_bundle_and_no_overwrite(self) -> None:
        self.invoke("model", "verify", "--path", self.recovered / "model.safetensors")
        self.invoke("model", "bundle", "verify", "--path", self.recovered / "bundle")
        self.invoke("checkpoint", "verify", "--path", self.recovered / "final.jyck")
        self.invoke("run", "verify", "--root", self.recovered / "run-store")
        replay = self.invoke("recipe", "resume", "--file", self.config_file, "--root", self.recovered)["data"]
        self.assertEqual(replay, self.resumed)
        self.invoke("recipe", "run", "--file", self.config_file, "--root", self.recovered, expected=2)
        database = sqlite3.connect((self.recovered / "run-store/runs.sqlite3").as_uri() + "?mode=ro", uri=True)
        try:
            self.assertEqual(database.execute("select count(*) from jyppx_runs").fetchone()[0], 1)
        finally:
            database.close()
        manifest = json.loads((self.recovered / "bundle/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["signatures"], [])
        for item in manifest["files"]:
            content = (self.recovered / "bundle" / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), item["artifact"]["digest"]["hex"])
        self.assertFalse(self.resumed["bundle"]["transitive_sbom_complete"])

    def test_03_schema_is_shared_with_config_resolver_and_unknown_fields_fail(self) -> None:
        for arguments in (("recipe",), ("recipe", "schema", "extra"), ("recipe", "unknown")):
            self.invoke(*arguments, expected=2)
        schema_file = self.base / "schema.json"
        schema_file.write_text(json.dumps(self.schema), encoding="utf-8")
        self.invoke("config", "lint", "--schema", schema_file, "--file", self.config_file)
        bad = self.base / "bad.json"
        for patch in ({"epochs": "12"}, {"epochs": 0}, {"minimum_accuracy": 0.0},
                      {"batch_size": 1}, {"batch_size": 15}, {"surprise": True},
                      {"device": "auto"}, {"train_root": "images/validation"}):
            bad.write_text(json.dumps(self.config | patch), encoding="utf-8")
            self.invoke("recipe", "check", "--file", bad, expected=2)
        bad.write_text(json.dumps(self.config)[:-1] + ', "seed": 1}', encoding="utf-8")
        self.invoke("recipe", "check", "--file", bad, expected=2)

    def test_04_content_configuration_class_and_build_changes_are_rejected(self) -> None:
        bad = self.base / "changed.json"
        bad.write_text(json.dumps(self.config | {"seed": 7}), encoding="utf-8")
        self.invoke("recipe", "resume", "--file", bad, "--root", self.recovered, expected=2)
        image = self.base / "images/train/green/0.png"
        before = image.read_bytes()
        try:
            image.write_bytes(png(1, 123))
            self.invoke("recipe", "resume", "--file", self.config_file, "--root", self.recovered, expected=2)
        finally:
            image.write_bytes(before)
        folder = self.base / "images/validation/red"
        folder.rename(folder.with_name("renamed"))
        try:
            self.invoke("recipe", "check", "--file", self.config_file, expected=2)
        finally:
            folder.with_name("renamed").rename(folder)
        lineage_path = self.recovered / "lineage.json"
        before = lineage_path.read_bytes()
        lineage = json.loads(before)
        lineage["build"]["libtorch_version"] = "0.0.0"
        try:
            lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
            self.invoke("recipe", "resume", "--file", self.config_file, "--root", self.recovered, expected=2)
        finally:
            lineage_path.write_bytes(before)

    def test_05_diagnostic_and_logical_resource_bounds(self) -> None:
        diagnostic = json.loads((self.recovered / "diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(diagnostic["identity"]["configuration"]["hex"], self.resumed["lineage"]["config_sha256"])
        self.assertEqual(diagnostic["identity"]["dataset"]["hex"], self.preflight["dataset_sha256"])
        for value in self.resumed["image_resource_payload"].values():
            self.assertLessEqual(value["peak"], value["limit"])

    def test_06_failed_quality_is_completed_and_replays_with_exit_four(self) -> None:
        # Identical validation pixels with contradictory labels cannot both be
        # correct for a deterministic classifier. This is a separate negative
        # task; the positive task retains its frozen 48-update/0.875 gate.
        for name in ("green", "red"):
            folder = self.base / "contradictory-validation" / name
            folder.mkdir(parents=True)
            (folder / "same.png").write_bytes(png(0, 0))
        config = self.base / "quality-failure.json"
        config.write_text(json.dumps(self.config | {
            "validation_root": "contradictory-validation", "epochs": 1,
            "minimum_accuracy": 1.0,
        }), encoding="utf-8")
        root = self.base / "quality-failure"
        result = self.invoke("recipe", "run", "--file", config, "--root", root, expected=4)["data"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["global_step"], 4)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertFalse(result["quality_passed"])
        self.assertEqual(self.invoke("recipe", "resume", "--file", config,
                                     "--root", root, expected=4)["data"], result)
        if output := os.environ.get("JYPPX_LIUDING_RECIPE_EVIDENCE_DIR"):
            path = Path(output) / self.device / "quality-failed.json"
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments, rest = parser.parse_known_args()
    ImageRecipe.cli = arguments.cli.resolve()
    ImageRecipe.device = arguments.device
    unittest.main(argv=[__file__, *rest])
