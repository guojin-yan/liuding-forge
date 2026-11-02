# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Cancel a real native image trainer after its first durable checkpoint.

Python supplies self-authored images and checks native lifecycle/CAS facts.
This interruption fixture has a 96-update ceiling, not a quality experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

from test_image_recipe import PARTICIPANTS, png
from test_local_attempt import alive


class LocalRecipeLifecycle(unittest.TestCase):
    cli: Path
    device: str

    def command(self, *arguments: object) -> list[str]:
        return [str(self.cli), *map(str, arguments), "--format", "json"]

    def invoke(self, *arguments: object, expected: int = 0) -> dict:
        result = subprocess.run(self.command(*arguments), capture_output=True,
                                text=True, encoding="utf-8", timeout=60,
                                env=self.environment, check=False)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        envelope = json.loads(result.stdout)
        return envelope.get("data", envelope)

    def references(self, run: str) -> dict:
        return {row["reference_id"]: row for row in self.invoke(
            "run", "show", "--root", self.store, "--id", run)["references"]}

    def blob(self, reference: dict) -> bytes:
        row = reference["artifact"]
        identity = row["sha256"]
        data = (self.store / "artifacts/objects" / identity[:2] / identity[2:]).read_bytes()
        self.assertEqual(len(data), row["size_bytes"])
        self.assertEqual(hashlib.sha256(data).hexdigest(), identity)
        return data

    def test_real_training_cancel_releases_admission_without_retry_or_success(self) -> None:
        self.environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                CUBLAS_WORKSPACE_CONFIG=":4096:8")
        with tempfile.TemporaryDirectory(prefix="liuding-recipe-cancel-") as temporary:
            base = Path(temporary) / "真实取消"
            base.mkdir()
            self.store = base / "store"
            work, spec_path = base / "training", base / "local.json"
            for split, count in (("train", 8), ("validation", 4)):
                for label, name in enumerate(("green", "red")):
                    folder = base / "images" / split / name
                    folder.mkdir(parents=True)
                    for number in range(count):
                        (folder / f"{number}.png").write_bytes(png(label, number))
            config = {
                "schema_version": 1, "recipe": "jyppx.liuding.imagefolder.resnet18/v1",
                "train_root": "images/train", "validation_root": "images/validation",
                "data_license": "CC0-1.0; self-authored RGB fixture", "device": self.device,
                "seed": 20261028, "epochs": 24, "batch_size": 4, "image_size": 32,
                "workers": 2, "prefetch_batches": 2, "learning_rate": 0.01,
                "momentum": 0.9, "weight_decay": 0.0001, "lr_gamma": 0.9,
                "checkpoint_every": 16, "minimum_accuracy": 0.875,
                "max_image_bytes": 1048576, "max_data_bytes": 10485760,
                "max_samples": 1024, "image_budget_bytes": 67108864,
            }
            config_path = base / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            plan = self.invoke("recipe", "local-spec", "--file", config_path,
                               "--root", work, "--store", self.store, "--spec", spec_path,
                               "--executable", self.cli, "--pause-after", 80, "--timeout-ms", 120000)
            spec = plan["specification"]
            memory = 2 << 30
            spec["resources"] = {
                "capacity": {"cpu_slots": 1, "memory_bytes": memory,
                             "gpus": {"gpu0": memory} if self.device == "cuda" else {}},
                "request": {"cpu_slots": 1, "memory_bytes": memory,
                            "gpu": {"id": "gpu0", "memory_bytes": memory}
                            if self.device == "cuda" else None},
            }
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            probe = json.loads(json.dumps(spec))
            probe["arguments"][5] = str(base / "never-trained")
            probe["training"]["model_id"] = "wrong-model"
            probe_path = base / "blocked.json"
            probe_path.write_text(json.dumps(probe), encoding="utf-8")
            started = time.monotonic()
            process = subprocess.Popen(self.command("local", "execute", "--root", self.store,
                                       "--id", "real-training", "--spec", spec_path),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       env=self.environment)
            try:
                deadline = time.monotonic() + 90
                while not (work / "latest.jyck").is_file():
                    self.assertIsNone(process.poll(), "Training ended before its first checkpoint")
                    self.assertLess(time.monotonic(), deadline, "No durable training checkpoint")
                    time.sleep(0.01)
                observed_checkpoint = time.monotonic() - started
                # The native callback first writes latest after update 16. The
                # competitor is denied before an attempt or model is created.
                blocked = self.invoke("local", "execute", "--root", self.store,
                                      "--id", "competitor", "--spec", probe_path, expected=7)
                self.assertEqual(blocked["status"], "resource_unavailable")
                self.assertEqual(blocked["attempts"], [])
                cancellation = self.invoke("local", "cancel", "--root", self.store,
                                           "--id", "real-training")
                self.assertTrue(cancellation["cancel_requested"])
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 130, stdout + stderr)
            finally:
                if process.poll() is None:
                    self.invoke("local", "cancel", "--root", self.store, "--id", "real-training")
                    try:
                        process.communicate(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate(timeout=10)
            cancelled = json.loads(stdout)["data"]
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertEqual(len(cancelled["attempts"]), 1)
            attempt = cancelled["attempts"][0]
            result = attempt["result"]
            self.assertEqual(result["status"], "cancelled")
            self.assertTrue(result["process_tree_cleanup_confirmed"])
            self.assertTrue(attempt["resource_released"])
            self.assertFalse(result.get("training_outputs_confirmed", False))
            self.assertFalse(alive(result["process_id"]))
            self.assertFalse((work / "result.json").exists())
            self.assertFalse((work / "final.jyck").exists())
            references = self.references(attempt["attempt_id"])
            self.assertTrue({"checkpoint.last", "model.weights", "training.result",
                             "local.training.outputs"}.isdisjoint(references))
            job_refs = self.references("real-training")
            release = json.loads(self.blob(job_refs["local.resource.release.1"]))
            for role in ("intent", "owner", "result", "stdout", "stderr"):
                self.assertEqual(release[role + "_sha256"],
                                 references["local." + role]["artifact"]["sha256"])
            snapshot = self.invoke("checkpoint", "inspect", "--path", work / "latest.jyck")["manifest"]
            self.assertEqual({p["state"]["participant_id"] for p in snapshot["participants"]}, PARTICIPANTS)
            before = {str(p.relative_to(work)): (p.stat().st_size, p.stat().st_mtime_ns)
                      for p in work.rglob("*") if p.is_file()}
            for action in ("recover", "execute"):
                options = ("--spec", spec_path) if action == "execute" else ()
                replay = self.invoke("local", action, "--root", self.store,
                                     "--id", "real-training", *options, expected=130)
                self.assertEqual(replay, cancelled)
            after = {str(p.relative_to(work)): (p.stat().st_size, p.stat().st_mtime_ns)
                     for p in work.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            # The same reservation can now be acquired. Its intentionally
            # wrong model is rejected by native preflight, with zero updates.
            admitted = self.invoke("local", "recover", "--root", self.store,
                                   "--id", "competitor", expected=1)
            self.assertEqual(admitted["status"], "failed")
            self.assertEqual(len(admitted["attempts"]), 1)
            rejection = admitted["attempts"][0]
            self.assertEqual(rejection["result"]["exit_code"], 2)
            self.assertTrue(rejection["result"]["process_tree_cleanup_confirmed"])
            self.assertTrue(rejection["resource_released"])
            self.assertFalse((base / "never-trained").exists())
            self.invoke("run", "verify", "--root", self.store)
            if output := os.environ.get("JYPPX_LIUDING_RECIPE_EVIDENCE_DIR"):
                evidence = Path(output) / self.device
                evidence.mkdir(parents=True, exist_ok=False)
                values = {"spec": spec, "blocked": blocked, "cancel-request": cancellation,
                          "cancelled": cancelled, "admitted-rejection": admitted,
                          "checkpoint": snapshot, "release": release,
                          "observation": {"first_checkpoint_seconds": observed_checkpoint,
                                          "checkpoint_update_lower_bound": 16,
                                          "configured_update_ceiling": 96,
                                          "exact_interrupted_update_count_known": False,
                                          "actual_training_processes": 1,
                                          "post_cancel_native_PID_alive": False,
                                          "workspace_metadata_replay_equal": before == after,
                                          "resource_limits_are_logical": True}}
                for name, value in values.items():
                    (evidence / (name + ".json")).write_text(
                        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                for name in ("local.intent", "local.owner", "local.result", "local.stdout",
                             "local.stderr", "local.training.inputs"):
                    (evidence / (name + ".raw")).write_bytes(self.blob(references[name]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args, rest = parser.parse_known_args()
    LocalRecipeLifecycle.cli, LocalRecipeLifecycle.device = args.cli.resolve(), args.device
    unittest.main(argv=[__file__, *rest])
