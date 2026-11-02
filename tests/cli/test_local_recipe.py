# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native recipe -> durable attempt -> exact checkpoint retry integration.

The positive fixture retains the existing 48-update / 0.875 quality gate.
Python only supervises the C++ CLI, makes self-authored PNG test inputs and
independently checks results/CAS; no Python model or training runtime is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_image_recipe import PARTICIPANTS, png
from test_local_attempt import sparse_resize


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalRecipe(unittest.TestCase):
    cli: Path
    device: str

    @classmethod
    def invoke(cls, *args: object, expected: int = 0) -> dict:
        result = subprocess.run(
            [str(cls.cli), *map(str, args), "--format", "json"],
            capture_output=True, text=True, encoding="utf-8", timeout=420,
            env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"), check=False)
        if result.returncode != expected:
            raise AssertionError(f"{args!r}\nexit={result.returncode}, expected={expected}"
                                 f"\n{result.stdout}\n{result.stderr}")
        envelope = json.loads(result.stdout)
        return envelope.get("data", envelope)

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="liuding-local-recipe-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name) / "本地 recipe"
        cls.base.mkdir()
        for split, count in (("train", 8), ("validation", 4)):
            for label, name in enumerate(("green", "red")):
                folder = cls.base / "images" / split / name
                folder.mkdir(parents=True)
                for number in range(count):
                    (folder / f"{number}.png").write_bytes(
                        png(label, number + (20 if split == "validation" else 0)))
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
        cls.config_path = cls.base / "config.json"
        cls.config_path.write_text(json.dumps(cls.config), encoding="utf-8")
        cls.store = cls.base / "store"
        cls.work = cls.base / "resumed"
        cls.spec_path = cls.base / "retry.json"
        cls.preflight = cls.invoke("recipe", "check", "--file", cls.config_path)
        cls.plan = cls.prepare(cls.config_path, cls.work, cls.spec_path, pause=19)

    @classmethod
    def train(cls) -> None:
        if hasattr(cls, "completed"):
            return
        cls.baseline_root = cls.base / "continuous"
        cls.baseline = cls.invoke("recipe", "run", "--file", cls.config_path,
                                  "--root", cls.baseline_root)
        cls.completed = cls.invoke("local", "execute", "--root", cls.store,
                                   "--id", "recipe-retry", "--spec", cls.spec_path)
        cls.resumed = json.loads((cls.work / "result.json").read_bytes())
        cls.snapshots = {}
        for name, root in (("continuous", cls.baseline_root), ("resumed", cls.work)):
            cls.snapshots[name] = cls.invoke("checkpoint", "inspect", "--path",
                                             root / "final.jyck")["manifest"]
        if output := os.environ.get("JYPPX_LIUDING_RECIPE_EVIDENCE_DIR"):
            evidence = Path(output) / cls.device
            evidence.mkdir(parents=True, exist_ok=False)
            cls.evidence = evidence
            for name, value in (("preflight", cls.preflight), ("plan", cls.plan),
                                ("continuous", cls.baseline), ("local", cls.completed),
                                ("resumed", cls.resumed), ("checkpoints", cls.snapshots)):
                (evidence / f"{name}.json").write_text(
                    json.dumps(value, indent=2) + "\n", encoding="utf-8")
            for segment in cls.completed["attempts"]:
                references = cls.refs(segment["attempt_id"])
                for name in ("local.result", "local.stdout", "local.training.inputs",
                             "local.training.outputs"):
                    if name in references:
                        (evidence / f"attempt-{segment['index']}-{name}.json").write_bytes(
                            cls.blob(references[name]))

    @classmethod
    def prepare(cls, config: Path, work: Path, spec: Path, pause: int = 0) -> dict:
        return cls.invoke("recipe", "local-spec", "--file", config, "--root", work,
                          "--store", cls.store, "--spec", spec, "--executable", cls.cli,
                          "--pause-after", pause, "--timeout-ms", 360000)

    @classmethod
    def refs(cls, run: str) -> dict:
        return {row["reference_id"]: row for row in cls.invoke(
            "run", "show", "--root", cls.store, "--id", run)["references"]}

    @classmethod
    def blob(cls, ref: dict) -> bytes:
        row = ref["artifact"]
        identity = row["sha256"]
        content = (cls.store / "artifacts/objects" / identity[:2] / identity[2:]).read_bytes()
        if len(content) != row["size_bytes"] or hashlib.sha256(content).hexdigest() != identity:
            raise AssertionError("CAS identity mismatch")
        return content

    def test_01_native_spec_pins_resolved_inputs_and_program(self) -> None:
        self.train()
        spec = self.plan["specification"]
        self.assertEqual(spec, json.loads(self.spec_path.read_bytes()))
        self.assertEqual(spec["executable_sha256"], sha(self.cli))
        self.assertEqual(spec["max_attempts"], 2)
        self.assertEqual(spec["retry_exit_codes"], [75])
        for name, actual in (("configuration", self.preflight["config"]),
                             ("data_manifest", self.preflight["dataset"])):
            binding = spec["training"][name]
            ref = self.refs(binding["run_id"])[binding["reference_id"]]
            self.assertEqual(json.loads(self.blob(ref)), actual)

    def test_02_real_quality_complete_state_and_rng_match_continuous(self) -> None:
        self.train()
        self.assertEqual(self.completed["status"], "completed")
        self.assertEqual(len(self.completed["attempts"]), 2)
        first, second = self.completed["attempts"]
        self.assertNotEqual(first["attempt_id"], second["attempt_id"])
        self.assertEqual(first["result"]["exit_code"], 75)
        self.assertEqual(second["result"]["exit_code"], 0)
        for segment in (first, second):
            self.assertTrue(segment["result"]["process_tree_cleanup_confirmed"])
        paused = json.loads(self.blob(self.refs(first["attempt_id"])["local.stdout"]))["data"]
        self.assertEqual((paused["status"], paused["global_step"]), ("paused", 19))
        for name in ("global_step", "optimizer_steps", "completed_epochs", "scheduler_steps",
                     "training_metrics", "validation_metrics", "model_sha256", "accuracy",
                     "confusion", "sampling", "next_rng_sha256", "next_cpu_rng_sha256"):
            self.assertEqual(self.baseline[name], self.resumed[name], name)
        self.assertEqual(self.resumed["global_step"], 48)
        self.assertGreaterEqual(self.resumed["accuracy"], 0.875)
        self.assertTrue(self.resumed["quality_passed"])
        self.assertEqual(sha(self.work / "final.jyck"), sha(self.baseline_root / "final.jyck"))
        self.assertEqual(self.snapshots["continuous"], self.snapshots["resumed"])
        self.assertEqual({p["state"]["participant_id"] for p in
                          self.snapshots["resumed"]["participants"]}, PARTICIPANTS)

    def test_03_retry_input_and_three_output_cas_bindings(self) -> None:
        self.train()
        first, second = self.completed["attempts"]
        old_refs, new_refs = self.refs(first["attempt_id"]), self.refs(second["attempt_id"])
        inputs = json.loads(self.blob(new_refs["local.training.inputs"]))
        self.assertEqual(inputs["training"]["resume"]["checkpoint"]["sha256"],
                         old_refs["checkpoint.last"]["artifact"]["sha256"])
        self.assertEqual(inputs["previous_result_sha256"], old_refs["local.result"]["artifact"]["sha256"])
        self.assertTrue(second["result"]["training_outputs_confirmed"])
        self.assertNotIn("local.training.outputs", old_refs)
        for name, path in (("checkpoint.last", "final.jyck"),
                           ("model.weights", "model.safetensors"),
                           ("training.result", "result.json")):
            self.assertEqual(new_refs[name]["artifact"]["sha256"], sha(self.work / path))
            self.assertEqual(len(self.blob(new_refs[name])), (self.work / path).stat().st_size)
        self.invoke("run", "verify", "--root", self.store)

    def test_04_confirmed_replay_does_not_train_or_change_recipe_files(self) -> None:
        self.train()
        before = {str(p.relative_to(self.work)): (sha(p), p.stat().st_mtime_ns)
                  for p in self.work.rglob("*") if p.is_file()}
        for action in ("recover", "execute"):
            options = ("--spec", self.spec_path) if action == "execute" else ()
            self.assertEqual(self.invoke("local", action, "--root", self.store,
                                         "--id", "recipe-retry", *options), self.completed)
        after = {str(p.relative_to(self.work)): (sha(p), p.stat().st_mtime_ns)
                 for p in self.work.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_05_changed_model_config_data_and_outputs_fail_before_training(self) -> None:
        if output := os.environ.get("JYPPX_LIUDING_RECIPE_EVIDENCE_DIR"):
            if not hasattr(self, "evidence"):
                self.evidence = Path(output) / (self.device + "-inputs-only")
                self.evidence.mkdir(parents=True, exist_ok=False)
        for name in ("model", "config", "data", "outputs"):
            with self.subTest(name=name):
                root, path = self.base / ("bad-" + name), self.base / ("bad-spec-" + name + ".json")
                spec = self.prepare(self.config_path, root, path)["specification"]
                if name == "model":
                    spec["training"]["model_id"] = "resnet50"
                elif name == "outputs":
                    spec["training"]["outputs"][1]["role"] = "wrong"
                elif name == "config":
                    changed = self.base / "changed-config.json"
                    changed.write_text(json.dumps(self.config | {"seed": 7}), encoding="utf-8")
                    spec["arguments"][3] = str(changed)
                original = None
                image = self.base / "images/train/green/0.png"
                if name == "data":
                    original = image.read_bytes()
                    image.write_bytes(png(1, 123))
                try:
                    path.write_text(json.dumps(spec), encoding="utf-8")
                    result = self.invoke("local", "execute", "--root", self.store,
                                          "--id", "bad-" + name, "--spec", path, expected=1)
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(len(result["attempts"]), 1)
                    attempt = result["attempts"][0]
                    self.assertEqual(attempt["result"]["exit_code"], 2)
                    self.assertTrue(attempt["result"]["process_tree_cleanup_confirmed"])
                    self.assertNotIn("checkpoint.last", self.refs(attempt["attempt_id"]))
                    self.assertFalse(root.exists())
                    if hasattr(self, "evidence"):
                        (self.evidence / f"rejected-{name}.json").write_text(
                            json.dumps(result, indent=2) + "\n", encoding="utf-8")
                finally:
                    if original is not None:
                        image.write_bytes(original)

    def test_06_local_run_requires_ownership_and_spec_never_overwrites(self) -> None:
        root = self.base / "unowned"
        self.invoke("recipe", "local-run", "--file", self.config_path,
                    "--root", root, expected=2)
        self.assertFalse(root.exists())
        before = self.spec_path.read_bytes()
        self.invoke("recipe", "local-spec", "--file", self.config_path,
                    "--root", root, "--store", self.store, "--spec", self.spec_path,
                    "--executable", self.cli, expected=2)
        self.assertEqual(self.spec_path.read_bytes(), before)

    def test_07_executable_identity_has_an_independent_byte_limit(self) -> None:
        accepted = self.base / "large-executable.bin"
        accepted.touch()
        sparse_resize(accepted, (256 << 20) + 1)
        spec_path, work = self.base / "large-executable.json", self.base / "large-work"
        planned = self.invoke(
            "recipe", "local-spec", "--file", self.config_path, "--root", work,
            "--store", self.store, "--spec", spec_path, "--executable", accepted)
        self.assertEqual(planned["specification"]["executable_sha256"], sha(accepted))
        self.assertFalse(work.exists())
        oversized = self.base / "oversized-executable.bin"
        oversized.touch()
        sparse_resize(oversized, (1 << 30) + 1)
        rejected = self.base / "oversized-executable.json"
        result = self.invoke(
            "recipe", "local-spec", "--file", self.config_path, "--root", work,
            "--store", self.store, "--spec", rejected, "--executable", oversized, expected=2)
        self.assertEqual(result["error"]["name"], "invalid_argument")
        self.assertIn("byte limit", result["error"]["message"])
        self.assertFalse(rejected.exists())
        self.assertFalse(work.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args, rest = parser.parse_known_args()
    LocalRecipe.cli, LocalRecipe.device = args.cli.resolve(), args.device
    unittest.main(argv=[__file__, *rest])
