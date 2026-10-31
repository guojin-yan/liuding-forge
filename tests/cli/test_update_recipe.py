# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Independent update-driven recipe/checkpoint acceptance, using stdlib only.

The self-authored 16/8 RGB task retains 48 updates, seed 20261028 and
accuracy >= 0.875. It does not qualify pretrained transfer, DINO quality,
real-dataset accuracy or performance. Transfer byte-preflight uses deliberately
invalid model bytes and never claims executable weights or group-rate training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest


_fixture_spec = importlib.util.spec_from_file_location(
    "_liuding_image_fixture", Path(__file__).resolve().with_name("test_image_recipe.py"))
assert _fixture_spec is not None and _fixture_spec.loader is not None
_fixture = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture)
png = _fixture.png
PARTICIPANTS = frozenset(_fixture.PARTICIPANTS)

EVIDENCE: Path | None = None
_invocation = 0
SCHEDULE = {"lr_schedule": "warmup_linear", "warmup_updates": 5,
            "warmup_start_factor": 0.5, "lr_end_factor": 0.1}


def save_evidence(name: str, value: object) -> None:
    if EVIDENCE is not None:
        (EVIDENCE / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")


def configuration(base: Path, device: str) -> dict:
    for split, count in (("train", 8), ("validation", 4)):
        for label, name in enumerate(("green", "red")):
            folder = base / "images" / split / name
            folder.mkdir(parents=True)
            for number in range(count):
                (folder / f"{number}.png").write_bytes(
                    png(label, number + (20 if split == "validation" else 0)))
    return {
        "schema_version": 1, "recipe": "jyppx.liuding.imagefolder.resnet18/v1",
        "train_root": "images/train", "validation_root": "images/validation",
        "data_license": "CC0-1.0; self-authored RGB fixture", "device": device,
        "seed": 20261028, "epochs": 12, "batch_size": 4, "image_size": 32,
        "workers": 2, "prefetch_batches": 2, "learning_rate": 0.01,
        "momentum": 0.9, "weight_decay": 0.0001, "lr_gamma": 1.0,
        "checkpoint_every": 16, "minimum_accuracy": 0.875,
        "max_image_bytes": 1048576, "max_data_bytes": 10485760,
        "max_samples": 1024, "image_budget_bytes": 67108864,
    }


class Cursor:
    """Bounded independent decoder for the persisted scheduler format."""

    def __init__(self, data: bytes) -> None:
        self.data, self.offset = data, 0

    def read(self, count: int) -> bytes:
        assert 0 <= count <= len(self.data) - self.offset, "Truncated checkpoint metadata"
        result = self.data[self.offset:self.offset + count]
        self.offset += count
        return result

    def number(self, kind: str) -> int | float:
        return struct.unpack("<" + kind, self.read(struct.calcsize("<" + kind)))[0]

    def count(self) -> int:
        value = self.number("I")
        assert isinstance(value, int) and 0 <= value <= 1024
        return value

    def string(self) -> str:
        length = self.number("I")
        assert isinstance(length, int) and 0 <= length <= 4096
        return self.read(length).decode("utf-8")

    def value(self) -> object:
        tag = self.number("B")
        if tag == 1:
            value = self.number("B")
            assert value in (0, 1)
            return bool(value)
        if tag in (2, 3):
            result = self.number("q" if tag == 2 else "d")
            assert math.isfinite(result)
            return result
        if tag == 4:
            return self.string()
        assert tag in (5, 6, 7), f"Unknown scheduler value tag {tag}"
        return [(self.string() if tag == 7 else self.number("q" if tag == 5 else "d"))
                for _ in range(self.count())]

    def mapping(self) -> dict:
        result = {}
        for _ in range(self.count()):
            key = self.string()
            assert key not in result
            result[key] = self.value()
        return result

    def descriptor(self, depth: int = 0) -> dict:
        assert depth <= 2
        result = {"schema_version": self.number("I"), "type_id": self.string(),
                  "step_mode": self.number("B"), "config": self.mapping(),
                  "optimizer_id": self.string()}
        result["optimizer_group_ids"] = [self.string() for _ in range(self.count())]
        result["optimizer_group_signature"] = self.string()
        assert self.number("B") == 0, "Recipe scheduler must not use a callable"
        result["children"] = [self.descriptor(depth + 1) for _ in range(self.count())]
        result["fingerprint"] = self.string()
        return result

    def state(self, depth: int = 0) -> dict:
        assert depth <= 2
        return {"completed_steps": self.number("Q"), "control": self.mapping(),
                "children": [self.state(depth + 1) for _ in range(self.count())]}

    def finish(self) -> None:
        assert self.offset == len(self.data), "Trailing checkpoint metadata"


def checkpoint(path: Path) -> dict:
    """Read only small control blobs; seek over model/RNG/optimizer tensors."""
    blobs = {}
    with path.open("rb") as stream:
        assert stream.read(8) == b"JYPPXCK1"
        assert struct.unpack("<I", stream.read(4))[0] == 1
        manifest_size = struct.unpack("<Q", stream.read(8))[0]
        assert 0 < manifest_size <= 8 * 1024 * 1024
        manifest = json.loads(stream.read(manifest_size))
        file_size = path.stat().st_size
        for participant in manifest["participants"]:
            identity = participant["state"]["participant_id"]
            for blob in participant["blobs"]:
                size = blob["size_bytes"]
                assert isinstance(size, int) and 0 <= size <= file_size - stream.tell()
                key = (identity, blob["field_name"])
                if key in (("recipe.lr", "topology"), ("recipe.lr", "control"),
                           ("recipe.sgd", "control")):
                    assert size <= 65536 and key not in blobs
                    data = stream.read(size)
                    assert hashlib.sha256(data).hexdigest() == blob["digest"]["hex"]
                    blobs[key] = data
                else:
                    stream.seek(size, 1)
        # The archive ends with its 32-byte whole-file SHA-256 trailer. The
        # native verifier checks that framing; this reader hashes selected blobs.
        assert stream.tell() + 32 == file_size
        assert len(stream.read(32)) == 32 and not stream.read(1)
    topology = Cursor(blobs[("recipe.lr", "topology")])
    assert topology.read(8) == b"JYPPXSLD" and topology.number("I") == 1
    assert topology.string() == "recipe.lr"
    descriptor = topology.descriptor()
    topology.finish()
    control = Cursor(blobs[("recipe.lr", "control")])
    assert control.read(8) == b"JYPPXSLC" and control.number("I") == 3
    state = control.state()
    rates = [control.number("d") for _ in range(control.count())]
    control.finish()
    sgd = Cursor(blobs[("recipe.sgd", "control")])
    assert sgd.read(8) == b"JYPPXSGC" and sgd.number("I") == 1
    steps, groups, parameters = sgd.number("Q"), sgd.count(), sgd.number("I")
    assert isinstance(parameters, int) and 0 < parameters <= 1000000
    sgd_rates = [sgd.number("d") for _ in range(groups)]
    sgd.read(parameters)  # Momentum-initialized flags, one per parameter.
    sgd.finish()
    return {"manifest": manifest, "descriptor": descriptor, "scheduler_state": state,
            "scheduler_rates": rates, "optimizer": {"completed_steps": steps,
            "parameters": parameters, "learning_rates": sgd_rates}}


def schedule_factor(completed: int, total: int = 48, warmup: int = 5) -> float:
    """Independent endpoint interpolation; never reads reported current rates."""
    if completed <= warmup:
        fraction = completed / warmup
        return (1 - fraction) * SCHEDULE["warmup_start_factor"] + fraction
    fraction = (completed - warmup) / (total - warmup)
    return (1 - fraction) + fraction * SCHEDULE["lr_end_factor"]


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def collect_root_evidence(arguments: tuple[object, ...], invocation: int) -> None:
    """Retain small failure artifacts before class cleanup removes its temp tree."""
    if EVIDENCE is None:
        return
    values = [str(value) for value in arguments]
    if "--root" not in values:
        return
    artifacts = {}
    try:
        root = Path(values[values.index("--root") + 1])
        if not root.is_dir():
            return
        for name in ("config.json", "lineage.json", "dataset.json", "build.json",
                     "weight-input.json", "result.json", "evaluation-suite.json",
                     "evaluation.json", "diagnostics.json", "bundle/manifest.json",
                     "bundle/metadata/provenance.json"):
            path = root / name
            try:
                if not path.is_file():
                    continue
                size = path.stat().st_size
                if size > 8 * 1024 * 1024:
                    artifacts[name] = {"capture_error": "Small JSON exceeds 8 MiB capture limit",
                                       "size_bytes": size}
                    continue
                raw = path.read_bytes()
                entry = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                    json.dumps(parsed, allow_nan=False)
                    entry["data"] = parsed
                except (UnicodeError, ValueError) as error:
                    entry.update(capture_error=repr(error), raw_hex=raw.hex())
                artifacts[name] = entry
            except Exception as error:
                artifacts[name] = {"capture_error": repr(error)}
        for name in ("latest.jyck", "final.jyck"):
            path = root / name
            try:
                if path.is_file():
                    artifacts[name] = checkpoint(path)
            except Exception as error:
                artifacts[name] = {"capture_error": repr(error)}
    except Exception as error:
        artifacts["capture_error"] = repr(error)
    # Capture failures must not replace a CLI failure or a later assertion.
    try:
        save_evidence(f"command-{invocation:03d}-artifacts", artifacts)
    except Exception as error:
        print(f"Small-artifact capture failed for command {invocation}: {error!r}",
              file=sys.stderr)


class RecipeCase(unittest.TestCase):
    cli: Path
    repo: Path
    device: str

    @classmethod
    def invoke(cls, *arguments: object, expected: int | tuple[int, ...] = 0) -> dict:
        global _invocation
        _invocation += 1
        try:
            result = subprocess.run(
                [str(cls.cli), *(str(value) for value in arguments), "--format", "json"],
                capture_output=True, text=True, encoding="utf-8", timeout=300,
                env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"), check=False)
        except Exception:
            collect_root_evidence(arguments, _invocation)
            raise
        save_evidence(f"command-{_invocation:03d}", {
            "arguments": [str(value) for value in arguments], "expected_exit": expected,
            "exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        collect_root_evidence(arguments, _invocation)
        accepted = (expected,) if isinstance(expected, int) else expected
        if result.returncode not in accepted:
            raise AssertionError(f"{arguments!r}: exit {result.returncode}, expected {expected}"
                                 f"\nstdout={result.stdout}\nstderr={result.stderr}")
        envelope = json.loads(result.stdout)
        assert envelope["schema_version"] == 1
        if accepted == (0, 4):
            assert result.returncode == (0 if envelope["data"]["quality_passed"] else 4)
        return envelope

    @classmethod
    def setup_fixture(cls, name: str) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix=f"liuding-update-{name}-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name) / "逐更新配方"
        cls.base.mkdir()
        cls.config = configuration(cls.base, cls.device)


class UpdateConfiguration(RecipeCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setup_fixture("config")
        cls.cases = {"image": cls.config}
        weights = cls.base / "not-model-weights.safetensors"
        weights.write_bytes(b"Deliberately not executable model weights")
        descriptor = json.loads((cls.repo / "recipes/transfer-learning/resnet50-a1.weights.json").read_text(encoding="utf-8"))
        descriptor.update(sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                          size_bytes=weights.stat().st_size)
        metadata = cls.base / "weights.json"
        metadata.write_text(json.dumps(descriptor), encoding="utf-8")
        cls.cases["transfer"] = cls.config | {
            "recipe": "jyppx.liuding.imagefolder.transfer/v1", "image_size": 224,
            "model": "resnet50", "weights_file": weights.name,
            "weights_metadata": metadata.name, "backbone_mode": "frozen",
            "batch_norm": "eval", "image_budget_bytes": 134217728}

    def test_01_optional_schema_default_identity_and_valid_explicit_modes(self) -> None:
        for name, config in self.cases.items():
            with self.subTest(recipe=name):
                arguments = ("recipe", "schema") + (() if name == "image" else ("--name", "transfer"))
                schema = self.invoke(*arguments)["data"]
                for field in SCHEDULE:
                    self.assertFalse(schema["root"]["fields"][field]["required"])
                path = self.base / f"{name}.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                baseline = self.invoke("recipe", "check", "--file", path)["data"]
                self.assertEqual(baseline["total_steps"], 48)
                for field in SCHEDULE:
                    self.assertNotIn(field, baseline["config"])
                hashes = {baseline["config_sha256"]}
                for patch in ({"lr_schedule": "epoch_step"}, SCHEDULE,
                              SCHEDULE | {"warmup_updates": 1},
                              SCHEDULE | {"warmup_start_factor": 1.0, "lr_end_factor": 1.0}):
                    path.write_text(json.dumps(config | patch), encoding="utf-8")
                    accepted = self.invoke("recipe", "check", "--file", path)["data"]
                    self.assertNotIn(accepted["config_sha256"], hashes)
                    hashes.add(accepted["config_sha256"])
                    for field, value in patch.items():
                        self.assertEqual(accepted["config"][field], value)
                schema_file = self.base / f"{name}-schema.json"
                schema_file.write_text(json.dumps(schema), encoding="utf-8")
                self.invoke("config", "lint", "--schema", schema_file, "--file", path)

    def test_02_invalid_closed_types_modes_bounds_rejected_before_root(self) -> None:
        patches = [{"lr_schedule": value} for value in (True, None, 1, "auto", "")]
        patches += [{field: value} for field, values in (
            ("warmup_updates", (True, None, "5", 5.0, 0, -1, 48, 49)),
            ("warmup_start_factor", (True, None, "0.5", 0.0, -0.1, 1.01, 1e-310,
                                      float("nan"), float("inf"), float("-inf"))),
            ("lr_end_factor", (True, None, "0.1", 0.0, -0.1, 1.01,
                                float("nan"), float("inf"), float("-inf"))),
        ) for value in values]
        patches += [{"lr_gamma": 0.9}, {"epochs": 1}, {"surprise": 1}]
        for name, config in self.cases.items():
            invalid = [(config | SCHEDULE | patch) for patch in patches]
            invalid += [({key: value for key, value in (config | SCHEDULE).items() if key != field})
                        for field in ("warmup_updates", "warmup_start_factor", "lr_end_factor")]
            for field in ("warmup_updates", "warmup_start_factor", "lr_end_factor"):
                invalid.extend((config | {field: SCHEDULE[field]},
                                config | {"lr_schedule": "epoch_step", field: SCHEDULE[field]}))
            for index, candidate in enumerate(invalid):
                with self.subTest(recipe=name, invalid=index):
                    path = self.base / "invalid.json"
                    path.write_text(json.dumps(candidate), encoding="utf-8")
                    root = self.base / f"rejected-{name}-{index}"
                    self.invoke("recipe", "run", "--file", path, "--root", root, expected=2)
                    self.assertFalse(root.exists(), "Invalid configuration created its output root")
            path.write_text(json.dumps(config | SCHEDULE)[:-1] + ', "warmup_updates": 6}', encoding="utf-8")
            root = self.base / f"duplicate-{name}"
            self.invoke("recipe", "run", "--file", path, "--root", root, expected=2)
            self.assertFalse(root.exists())

    def test_03_transfer_group_configuration_preserves_identity_without_claiming_execution(self) -> None:
        config = self.cases["transfer"] | SCHEDULE | {"backbone_mode": "finetune"}
        path = self.base / "groups.json"
        hashes = set()
        for rates in ({}, {"head_learning_rate": 0.02}, {"backbone_learning_rate": 0.0001},
                      {"head_learning_rate": 0.02, "backbone_learning_rate": 0.0001}):
            path.write_text(json.dumps(config | rates), encoding="utf-8")
            accepted = self.invoke("recipe", "check", "--file", path)["data"]
            self.assertEqual(accepted["total_steps"], 48)
            self.assertNotIn(accepted["config_sha256"], hashes)
            hashes.add(accepted["config_sha256"])
            for field in ("head_learning_rate", "backbone_learning_rate"):
                if field in rates:
                    self.assertEqual(accepted["config"][field], rates[field])
                else:
                    self.assertNotIn(field, accepted["config"])


class UpdateRecipe(RecipeCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setup_fixture("training")
        cls.config |= SCHEDULE
        cls.file = cls.base / "config.json"
        cls.file.write_text(json.dumps(cls.config), encoding="utf-8")
        cls.preflight = cls.invoke("recipe", "check", "--file", cls.file)["data"]
        save_evidence("training-config", cls.config)
        save_evidence("preflight", cls.preflight)
        cls.roots = {name: cls.base / name for name in ("continuous", "resumed", "killed")}
        cls.results, cls.checkpoints = {}, {}
        cls.results["continuous"] = cls.invoke(
            "recipe", "run", "--file", cls.file, "--root", cls.roots["continuous"])["data"]
        for action, step in (("run", 3), ("resume", 7)):
            name = f"paused-{step}"
            cls.results[name] = cls.invoke("recipe", action, "--file", cls.file,
                "--root", cls.roots["resumed"], "--pause-after", step)["data"]
            cls.checkpoints[name] = checkpoint(cls.roots["resumed"] / "latest.jyck")
            save_evidence(name, cls.results[name])
            save_evidence(f"{name}-checkpoint", cls.checkpoints[name])
        cls.results["resumed"] = cls.invoke(
            "recipe", "resume", "--file", cls.file, "--root", cls.roots["resumed"])["data"]
        root = cls.roots["killed"]
        process = subprocess.Popen(
            [str(cls.cli), "recipe", "run", "--file", str(cls.file),
             "--root", str(root), "--format", "json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"))
        try:
            deadline = time.monotonic() + 120
            while not (root / "latest.jyck").is_file():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise AssertionError("Training exited/timed out before its first durable checkpoint")
                time.sleep(0.01)
            if process.poll() is not None:
                raise AssertionError("Training completed before process-kill recovery could be exercised")
            process.kill()
            stdout, stderr = process.communicate(timeout=30)
            save_evidence("killed-process", {"returncode": process.returncode,
                "stdout": stdout.decode("utf-8"), "stderr": stderr.decode("utf-8")})
            assert process.returncode != 0 and not (root / "final.jyck").exists()
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=30)
        cls.checkpoints["killed-durable"] = checkpoint(root / "latest.jyck")
        save_evidence("killed-durable-checkpoint", cls.checkpoints["killed-durable"])
        cls.results["killed"] = cls.invoke(
            "recipe", "resume", "--file", cls.file, "--root", root)["data"]
        for name, root in cls.roots.items():
            cls.checkpoints[name] = checkpoint(root / "final.jyck")
            save_evidence(name, cls.results[name])
            save_evidence(f"{name}-checkpoint", cls.checkpoints[name])
        if EVIDENCE is not None:
            for name in ("lineage.json", "dataset.json", "evaluation.json", "diagnostics.json"):
                shutil.copy2(cls.roots["resumed"] / name, EVIDENCE / name)
            shutil.copy2(cls.roots["resumed"] / "bundle/metadata/provenance.json",
                         EVIDENCE / "provenance.json")

    def assert_schedule(self, result: dict, completed: int) -> None:
        optimization = result["optimization"]
        self.assertEqual(optimization["optimizer_id"], "recipe.sgd")
        scheduler = optimization["scheduler"]
        for name, expected in ({"scheduler_id": "recipe.lr", "kind": "warmup_linear",
            "step_mode": "batch", "total_updates": 48, "completed_steps": completed} |
            {name: value for name, value in SCHEDULE.items() if name != "lr_schedule"}).items():
            self.assertEqual(scheduler[name], expected, name)
        groups = optimization["groups"]
        self.assertEqual([group["group_id"] for group in groups], ["main"])
        self.assertTrue(groups[0]["parameter_ids"])
        self.assertEqual(len(groups[0]["parameter_ids"]), len(set(groups[0]["parameter_ids"])))
        self.assertEqual(groups[0]["initial_learning_rate"], 0.01)
        # This checks the descriptor claim. Actual first-update ordering is
        # qualified separately by the C++ Trainer/scheduler integration gate.
        self.assertEqual(groups[0]["first_learning_rate"], 0.005)
        self.assertAlmostEqual(groups[0]["final_learning_rate"], 0.01 * schedule_factor(completed), places=14)

    def test_01_real_quality_and_cross_process_exact_recovery(self) -> None:
        self.assertEqual(self.preflight["total_steps"], 48)
        for completed in (3, 7):
            paused = self.results[f"paused-{completed}"]
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["global_step"], completed)
            self.assert_schedule(paused, completed)
        fields = ("model_sha256", "next_rng_sha256", "next_cpu_rng_sha256", "training_metrics",
                  "validation_metrics", "accuracy", "confusion", "global_step", "optimizer_steps",
                  "scheduler_steps", "completed_epochs", "optimization")
        for name in self.roots:
            result = self.results[name]
            self.assertEqual(result["status"], "completed")
            for field in ("global_step", "optimizer_steps", "scheduler_steps"):
                self.assertEqual(result[field], 48)
            self.assertEqual(result["completed_epochs"], 12)
            self.assertEqual(result["validation_samples"], 8)
            self.assertGreaterEqual(result["accuracy"], 0.875)
            self.assertTrue(result["quality_passed"])
            self.assert_schedule(result, 48)
            for field in fields:
                self.assertEqual(result[field], self.results["continuous"][field], (name, field))
            participants = self.checkpoints[name]["manifest"]["participants"]
            self.assertEqual({item["state"]["participant_id"] for item in participants}, PARTICIPANTS)
            self.assertEqual(participants, self.checkpoints["continuous"]["manifest"]["participants"])
            self.assertEqual(file_sha256(self.roots[name] / "model.safetensors"),
                             self.results["continuous"]["model_sha256"])

    def test_02_persisted_optimizer_rates_scheduler_tree_and_counters(self) -> None:
        for name, saved in self.checkpoints.items():
            with self.subTest(checkpoint=name):
                steps = saved["optimizer"]["completed_steps"]
                if name.startswith("paused-"):
                    self.assertEqual(steps, int(name.removeprefix("paused-")))
                elif name == "killed-durable":
                    self.assertGreaterEqual(steps, 16)
                    self.assertLess(steps, 48)
                else:
                    self.assertEqual(steps, 48)
                state, descriptor = saved["scheduler_state"], saved["descriptor"]
                self.assertEqual(state["completed_steps"], steps)
                self.assertEqual([node["completed_steps"] for node in state["children"]],
                                 [min(steps, 5), max(steps - 5, 0)])
                self.assertEqual(descriptor["type_id"], "jyppx.scheduler.sequential_lr/v1")
                self.assertEqual(descriptor["config"]["milestones"], [5])
                self.assertEqual([node["type_id"] for node in descriptor["children"]],
                    ["jyppx.scheduler.warmup_lr/v1", "jyppx.scheduler.linear_lr/v1"])
                for node in (descriptor, *descriptor["children"]):
                    self.assertEqual(node["step_mode"], 1)
                    self.assertEqual(node["optimizer_id"], "recipe.sgd")
                    self.assertEqual(node["optimizer_group_ids"], ["main"])
                self.assertEqual(descriptor["children"][0]["config"]["warmup_iters"], 5)
                self.assertEqual(descriptor["children"][1]["config"]["total_iters"], 43)
                self.assertEqual(saved["scheduler_rates"], saved["optimizer"]["learning_rates"])
                self.assertEqual(len(saved["scheduler_rates"]), 1)
                self.assertAlmostEqual(saved["scheduler_rates"][0], 0.01 * schedule_factor(steps), places=14)
        self.assertEqual(self.checkpoints["continuous"]["descriptor"], self.checkpoints["resumed"]["descriptor"])
        self.assertEqual(self.checkpoints["continuous"]["descriptor"], self.checkpoints["killed"]["descriptor"])

    def test_03_full_checkpoint_readers_provenance_and_configuration_binding(self) -> None:
        root = self.roots["resumed"]
        self.invoke("checkpoint", "verify", "--path", root / "final.jyck")
        self.invoke("model", "verify", "--path", root / "model.safetensors")
        self.invoke("model", "bundle", "verify", "--path", root / "bundle")
        provenance = json.loads((root / "bundle/metadata/provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["optimization"], self.results["resumed"]["optimization"])
        self.assertEqual(self.results["resumed"]["lineage"]["config_sha256"], self.preflight["config_sha256"])
        self.assertEqual(self.invoke("recipe", "resume", "--file", self.file, "--root", root)["data"], self.results["resumed"])
        changed = self.base / "changed.json"
        for patch in ({"warmup_updates": 6}, {"warmup_start_factor": 0.25}, {"lr_end_factor": 0.2}):
            changed.write_text(json.dumps(self.config | patch), encoding="utf-8")
            self.invoke("recipe", "resume", "--file", changed, "--root", root, expected=2)
        epoch = {key: value for key, value in self.config.items() if key not in SCHEDULE}
        epoch["lr_schedule"] = "epoch_step"
        changed.write_text(json.dumps(epoch), encoding="utf-8")
        self.invoke("recipe", "resume", "--file", changed, "--root", root, expected=2)


class TransferUpdateRecipe(RecipeCase):
    """Real fixed DINO weights; schedule/state-policy coverage, not task quality."""
    weights_root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="liuding-transfer-update-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name) / "迁移逐更新"
        cls.base.mkdir()
        for split in ("train", "validation"):
            for label, name in enumerate(("green", "red")):
                folder = cls.base / "images" / split / name
                folder.mkdir(parents=True)
                for number in range(2):
                    (folder / f"{number}.png").write_bytes(
                        png(label, number + (20 if split == "validation" else 0)))
        weights = cls.weights_root / "dinov2-small.liuding.safetensors"
        assert weights.is_file(), weights
        metadata = cls.base / "dinov2-small.weights.json"
        metadata.write_bytes((cls.repo / "recipes/transfer-learning/dinov2-small.weights.json").read_bytes())
        template = {
            "schema_version": 1, "recipe": "jyppx.liuding.imagefolder.transfer/v1",
            "train_root": "images/train", "validation_root": "images/validation",
            "data_license": "CC0-1.0; self-authored RGB fixture", "device": cls.device,
            "seed": 20261029, "epochs": 2, "batch_size": 2, "image_size": 224,
            "workers": 1, "prefetch_batches": 1, "learning_rate": 0.01,
            "momentum": 0.9, "weight_decay": 0.0001, "lr_gamma": 1.0,
            "checkpoint_every": 2, "minimum_accuracy": 0.5,
            "max_image_bytes": 1048576, "max_data_bytes": 10485760,
            "max_samples": 1024, "image_budget_bytes": 134217728,
            "model": "vit_small_patch14_dinov2", "weights_file": str(weights),
            "weights_metadata": metadata.name, "batch_norm": "eval",
            "head_learning_rate": 0.01,
        } | SCHEDULE | {"warmup_updates": 2}
        cls.cases = {}
        for mode in ("frozen", "finetune"):
            config = template | {"backbone_mode": mode}
            if mode == "finetune":
                config["backbone_learning_rate"] = 0.00005
            file = cls.base / f"{mode}.json"
            file.write_text(json.dumps(config), encoding="utf-8")
            roots = {name: cls.base / f"{mode}-{name}" for name in ("continuous", "resumed")}
            results, checkpoints = {}, {}
            preflight = cls.invoke("recipe", "check", "--file", file)["data"]
            results["continuous"] = cls.invoke(
                "recipe", "run", "--file", file, "--root", roots["continuous"], expected=(0, 4))["data"]
            for action, step in (("run", 1), ("resume", 3)):
                name = f"paused-{step}"
                results[name] = cls.invoke("recipe", action, "--file", file,
                    "--root", roots["resumed"], "--pause-after", step)["data"]
                checkpoints[name] = checkpoint(roots["resumed"] / "latest.jyck")
                save_evidence(f"{mode}-{name}", {"result": results[name], "checkpoint": checkpoints[name]})
            results["resumed"] = cls.invoke("recipe", "resume", "--file", file,
                "--root", roots["resumed"], expected=(0, 4))["data"]
            for name, root in roots.items():
                checkpoints[name] = checkpoint(root / "final.jyck")
                save_evidence(f"{mode}-{name}", {"result": results[name], "checkpoint": checkpoints[name]})
            cls.cases[mode] = {"config": config, "file": file, "roots": roots,
                              "preflight": preflight, "results": results, "checkpoints": checkpoints}
            save_evidence(f"{mode}-inputs", {"config": config, "preflight": preflight,
                "scope": "Actual LR groups, frozen/finetune state policy and complete recovery; not DINO task quality"})

    def test_01_actual_group_rates_and_scheduler_state_across_warmup(self) -> None:
        for mode, case in self.cases.items():
            config = case["config"]
            ids = ["backbone", "head"] if mode == "finetune" else ["head"]
            bases = [config[identity + "_learning_rate"] for identity in ids]
            self.assertEqual(case["preflight"]["total_steps"], 4)
            for name, result in case["results"].items():
                with self.subTest(mode=mode, route=name):
                    completed = int(name.removeprefix("paused-")) if name.startswith("paused-") else 4
                    self.assertEqual(result["global_step"], completed)
                    self.assertEqual(result["status"], "paused" if name.startswith("paused-") else "completed")
                    scheduler = result["optimization"]["scheduler"]
                    for field, value in {"scheduler_id": "recipe.lr", "kind": "warmup_linear",
                        "step_mode": "batch", "total_updates": 4, "warmup_updates": 2,
                        "warmup_start_factor": 0.5, "lr_end_factor": 0.1, "completed_steps": completed}.items():
                        self.assertEqual(scheduler[field], value, field)
                    groups = result["optimization"]["groups"]
                    self.assertEqual([group["group_id"] for group in groups], ids)
                    bindings = [parameter for group in groups for parameter in group["parameter_ids"]]
                    self.assertEqual(len(bindings), len(set(bindings)))
                    for group, base in zip(groups, bases):
                        self.assertTrue(group["parameter_ids"])
                        self.assertTrue(all(parameter.startswith(group["group_id"] + ".") for parameter in group["parameter_ids"]))
                        self.assertEqual(group["initial_learning_rate"], base)
                        self.assertEqual(group["first_learning_rate"], base * 0.5)
                        self.assertAlmostEqual(group["final_learning_rate"], base * schedule_factor(completed, 4, 2), places=14)
                    saved = case["checkpoints"][name]
                    optimizer, state = saved["optimizer"], saved["scheduler_state"]
                    self.assertEqual(optimizer["completed_steps"], completed)
                    self.assertEqual(optimizer["parameters"], len(bindings))
                    self.assertEqual(saved["scheduler_rates"], optimizer["learning_rates"])
                    self.assertEqual(len(optimizer["learning_rates"]), len(ids))
                    for actual, base in zip(optimizer["learning_rates"], bases):
                        self.assertAlmostEqual(actual, base * schedule_factor(completed, 4, 2), places=14)
                    if mode == "finetune":
                        self.assertAlmostEqual(optimizer["learning_rates"][1] / optimizer["learning_rates"][0], 200.0, places=10)
                    self.assertEqual(state["completed_steps"], completed)
                    self.assertEqual([node["completed_steps"] for node in state["children"]],
                                     [min(completed, 2), max(completed - 2, 0)])
                    descriptor = saved["descriptor"]
                    self.assertEqual(descriptor["type_id"], "jyppx.scheduler.sequential_lr/v1")
                    self.assertEqual(descriptor["config"]["milestones"], [2])
                    self.assertEqual([node["type_id"] for node in descriptor["children"]],
                                     ["jyppx.scheduler.warmup_lr/v1", "jyppx.scheduler.linear_lr/v1"])
                    for node in (descriptor, *descriptor["children"]):
                        self.assertEqual(node["step_mode"], 1)
                        self.assertEqual(node["optimizer_id"], "recipe.sgd")
                        self.assertEqual(node["optimizer_group_ids"], ids)

    def test_02_complete_recovery_frozen_policy_and_truthful_quality_result(self) -> None:
        for mode, case in self.cases.items():
            with self.subTest(mode=mode):
                continuous, resumed = case["results"]["continuous"], case["results"]["resumed"]
                for field in ("model_sha256", "next_rng_sha256", "next_cpu_rng_sha256", "accuracy",
                              "confusion", "training_metrics", "validation_metrics", "optimization",
                              "global_step", "optimizer_steps", "scheduler_steps", "completed_epochs", "transfer"):
                    self.assertEqual(continuous[field], resumed[field], field)
                for result in (continuous, resumed):
                    self.assertEqual(result["global_step"], 4)
                    self.assertEqual(result["optimizer_steps"], 4)
                    self.assertEqual(result["scheduler_steps"], 4)
                    self.assertEqual(result["completed_epochs"], 2)
                    self.assertEqual(result["quality_passed"], result["accuracy"] >= 0.5)
                    initial, final = result["transfer"]["initial"], result["transfer"]["final"]
                    self.assertNotEqual(initial["head_sha256"], final["head_sha256"])
                    if mode == "frozen":
                        self.assertEqual(initial["backbone_sha256"], final["backbone_sha256"])
                        self.assertEqual(initial["trainable_parameters"], (384 + 1) * 2)
                        self.assertGreater(initial["frozen_parameters"], 1000000)
                    else:
                        self.assertNotEqual(initial["backbone_sha256"], final["backbone_sha256"])
                        self.assertEqual(initial["frozen_parameters"], 0)
                baseline = case["checkpoints"]["continuous"]["manifest"]["participants"]
                self.assertEqual({item["state"]["participant_id"] for item in baseline}, PARTICIPANTS)
                self.assertEqual(case["checkpoints"]["resumed"]["manifest"]["participants"], baseline)
                for root in case["roots"].values():
                    self.assertEqual(file_sha256(root / "model.safetensors"), continuous["model_sha256"])

    def test_03_bundle_provenance_readers_and_changed_group_rejected(self) -> None:
        for mode, case in self.cases.items():
            with self.subTest(mode=mode):
                root = case["roots"]["resumed"]
                for arguments in (("checkpoint", "verify", "--path", root / "final.jyck"),
                                  ("model", "bundle", "verify", "--path", root / "bundle")):
                    self.invoke(*arguments)
                result = case["results"]["resumed"]
                self.assertEqual(self.invoke("recipe", "resume", "--file", case["file"],
                    "--root", root, expected=(0, 4))["data"], result)
                provenance = json.loads((root / "bundle/metadata/provenance.json").read_text(encoding="utf-8"))
                self.assertEqual(provenance["optimization"], result["optimization"])
                self.assertEqual(result["lineage"]["config_sha256"], case["preflight"]["config_sha256"])
                self.assertEqual(provenance["model"]["transfer"]["optimizer_groups"], [
                    {"group_id": group["group_id"], "learning_rate": group["initial_learning_rate"]}
                    for group in result["optimization"]["groups"]])
                changed = self.base / f"{mode}-changed.json"
                changed.write_text(json.dumps(case["config"] | {"head_learning_rate": 0.02}), encoding="utf-8")
                self.invoke("recipe", "resume", "--file", changed, "--root", root, expected=2)
                save_evidence(f"{mode}-provenance", provenance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--weights-root", type=Path,
                        default=os.environ.get("JYPPX_LIUDING_PRETRAINED_REFERENCE_DIR"))
    parser.add_argument("--transfer-update", action="store_true")
    arguments, rest = parser.parse_known_args()
    if arguments.transfer_update and arguments.weights_root is None:
        print("SKIP: set JYPPX_LIUDING_PRETRAINED_REFERENCE_DIR to approved local DINOv2-S weights")
        raise SystemExit(77)
    RecipeCase.cli = arguments.cli.resolve()
    RecipeCase.repo = Path(__file__).resolve().parents[2]
    RecipeCase.device = arguments.device
    TransferUpdateRecipe.weights_root = arguments.weights_root.resolve() if arguments.weights_root else None
    if output := os.environ.get("JYPPX_LIUDING_UPDATE_RECIPE_EVIDENCE_DIR"):
        EVIDENCE = Path(output).resolve() / arguments.device
        if arguments.transfer_update:
            EVIDENCE /= "transfer"
        EVIDENCE.mkdir(parents=True, exist_ok=False)
    unittest.main(argv=[__file__, *rest], defaultTest=(
        "TransferUpdateRecipe" if arguments.transfer_update else ("UpdateConfiguration", "UpdateRecipe")))
