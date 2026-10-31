# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Native AdamW configuration, persisted state and process recovery acceptance.

The existing 16/8-image, 48-update task retains its 0.875 accuracy gate.
This stdlib checker never implements product training or imports Python Torch.
It does not qualify pretrained DINO quality, numerical parity or performance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import unittest

_spec = importlib.util.spec_from_file_location(
    "_liuding_update_fixture", Path(__file__).with_name("test_update_recipe.py"))
assert _spec is not None and _spec.loader is not None
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)
EVIDENCE: Path | None = None
COMMAND = 0
OPTIONS = {"optimizer": "adamw", "adam_beta1": 0.9, "adam_beta2": 0.999,
           "adam_epsilon": 1e-8, "gradient_clip_norm": 1.0,
           "lr_schedule": "linear", "lr_gamma": 1.0, "weight_decay": 0.0}


def save(name: str, value: object) -> None:
    if EVIDENCE is not None:
        with (EVIDENCE / (name + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


def adamw_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if key != "momentum"} | OPTIONS


def checkpoint(path: Path) -> dict:
    """Independently decode small persisted control, seeking past tensor blobs."""
    wanted = {("recipe.lr", "topology"), ("recipe.lr", "control"), ("recipe.adamw", "control")}
    blobs = {}
    with path.open("rb") as stream:
        assert stream.read(8) == b"JYPPXCK1" and struct.unpack("<I", stream.read(4))[0] == 1
        size = struct.unpack("<Q", stream.read(8))[0]
        assert 0 < size <= 8 * 1024 * 1024
        manifest = json.loads(stream.read(size))
        for participant in manifest["participants"]:
            identity = participant["state"]["participant_id"]
            for blob in participant["blobs"]:
                size = blob["size_bytes"]
                assert isinstance(size, int) and 0 <= size <= path.stat().st_size - stream.tell()
                key = identity, blob["field_name"]
                if key in wanted:
                    assert size <= 65536 and key not in blobs
                    data = stream.read(size)
                    assert hashlib.sha256(data).hexdigest() == blob["digest"]["hex"]
                    blobs[key] = data
                else:
                    stream.seek(size, 1)
        assert stream.tell() + 32 == path.stat().st_size
    assert set(blobs) == wanted
    topology = fixture.Cursor(blobs[("recipe.lr", "topology")])
    assert topology.read(8) == b"JYPPXSLD" and topology.number("I") == 1
    assert topology.string() == "recipe.lr"
    descriptor = topology.descriptor()
    topology.finish()
    control = fixture.Cursor(blobs[("recipe.lr", "control")])
    assert control.read(8) == b"JYPPXSLC" and control.number("I") == 3
    scheduler = control.state()
    scheduler_rates = [control.number("d") for _ in range(control.count())]
    control.finish()
    optimizer = fixture.Cursor(blobs[("recipe.adamw", "control")])
    assert optimizer.read(8) == b"JYPPXAWC" and optimizer.number("I") == 1
    steps, groups, parameters = optimizer.number("Q"), optimizer.count(), optimizer.number("I")
    assert isinstance(parameters, int) and 0 < parameters <= 1000000
    rates = [optimizer.number("d") for _ in range(groups)]
    states = [(optimizer.number("B"), optimizer.number("q")) for _ in range(parameters)]
    optimizer.finish()
    return {"manifest": manifest, "descriptor": descriptor, "scheduler": scheduler,
            "scheduler_rates": scheduler_rates, "optimizer_rates": rates,
            "completed_steps": steps, "parameter_states": states}


class Case(unittest.TestCase):
    cli: Path
    repo: Path
    device: str

    @classmethod
    def invoke(cls, *args: object, expected: int | tuple[int, ...] = 0) -> dict:
        global COMMAND
        COMMAND += 1
        result = subprocess.run([str(cls.cli), *(str(value) for value in args), "--format", "json"],
            text=True, encoding="utf-8", capture_output=True, timeout=300, check=False,
            env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"))
        save(f"command-{COMMAND:03d}", {"arguments": [str(value) for value in args],
            "exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        accepted = (expected,) if isinstance(expected, int) else expected
        assert result.returncode in accepted, (args, result.returncode, result.stdout, result.stderr)
        envelope = json.loads(result.stdout)
        assert envelope["schema_version"] == 1
        if accepted == (0, 4):
            assert result.returncode == (0 if envelope["data"]["quality_passed"] else 4)
        return envelope

    @classmethod
    def setup_fixture(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="liuding-adamw-recipe-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name) / "AdamW配方"
        cls.base.mkdir()
        cls.config = fixture.configuration(cls.base, cls.device)


class Configuration(Case):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setup_fixture()
        cls.cases = {"image": cls.config}
        weights = cls.base / "not-model-weights.safetensors"
        weights.write_bytes(b"Byte preflight only; not executable model weights")
        descriptor = json.loads((cls.repo / "recipes/transfer-learning/resnet50-a1.weights.json").read_text())
        descriptor.update(sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), size_bytes=weights.stat().st_size)
        metadata = cls.base / "weights.json"
        metadata.write_text(json.dumps(descriptor), encoding="utf-8")
        cls.cases["transfer"] = cls.config | {"recipe": "jyppx.liuding.imagefolder.transfer/v1",
            "model": "resnet50", "image_size": 224, "weights_file": weights.name,
            "weights_metadata": metadata.name, "backbone_mode": "frozen", "batch_norm": "eval",
            "image_budget_bytes": 134217728}

    def test_01_schema_and_explicit_adamw_configuration(self) -> None:
        for name, config in self.cases.items():
            with self.subTest(recipe=name):
                tail = ("--name", "transfer") if name == "transfer" else ()
                schema = self.invoke("recipe", "schema", *tail)["data"]["root"]["fields"]
                for field in ("optimizer", "momentum", "adam_beta1", "adam_beta2", "adam_epsilon", "gradient_clip_norm"):
                    self.assertFalse(schema[field]["required"])
                path = self.base / (name + ".json")
                path.write_text(json.dumps(config), encoding="utf-8")
                original = self.invoke("recipe", "check", "--file", path)["data"]
                self.assertNotIn("optimizer", original["config"])
                for options in (OPTIONS, OPTIONS | {"adam_beta1": 0.0, "adam_beta2": 0.0},
                                OPTIONS | {"lr_schedule": "epoch_step"}):
                    candidate = adamw_config(config) | options
                    path.write_text(json.dumps(candidate), encoding="utf-8")
                    checked = self.invoke("recipe", "check", "--file", path)["data"]
                    self.assertNotEqual(checked["config_sha256"], original["config_sha256"])
                    self.assertEqual(checked["config"]["optimizer"], "adamw")
                    self.assertNotIn("momentum", checked["config"])

    def test_02_invalid_optimizer_options_rejected_before_run_root(self) -> None:
        invalid = [{"optimizer": value} for value in ("auto", "adam", True, 1, None)]
        invalid += [{key: value} for key in ("adam_beta1", "adam_beta2")
                    for value in (-0.1, 1.0, True, "0.9", None, float("nan"), float("inf"))]
        invalid += [{key: value} for key in ("adam_epsilon", "gradient_clip_norm")
                    for value in (0, -0.1, True, "1.0", None, float("nan"), float("inf"))]
        invalid += [{"momentum": 0.9}, {"lr_gamma": 0.9}, {"warmup_updates": 1},
                    {"lr_end_factor": 0.0}, {"warmup_start_factor": 0.5}]
        for name, config in self.cases.items():
            base = adamw_config(config)
            candidates = [base | patch for patch in invalid]
            candidates += [{key: value for key, value in base.items() if key != missing}
                           for missing in ("adam_beta1", "adam_beta2", "adam_epsilon")]
            candidates += [config | {"adam_beta1": 0.9},
                           {key: value for key, value in config.items() if key != "momentum"}]
            path, absent = self.base / (name + "-bad.json"), self.base / (name + "-rejected-run")
            for index, candidate in enumerate(candidates):
                with self.subTest(recipe=name, case=index):
                    path.write_text(json.dumps(candidate), encoding="utf-8")
                    self.invoke("recipe", "run", "--file", path, "--root", absent, expected=2)
                    self.assertFalse(absent.exists())


class Recovery(Case):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setup_fixture()
        cls.config = adamw_config(cls.config) | {"learning_rate": 0.001}
        cls.file = cls.base / "config.json"
        cls.file.write_text(json.dumps(cls.config), encoding="utf-8")
        save("configuration", cls.config)
        cls.results, cls.saved = {}, {}
        cls.roots = {name: cls.base / name for name in ("continuous", "resumed", "killed")}
        cls.results["continuous"] = cls.invoke("recipe", "run", "--file", cls.file,
            "--root", cls.roots["continuous"], expected=(0, 4))["data"]
        for action, step in (("run", 3), ("resume", 7)):
            label = f"paused-{step}"
            cls.results[label] = cls.invoke("recipe", action, "--file", cls.file,
                "--root", cls.roots["resumed"], "--pause-after", step)["data"]
            cls.saved[label] = checkpoint(cls.roots["resumed"] / "latest.jyck")
            save(label + "-checkpoint", cls.saved[label])
        cls.results["resumed"] = cls.invoke("recipe", "resume", "--file", cls.file,
            "--root", cls.roots["resumed"], expected=(0, 4))["data"]
        root = cls.roots["killed"]
        process = subprocess.Popen([str(cls.cli), "recipe", "run", "--file", str(cls.file),
            "--root", str(root), "--format", "json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"))
        try:
            deadline = time.monotonic() + 120
            while not (root / "latest.jyck").is_file():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise AssertionError("No durable checkpoint before native process exit/timeout")
                time.sleep(0.01)
            assert process.poll() is None and not (root / "final.jyck").exists()
            process.kill()
            stdout, stderr = process.communicate(timeout=30)
            save("killed-process", {"exit": process.returncode, "stdout": stdout.decode("utf-8"),
                                    "stderr": stderr.decode("utf-8")})
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=30)
        cls.saved["killed-durable"] = checkpoint(root / "latest.jyck")
        save("killed-durable-checkpoint", cls.saved["killed-durable"])
        cls.results["killed"] = cls.invoke("recipe", "resume", "--file", cls.file,
            "--root", root, expected=(0, 4))["data"]
        for name, root in cls.roots.items():
            cls.saved[name] = checkpoint(root / "final.jyck")
            save(name + "-checkpoint", cls.saved[name])
            save(name + "-result", cls.results[name])

    def test_01_quality_and_complete_cross_process_recovery(self) -> None:
        expected = self.results["continuous"]
        participants = (set(fixture.PARTICIPANTS) - {"recipe.sgd"}) | {"recipe.adamw"}
        fields = ("model_sha256", "next_rng_sha256", "next_cpu_rng_sha256", "training_metrics",
            "validation_metrics", "accuracy", "confusion", "global_step", "optimizer_steps",
            "scheduler_steps", "completed_epochs", "optimization", "sampling")
        for name in self.roots:
            result = self.results[name]
            self.assertTrue(result["quality_passed"])
            self.assertGreaterEqual(result["accuracy"], .875)
            for field in fields:
                self.assertEqual(result[field], expected[field], (name, field))
            for field in ("global_step", "optimizer_steps", "scheduler_steps"):
                self.assertEqual(result[field], 48)
            self.assertEqual(result["completed_epochs"], 12)
            items = self.saved[name]["manifest"]["participants"]
            self.assertEqual({item["state"]["participant_id"] for item in items}, participants)
            self.assertEqual(items, self.saved["continuous"]["manifest"]["participants"])
            self.assertEqual(fixture.file_sha256(self.roots[name] / "model.safetensors"), result["model_sha256"])

    def test_02_actual_adamw_state_and_linear_endpoint(self) -> None:
        for name, saved in self.saved.items():
            with self.subTest(checkpoint=name):
                steps = saved["completed_steps"]
                if name.startswith("paused-"):
                    self.assertEqual(steps, int(name.removeprefix("paused-")))
                elif name == "killed-durable":
                    self.assertGreaterEqual(steps, 16)
                    self.assertLess(steps, 48)
                else:
                    self.assertEqual(steps, 48)
                self.assertTrue(saved["parameter_states"])
                self.assertTrue(all(present == 1 and count == steps for present, count in saved["parameter_states"]))
                descriptor = saved["descriptor"]
                self.assertEqual(descriptor["type_id"], "jyppx.scheduler.polynomial_lr/v1")
                self.assertEqual(descriptor["optimizer_id"], "recipe.adamw")
                self.assertEqual(descriptor["step_mode"], 1)
                self.assertEqual(descriptor["config"]["power"], 1.0)
                self.assertEqual(descriptor["config"]["total_iters"], 48)
                self.assertEqual(saved["scheduler"]["completed_steps"], steps)
                self.assertEqual(saved["scheduler_rates"], saved["optimizer_rates"])
                self.assertEqual(len(saved["optimizer_rates"]), 1)
                self.assertTrue(math.isclose(saved["optimizer_rates"][0], .001 * (48 - steps) / 48,
                                             rel_tol=1e-12, abs_tol=1e-15))

    def test_03_provenance_and_changed_optimizer_rejected_without_mutation(self) -> None:
        root = self.roots["resumed"]
        for args in (("checkpoint", "verify", "--path", root / "final.jyck"),
                     ("model", "bundle", "verify", "--path", root / "bundle")):
            self.invoke(*args)
        provenance = json.loads((root / "bundle/metadata/provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["optimization"], self.results["resumed"]["optimization"])
        save("bundle-provenance", provenance)
        checkpoint_sha = fixture.file_sha256(root / "final.jyck")
        path = self.base / "changed.json"
        for patch in ({"adam_beta1": .8}, {"adam_beta2": .99}, {"adam_epsilon": 1e-7},
                      {"gradient_clip_norm": .5}, {"lr_schedule": "epoch_step"}):
            path.write_text(json.dumps(self.config | patch), encoding="utf-8")
            self.invoke("recipe", "resume", "--file", path, "--root", root, expected=2)
            self.assertEqual(fixture.file_sha256(root / "final.jyck"), checkpoint_sha)


class TransferRecovery(Case):
    """Four actual DINO updates qualify wiring/recovery, not task accuracy."""
    weights_root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="liuding-transfer-adamw-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name) / "AdamW迁移恢复"
        cls.base.mkdir()
        for split in ("train", "validation"):
            for label, name in enumerate(("green", "red")):
                folder = cls.base / "images" / split / name
                folder.mkdir(parents=True)
                for number in range(2):
                    (folder / f"{number}.png").write_bytes(
                        fixture.png(label, number + (20 if split == "validation" else 0)))
        weights = cls.weights_root / "dinov2-small.liuding.safetensors"
        assert weights.is_file(), weights
        metadata = cls.base / "dinov2-small.weights.json"
        metadata.write_bytes((cls.repo / "recipes/transfer-learning/dinov2-small.weights.json").read_bytes())
        template = {
            "schema_version": 1, "recipe": "jyppx.liuding.imagefolder.transfer/v1",
            "train_root": "images/train", "validation_root": "images/validation",
            "data_license": "CC0-1.0; self-authored RGB fixture", "device": cls.device,
            "seed": 20261029, "epochs": 2, "batch_size": 2, "image_size": 224,
            "workers": 1, "prefetch_batches": 1, "learning_rate": 0.001,
            "checkpoint_every": 2, "minimum_accuracy": 0.5,
            "max_image_bytes": 1048576, "max_data_bytes": 10485760,
            "max_samples": 1024, "image_budget_bytes": 134217728,
            "model": "vit_small_patch14_dinov2", "weights_file": str(weights),
            "weights_metadata": metadata.name, "batch_norm": "eval",
            "head_learning_rate": 0.001,
        } | OPTIONS
        cls.cases = {}
        for mode in ("frozen", "finetune"):
            config = template | {"backbone_mode": mode}
            if mode == "finetune":
                config["backbone_learning_rate"] = 0.00005
            file = cls.base / f"{mode}.json"
            file.write_text(json.dumps(config), encoding="utf-8")
            roots = {name: cls.base / f"{mode}-{name}" for name in ("continuous", "resumed")}
            results, saved = {}, {}
            cls.invoke("recipe", "check", "--file", file)
            results["continuous"] = cls.invoke("recipe", "run", "--file", file,
                "--root", roots["continuous"], expected=(0, 4))["data"]
            for action, step in (("run", 1), ("resume", 3)):
                name = f"paused-{step}"
                results[name] = cls.invoke("recipe", action, "--file", file,
                    "--root", roots["resumed"], "--pause-after", step)["data"]
                saved[name] = checkpoint(roots["resumed"] / "latest.jyck")
                save(f"{mode}-{name}", {"result": results[name], "checkpoint": saved[name]})
            results["resumed"] = cls.invoke("recipe", "resume", "--file", file,
                "--root", roots["resumed"], expected=(0, 4))["data"]
            for name, root in roots.items():
                saved[name] = checkpoint(root / "final.jyck")
                save(f"{mode}-{name}", {"result": results[name], "checkpoint": saved[name]})
            cls.cases[mode] = {"config": config, "file": file, "roots": roots,
                               "results": results, "saved": saved}
            save(mode + "-configuration", config)

    def test_01_actual_group_rates_and_adamw_states(self) -> None:
        for mode, case in self.cases.items():
            ids = ["backbone", "head"] if mode == "finetune" else ["head"]
            for name, result in case["results"].items():
                with self.subTest(mode=mode, checkpoint=name):
                    completed = int(name.removeprefix("paused-")) if name.startswith("paused-") else 4
                    self.assertEqual(result["global_step"], completed)
                    optimization, saved = result["optimization"], case["saved"][name]
                    self.assertEqual(optimization["optimizer_id"], "recipe.adamw")
                    self.assertEqual(optimization["gradient_clip_norm"], 1.0)
                    self.assertEqual(optimization["adamw"], {"beta1": .9, "beta2": .999,
                        "epsilon": 1e-8, "weight_decay": 0.0, "amsgrad": False})
                    groups = optimization["groups"]
                    self.assertEqual([group["group_id"] for group in groups], ids)
                    names = [parameter for group in groups for parameter in group["parameter_ids"]]
                    self.assertEqual(len(names), len(set(names)))
                    self.assertEqual(saved["completed_steps"], completed)
                    self.assertEqual(saved["scheduler"]["completed_steps"], completed)
                    self.assertEqual(saved["descriptor"]["optimizer_group_ids"], ids)
                    self.assertEqual(saved["descriptor"]["config"]["total_iters"], 4)
                    self.assertEqual(saved["scheduler_rates"], saved["optimizer_rates"])
                    self.assertEqual(len(saved["optimizer_rates"]), len(ids))
                    for group, rate in zip(groups, saved["optimizer_rates"]):
                        initial = case["config"][group["group_id"] + "_learning_rate"]
                        self.assertEqual(group["initial_learning_rate"], initial)
                        self.assertEqual(group["first_learning_rate"], initial)
                        self.assertEqual(group["final_learning_rate"], rate)
                        self.assertTrue(math.isclose(rate, initial * (4 - completed) / 4,
                                                     rel_tol=1e-12, abs_tol=1e-15))
                    self.assertEqual(len(saved["parameter_states"]), len(names))
                    missing = []
                    for parameter, (present, steps) in zip(names, saved["parameter_states"]):
                        if present:
                            self.assertEqual((present, steps), (1, completed))
                        else:
                            self.assertEqual(steps, 0)
                            missing.append(parameter)
                    # The native DINOv2 classification topology has no masked-image parameter.
                    self.assertEqual(len(names), 176 if mode == "finetune" else 2)
                    self.assertFalse(any(parameter.endswith("mask_token") for parameter in names))
                    self.assertEqual(missing, [])

    def test_02_complete_recovery_state_policy_and_provenance(self) -> None:
        participants = (set(fixture.PARTICIPANTS) - {"recipe.sgd"}) | {"recipe.adamw"}
        for mode, case in self.cases.items():
            with self.subTest(mode=mode):
                continuous, resumed = case["results"]["continuous"], case["results"]["resumed"]
                for field in ("model_sha256", "next_rng_sha256", "next_cpu_rng_sha256", "accuracy",
                              "confusion", "training_metrics", "validation_metrics", "optimization",
                              "global_step", "optimizer_steps", "scheduler_steps", "completed_epochs", "transfer", "sampling"):
                    self.assertEqual(continuous[field], resumed[field], field)
                baseline = case["saved"]["continuous"]["manifest"]["participants"]
                self.assertEqual({item["state"]["participant_id"] for item in baseline}, participants)
                self.assertEqual(case["saved"]["resumed"]["manifest"]["participants"], baseline)
                for name, result in (("continuous", continuous), ("resumed", resumed)):
                    self.assertEqual(result["quality_passed"], result["accuracy"] >= .5)
                    initial, final = result["transfer"]["initial"], result["transfer"]["final"]
                    self.assertNotEqual(initial["head_sha256"], final["head_sha256"])
                    self.assertEqual(initial["backbone_sha256"] == final["backbone_sha256"], mode == "frozen")
                    self.assertEqual(fixture.file_sha256(case["roots"][name] / "model.safetensors"), result["model_sha256"])
                root = case["roots"]["resumed"]
                self.invoke("checkpoint", "verify", "--path", root / "final.jyck")
                self.invoke("model", "bundle", "verify", "--path", root / "bundle")
                provenance = json.loads((root / "bundle/metadata/provenance.json").read_text(encoding="utf-8"))
                self.assertEqual(provenance["optimization"], resumed["optimization"])
                self.assertEqual(provenance["model"]["sampling"], resumed["sampling"])
                save(mode + "-provenance", provenance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--contracts-only", action="store_true")
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--weights-root", type=Path,
                        default=os.environ.get("JYPPX_LIUDING_PRETRAINED_REFERENCE_DIR"))
    args, rest = parser.parse_known_args()
    if args.transfer and args.weights_root is None:
        print("SKIP: set JYPPX_LIUDING_PRETRAINED_REFERENCE_DIR to approved local DINOv2-S weights")
        raise SystemExit(77)
    if args.contracts_only and args.transfer:
        parser.error("--contracts-only and --transfer are mutually exclusive")
    Case.cli, Case.repo, Case.device = args.cli.resolve(), Path(__file__).resolve().parents[2], args.device
    TransferRecovery.weights_root = args.weights_root.resolve() if args.weights_root else None
    if output := os.environ.get("JYPPX_LIUDING_ADAMW_RECIPE_EVIDENCE_DIR"):
        EVIDENCE = Path(output).resolve() / args.device
        if args.transfer:
            EVIDENCE /= "transfer"
        EVIDENCE.mkdir(parents=True, exist_ok=False)
    unittest.main(argv=[__file__, *rest], defaultTest=("Configuration" if args.contracts_only else
                  "TransferRecovery" if args.transfer else ("Configuration", "Recovery")))
