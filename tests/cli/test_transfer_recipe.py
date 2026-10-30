# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Pinned external-weight transfer through Trainer, including process recovery.

No network or Python Torch. The four-image color fixture checks execution,
parameter/buffer policy and complete recovery, not real-dataset qualification.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import unittest

from test_image_recipe import png


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def batch_norm_counters(path: Path, prefix: str = "") -> dict[str, int]:
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        result = {}
        for name, entry in header.items():
            if name.startswith(prefix) and name.endswith("num_batches_tracked"):
                assert entry["dtype"] == "I64" and entry["shape"] == []
                stream.seek(8 + size + entry["data_offsets"][0])
                result[name.removeprefix(prefix)] = struct.unpack("<q", stream.read(8))[0]
        return result


def checkpoint_sgd_control(path: Path) -> dict:
    """Read actual persisted rates, independently of the recipe's JSON report."""
    with path.open("rb") as stream:
        assert stream.read(8) == b"JYPPXCK1"
        assert struct.unpack("<I", stream.read(4))[0] == 1
        size = struct.unpack("<Q", stream.read(8))[0]
        assert size < 8 * 1024 * 1024
        manifest = json.loads(stream.read(size))
        for participant in manifest["participants"]:
            for blob in participant["blobs"]:
                if participant["state"]["participant_id"] == "recipe.sgd" and blob["field_name"] == "control":
                    assert blob["size_bytes"] < 65536
                    data = stream.read(blob["size_bytes"])
                    assert hashlib.sha256(data).hexdigest() == blob["digest"]["hex"]
                    magic, version, steps, groups, parameters = struct.unpack_from("<8sIQII", data)
                    assert magic == b"JYPPXSGC" and version == 1
                    assert len(data) == 28 + 8 * groups + parameters
                    return {"completed_steps": steps, "parameters": parameters,
                            "learning_rates": list(struct.unpack_from("<" + "d" * groups, data, 28))}
                stream.seek(blob["size_bytes"], 1)
    raise AssertionError("Checkpoint has no SGD control")


class TransferRecipe(unittest.TestCase):
    @classmethod
    def invoke(cls, *arguments: object, expected: int | tuple[int, ...] = 0) -> dict:
        result = subprocess.run(
            [str(cls.cli), *(str(value) for value in arguments), "--format", "json"],
            capture_output=True, text=True, encoding="utf-8", timeout=300,
            env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"), check=False,
        )
        accepted = (expected,) if isinstance(expected, int) else expected
        if result.returncode not in accepted:
            raise AssertionError(f"{arguments!r}: exit {result.returncode}, expected {expected}"
                                 f"\n{result.stdout}\n{result.stderr}")
        envelope = json.loads(result.stdout)
        if accepted == (0, 4):
            assert result.returncode == (0 if envelope["data"]["quality_passed"] else 4)
        return envelope

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="liuding-transfer-recipe-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name) / "迁移学习"
        cls.base.mkdir()
        for split in ("train", "validation"):
            for label, name in enumerate(("green", "red")):
                folder = cls.base / "images" / split / name
                folder.mkdir(parents=True)
                for number in range(2):
                    (folder / f"{number}.png").write_bytes(png(label, number + (20 if split == "validation" else 0)))
        cls.schema = cls.invoke("recipe", "schema", "--name", "transfer")["data"]
        cls.config = {
            "schema_version": 1, "recipe": "jyppx.liuding.imagefolder.transfer/v1",
            "train_root": "images/train", "validation_root": "images/validation",
            "data_license": "CC0-1.0; self-authored RGB fixture", "device": cls.device,
            "seed": 20261029, "epochs": 2, "batch_size": 2, "image_size": 224,
            "workers": 1, "prefetch_batches": 1, "learning_rate": 0.01,
            "momentum": 0.9, "weight_decay": 0.0001, "lr_gamma": 0.9,
            "checkpoint_every": 2, "minimum_accuracy": 0.5,
            "max_image_bytes": 1048576, "max_data_bytes": 10485760,
            "max_samples": 1024, "image_budget_bytes": 134217728,
            "backbone_mode": "frozen", "batch_norm": "eval",
        }
        cls.cases = {}
        for model, filename, descriptor in (
            ("resnet50", "resnet50.a1_in1k.safetensors", "resnet50-a1.weights.json"),
            ("vit_small_patch14_dinov2", "dinov2-small.liuding.safetensors", "dinov2-small.weights.json"),
        ):
            weights = cls.weights_root / filename
            assert weights.is_file(), weights
            metadata = cls.base / descriptor
            metadata.write_bytes((cls.repo / "recipes/transfer-learning" / descriptor).read_bytes())
            for mode in ("frozen", "finetune"):
                name = model + "-" + mode
                config = cls.config | {
                    "model": model, "weights_file": str(weights),
                    "weights_metadata": descriptor, "backbone_mode": mode,
                    "learning_rate": 0.01 if mode == "frozen" else 0.0001,
                }
                if cls.group_rates:
                    config["head_learning_rate"] = 0.01
                    if mode == "frozen":
                        config["learning_rate"] = 0.002
                    elif model == "vit_small_patch14_dinov2":
                        config["backbone_learning_rate"] = 0.00005
                config_path = cls.base / (name + ".json")
                config_path.write_text(json.dumps(config), encoding="utf-8")
                preflight = cls.invoke("recipe", "check", "--file", config_path)["data"]
                roots = {kind: cls.base / (name + "-" + kind)
                         for kind in ("continuous", "resumed", "killed")}
                continuous = cls.invoke("recipe", "run", "--file", config_path,
                                        "--root", roots["continuous"])["data"]
                pause = cls.invoke("recipe", "run", "--file", config_path,
                                   "--root", roots["resumed"], "--pause-after", 1)["data"]
                paused_optimizer = checkpoint_sgd_control(roots["resumed"] / "latest.jyck")
                if cls.group_rates:
                    changed_rate = cls.base / (name + "-changed-paused-rate.json")
                    changed_rate.write_text(json.dumps(config | {"head_learning_rate": 0.02}), encoding="utf-8")
                    cls.invoke("recipe", "resume", "--file", changed_rate,
                               "--root", roots["resumed"], expected=2)
                resumed = cls.invoke("recipe", "resume", "--file", config_path,
                                     "--root", roots["resumed"])["data"]
                with (cls.base / (name + "-killed.log")).open("wb") as log:
                    process = subprocess.Popen(
                        [str(cls.cli), "recipe", "run", "--file", str(config_path),
                         "--root", str(roots["killed"]), "--format", "json"],
                        stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"),
                    )
                    try:
                        deadline = time.monotonic() + 240
                        while not (roots["killed"] / "latest.jyck").is_file():
                            if process.poll() is not None or time.monotonic() > deadline:
                                raise AssertionError("No durable checkpoint before exit/timeout")
                            time.sleep(0.01)
                        process.kill()
                        process.wait(timeout=30)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=30)
                killed = cls.invoke("recipe", "resume", "--file", config_path,
                                    "--root", roots["killed"])["data"]
                checkpoints = {
                    kind: cls.invoke("checkpoint", "inspect", "--path", root / "final.jyck")["data"]["manifest"]
                    for kind, root in roots.items()
                }
                cls.cases[name] = {
                    "config": config, "file": config_path, "metadata": metadata,
                    "preflight": preflight, "roots": roots, "pause": pause,
                    "continuous": continuous, "resumed": resumed, "killed": killed,
                    "checkpoints": checkpoints,
                    "optimizer_controls": {"paused": paused_optimizer} | {
                        route: checkpoint_sgd_control(root / "final.jyck")
                        for route, root in roots.items()
                    },
                }
                if cls.evidence:
                    target = cls.evidence / cls.device / name
                    target.mkdir(parents=True, exist_ok=False)
                    for key in ("config", "preflight", "pause", "continuous", "resumed", "killed", "checkpoints", "optimizer_controls"):
                        (target / (key + ".json")).write_text(json.dumps(cls.cases[name][key], indent=2) + "\n", encoding="utf-8")
                    for filename in ("lineage.json", "diagnostics.json"):
                        (target / filename).write_bytes((roots["resumed"] / filename).read_bytes())
                    (target / "bundle-manifest.json").write_bytes((roots["resumed"] / "bundle/manifest.json").read_bytes())

    def test_01_complete_process_recovery_and_readers(self) -> None:
        keys = ("model_sha256", "next_rng_sha256", "next_cpu_rng_sha256", "training_metrics",
                "validation_metrics", "accuracy", "confusion", "global_step", "optimizer_steps",
                "scheduler_steps", "completed_epochs", "transfer")
        if self.group_rates:
            keys += ("optimization",)
        for name, case in self.cases.items():
            with self.subTest(name=name):
                self.assertEqual(case["pause"]["global_step"], 1)
                self.assertEqual(case["preflight"]["total_steps"], 4)
                for route in ("continuous", "resumed", "killed"):
                    result = case[route]
                    self.assertEqual(result["global_step"], 4)
                    self.assertTrue(result["quality_passed"])
                    self.assertGreaterEqual(result["accuracy"], 0.5)
                    for key in keys:
                        self.assertEqual(result[key], case["continuous"][key], (route, key))
                    participants = case["checkpoints"][route]["participants"]
                    self.assertEqual(len(participants), 10)
                    self.assertEqual(participants, case["checkpoints"]["continuous"]["participants"])
                    root = case["roots"][route]
                    self.assertEqual(digest(root / "model.safetensors"), result["model_sha256"])
                    for args in (("model", "verify", "--path", root / "model.safetensors"),
                                 ("model", "bundle", "verify", "--path", root / "bundle"),
                                 ("checkpoint", "verify", "--path", root / "final.jyck"),
                                 ("run", "verify", "--root", root / "run-store")):
                        self.invoke(*args)
                self.assertEqual(self.invoke("recipe", "resume", "--file", case["file"],
                                             "--root", case["roots"]["resumed"])["data"], case["resumed"])

    def test_02_frozen_and_finetuned_parameter_and_buffer_policy(self) -> None:
        for name, case in self.cases.items():
            with self.subTest(name=name):
                state = case["continuous"]["transfer"]
                initial, final = state["initial"], state["final"]
                self.assertNotEqual(initial["head_sha256"], final["head_sha256"])
                if case["config"]["backbone_mode"] == "frozen":
                    self.assertEqual(initial["backbone_sha256"], final["backbone_sha256"])
                    features = 2048 if case["config"]["model"] == "resnet50" else 384
                    self.assertEqual(initial["trainable_parameters"], (features + 1) * 2)
                    self.assertGreater(initial["frozen_parameters"], 1000000)
                else:
                    self.assertNotEqual(initial["backbone_sha256"], final["backbone_sha256"])
                    self.assertEqual(initial["frozen_parameters"], 0)
                if case["config"]["model"] == "resnet50":
                    counters = batch_norm_counters(Path(case["config"]["weights_file"]))
                    self.assertEqual(len(counters), 53)
                    self.assertEqual(counters, batch_norm_counters(
                        case["roots"]["continuous"] / "model.safetensors", "buffer.backbone."))

    def test_03_closed_config_descriptor_and_lineage(self) -> None:
        case = self.cases["resnet50-frozen"]
        bad = self.base / "invalid.json"
        for patch in ({"surprise": 1}, {"backbone_mode": "auto"}, {"batch_norm": "auto"},
                      {"batch_norm": "train"}, {"model": "resnet18"}, {"image_size": 32}):
            bad.write_text(json.dumps(case["config"] | patch), encoding="utf-8")
            self.invoke("recipe", "check", "--file", bad, expected=2)
        original = case["metadata"].read_bytes()
        try:
            descriptor = json.loads(original)
            for patch in ({"sha256": "0" * 64}, {"size_bytes": 1}, {"surprise": 1},
                          {"source_num_classes": -1}, {"license_spdx": "NOASSERTION"},
                          {"schema_version": True}, {"position_antialias": "false"},
                          {"image": descriptor["image"] | {"mean": [0, 0]}},
                          {"image": descriptor["image"] | {"std": [0, 1, 1]}}):
                case["metadata"].write_text(json.dumps(descriptor | patch), encoding="utf-8")
                self.invoke("recipe", "check", "--file", case["file"], expected=2)
            # A still-valid change in provenance must invalidate even completed resume.
            changed = copy.deepcopy(descriptor)
            changed["source_revision"] = "different-revision"
            case["metadata"].write_text(json.dumps(changed), encoding="utf-8")
            self.invoke("recipe", "resume", "--file", case["file"],
                        "--root", case["roots"]["resumed"], expected=2)
            case["metadata"].write_bytes(original[:-2] + b', "id": "duplicate"}')
            self.invoke("recipe", "check", "--file", case["file"], expected=2)
        finally:
            case["metadata"].write_bytes(original)

    def test_04_explicit_training_batch_norm_updates(self) -> None:
        case = self.cases["resnet50-finetune"]
        config = self.base / "batch-norm-train.json"
        options = case["config"] | {"batch_norm": "train"}
        if self.group_rates:
            options.pop("head_learning_rate", None)
            options["backbone_learning_rate"] = 0.00005
        config.write_text(json.dumps(options), encoding="utf-8")
        root = self.base / "batch-norm-train"
        # This extra case checks BN state transitions, not convergence. Preserve
        # its 0.5 quality threshold and assert truthful quality-failure reporting.
        result = self.invoke("recipe", "run", "--file", config, "--root", root,
                             expected=(0, 4))["data"]
        self.assertEqual(result["quality_passed"], result["accuracy"] >= 0.5)
        if self.group_rates:
            groups = {group["group_id"]: group for group in result["optimization"]["groups"]}
            self.assertEqual(groups["head"]["initial_learning_rate"], options["learning_rate"])
            self.assertEqual(groups["backbone"]["initial_learning_rate"], options["backbone_learning_rate"])
        before = batch_norm_counters(Path(case["config"]["weights_file"]))
        after = batch_norm_counters(root / "model.safetensors", "buffer.backbone.")
        self.assertEqual(set(before), set(after))
        for name in before:
            self.assertEqual(after[name], before[name] + result["global_step"])
        if self.evidence:
            target = self.evidence / self.device / "batch-norm-train.json"
            target.write_text(json.dumps({"result": result, "before": before, "after": after,
                                          "scope": "BN state policy; task quality not qualified"},
                                         indent=2) + "\n", encoding="utf-8")

    def test_05_schema_and_delivery_describe_actual_preprocessing(self) -> None:
        self.assertEqual(self.schema["schema_id"], self.config["recipe"])
        for name in ("model", "weights_file", "weights_metadata", "backbone_mode", "batch_norm"):
            self.assertIn(name, self.schema["root"]["fields"])
        for case in self.cases.values():
            root = case["roots"]["resumed"]
            pixels = json.loads((root / "bundle/metadata/preprocess.json").read_text(encoding="utf-8"))
            expected = json.loads(case["metadata"].read_text(encoding="utf-8"))["image"]
            self.assertEqual(pixels, expected | {"schema_version": 1})
            for resource in case["resumed"]["image_resource_payload"].values():
                self.assertLessEqual(resource["peak"], resource["limit"])

    def test_06_groups_actual_checkpoint_rates_and_epoch_decay(self) -> None:
        for name, case in self.cases.items():
            with self.subTest(name=name):
                config = case["config"]
                if not self.group_rates:
                    controls = case["optimizer_controls"]
                    self.assertEqual(controls["paused"]["completed_steps"], 1)
                    self.assertEqual(controls["paused"]["learning_rates"], [config["learning_rate"]])
                    for route in ("continuous", "resumed", "killed"):
                        self.assertNotIn("optimization", case[route])
                        self.assertEqual(controls[route]["completed_steps"], 4)
                        self.assertAlmostEqual(controls[route]["learning_rates"][0],
                                               config["learning_rate"] * config["lr_gamma"] ** config["epochs"], places=14)
                        self.assertEqual(len(controls[route]["learning_rates"]), 1)
                    self.assertNotIn("optimization", case["pause"])
                    provenance = json.loads((case["roots"]["resumed"] / "bundle/metadata/provenance.json").read_text(encoding="utf-8"))
                    self.assertNotIn("optimization", provenance)
                    self.assertNotIn("optimizer_groups", provenance["model"]["transfer"])
                    continue
                expected_ids = (["backbone", "head"] if config["backbone_mode"] == "finetune"
                                else ["head"]) if self.group_rates else ["main"]
                groups = case["continuous"]["optimization"]["groups"]
                self.assertEqual([group["group_id"] for group in groups], expected_ids)
                bindings = [parameter for group in groups for parameter in group["parameter_ids"]]
                self.assertEqual(len(bindings), len(set(bindings)))
                with (case["roots"]["continuous"] / "model.safetensors").open("rb") as stream:
                    size = struct.unpack("<Q", stream.read(8))[0]
                    header = json.loads(stream.read(size))
                expected_parameters = {key.removeprefix("parameter.") for key in header
                                       if key.startswith("parameter.") and
                                       (config["backbone_mode"] == "finetune" or key.startswith("parameter.head."))}
                self.assertEqual(set(bindings), expected_parameters)
                initial_rates = []
                final_rates = []
                for group in groups:
                    identity = group["group_id"]
                    initial = config.get(identity + "_learning_rate", config["learning_rate"])
                    final = initial * config["lr_gamma"] ** config["epochs"]
                    self.assertTrue(group["parameter_ids"])
                    if self.group_rates:
                        self.assertTrue(all(value.startswith(identity + ".") for value in group["parameter_ids"]))
                    self.assertEqual(group["initial_learning_rate"], initial)
                    self.assertAlmostEqual(group["final_learning_rate"], final, places=14)
                    initial_rates.append(initial)
                    final_rates.append(group["final_learning_rate"])
                controls = case["optimizer_controls"]
                self.assertEqual(controls["paused"]["completed_steps"], 1)
                self.assertEqual(controls["paused"]["learning_rates"], initial_rates)
                self.assertEqual(controls["paused"]["parameters"], len(bindings))
                for route in ("continuous", "resumed", "killed"):
                    self.assertEqual(controls[route]["completed_steps"], 4)
                    self.assertEqual(controls[route]["learning_rates"], final_rates)
                    self.assertEqual(controls[route]["parameters"], len(bindings))
                    self.assertEqual(case[route]["optimization"], case["continuous"]["optimization"])
                for group, paused in zip(groups, case["pause"]["optimization"]["groups"]):
                    self.assertEqual(paused["parameter_ids"], group["parameter_ids"])
                    self.assertEqual(paused["final_learning_rate"], group["initial_learning_rate"])
                root = case["roots"]["resumed"]
                provenance = json.loads((root / "bundle/metadata/provenance.json").read_text(encoding="utf-8"))
                self.assertEqual(provenance["optimization"], case["resumed"]["optimization"])
                self.assertEqual(provenance["model"]["transfer"]["optimizer_groups"], [
                    {"group_id": group["group_id"], "learning_rate": group["initial_learning_rate"]}
                    for group in groups
                ])
                changed = self.base / "changed-rate.json"
                changed.write_text(json.dumps(config | {"head_learning_rate": 0.02}), encoding="utf-8")
                self.invoke("recipe", "resume", "--file", changed, "--root", root, expected=2)


class TransferConfiguration(unittest.TestCase):
    """Always available: no downloaded weights, model training or network."""
    invoke = classmethod(TransferRecipe.invoke.__func__)

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="liuding-transfer-config-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name)
        for split in ("train", "validation"):
            for label, name in enumerate(("green", "red")):
                folder = cls.base / split / name
                folder.mkdir(parents=True)
                (folder / "sample.png").write_bytes(png(label, 0))
        # Deliberately not executable model weights. A byte preflight does not
        # claim state topology, strict import, training or task qualification.
        cls.weights = cls.base / "not-model-weights.safetensors"
        cls.weights.write_bytes(b"this is not a model")
        cls.descriptor = json.loads((cls.repo / "recipes/transfer-learning/resnet50-a1.weights.json").read_text())
        cls.descriptor.update(sha256=digest(cls.weights), size_bytes=cls.weights.stat().st_size)
        cls.metadata = cls.base / "weights.json"
        cls.metadata.write_text(json.dumps(cls.descriptor), encoding="utf-8")
        cls.config = json.loads((cls.repo / "recipes/transfer-learning/resnet50.json").read_text())
        cls.config.update(train_root="train", validation_root="validation", batch_size=2,
                          weights_file=cls.weights.name, weights_metadata=cls.metadata.name)
        cls.file = cls.base / "config.json"
        cls.file.write_text(json.dumps(cls.config), encoding="utf-8")

    def test_01_schema_and_byte_identity_preflight(self) -> None:
        schema = self.invoke("recipe", "schema", "--name", "transfer")["data"]
        self.assertEqual(schema["schema_id"], self.config["recipe"])
        preflight = self.invoke("recipe", "check", "--file", self.file)["data"]
        self.assertEqual(preflight["transfer"]["descriptor"], self.descriptor)
        self.assertEqual(preflight["total_steps"], self.config["epochs"])

    def test_02_closed_types_modes_and_paths(self) -> None:
        bad = self.base / "bad.json"
        for patch in ({"surprise": True}, {"backbone_mode": "auto"}, {"backbone_mode": True},
                      {"batch_norm": "train"}, {"batch_norm": "auto"}, {"model": "resnet18"},
                      {"image_size": 32}, {"weights_file": "missing"}):
            with self.subTest(patch=patch):
                bad.write_text(json.dumps(self.config | patch), encoding="utf-8")
                self.invoke("recipe", "check", "--file", bad, expected=2)

    def test_03_descriptor_integrity_and_duplicate_keys(self) -> None:
        try:
            for patch in ({"surprise": True}, {"size_bytes": 1}, {"sha256": "0" * 64},
                          {"position_antialias": "false"}, {"schema_version": True},
                          {"license_spdx": "NOASSERTION"}, {"source_num_classes": -1},
                          {"image": self.descriptor["image"] | {"std": [0, 1, 1]}}):
                with self.subTest(patch=patch):
                    self.metadata.write_text(json.dumps(self.descriptor | patch), encoding="utf-8")
                    self.invoke("recipe", "check", "--file", self.file, expected=2)
            self.metadata.write_text(json.dumps(self.descriptor)[:-1] + ', "id": "duplicate"}', encoding="utf-8")
            self.invoke("recipe", "check", "--file", self.file, expected=2)
        finally:
            self.metadata.write_text(json.dumps(self.descriptor), encoding="utf-8")

    def test_04_optional_group_rate_types_bounds_fallback_and_identity(self) -> None:
        schema = self.invoke("recipe", "schema", "--name", "transfer")["data"]
        for name in ("backbone_learning_rate", "head_learning_rate"):
            self.assertFalse(schema["root"]["fields"][name]["required"])
        baseline = self.invoke("recipe", "check", "--file", self.file)["data"]
        self.assertNotIn("backbone_learning_rate", baseline["config"])
        self.assertNotIn("head_learning_rate", baseline["config"])
        path = self.base / "rates.json"
        for patch in ({"head_learning_rate": 0.02},
                      {"head_learning_rate": 1.0},
                      {"backbone_mode": "finetune", "backbone_learning_rate": 0.0001},
                      {"backbone_mode": "finetune", "head_learning_rate": 0.01},
                      {"backbone_mode": "finetune", "backbone_learning_rate": 0.0001, "head_learning_rate": 0.01}):
            with self.subTest(patch=patch):
                path.write_text(json.dumps(self.config | patch), encoding="utf-8")
                resolved = self.invoke("recipe", "check", "--file", path)["data"]
                self.assertNotEqual(resolved["config_sha256"], baseline["config_sha256"])
                for key, value in patch.items():
                    self.assertEqual(resolved["config"][key], value)
        for patch in ({"backbone_learning_rate": 0.0001}, {"head_learning_rate": 0.0},
                      {"head_learning_rate": -0.1}, {"head_learning_rate": 1.01},
                      {"head_learning_rate": True}, {"head_learning_rate": "0.01"},
                      {"head_learning_rate": None}, {"head_learning_rate": float("nan")},
                      {"head_learning_rate": float("inf")}, {"head_learning_rate": float("-inf")},
                      {"backbone_mode": "finetune", "backbone_learning_rate": 0.0}):
            with self.subTest(patch=patch):
                path.write_text(json.dumps(self.config | patch), encoding="utf-8")
                self.invoke("recipe", "check", "--file", path, expected=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--weights-root", type=Path,
                        default=os.environ.get("JYPPX_LIUDING_PRETRAINED_REFERENCE_DIR"))
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--contracts-only", action="store_true")
    parser.add_argument("--group-rates", action="store_true")
    arguments, rest = parser.parse_known_args()
    if arguments.weights_root is None and not arguments.contracts_only:
        print("SKIP: set JYPPX_LIUDING_PRETRAINED_REFERENCE_DIR to approved local weights")
        raise SystemExit(77)
    TransferRecipe.cli = arguments.cli.resolve()
    TransferRecipe.repo = Path(__file__).resolve().parents[2]
    TransferRecipe.weights_root = arguments.weights_root.resolve() if arguments.weights_root else None
    TransferRecipe.device = arguments.device
    TransferRecipe.group_rates = arguments.group_rates
    TransferRecipe.evidence = arguments.evidence
    TransferConfiguration.cli = TransferRecipe.cli
    TransferConfiguration.repo = TransferRecipe.repo
    unittest.main(argv=[__file__, *rest], defaultTest=(
        "TransferConfiguration" if arguments.contracts_only else "TransferRecipe"))
