# Copyright 2026 JYPPX Training Framework Contributors
# Product: Liuding Forge
# Licensed under the Apache License, Version 2.0.
"""Typed transfer contracts and explicitly selected native task recovery.

The default contracts task performs no training. The fixture task retains the
existing four-update policy/recovery qualification; real retains the original
375-update Fashion-MNIST task. Python uses stdlib and invokes native C++ only.
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

from test_image_recipe import png

ROOT = Path(__file__).resolve().parents[2]
PARTICIPANTS = {"callbacks.dispatcher", "model.main", "recipe.lr", "recipe.sgd",
    "recipe.trainer", "recipe.training", "recipe.validation", "rng.global",
    "strategy.single", "training.diagnostics"}
RESULT_FIELDS = ("model_sha256", "next_rng_sha256", "next_cpu_rng_sha256",
    "training_metrics", "validation_metrics", "accuracy", "confusion", "global_step",
    "optimizer_steps", "scheduler_steps", "completed_epochs", "transfer", "sampling",
    "numerics", "validation_samples")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalTransferRecipe(unittest.TestCase):
    @classmethod
    def command_evidence(cls, index: int, command: list[str], expected: int,
                         status: str, stdout: bytes = b"", stderr: bytes = b"",
                         returncode: int | None = None) -> None:
        if cls.evidence is None:
            return
        prefix = cls.evidence / f"command-{index:03}"
        prefix.with_suffix(".json").write_text(json.dumps({"command": command,
            "expected_exit": expected, "status": status, "returncode": returncode},
            indent=2) + "\n", encoding="utf-8")
        prefix.with_suffix(".stdout").write_bytes(stdout)
        prefix.with_suffix(".stderr").write_bytes(stderr)

    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        command = [str(cls.cli), *(str(value) for value in arguments), "--format", "json"]
        cls.command_index += 1
        index = cls.command_index
        cls.command_evidence(index, command, expected, "started")
        try:
            result = subprocess.run(command, capture_output=True, timeout=cls.command_timeout,
                env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"), check=False)
        except subprocess.TimeoutExpired as error:
            cls.command_evidence(index, command, expected, "timeout",
                                 error.stdout or b"", error.stderr or b"")
            raise
        cls.command_evidence(index, command, expected, "returned", result.stdout,
                             result.stderr, result.returncode)
        if result.returncode != expected:
            raise AssertionError(f"args={arguments!r} exit={result.returncode} expected={expected}"
                f"\nstdout={result.stdout.decode('utf-8', errors='replace')}"
                f"\nstderr={result.stderr.decode('utf-8', errors='replace')}")
        envelope = json.loads(result.stdout)
        if envelope["schema_version"] != 1:
            raise AssertionError(envelope)
        return envelope.get("data", envelope)

    @classmethod
    def preserve_workspace(cls) -> None:
        for name, attribute in (("config", "config"), ("descriptor", "descriptor"),
                ("preflight", "preflight"), ("plan", "prepared"),
                ("continuous", "baseline"), ("local", "completed"),
                ("resumed", "resumed"), ("checkpoints", "manifests")):
            if hasattr(cls, attribute):
                (cls.evidence / f"{name}.json").write_text(json.dumps(
                    getattr(cls, attribute), indent=2) + "\n", encoding="utf-8")
        roots = {"workspace": cls.base}
        if hasattr(cls, "baseline_root") and cls.base not in cls.baseline_root.parents:
            roots["baseline"] = cls.baseline_root
        inventory = {}
        for name, root in roots.items():
            inventory[name] = {"path": str(root), "files": [
                {"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                 "sha256": file_sha(path)} for path in sorted(root.rglob("*")) if path.is_file()]}
        (cls.evidence / "workspace-inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def setUpClass(cls) -> None:
        output = os.environ.get("JYPPX_LIUDING_RECIPE_EVIDENCE_DIR")
        cls.evidence = Path(output) / (f"transfer-local-{cls.task}-{cls.device}") if output else None
        cls.command_index = 0
        if cls.evidence is not None:
            cls.evidence.mkdir(parents=True, exist_ok=False)
            retained = Path(tempfile.mkdtemp(prefix="liuding-local-transfer-"))
            cls.base = retained / "typed transfer"
            cls.addClassCleanup(cls.preserve_workspace)
            (cls.evidence / "workspace-source.json").write_text(json.dumps({
                "retained_temp": str(retained), "workspace": str(cls.base),
                "retained_for_root_collection": True,
                "large_workspace_copied": False}, indent=2) + "\n", encoding="utf-8")
        else:
            cls.temporary = tempfile.TemporaryDirectory(prefix="liuding-local-transfer-")
            cls.addClassCleanup(cls.temporary.cleanup)
            cls.base = Path(cls.temporary.name) / "typed transfer"
        cls.base.mkdir()
        cls.descriptor = json.loads((ROOT / "recipes/transfer-learning/resnet50-a1.weights.json").read_bytes())
        if cls.task == "contracts":
            # This input permits byte/descriptor preflight only. No model claim.
            cls.weights = cls.base / "not-model-weights.safetensors"
            cls.weights.write_bytes(b"typed-transfer-byte-identity-only")
            cls.descriptor.update(sha256=file_sha(cls.weights), size_bytes=cls.weights.stat().st_size)
        else:
            cls.weights = cls.weights_file.resolve()
            if not cls.weights.is_file():
                raise AssertionError(f"Missing approved local weights: {cls.weights}")
            if (cls.weights.stat().st_size != cls.descriptor["size_bytes"] or
                    file_sha(cls.weights) != cls.descriptor["sha256"]):
                raise AssertionError("Approved full pretrained weight identity differs")
            # Preserve workspace evidence without duplicating the external weights.
            if cls.base == cls.weights or cls.base in cls.weights.parents:
                raise AssertionError("Keep external weights outside the disposable workspace")
        cls.metadata = cls.base / "weights.json"
        cls.metadata.write_text(json.dumps(cls.descriptor), encoding="utf-8")
        common = {"schema_version": 1, "recipe": "jyppx.liuding.imagefolder.transfer/v1",
            "device": cls.device, "seed": 20261029, "batch_size": 2, "image_size": 224,
            "workers": 1, "prefetch_batches": 1, "learning_rate": .01, "momentum": .9,
            "weight_decay": .0001, "lr_gamma": .9, "checkpoint_every": 2,
            "max_image_bytes": 1048576, "max_data_bytes": 10485760,
            "max_samples": 1024, "image_budget_bytes": 134217728,
            "model": "resnet50", "weights_file": str(cls.weights),
            "weights_metadata": str(cls.metadata), "backbone_mode": "frozen", "batch_norm": "eval"}
        if cls.task == "real":
            data = cls.data_root.resolve()
            if cls.base == data or cls.base in data.parents:
                raise AssertionError("Keep real data outside the disposable workspace")
            common.update(train_root=str(data / "train"), validation_root=str(data / "validation"),
                data_license="MIT; Fashion-MNIST fixed b2617bb6d3ffa2e429640350f613e3291e10b141, official test used as validation",
                epochs=3, batch_size=8, workers=2, prefetch_batches=1, checkpoint_every=125,
                minimum_accuracy=.65, max_data_bytes=104857600, max_samples=2000,
                image_budget_bytes=268435456)
            cls.total_steps, cls.pause, cls.minimum_accuracy, cls.samples = 375, 47, .65, 1000
        else:
            for split in ("train", "validation"):
                for label, name in enumerate(("green", "red")):
                    folder = cls.base / "images" / split / name
                    folder.mkdir(parents=True)
                    for number in range(2):
                        (folder / f"{number}.png").write_bytes(png(
                            label, number + (20 if split == "validation" else 0)))
            common.update(train_root=str(cls.base / "images/train"),
                validation_root=str(cls.base / "images/validation"),
                data_license="CC0-1.0; self-authored RGB fixture", epochs=2, minimum_accuracy=.5)
            cls.total_steps, cls.pause, cls.minimum_accuracy, cls.samples = 4, 1, .5, 4
        cls.config = common
        cls.config_file = cls.base / "config.json"
        cls.config_file.write_text(json.dumps(cls.config), encoding="utf-8")
        cls.preflight = cls.invoke("recipe", "check", "--file", cls.config_file)
        if cls.preflight["total_steps"] != cls.total_steps:
            raise AssertionError("The fixed task update budget differs")
        for split in ("train", "validation"):
            if cls.preflight["dataset"][split]["samples"] != cls.samples:
                raise AssertionError("The fixed task sample count differs")
        cls.store, cls.work = cls.base / "store", cls.base / "resumed"
        cls.spec_path = cls.base / "retry.json"
        cls.training_started = False

    @classmethod
    def prepare(cls, work: Path, spec: Path, *, expected: int = 0) -> dict:
        return cls.invoke("recipe", "local-spec", "--file", cls.config_file,
            "--root", work, "--store", cls.store, "--spec", spec, "--executable", cls.cli,
            "--pause-after", cls.pause, "--timeout-ms", cls.command_timeout * 1000, expected=expected)

    @classmethod
    def plan(cls) -> dict:
        if not hasattr(cls, "prepared"):
            cls.prepared = cls.prepare(cls.work, cls.spec_path)
        return cls.prepared

    @classmethod
    def refs(cls, run: str) -> dict:
        return {row["reference_id"]: row for row in cls.invoke(
            "run", "show", "--root", cls.store, "--id", run)["references"]}

    @classmethod
    def blob(cls, reference: dict) -> bytes:
        artifact = reference["artifact"]
        identity = artifact["sha256"]
        content = (cls.store / "artifacts/objects" / identity[:2] / identity[2:]).read_bytes()
        if len(content) != artifact["size_bytes"] or hashlib.sha256(content).hexdigest() != identity:
            raise AssertionError("Typed transfer CAS content differs")
        return content

    @classmethod
    def train(cls) -> None:
        if cls.task == "contracts":
            raise AssertionError("The byte-only contract task cannot train; select fixture or real explicitly")
        if cls.training_started:
            if not hasattr(cls, "completed"):
                raise AssertionError("The previous training group failed; automatic rerun is forbidden")
            return
        cls.plan()
        cls.training_started = True
        if cls.baseline_parent is None:
            cls.baseline_root = cls.base / "continuous"
        else:
            if cls.evidence is not None:
                directory = Path(tempfile.mkdtemp(prefix="liuding-transfer-baseline-",
                                                  dir=cls.baseline_parent))
            else:
                temporary = tempfile.TemporaryDirectory(prefix="liuding-transfer-baseline-",
                                                        dir=cls.baseline_parent)
                cls.addClassCleanup(temporary.cleanup)
                directory = Path(temporary.name)
            cls.baseline_root = directory / "continuous"
            if cls.evidence is not None:
                (cls.evidence / "baseline-source.json").write_text(json.dumps({
                    "path": str(cls.baseline_root), "retained_for_root_collection": True,
                    "large_workspace_copied": False}, indent=2) + "\n", encoding="utf-8")
        cls.baseline = cls.invoke("recipe", "run", "--file", cls.config_file,
                                  "--root", cls.baseline_root)
        completed = cls.invoke("local", "execute", "--root", cls.store,
            "--id", "transfer-retry", "--spec", cls.spec_path)
        cls.resumed = json.loads((cls.work / "result.json").read_bytes())
        cls.manifests = {name: cls.invoke("checkpoint", "inspect", "--path", root / "final.jyck")["manifest"]
            for name, root in (("continuous", cls.baseline_root), ("resumed", cls.work))}
        cls.completed = completed

    def test_01_local_spec_freezes_weights_descriptor_inputs_and_program(self) -> None:
        spec = self.plan()["specification"]
        self.assertEqual(spec, json.loads(self.spec_path.read_bytes()))
        self.assertEqual(spec["executable_sha256"], file_sha(self.cli))
        self.assertEqual(spec["experiment_id"], "transfer-learning")
        self.assertEqual(spec["training"]["model_id"], "resnet50")
        self.assertEqual(spec["max_attempts"], 2)
        self.assertEqual(spec["retry_exit_codes"], [75])
        self.assertFalse(self.work.exists())
        for name, expected in (("configuration", {
                "format": "jyppx.liuding.local_transfer_configuration", "schema_version": 1,
                "configuration": self.preflight["config"], "transfer": self.preflight["transfer"]}),
                ("data_manifest", self.preflight["dataset"])):
            binding = spec["training"][name]
            reference = self.refs(binding["run_id"])[binding["reference_id"]]
            self.assertEqual(json.loads(self.blob(reference)), expected)
            self.assertEqual(self.invoke("run", "show", "--root", self.store,
                "--id", binding["run_id"])["run"]["experiment_id"], "transfer-learning")
        before = self.spec_path.read_bytes()
        self.prepare(self.base / "unused", self.spec_path, expected=2)
        self.assertEqual(self.spec_path.read_bytes(), before)

    def test_02_native_exact_retry_outputs_and_frozen_policy_match_continuous(self) -> None:
        self.train()
        for result in (self.baseline, self.resumed):
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["global_step"], self.total_steps)
            self.assertEqual(result["optimizer_steps"], self.total_steps)
            self.assertEqual(result["completed_epochs"], self.config["epochs"])
            self.assertEqual(result["scheduler_steps"], self.config["epochs"])
            self.assertEqual(result["validation_samples"], self.samples)
            self.assertGreaterEqual(result["accuracy"], self.minimum_accuracy)
            self.assertTrue(result["quality_passed"])
            transfer = result["transfer"]
            self.assertEqual(transfer["initial"]["backbone_sha256"], transfer["final"]["backbone_sha256"])
            self.assertNotEqual(transfer["initial"]["head_sha256"], transfer["final"]["head_sha256"])
            self.assertEqual(transfer["initial"]["trainable_parameters"], 2049 * 2 if self.task == "fixture" else 2049 * 10)
            self.assertEqual(transfer["weight_input_sha256"], self.preflight["transfer"]["weight_input_sha256"])
        for field in RESULT_FIELDS:
            self.assertEqual(self.resumed[field], self.baseline[field], field)
        self.assertEqual((self.work / "model.safetensors").read_bytes(),
                         (self.baseline_root / "model.safetensors").read_bytes())
        states = {name: {entry["state"]["participant_id"]: entry for entry in manifest["participants"]}
                  for name, manifest in self.manifests.items()}
        self.assertEqual(set(states["resumed"]), PARTICIPANTS)
        self.assertEqual(states["resumed"], states["continuous"])
        self.assertEqual(self.completed["status"], "completed")
        self.assertEqual(len(self.completed["attempts"]), 2)
        first, second = self.completed["attempts"]
        self.assertEqual(first["result"]["exit_code"], 75)
        self.assertEqual(second["result"]["exit_code"], 0)
        for attempt in (first, second):
            self.assertTrue(attempt["result"]["process_tree_cleanup_confirmed"])
            refs = self.refs(attempt["attempt_id"])
            for identity in (attempt["training_sha256"], attempt["result"]["training_sha256"]):
                self.assertEqual(identity, refs["local.training.inputs"]["artifact"]["sha256"])
        old_refs, new_refs = self.refs(first["attempt_id"]), self.refs(second["attempt_id"])
        self.assertIn("checkpoint.last", old_refs)
        for name in ("model.weights", "training.result", "local.training.outputs"):
            self.assertNotIn(name, old_refs)
        paused = json.loads(self.blob(old_refs["local.stdout"]))["data"]
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["global_step"], self.pause)
        first_checkpoint = old_refs["checkpoint.last"]["artifact"]
        checkpoint_sha = first_checkpoint["sha256"]
        checkpoint_path = self.store / "artifacts/objects" / checkpoint_sha[:2] / checkpoint_sha[2:]
        self.assertEqual(checkpoint_path.stat().st_size, first_checkpoint["size_bytes"])
        self.assertEqual(file_sha(checkpoint_path), checkpoint_sha)
        self.invoke("checkpoint", "verify", "--path", checkpoint_path)
        training = dict(self.plan()["specification"]["training"])
        del training["retry_resume"]
        checkpoint = old_refs["checkpoint.last"]["artifact"]
        training["resume"] = {"restore_policy": "exact_only", "checkpoint": {
            "run_id": first["attempt_id"], "reference_id": "checkpoint.last",
            "sha256": checkpoint["sha256"], "size_bytes": checkpoint["size_bytes"]}}
        spec_sha = self.refs("transfer-retry")["local.spec"]["artifact"]["sha256"]
        self.assertEqual(json.loads(self.blob(new_refs["local.training.inputs"])), {
            "format": "jyppx.liuding.local_training_inputs", "schema_version": 1,
            "job_id": "transfer-retry", "attempt_id": second["attempt_id"], "index": 2,
            "spec_sha256": spec_sha, "training": training,
            "previous_result_sha256": old_refs["local.result"]["artifact"]["sha256"]})
        self.assertTrue(second["result"]["training_outputs_confirmed"])
        receipt = json.loads(self.blob(new_refs["local.training.outputs"]))
        self.assertEqual(new_refs["local.training.outputs"]["artifact"]["sha256"],
                         second["result"]["training_outputs_sha256"])
        native = dict(second["result"])
        del native["training_outputs_sha256"], native["training_outputs_confirmed"]
        self.assertEqual(receipt, {"format": "jyppx.liuding.local_training_outputs",
            "schema_version": 1, "job_id": "transfer-retry", "attempt_id": second["attempt_id"],
            "index": 2, "spec_sha256": spec_sha,
            "intent_sha256": new_refs["local.intent"]["artifact"]["sha256"],
            "owner_sha256": new_refs["local.owner"]["artifact"]["sha256"],
            "native_result_sha256": hashlib.sha256(json.dumps(native, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "training": {key: value for key, value in training.items() if key != "outputs"},
            "outputs": [dict(row, sha256=new_refs[row["reference_id"]]["artifact"]["sha256"],
                size_bytes=new_refs[row["reference_id"]]["artifact"]["size_bytes"],
                confirmed=True, failure_reason=None) for row in training["outputs"]],
            "confirmed": True, "failure_reason": None})
        for name, filename in (("checkpoint.last", "final.jyck"),
                ("model.weights", "model.safetensors"), ("training.result", "result.json")):
            self.assertEqual(new_refs[name]["artifact"]["sha256"], file_sha(self.work / filename))
            self.assertEqual(self.blob(new_refs[name]), (self.work / filename).read_bytes())
        for command, path in ((('model', 'verify'), self.work / 'model.safetensors'),
                (('model', 'bundle', 'verify'), self.work / 'bundle'),
                (('checkpoint', 'verify'), self.work / 'final.jyck')):
            self.invoke(*command, "--path", path)
        for root in (self.store, self.work / "run-store"):
            self.assertTrue(self.invoke("run", "verify", "--root", root)["valid"])
        provenance = json.loads((self.work / "bundle/metadata/provenance.json").read_bytes())
        self.assertEqual(provenance["model"]["transfer"]["descriptor"], self.descriptor)

    def test_03_terminal_replay_does_not_train_or_change_workspace(self) -> None:
        self.train()
        before = {str(p.relative_to(self.work)): (file_sha(p), p.stat().st_mtime_ns)
                  for p in self.work.rglob("*") if p.is_file()}
        for action in ("recover", "execute"):
            options = ("--spec", self.spec_path) if action == "execute" else ()
            self.assertEqual(self.invoke("local", action, "--root", self.store,
                "--id", "transfer-retry", *options), self.completed)
        self.assertEqual({str(p.relative_to(self.work)): (file_sha(p), p.stat().st_mtime_ns)
            for p in self.work.rglob("*") if p.is_file()}, before)

    def test_04_descriptor_drift_and_invalid_ownership_fail_before_workspace(self) -> None:
        # Each spec freezes the original descriptor before it changes in place.
        for name in ("descriptor", "model", "outputs"):
            with self.subTest(name=name):
                work, path = self.base / ("bad-" + name), self.base / ("bad-spec-" + name + ".json")
                spec = self.prepare(work, path)["specification"]
                metadata_before = self.metadata.read_bytes()
                if name == "descriptor":
                    changed = json.loads(metadata_before)
                    changed["image"]["mean"][0] += .01
                    self.metadata.write_text(json.dumps(changed), encoding="utf-8")
                    # The changed descriptor is otherwise valid: rejection must
                    # come from the frozen local binding, not its own schema.
                elif name == "model":
                    spec["training"]["model_id"] = "not-resnet50"
                else:
                    spec["training"]["outputs"][1]["role"] = "wrong"
                try:
                    if name == "descriptor":
                        self.invoke("recipe", "check", "--file", self.config_file)
                    path.write_text(json.dumps(spec), encoding="utf-8")
                    rejected = self.invoke("local", "execute", "--root", self.store,
                        "--id", "bad-" + name, "--spec", path, expected=1)
                    self.assertEqual(rejected["status"], "failed")
                    self.assertEqual(len(rejected["attempts"]), 1)
                    attempt = rejected["attempts"][0]
                    self.assertEqual(attempt["result"]["exit_code"], 2)
                    self.assertTrue(attempt["result"]["process_tree_cleanup_confirmed"])
                    refs = self.refs(attempt["attempt_id"])
                    self.assertNotIn("checkpoint.last", refs)
                    diagnostic = json.loads(self.blob(refs["local.stdout"]))
                    self.assertEqual(diagnostic["error"]["name"], "invalid_argument")
                    if name == "descriptor":
                        self.assertIn("Recipe configuration or dataset differs from frozen local input",
                                      diagnostic["error"]["message"])
                    self.assertFalse(work.exists())
                finally:
                    self.metadata.write_bytes(metadata_before)
        unowned = self.base / "unowned"
        self.invoke("recipe", "local-run", "--file", self.config_file, "--root", unowned, expected=2)
        self.assertFalse(unowned.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--task", choices=("contracts", "fixture", "real"), default="contracts")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--weights-file", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--command-timeout", type=int, default=300)
    parser.add_argument("--baseline-parent", type=Path)
    arguments, remaining = parser.parse_known_args()
    if arguments.task != "contracts" and arguments.weights_file is None:
        parser.error("fixture/real requires explicit approved --weights-file")
    if arguments.task == "real" and arguments.data_root is None:
        parser.error("real requires the existing first100-per-class Fashion --data-root")
    if arguments.command_timeout < 1 or arguments.command_timeout > 86400:
        parser.error("--command-timeout must be 1..86400 seconds")
    LocalTransferRecipe.cli = arguments.cli.resolve()
    LocalTransferRecipe.task = arguments.task
    LocalTransferRecipe.device = arguments.device
    LocalTransferRecipe.weights_file = arguments.weights_file
    LocalTransferRecipe.data_root = arguments.data_root
    LocalTransferRecipe.command_timeout = arguments.command_timeout
    LocalTransferRecipe.baseline_parent = arguments.baseline_parent
    selected = remaining or (["LocalTransferRecipe.test_01_local_spec_freezes_weights_descriptor_inputs_and_program",
        "LocalTransferRecipe.test_04_descriptor_drift_and_invalid_ownership_fail_before_workspace"]
        if arguments.task == "contracts" else ["LocalTransferRecipe"])
    unittest.main(argv=[__file__, *selected])
