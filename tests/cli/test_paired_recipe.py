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


def copytree_windows_safe(source: Path, destination: Path) -> None:
    """Include descendants in long-path handling, even with a short tree root."""
    def extended(path: Path) -> str:
        absolute = os.path.abspath(os.fspath(path))
        if os.name != "nt" or absolute.startswith("\\\\?\\"):
            return absolute
        if absolute.startswith("\\\\"):
            return "\\\\?\\UNC\\" + absolute[2:]
        return "\\\\?\\" + absolute

    shutil.copytree(extended(source), extended(destination))


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
        prepared = cls.invoke("data", "prepare", "--name", "paired-affine",
                              "--data-root", cls.base / "datasets")["data"]
        cls.data = Path(prepared["destination"])
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
                copytree_windows_safe(root, cls.evidence / ("run-" + name))

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


class LocalPairedRecipe(unittest.TestCase):
    """The existing full paired task through typed durable attempts."""
    cli: Path
    device: str
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
            result = subprocess.run(command, capture_output=True, timeout=180,
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
        return envelope

    @classmethod
    def preserve_workspace(cls) -> None:
        # Evidence mode retains the original short TEMP as well as the copy,
        # including when a command or checker fails before train() completes.
        for name, attribute in (("config", "config"), ("preflight", "preflight"),
                                ("plan", "prepared"), ("continuous", "baseline"),
                                ("local", "completed"), ("resumed", "resumed"),
                                ("checkpoints", "manifests")):
            if hasattr(cls, attribute):
                (cls.evidence / f"{name}.json").write_text(
                    json.dumps(getattr(cls, attribute), indent=2) + "\n", encoding="utf-8")
        if cls.base.is_dir():
            copytree_windows_safe(cls.base, cls.evidence / "workspace")


    @staticmethod
    def file_sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def setUpClass(cls) -> None:
        output = os.environ.get("JYPPX_LIUDING_RECIPE_EVIDENCE_DIR")
        cls.evidence = Path(output) / ("paired-local-" + cls.device) if output else None
        cls.command_index = 0
        if cls.evidence is not None:
            cls.evidence.mkdir(parents=True, exist_ok=False)
            retained = Path(tempfile.mkdtemp(prefix="liuding-local-paired-"))
            cls.base = retained / "typed paired recipe"
            cls.addClassCleanup(cls.preserve_workspace)
            (cls.evidence / "workspace-source.json").write_text(json.dumps({
                "retained_temp": str(retained), "workspace": str(cls.base),
                "retained_for_root_collection": True,
                "copy": str(cls.evidence / "workspace")}, indent=2) + "\n", encoding="utf-8")
        else:
            cls.temporary = tempfile.TemporaryDirectory(prefix="liuding-local-paired-")
            cls.addClassCleanup(cls.temporary.cleanup)
            cls.base = Path(cls.temporary.name) / "typed paired recipe"
        cls.base.mkdir()
        prepared = cls.invoke("data", "prepare", "--name", "paired-affine",
                              "--data-root", cls.base / "datasets")["data"]
        cls.data = Path(prepared["destination"])
        cls.config = json.loads((ROOT / "recipes/multi-io/paired-affine.json").read_text(encoding="utf-8"))
        cls.config.update(train_file=str(cls.data / "train.jsonl"),
                          validation_file=str(cls.data / "validation.jsonl"), device=cls.device)
        cls.config_file = cls.base / "config.json"
        cls.config_file.write_text(json.dumps(cls.config), encoding="utf-8")
        cls.preflight = cls.invoke("recipe", "check", "--file", cls.config_file)["data"]
        cls.store, cls.work = cls.base / "store", cls.base / "resumed"
        cls.spec_path = cls.base / "retry.json"
        cls.training_started = False

    @classmethod
    def prepare(cls, config: Path, work: Path, spec: Path, *, pause: int = 259,
                expected: int = 0) -> dict:
        return cls.invoke("recipe", "local-spec", "--file", config, "--root", work,
                          "--store", cls.store, "--spec", spec, "--executable", cls.cli,
                          "--pause-after", pause, "--timeout-ms", 360000, expected=expected)

    @classmethod
    def plan(cls) -> dict:
        if not hasattr(cls, "prepared"):
            cls.prepared = cls.prepare(cls.config_file, cls.work, cls.spec_path)["data"]
        return cls.prepared

    @classmethod
    def refs(cls, run: str) -> dict:
        return {row["reference_id"]: row for row in cls.invoke(
            "run", "show", "--root", cls.store, "--id", run)["data"]["references"]}

    @classmethod
    def blob(cls, reference: dict) -> bytes:
        artifact = reference["artifact"]
        identity = artifact["sha256"]
        content = (cls.store / "artifacts/objects" / identity[:2] / identity[2:]).read_bytes()
        if len(content) != artifact["size_bytes"] or hashlib.sha256(content).hexdigest() != identity:
            raise AssertionError("Typed paired CAS content differs from its reference")
        return content

    @classmethod
    def train(cls) -> None:
        if cls.training_started:
            if not hasattr(cls, "completed"):
                raise AssertionError("The previous native training group failed; its budget forbids an automatic rerun")
            return
        cls.plan()
        cls.training_started = True
        cls.baseline_root = cls.base / "continuous"
        cls.baseline = cls.invoke("recipe", "run", "--file", cls.config_file,
                                  "--root", cls.baseline_root)["data"]
        completed = cls.invoke("local", "execute", "--root", cls.store,
                               "--id", "paired-retry", "--spec", cls.spec_path)["data"]
        cls.resumed = json.loads((cls.work / "result.json").read_bytes())
        cls.manifests = {name: cls.invoke("checkpoint", "inspect", "--path",
            root / "final.jyck")["data"]["manifest"] for name, root in (
                ("continuous", cls.baseline_root), ("resumed", cls.work))}
        cls.completed = completed
        if cls.evidence is not None:
            evidence = cls.evidence
            for name, value in (("preflight", cls.preflight), ("plan", cls.prepared),
                                ("continuous", cls.baseline), ("local", cls.completed),
                                ("resumed", cls.resumed), ("checkpoints", cls.manifests)):
                (evidence / f"{name}.json").write_text(
                    json.dumps(value, indent=2) + "\n", encoding="utf-8")
            for attempt in cls.completed["attempts"]:
                references = cls.refs(attempt["attempt_id"])
                for name in ("local.result", "local.training.inputs", "local.training.outputs"):
                    if name in references:
                        (evidence / f"attempt-{attempt['index']}-{name}.json").write_bytes(
                            cls.blob(references[name]))

    def test_01_local_spec_freezes_paired_inputs_and_program(self) -> None:
        spec = self.plan()["specification"]
        self.assertEqual(spec, json.loads(self.spec_path.read_bytes()))
        self.assertEqual(spec["executable_sha256"], self.file_sha(self.cli))
        self.assertEqual(spec["experiment_id"], "paired-affine")
        self.assertEqual(spec["training"]["model_id"], "paired-affine")
        self.assertEqual(spec["max_attempts"], 2)
        self.assertEqual(spec["retry_exit_codes"], [75])
        self.assertEqual(spec["training"]["retry_resume"],
                         {"checkpoint_reference_id": "checkpoint.last", "restore_policy": "exact_only"})
        self.assertFalse(self.work.exists())
        for name, actual in (("configuration", self.preflight["config"]),
                             ("data_manifest", self.preflight["dataset"])):
            binding = spec["training"][name]
            references = self.refs(binding["run_id"])
            self.assertEqual(json.loads(self.blob(references[binding["reference_id"]])), actual)
            source = self.invoke("run", "show", "--root", self.store,
                                 "--id", binding["run_id"])["data"]["run"]
            self.assertEqual(source["experiment_id"], "paired-affine")
        before = self.spec_path.read_bytes()
        self.prepare(self.config_file, self.base / "unused-work", self.spec_path, expected=2)
        self.assertEqual(self.spec_path.read_bytes(), before)
        single = self.prepare(self.config_file, self.base / "single-work",
                              self.base / "single.json", pause=0)["data"]["specification"]
        self.assertEqual(single["max_attempts"], 1)
        self.assertEqual(single["retry_exit_codes"], [])
        self.assertNotIn("retry_resume", single["training"])

    def test_02_full_task_exact_retry_and_typed_outputs_match_continuous(self) -> None:
        self.train()
        self.assertEqual(self.config["seed"], 20261028)
        self.assertEqual(self.preflight["total_steps"], 1280)
        self.assertEqual(self.preflight["dataset"]["train"]["samples"], 1024)
        self.assertEqual(self.preflight["dataset"]["validation"]["samples"], 256)
        for result in (self.baseline, self.resumed):
            self.assertEqual(result["global_step"], 1280)
            self.assertEqual(result["optimizer_steps"], 1280)
            self.assertEqual(result["scheduler_steps"], 40)
            self.assertEqual(result["completed_epochs"], 40)
            self.assertEqual(result["validation_samples"], 256)
            self.assertGreaterEqual(result["accuracy"], .95)
            self.assertLessEqual(result["mse"], .005)
            self.assertTrue(result["quality_passed"])
        for field in RESULT_FIELDS:
            self.assertEqual(self.resumed[field], self.baseline[field], field)
        self.assertEqual((self.work / "model.safetensors").read_bytes(),
                         (self.baseline_root / "model.safetensors").read_bytes())
        states = {name: {entry["state"]["participant_id"]: entry
                  for entry in manifest["participants"]} for name, manifest in self.manifests.items()}
        self.assertEqual(set(states["resumed"]), PARTICIPANTS)
        self.assertEqual(states["resumed"], states["continuous"])
        self.assertEqual(self.completed["status"], "completed")
        self.assertEqual(len(self.completed["attempts"]), 2)
        first, second = self.completed["attempts"]
        self.assertNotEqual(first["attempt_id"], second["attempt_id"])
        self.assertEqual(first["result"]["exit_code"], 75)
        self.assertEqual(second["result"]["exit_code"], 0)
        for attempt in (first, second):
            self.assertTrue(attempt["result"]["process_tree_cleanup_confirmed"])
        old_refs, new_refs = self.refs(first["attempt_id"]), self.refs(second["attempt_id"])
        self.assertIn("checkpoint.last", old_refs)
        self.assertNotIn("model.weights", old_refs)
        self.assertNotIn("training.result", old_refs)
        self.assertNotIn("local.training.outputs", old_refs)
        paused = json.loads(self.blob(old_refs["local.stdout"]))["data"]
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["global_step"], 259)
        inputs = json.loads(self.blob(new_refs["local.training.inputs"]))
        checkpoint = old_refs["checkpoint.last"]["artifact"]
        paused_checkpoint = self.base / "paused.jyck"
        paused_checkpoint.write_bytes(self.blob(old_refs["checkpoint.last"]))
        self.invoke("checkpoint", "verify", "--path", paused_checkpoint)
        expected_training = dict(self.plan()["specification"]["training"])
        del expected_training["retry_resume"]
        expected_training["resume"] = {"restore_policy": "exact_only", "checkpoint": {
            "run_id": first["attempt_id"], "reference_id": "checkpoint.last",
            "sha256": checkpoint["sha256"], "size_bytes": checkpoint["size_bytes"]}}
        spec_sha = self.refs("paired-retry")["local.spec"]["artifact"]["sha256"]
        self.assertEqual(inputs, {"format": "jyppx.liuding.local_training_inputs",
            "schema_version": 1, "job_id": "paired-retry", "attempt_id": second["attempt_id"],
            "index": 2, "spec_sha256": spec_sha, "training": expected_training,
            "previous_result_sha256": old_refs["local.result"]["artifact"]["sha256"]})
        self.assertEqual(second["result"]["training_sha256"],
                         new_refs["local.training.inputs"]["artifact"]["sha256"])
        self.assertTrue(second["result"]["training_outputs_confirmed"])
        receipt = json.loads(self.blob(new_refs["local.training.outputs"]))
        self.assertEqual(new_refs["local.training.outputs"]["artifact"]["sha256"],
                         second["result"]["training_outputs_sha256"])
        native = dict(second["result"])
        del native["training_outputs_sha256"], native["training_outputs_confirmed"]
        expected_rows = [dict(row, sha256=new_refs[row["reference_id"]]["artifact"]["sha256"],
            size_bytes=new_refs[row["reference_id"]]["artifact"]["size_bytes"],
            confirmed=True, failure_reason=None) for row in expected_training["outputs"]]
        self.assertEqual(receipt, {"format": "jyppx.liuding.local_training_outputs",
            "schema_version": 1, "job_id": "paired-retry", "attempt_id": second["attempt_id"],
            "index": 2, "spec_sha256": spec_sha,
            "intent_sha256": new_refs["local.intent"]["artifact"]["sha256"],
            "owner_sha256": new_refs["local.owner"]["artifact"]["sha256"],
            "native_result_sha256": hashlib.sha256(json.dumps(native, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "training": {key: value for key, value in expected_training.items() if key != "outputs"},
            "outputs": expected_rows, "confirmed": True, "failure_reason": None})
        for name, filename in (("checkpoint.last", "final.jyck"),
                               ("model.weights", "model.safetensors"), ("training.result", "result.json")):
            self.assertEqual(new_refs[name]["artifact"]["sha256"], self.file_sha(self.work / filename))
            self.assertEqual(self.blob(new_refs[name]), (self.work / filename).read_bytes())
        for command, path in ((('model', 'verify'), self.work / 'model.safetensors'),
                              (('model', 'bundle', 'verify'), self.work / 'bundle'),
                              (('checkpoint', 'verify'), self.work / 'final.jyck')):
            self.invoke(*command, "--path", path)
        for root in (self.store, self.work / "run-store"):
            self.assertTrue(self.invoke("run", "verify", "--root", root)["data"]["valid"])
        input_schema = json.loads((self.work / "bundle/metadata/input.json").read_text(encoding="utf-8"))
        output_schema = json.loads((self.work / "bundle/metadata/output.json").read_text(encoding="utf-8"))
        self.assertEqual([entry["name"] for entry in input_schema["positional"]], ["left", "right"])
        self.assertEqual([entry["shape"] for entry in output_schema["tuple"]], [[None, 2], [None, 1]])

    def test_03_confirmed_terminal_replay_does_not_train_or_change_files(self) -> None:
        self.train()
        before = {str(p.relative_to(self.work)): (self.file_sha(p), p.stat().st_mtime_ns)
                  for p in self.work.rglob("*") if p.is_file()}
        for action in ("recover", "execute"):
            options = ("--spec", self.spec_path) if action == "execute" else ()
            replay = self.invoke("local", action, "--root", self.store,
                                 "--id", "paired-retry", *options)["data"]
            self.assertEqual(replay, self.completed)
        after = {str(p.relative_to(self.work)): (self.file_sha(p), p.stat().st_mtime_ns)
                 for p in self.work.rglob("*") if p.is_file()}
        self.assertEqual(after, before)

    def test_04_changed_inputs_and_unowned_local_run_fail_before_training(self) -> None:
        for name in ("model", "config", "data", "outputs"):
            with self.subTest(name=name):
                work, path = self.base / ("bad-" + name), self.base / ("bad-spec-" + name + ".json")
                spec = self.prepare(self.config_file, work, path)["data"]["specification"]
                original = None
                if name == "model":
                    spec["training"]["model_id"] = "not-paired-affine"
                elif name == "outputs":
                    spec["training"]["outputs"][1]["role"] = "wrong"
                elif name == "config":
                    changed = self.base / "changed-config.json"
                    changed.write_text(json.dumps(self.config | {"seed": 7}), encoding="utf-8")
                    spec["arguments"][3] = str(changed)
                else:
                    data = self.data / "train.jsonl"
                    original = data.read_bytes()
                    head, tail = original.split(b"\n", 1)
                    row = json.loads(head)
                    row["left"][0] = -row["left"][0] if row["left"][0] else .5
                    data.write_bytes(json.dumps(row).encode("utf-8") + b"\n" + tail)
                try:
                    path.write_text(json.dumps(spec), encoding="utf-8")
                    rejected = self.invoke("local", "execute", "--root", self.store,
                        "--id", "bad-" + name, "--spec", path, expected=1)["data"]
                    self.assertEqual(rejected["status"], "failed")
                    self.assertEqual(len(rejected["attempts"]), 1)
                    attempt = rejected["attempts"][0]
                    self.assertEqual(attempt["result"]["exit_code"], 2)
                    self.assertTrue(attempt["result"]["process_tree_cleanup_confirmed"])
                    self.assertNotIn("checkpoint.last", self.refs(attempt["attempt_id"]))
                    self.assertFalse(work.exists())
                finally:
                    if original is not None:
                        data.write_bytes(original)
        unowned = self.base / "unowned"
        self.invoke("recipe", "local-run", "--file", self.config_file, "--root", unowned, expected=2)
        self.assertFalse(unowned.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments, remaining = parser.parse_known_args()
    PairedRecipe.cli = arguments.cli.resolve()
    PairedRecipe.device = arguments.device
    LocalPairedRecipe.cli = arguments.cli.resolve()
    LocalPairedRecipe.device = arguments.device
    unittest.main(argv=[__file__, *remaining])
