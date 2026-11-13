# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Native ImageFolder fault diagnostics; optionally use pinned real image data."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import unittest
import zlib

FAULT_ENV = "JYPPX_LIUDING_TEST_IMAGE_DIAGNOSTIC_FAULT"
FAULTS = ("training.nan", "training.positive_infinity", "training.negative_infinity",
          "validation.nan", "validation.positive_infinity", "validation.negative_infinity",
          "gradient.nan", "gradient.infinity")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fixture_png(label, index):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    row = bytes((40 + label * 15, 60 + index * 20, 160 - label * 10)) * 32
    pixels = (b"\0" + row) * 32
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


class ImageFailureDiagnostics(unittest.TestCase):
    fault_environment = FAULT_ENV
    run_id = "run.image-classification"

    @classmethod
    def expected_model_identity(cls, lineage):
        return lineage["model_contract_sha256"]

    @classmethod
    def invoke(cls, *arguments, expected=(0,), fault=None):
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
        env.pop(cls.fault_environment, None)
        if fault:
            env[cls.fault_environment] = fault
        result = subprocess.run([str(cls.cli), *map(str, arguments), "--format", "json"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=60, env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode not in expected:
            raise AssertionError((result.returncode, result.stdout.decode(errors="replace"),
                                  result.stderr.decode(errors="replace")))
        if len(result.stdout) + len(result.stderr) > (1 << 20):
            raise AssertionError("CLI diagnostic output exceeded bound")
        value = json.loads(result.stdout)
        cls.commands.append({"arguments": list(map(str, arguments)), "fault": fault,
                             "exit": result.returncode, "response": value,
                             "stderr": result.stderr.decode(errors="replace")})
        return value, result.stderr.decode(errors="replace")

    @classmethod
    def setUpClass(cls):
        cls.commands = []
        cls.evidence = {"source": "explicit image cache" if cls.data_root else "self-authored RGB fixture",
                        "not_full_training_or_quality_qualification": True}
        cls.base = Path(tempfile.mkdtemp(prefix="image-failure-", dir=cls.work_root))
        manifest = []
        for role, count in (("train", 2), ("validation", 1)):
            for label in range(10):
                category = cls.categories[label] if cls.data_root else f"class-{label:02d}"
                folder = cls.base / "images" / role / category
                folder.mkdir(parents=True)
                if cls.data_root:
                    candidates = sorted((cls.data_root / role / category).glob("*.png"))
                    if len(candidates) < count:
                        raise AssertionError("required image cache is incomplete")
                    for source in candidates[:count]:
                        target = folder / source.name
                        shutil.copyfile(source, target)
                        manifest.append({"source": str(source), "copy": str(target.relative_to(cls.base)),
                                         "bytes": source.stat().st_size, "sha256": sha(source)})
                else:
                    for index in range(count):
                        target = folder / f"{index}.png"
                        target.write_bytes(fixture_png(label, index))
                        manifest.append({"copy": str(target.relative_to(cls.base)),
                                         "bytes": target.stat().st_size, "sha256": sha(target)})
        cls.evidence["input_images"] = manifest
        config = {"schema_version": 1, "recipe": "jyppx.liuding.imagefolder.resnet18/v1",
                  "train_root": "images/train", "validation_root": "images/validation",
                  "data_license": "CIFAR-10, explicit cached diagnostic subset" if cls.data_root else "CC0-1.0",
                  "device": "cpu", "seed": 20261028, "epochs": 2, "batch_size": 10,
                  "image_size": 32, "workers": 0, "prefetch_batches": 1,
                  "learning_rate": .1, "momentum": .9, "weight_decay": .0001,
                  "lr_gamma": .95, "checkpoint_every": 1, "minimum_accuracy": .65,
                  "max_image_bytes": 1048576, "max_data_bytes": 10485760,
                  "max_samples": 100, "image_budget_bytes": 67108864}
        cls.config = cls.base / "config.json"
        cls.config.write_text(json.dumps(config), encoding="utf-8")
        cls.prefix = cls.base / "prefix"
        value, _ = cls.invoke("recipe", "run", "--file", cls.config, "--root", cls.prefix, "--pause-after", 1)
        if value["data"]["status"] != "paused" or value["data"]["global_step"] != 1:
            raise AssertionError("finite diagnostic prefix did not persist exactly one step")
        cls.evidence["prefix_checkpoint_sha256"] = sha(cls.prefix / "latest.jyck")
        cls.cases = {}

    @classmethod
    def tearDownClass(cls):
        cls.evidence["commands"] = cls.commands
        cls.evidence["cases"] = cls.cases
        (cls.base / "evidence.json").write_text(json.dumps(cls.evidence, indent=2) + "\n", encoding="utf-8")

    def references(self, root):
        database = (root / "run-store/runs.sqlite3").resolve()
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchall(), [("ok",)])
            self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
            rows = db.execute("SELECT reference_id,digest,logical_name,media_type,producer FROM jyppx_run_artifacts "
                              "WHERE role='training.diagnostics.failure'").fetchall()
            self.assertEqual(db.execute("SELECT status FROM jyppx_runs").fetchall(), [(2,)])
        return rows

    def root_for(self, fault):
        root = self.base / fault
        shutil.copytree(self.prefix, root)
        return root

    def check_fault(self, fault):
        root = self.root_for(fault)
        before = sha(root / "latest.jyck")
        value, stderr = self.invoke("recipe", "resume", "--file", self.config,
                                    "--root", root, expected=(3,), fault=fault)
        self.assertEqual(value["error"]["name"], "numerical")
        self.assertEqual(value["error"]["component"], "training.trainer")
        self.assertNotIn("Could not persist", stderr)
        rows = self.references(root)
        self.assertEqual(len(rows), 1)
        reference, digest, logical_name, media_type, producer = rows[0]
        self.assertEqual(reference, "diagnostic.failure." + digest)
        self.assertEqual((logical_name, media_type, producer),
                         ("training-diagnostics-failure.json",
                          "application/vnd.jyppx.training-diagnostics+json;version=1", "training.diagnostics/v1"))
        cas = root / "run-store/artifacts/objects" / digest[:2] / digest[2:]
        self.assertEqual(sha(cas), digest)
        report = json.loads(cas.read_bytes())
        self.assertEqual(report["outcome"], "failed")
        lineage = json.loads((root / "lineage.json").read_bytes())
        for key, expected in (("model", self.expected_model_identity(lineage)),
                              ("configuration", lineage["config_sha256"]),
                              ("dataset", lineage["dataset_sha256"]),
                              ("checkpoint", sha(root / "latest.jyck"))):
            self.assertEqual(report["identity"][key], {"algorithm": "sha256", "hex": expected})
        self.assertEqual(report["identity"]["run_id"], self.run_id)
        self.assertFalse(report["options"]["measure_data_wait"])
        self.assertEqual(report["gradient"]["state"], "disabled")
        is_validation = fault.startswith("validation.")
        is_gradient = fault.startswith("gradient.")
        self.assertEqual(report["loss"]["training"]["observations"], 2)
        self.assertEqual(report["loss"]["validation"]["observations"], int(is_validation))
        for role in ("training", "validation"):
            affected = (role == "validation") == is_validation and not is_gradient
            self.assertEqual(report["loss"][role]["nan"], int(affected and fault.endswith("nan")))
            self.assertEqual(report["loss"][role]["infinity"], int(affected and "infinity" in fault))
        self.assertEqual(report["gradient"]["update_attempts"], 2 if is_gradient or is_validation else 1)
        self.assertEqual(report["gradient"]["nonfinite_attempts"], int(is_gradient))
        self.assertEqual(report["gradient"]["amp_skipped_attempts"], 0)
        self.assertEqual(report["identity"]["attempt_id"], "attempt.fit.step-" + ("2" if is_validation else "1"))
        if not is_validation:
            self.assertEqual(sha(root / "latest.jyck"), before)
        self.assertFalse((root / "result.json").exists())
        self.assertFalse((root / "final.jyck").exists())
        if fault == "training.nan":
            old = (cas.stat().st_ino, cas.stat().st_mtime_ns, cas.read_bytes())
            repeated, _ = self.invoke("recipe", "resume", "--file", self.config,
                                      "--root", root, expected=(3,), fault=fault)
            self.assertEqual(repeated["error"], value["error"])
            self.assertEqual(self.references(root), rows)
            self.assertEqual((cas.stat().st_ino, cas.stat().st_mtime_ns, cas.read_bytes()), old)
        self.cases[fault] = {"root": str(root), "report": report, "reference": reference,
                             "report_sha256": digest, "checkpoint_sha256": sha(root / "latest.jyck"),
                             "original_error": value["error"], "passed": True}

    def test_01_nonfinite_real_training_and_validation_losses(self):
        for fault in FAULTS[:6]:
            with self.subTest(fault=fault):
                self.check_fault(fault)

    def test_02_finite_loss_nonfinite_backward_gradients(self):
        for fault in FAULTS[6:]:
            with self.subTest(fault=fault):
                self.check_fault(fault)

    def test_03_diagnostic_storage_failure_preserves_original_training_error(self):
        root = self.root_for("storage.failure")
        before = sha(root / "latest.jyck")
        try:
            value, stderr = self.invoke("recipe", "resume", "--file", self.config,
                                        "--root", root, expected=(3,), fault="storage.failure")
            self.assertEqual(value["error"]["name"], "numerical")
            self.assertEqual(value["error"]["operation"], "step.loss")
            self.assertIn("Could not persist training failure diagnostics", stderr)
        finally:
            objects = root / "run-store/artifacts/objects"
            if objects.is_file():
                objects.unlink()
                (objects.parent / "objects.saved").rename(objects)
        self.assertEqual(sha(root / "latest.jyck"), before)
        self.assertEqual(self.references(root), [])
        self.cases["storage.failure"] = {"passed": True, "original_error": value["error"],
                                          "warning": stderr, "checkpoint_sha256": before}

    def test_04_safe_checkpoint_recovers_without_failed_update_or_extra_counts(self):
        root = self.base / "gradient.infinity"
        self.assertTrue(root.is_dir(), "gradient fault must run before recovery")
        baseline, _ = self.invoke("recipe", "resume", "--file", self.config, "--root", self.prefix, expected=(0, 4))
        resumed, _ = self.invoke("recipe", "resume", "--file", self.config, "--root", root, expected=(0, 4))
        for name in ("model.safetensors", "final.jyck", "diagnostics.json"):
            self.assertEqual(sha(root / name), sha(self.prefix / name))
        self.assertEqual(baseline["data"], resumed["data"])
        self.assertEqual(resumed["data"]["global_step"], 4)
        report = json.loads((root / "diagnostics.json").read_bytes())
        self.assertEqual(report["gradient"]["update_attempts"], 4)
        self.assertEqual(report["gradient"]["nonfinite_attempts"], 0)
        self.assertEqual(report["loss"]["training"]["nan"], 0)
        self.assertEqual(report["loss"]["training"]["infinity"], 0)
        self.evidence["recovery"] = {"passed": True, "global_step": 4,
                                      "model_sha256": sha(root / "model.safetensors"),
                                      "checkpoint_sha256": sha(root / "final.jyck"),
                                      "diagnostic_sha256": sha(root / "diagnostics.json"),
                                      "quality_passed": resumed["data"]["quality_passed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    data = args.data_root or os.environ.get("JYPPX_LIUDING_TEST_DIAGNOSTIC_IMAGE_ROOT")
    ImageFailureDiagnostics.data_root = Path(data).resolve() if data else None
    if data:
        ImageFailureDiagnostics.categories = sorted(p.name for p in (Path(data) / "train").iterdir() if p.is_dir())
        if len(ImageFailureDiagnostics.categories) != 10:
            parser.error("--data-root requires all ten ImageFolder classes")
    temporary = None
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        root = args.work_dir.resolve()
    else:
        temporary = tempfile.TemporaryDirectory(prefix="image-failure-diagnostics-")
        root = Path(temporary.name)
    ImageFailureDiagnostics.cli = args.cli.resolve()
    ImageFailureDiagnostics.work_root = root
    try:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ImageFailureDiagnostics))
        passed = result.wasSuccessful() and result.testsRun == 4 and not result.skipped
        print(json.dumps({"passed": passed, "tests_run": result.testsRun,
                          "skipped": len(result.skipped), "work_root": str(root),
                          "actual_image_cache": bool(data)}))
        return 0 if passed else 1
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
