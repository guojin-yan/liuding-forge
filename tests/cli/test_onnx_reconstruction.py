# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native checkpoint -> generic exporter -> independent ORT -> atomic bundle."""
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
import unittest


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OnnxReconstructionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_fixture = tempfile.TemporaryDirectory(prefix="liuding-source-fixture-")
        cls.addClassCleanup(cls.temporary_fixture.cleanup)
        cls.fixture = Path(cls.temporary_fixture.name)
        result = subprocess.run([str(cls.exporter), "--gtest_filter=NativeOnnxSource.ProducesCheckpointAndDefinitionReferences"],
            env=dict(os.environ, JYPPX_LIUDING_ONNX_SOURCE_FIXTURE=str(cls.fixture), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"),
            capture_output=True, text=True, encoding="utf-8", timeout=180, check=False)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.summary = json.loads((cls.fixture / "source-summary.json").read_bytes())

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="liuding-source-cli-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "原训练 run"
        shutil.copytree(self.fixture / "run-store", self.root)
        self.output = self.directory / "deployment bundle 中文"
        self.inputs = self.directory / "inputs.json"
        self.tensors = self.directory / "inputs.safetensors"
        self.metadata_path = self.directory / "delivery.json"
        self.prepare("linear")

    def invoke(self, *arguments: object, expected: int = 0) -> dict:
        result = subprocess.run([str(self.cli), *map(str, arguments), "--format", "json"],
            env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"), capture_output=True,
            text=True, encoding="utf-8", timeout=120, check=False)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def prepare(self, model: str) -> None:
        self.model = model
        payload, header, cases = bytearray(), {}, []
        for batch in (1, 2, 5):
            shape = [batch, 3] if model == "linear" else [batch, 3, 32, 32]
            count = 1
            for extent in shape: count *= extent
            values = [1.] * count
            start = len(payload)
            payload += struct.pack(f"<{count}f", *values)
            key = f"batch{batch}"
            header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [start, len(payload)]}
            cases.append({"id": key, "inputs": [{"name": "image", "tensor": key}]})
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        self.tensors.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        self.spec = {"format": "jyppx.liuding.onnx_export_inputs", "schema_version": 1,
            "input_names": ["image"], "output_names": ["logits"], "minimum_batch": 1,
            "maximum_batch": 8, "dynamic_batch": True, "absolute_tolerance": 1e-5,
            "relative_tolerance": 1e-4, "cases": cases}
        self.inputs.write_text(json.dumps(self.spec), encoding="utf-8")
        def owner(name: str) -> dict:
            return {"schema_id": "independent.source." + name, "schema_version": 1, "content": {"fixture": True}}
        self.metadata = {"format": "jyppx.liuding.onnx_delivery_metadata", "schema_version": 1,
            "bundle_id": "reconstructed-source", "checkpoint": {"sha256": self.summary[model]["checkpoint"],
            "size_bytes": self.summary[model]["size_bytes"]}, "preprocessing": owner("pre"),
            "postprocessing": owner("post"), "labels": None, "evaluation": owner("evaluation"), "sbom": owner("sbom")}
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")

    def export(self, expected: int = 0, definition: str | None = None, checkpoint: str | None = None,
               source: str = "training-source", inputs_digest: str | None = None, provider: str = "CPUExecutionProvider") -> dict:
        return self.invoke("model", "onnx", "export", "--root", self.root, "--source-run", source,
            "--checkpoint-reference", checkpoint or self.model + ".checkpoint",
            "--definition-reference", definition or self.model + ".definition",
            "--inputs", self.inputs, "--inputs-sha256", inputs_digest or digest(self.inputs),
            "--tensors", self.tensors, "--tensors-sha256", digest(self.tensors),
            "--metadata", self.metadata_path, "--metadata-sha256", digest(self.metadata_path),
            "--provider", provider, "--output", self.output, expected=expected)

    def smoke(self) -> dict:
        return self.invoke("model", "onnx", "smoke", "--bundle", self.output,
            "--bundle-sha256", digest(self.output / "manifest.json"), "--model-id", "primary",
            "--report-id", "export-report", "--cases-id", "cases", "--tensors-id", "tensors",
            "--provider", "CPUExecutionProvider")["data"]

    def test_linear_export_uses_exact_checkpoint_and_scalar_oracle(self) -> None:
        source_before = sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file())
        def database_contents() -> list[str]:
            connection = sqlite3.connect((self.root / "runs.sqlite3").as_uri() + "?mode=ro", uri=True)
            try:
                return list(connection.iterdump())
            finally:
                connection.close()
        database_before = database_contents()
        data = self.export()["data"]
        self.assertTrue(data["published"])
        self.assertEqual(data["operation"], "model.onnx.export")
        proof = data["source_reconstruction"]
        self.assertTrue(proof["exact_model_state_restored"])
        self.assertTrue(proof["checkpoint_run_association_verified"])
        self.assertTrue(proof["definition_run_association_verified"])
        self.assertFalse(proof["source_authenticated"])
        self.assertFalse(proof["original_config_semantics_verified"])
        self.assertEqual(proof["source_case_count"], 3)
        manifest = json.loads((self.output / "manifest.json").read_bytes())
        self.assertEqual(len(manifest["evidence"]), 11)
        self.assertEqual((self.output / "evidence/source-inputs.json").read_bytes(), self.inputs.read_bytes())
        self.assertEqual((self.output / "evidence/source-inputs.safetensors").read_bytes(), self.tensors.read_bytes())
        report = json.loads((self.output / "evidence/export-report.json").read_bytes())
        state = {row["name"]: row for row in report["state"]}
        native_weight = struct.pack("<6f", *(value / 7 for value in range(6)))
        self.assertEqual(state["state.weight"]["digest"]["hex"], hashlib.sha256(native_weight).hexdigest())
        tensor_bytes = (self.output / "evidence/cases.safetensors").read_bytes()
        header_size = struct.unpack_from("<Q", tensor_bytes)[0]
        header = json.loads(tensor_bytes[8:8 + header_size])
        for name, row in header.items():
            if ".out" not in name: continue
            start, end = row["data_offsets"]
            values = struct.unpack(f"<{(end-start)//4}f", tensor_bytes[8 + header_size + start:8 + header_size + end])
            for index, value in enumerate(values):
                expected = sum((index % 2 * 3 + col) / 7 for col in range(3)) + (0.25 if index % 2 == 0 else -0.5)
                self.assertAlmostEqual(value, expected, places=6)
        self.assertEqual(self.smoke()["numerical"], "passed")
        self.assertEqual(source_before, sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()))
        self.assertEqual(database_before, database_contents())

    def test_bundle_runs_after_original_run_and_inputs_are_removed(self) -> None:
        self.export()
        shutil.rmtree(self.root)
        for path in (self.inputs, self.tensors, self.metadata_path): path.unlink()
        self.assertEqual(self.smoke()["numerical"], "passed")

    def test_checkpoint_and_definition_must_belong_to_selected_run(self) -> None:
        for kwargs in ({"source": "missing-run"}, {"checkpoint": "absent"}, {"definition": "absent"},
                       {"checkpoint": "linear.definition"}, {"definition": "linear.wrong-role"}):
            with self.subTest(kwargs=kwargs):
                self.export(expected=2, **kwargs)
                self.assertFalse(self.output.exists())

    def test_exact_participant_and_closed_definition_are_required(self) -> None:
        for definition in ("linear.missing-participant", "linear.unknown-field"):
            with self.subTest(definition=definition):
                self.export(expected=2, definition=definition)
                self.assertFalse(self.output.exists())

    def test_corrupt_checkpoint_cas_never_exports(self) -> None:
        identity = self.summary["linear"]["checkpoint"]
        path = self.root / "artifacts/objects" / identity[:2] / identity[2:]
        content = bytearray(path.read_bytes())
        content[len(content) // 2] ^= 1
        path.write_bytes(content)
        self.export(expected=2)
        self.assertFalse(self.output.exists())

    def test_metadata_cannot_substitute_an_unrestored_checkpoint(self) -> None:
        self.metadata["checkpoint"]["sha256"] = "e" * 64
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")
        self.export(expected=2)
        self.assertFalse(self.output.exists())

    def test_source_inputs_are_closed_hashed_and_exactly_bound(self) -> None:
        self.export(expected=2, inputs_digest="a" * 64)
        for change in ("version", "unknown", "binding", "duplicate", "nonfinite"):
            with self.subTest(change=change):
                self.prepare("linear")
                if change == "version": self.spec["schema_version"] = 2
                elif change == "unknown": self.spec["future"] = 1
                elif change == "binding": self.spec["cases"][0]["inputs"][0]["name"] = "wrong"
                elif change == "duplicate": self.spec["cases"][1]["id"] = self.spec["cases"][0]["id"]
                else:
                    content = bytearray(self.tensors.read_bytes()); content[-4:] = struct.pack("<f", float("inf")); self.tensors.write_bytes(content)
                self.inputs.write_text(json.dumps(self.spec), encoding="utf-8")
                self.export(expected=2)
                self.assertFalse(self.output.exists())

    def test_missing_provider_and_existing_output_do_not_publish(self) -> None:
        self.export(expected=2, provider="UnavailableExecutionProvider")
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_bytes(b"preserve")
        self.export(expected=3)
        self.assertEqual(marker.read_bytes(), b"preserve")

    def test_all_source_cases_match_profile_before_native_execution(self) -> None:
        for change in ("late-batch", "late-width", "fixed-batch", "zero-minimum"):
            with self.subTest(change=change):
                self.prepare("linear")
                if change in ("late-batch", "zero-minimum"):
                    self.spec["maximum_batch"] = 4
                    if change == "zero-minimum":
                        self.spec["minimum_batch"] = 0
                elif change == "fixed-batch":
                    self.spec["dynamic_batch"] = False
                else:
                    content = self.tensors.read_bytes()
                    size = struct.unpack_from("<Q", content)[0]
                    header = json.loads(content[8:8 + size])
                    header["batch5"]["shape"] = [3, 5]
                    encoded = json.dumps(header).encode()
                    encoded += b" " * (-len(encoded) % 8)
                    self.tensors.write_bytes(struct.pack("<Q", len(encoded)) + encoded + content[8 + size:])
                self.inputs.write_text(json.dumps(self.spec), encoding="utf-8")
                self.export(expected=2)
                self.assertFalse(self.output.exists())

    def test_linear_output_expansion_is_bounded_before_forward(self) -> None:
        self.prepare("wide")
        payload = struct.pack("<387f", *([1.] * 387))
        encoded = json.dumps({"sample": {"dtype": "F32", "shape": [129, 3],
                            "data_offsets": [0, len(payload)]}}).encode()
        encoded += b" " * (-len(encoded) % 8)
        self.tensors.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        self.spec["maximum_batch"] = 256
        self.spec["cases"] = [{"id": "bounded-expansion", "inputs": [{"name": "image", "tensor": "sample"}]}]
        self.inputs.write_text(json.dumps(self.spec), encoding="utf-8")
        result = self.export(expected=3)
        self.assertEqual(result["error"]["name"], "resource_exhausted")
        self.assertIn("shared 64 MiB payload budget", json.dumps(result))
        self.assertFalse(self.output.exists())

    def test_registered_resnet_reconstruction_uses_generic_exporter(self) -> None:
        if "resnet" not in self.summary:
            self.skipTest("Models component disabled in this lane")
        self.prepare("resnet")
        result = self.export()["data"]
        self.assertTrue(result["source_run_lineage_verified"])
        model = json.loads((self.output / "evidence/model-definition.json").read_bytes())
        self.assertEqual(model["factory"]["name"], "resnet18")
        packed = (self.output / "evidence/cases.safetensors").read_bytes()
        size = struct.unpack_from("<Q", packed)[0]
        header = json.loads(packed[8:8 + size])
        start, end = header["case1.out0"]["data_offsets"]
        source = (self.fixture / "resnet-output.bin").read_bytes()
        observed = packed[8 + size + start:8 + size + end]
        self.assertEqual(observed, source)
        self.assertEqual(self.smoke()["numerical"], "passed")

    def test_actual_image_training_checkpoint_can_be_exported(self) -> None:
        if not self.image_recipe:
            self.skipTest("ImageFolder producer is not enabled in this profile")
        from test_image_recipe import png

        for split, count in (("train", 8), ("validation", 4)):
            for label, name in enumerate(("green", "red")):
                folder = self.directory / "images" / split / name
                folder.mkdir(parents=True)
                for number in range(count):
                    (folder / f"{number}.png").write_bytes(
                        png(label, number + (20 if split == "validation" else 0)))
        config = {
            "schema_version": 1, "recipe": "jyppx.liuding.imagefolder.resnet18/v1",
            "train_root": "images/train", "validation_root": "images/validation",
            "data_license": "CC0-1.0; self-authored RGB fixture", "device": "cpu",
            "seed": 20261028, "epochs": 12, "batch_size": 4, "image_size": 32,
            "workers": 2, "prefetch_batches": 2, "learning_rate": 0.01,
            "momentum": 0.9, "weight_decay": 0.0001, "lr_gamma": 0.9,
            "checkpoint_every": 16, "minimum_accuracy": 0.875,
            "max_image_bytes": 1048576, "max_data_bytes": 10485760,
            "max_samples": 1024, "image_budget_bytes": 67108864,
        }
        config_file = self.directory / "train.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")
        training = self.directory / "native image training"
        trained = self.invoke("recipe", "run", "--file", config_file, "--root", training)["data"]
        self.assertEqual(trained["global_step"], 48)
        self.assertGreaterEqual(trained["accuracy"], config["minimum_accuracy"])
        self.prepare("resnet")
        self.root = training / "run-store"
        checkpoint = training / "final.jyck"
        self.metadata["checkpoint"] = {"sha256": digest(checkpoint), "size_bytes": checkpoint.stat().st_size}
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")
        data = self.export(source=trained["run_id"], checkpoint="final-checkpoint", definition="model-definition")["data"]
        self.assertTrue(data["source_reconstruction"]["exact_model_state_restored"])
        self.assertEqual((self.output / "evidence/model-definition.json").read_bytes(),
                         (training / "model-definition.json").read_bytes())

        # Compare every exported parameter/buffer with the independently saved
        # training safetensors; this rejects a factory's fresh random state.
        content = (training / "model.safetensors").read_bytes()
        size = struct.unpack_from("<Q", content)[0]
        header = json.loads(content[8:8 + size])
        report = json.loads((self.output / "evidence/export-report.json").read_bytes())
        observed = {row["role"] + "." + row["name"].removeprefix("state."):
                    row["digest"]["hex"] for row in report["state"]}
        expected = {}
        for name, row in header.items():
            if name == "__metadata__":
                continue
            start, end = row["data_offsets"]
            expected[name] = hashlib.sha256(content[8 + size + start:8 + size + end]).hexdigest()
        self.assertEqual(observed, expected)
        self.assertEqual(self.smoke()["numerical"], "passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--image-recipe", action="store_true")
    arguments, rest = parser.parse_known_args()
    OnnxReconstructionTest.cli = arguments.cli.resolve()
    OnnxReconstructionTest.exporter = arguments.exporter.resolve()
    OnnxReconstructionTest.image_recipe = arguments.image_recipe
    unittest.main(argv=[__file__, *rest])
