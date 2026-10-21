# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Execute the installed offline reference workflow and audit its real outputs.

Usage: python test_offline_workflow.py --cli /absolute/path/to/jyppx --device cpu
Use --device cuda in the Ubuntu/CUDA chapter validation lane. No mock Trainer,
synthetic report, Python Torch, or network service is used by this contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class OfflineWorkflowContract(unittest.TestCase):
    cli: Path
    device: str

    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        process = subprocess.run(
            [str(cls.cli), *(str(value) for value in arguments), "--format", "json"],
            capture_output=True, text=True, encoding="utf-8", timeout=240,
            check=False,
        )
        if process.returncode != expected:
            raise AssertionError(
                f"argv={arguments!r}; exit={process.returncode}, expected={expected}"
                f"\nstdout={process.stdout}\nstderr={process.stderr}"
            )
        result = json.loads(process.stdout)
        if result["schema_version"] != 1:
            raise AssertionError(f"Unsupported CLI envelope: {result!r}")
        return result

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory(prefix="jyppx-offline-contract-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.parent = Path(cls.directory.name)
        cls.root = cls.parent / "离线 offline evidence"
        result = cls.invoke(
            "offline", "run", "--root", cls.root, "--device", cls.device,
            "--seed", 20261021,
        )
        if result["status"] != "ok":
            raise AssertionError(f"Offline workflow failed: {result!r}")
        cls.summary = result["data"]

    def load(self, name: str) -> dict | list:
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def test_actual_training_recovery_and_distinct_reconstruction(self) -> None:
        summary = self.summary
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["device"], "cuda:0" if self.device == "cuda" else "cpu")
        self.assertEqual(self.load("summary.json"), summary)
        evidence = self.load("recovery-check.json")
        self.assertEqual(summary["recovery"], evidence)
        self.assertEqual(evidence["boundary_global_step"], 6)
        self.assertTrue(evidence["destroyed_before_rebuild"])
        self.assertNotEqual(evidence["initial_seed"], evidence["rebuilt_seed"])
        self.assertNotEqual(evidence["initial_model_sha256"], evidence["rebuilt_model_sha256"])
        for key in (
            "parameters_and_buffers_exact", "metrics_exact", "sample_order_exact",
            "counters_exact", "next_rng_exact", "telemetry_stream_restored",
        ):
            self.assertTrue(evidence[key], key)
        self.assertEqual(evidence["pending_events"], 0)
        self.assertEqual(evidence["baseline"], evidence["recovery"])
        self.assertEqual(evidence["recovery"]["global_step"], 12)
        self.assertEqual(evidence["recovery"]["completed_epochs"], 3)
        self.assertEqual(len(evidence["recovery_sample_order"]), 24)
        self.assertEqual(evidence["baseline_sample_order"], evidence["recovery_sample_order"])
        for phase in ("training_metrics", "validation_metrics"):
            self.assertIn(phase.removesuffix("_metrics") + ".loss", evidence["recovery"][phase])
            self.assertIn(phase.removesuffix("_metrics") + ".f1", evidence["recovery"][phase])
        self.assertEqual(evidence["baseline_next_rng_sha256"], evidence["recovery_next_rng_sha256"])
        self.assertEqual(evidence["baseline_next_cpu_rng_sha256"], evidence["recovery_next_cpu_rng_sha256"])

    def test_model_and_checkpoint_outputs_pass_production_readers(self) -> None:
        model_hashes = []
        for prefix in ("baseline", "recovery"):
            model = self.root / f"{prefix}-model.safetensors"
            model_hashes.append(hashlib.sha256(model.read_bytes()).hexdigest())
            data = self.invoke("model", "verify", "--path", model)["data"]
            self.assertTrue(data["valid"])
            self.assertEqual(data["sha256"], model_hashes[-1])
            self.assertEqual(data["tensor_count"], 5)
            self.assertTrue(all(tensor["dtype"] == "F64" for tensor in data["tensors"]))
            for stage in ("boundary", "final"):
                data = self.invoke("checkpoint", "verify", "--path", self.root / f"{prefix}-{stage}.jyck")["data"]
                self.assertTrue(data["valid"])
                self.assertGreaterEqual(data["participant_count"], 10)
                self.assertEqual(data["verification"]["blob_digests"], "sha256_verified")
        self.assertEqual(model_hashes[0], model_hashes[1])

    def test_real_run_lineage_and_content_addressed_references(self) -> None:
        config_hash = hashlib.sha256((self.root / "config.json").read_bytes()).hexdigest()
        data_hash = hashlib.sha256((self.root / "dataset.json").read_bytes()).hexdigest()
        build = self.load("build.json")
        self.assertEqual(build, self.summary["build"])
        self.assertTrue(build["git_commit"])
        required_roles = {
            "config", "data_manifest", "build", "evaluation_suite", "checkpoint.boundary",
            "checkpoint.final", "model.weights", "evaluation.report", "recovery.evidence",
            "evaluation.comparison", "telemetry.projection",
        }
        for prefix in ("baseline", "recovery"):
            metadata = self.load(f"{prefix}-run.json")
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["lineage"]["config_digest"], config_hash)
            self.assertEqual(metadata["lineage"]["data_manifest_digest"], data_hash)
            self.assertEqual(metadata["lineage"]["source_identity"], build["git_commit"])
            self.assertEqual(json.loads(metadata["lineage"]["build_fingerprint"]), build)
            references = self.load(f"{prefix}-references.json")
            self.assertEqual({reference["role"] for reference in references}, required_roles)
            for reference in references:
                digest = reference["sha256"]
                blob = self.root / "store" / "artifacts" / "objects" / digest[:2] / digest[2:]
                content = blob.read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)
                self.assertEqual(len(content), reference["size_bytes"])

    def test_neutral_durable_events_match_canonical_jsonl_evidence(self) -> None:
        for prefix in ("baseline", "recovery"):
            stored = self.load(f"{prefix}-stored-events.json")
            observer = [json.loads(line) for line in (self.root / f"{prefix}-events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertGreater(len(stored), 20)
            self.assertEqual(len(stored), len(observer))
            streams = set()
            metrics = set()
            for sequence, (durable, projected) in enumerate(zip(stored, observer), 1):
                self.assertEqual(durable["run_sequence"], sequence)
                self.assertEqual(durable["event"], projected["event"])
                self.assertEqual(durable["sha256"], projected["content_digest"])
                canonical = bytes.fromhex(projected["canonical_hex"])
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), durable["sha256"])
                event = durable["event"]
                self.assertEqual(event["identity"]["sequence"], sequence)
                self.assertEqual(event["correlation"]["run_id"], f"run.offline.{prefix}")
                self.assertIsNotNone(event["time"]["wall_time_unix_ns"])
                streams.add(event["identity"]["stream_id"])
                self.assertNotIn(event["payload"]["name"], {"training.run_failed", "training.checkpoint_failed"})
                for point in event["payload"]["metrics"]:
                    metrics.add(point["definition"]["metric_id"])
            self.assertEqual(len(streams), 1)
            self.assertEqual(metrics, {"training.loss", "training.f1", "validation.loss", "validation.f1"})
            self.assertEqual(stored[-1]["event"]["payload"]["name"], "training.run_finished")
            self.assertTrue(stored[-1]["event"]["payload"]["terminal"])

    def test_versioned_evaluation_uses_actual_model_and_dataset(self) -> None:
        suite = self.load("evaluation-suite.json")
        baseline = self.load("baseline-quality.json")
        recovered = self.load("recovery-quality.json")
        comparison = self.load("quality-comparison.json")
        self.assertEqual(suite["version"], 1)
        self.assertEqual(baseline["suite"], suite)
        self.assertEqual(recovered["suite"], suite)
        self.assertEqual(baseline["results"], recovered["results"])
        self.assertEqual(comparison["verdict"], "pass")
        self.assertEqual(comparison["baseline_report_sha256"], hashlib.sha256((self.root / "baseline-quality.json").read_bytes()).hexdigest())
        self.assertEqual(comparison["candidate_report_sha256"], hashlib.sha256((self.root / "recovery-quality.json").read_bytes()).hexdigest())
        for result in recovered["results"]:
            self.assertEqual(result["observation"]["sample_count"], 8)
        for result in comparison["results"]:
            self.assertEqual(result["candidate_minus_baseline"], 0)
            self.assertEqual(result["regression"], 0)
        self.assertTrue((self.root / "recovery-quality.html").is_file())

    def test_store_integrity_and_real_backup_restoration(self) -> None:
        evidence = self.load("storage-check.json")
        self.assertEqual(evidence, self.summary["storage"])
        self.assertTrue(evidence["integrity_ok"])
        self.assertTrue(evidence["restored_integrity_ok"])
        self.assertTrue(evidence["restored_evidence_exact"])
        self.assertGreater(evidence["checked_artifacts"], 5)
        self.assertEqual(evidence["backup"], evidence["restore"])
        for path in (self.root / "store", self.root / "restored-store"):
            data = self.invoke("run", "verify", "--root", path)["data"]
            self.assertTrue(data["valid"])

    def test_existing_root_is_rejected_and_summary_is_preserved(self) -> None:
        previous = (self.root / "summary.json").read_bytes()
        result = self.invoke("offline", "run", "--root", self.root, expected=2)
        self.assertEqual(result["error"]["name"], "invalid_argument")
        self.assertEqual((self.root / "summary.json").read_bytes(), previous)
        self.assertFalse((self.root / "failure.json").exists())

    def test_invalid_options_fail_before_output_creation(self) -> None:
        invalid_root = self.parent / "must-not-be-created"
        for arguments in (
            ("--device", "cuda:1"), ("--seed", "-1"),
            ("--seed", "9223372036854775808"), ("--unknown", "value"),
            ("--root", self.parent / "duplicate"),
        ):
            with self.subTest(arguments=arguments):
                result = self.invoke("offline", "run", "--root", invalid_root, *arguments, expected=2)
                self.assertEqual(result["error"]["name"], "invalid_argument")
                self.assertFalse(invalid_root.exists())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    options = parser.parse_args()
    OfflineWorkflowContract.cli = options.cli.resolve(strict=True)
    OfflineWorkflowContract.device = options.device
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(OfflineWorkflowContract)
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
