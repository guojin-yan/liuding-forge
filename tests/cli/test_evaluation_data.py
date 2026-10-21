# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Real CLI contracts for evaluation, config and basic projected data quality."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest

CLI: Path


def digest(letter: str) -> dict:
    return {"algorithm": "sha256", "hex": letter * 64}


def definition() -> dict:
    return {"schema_version": 1, "metric_id": "classification.accuracy", "version": 1,
            "unit": 5, "aggregation": 3, "direction": 3,
            "parameters": {"average": "micro"}, "context_keys": ["split"]}


def suite() -> dict:
    return {"schema_version": 1, "suite_id": "classification.smoke", "version": 1,
            "task": "classification", "dataset_digest": digest("a"),
            "environment_id": "cpu-f64-test", "seed": 71,
            "rules": [{"rule_id": "validation.accuracy", "definition": definition(),
                       "context": {"split": "validation"}, "minimum_samples": 10,
                       "minimum": 0.8, "maximum": 1.0,
                       "max_absolute_regression": 0.01, "max_relative_regression": 0.02}]}


def measured(value: float = 0.9) -> dict:
    return {"schema_version": 1,
            "identity": {"evaluation_id": "eval-1", "run_id": "run-1", "model_digest": digest("b"),
                         "build_fingerprint": "actual-measurement-producer", "adapter_id": "classification.local",
                         "adapter_version": 1, "config_digest": digest("c")},
            "observations": [{"definition": definition(), "step": {"kind": 4, "value": 10},
                              "observation": {"status": 1, "value": value, "reason": ""},
                              "context": {"split": "validation"}, "sample_count": 10}]}


class EvaluationDataCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="jyppx-evaluation-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "中文 evidence"
        self.root.mkdir()
        self.suite = self.write("suite.json", suite())
        self.input = self.write("input.json", measured())

    def write(self, name: str, data: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def invoke(self, *args, expected: int = 0, text: bool = False) -> dict | str:
        result = subprocess.run([str(CLI), *map(str, args), "--format", "text" if text else "json"],
                                capture_output=True, text=True, encoding="utf-8", timeout=45)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertFalse(result.stderr, result.stderr)
        if text:
            self.assertFalse(result.stdout.lstrip().startswith("{"))
            return result.stdout
        value = json.loads(result.stdout)
        self.assertEqual(value["schema_version"], 1)
        return value

    def evaluate(self, *options, expected: int = 0) -> dict:
        return self.invoke("evaluation", "run", "--suite", self.suite, "--input", self.input,
                           *options, expected=expected)["data"]

    def test_report_canonical_hash_html_replay_and_text(self) -> None:
        output, html = self.root / "quality.json", self.root / "quality.html"
        first = self.evaluate("--output", output, "--html", html)
        self.assertEqual(first["verdict"], "pass")
        self.assertEqual(first["report_digest"]["hex"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(first["report"]["identity"], measured()["identity"])
        self.assertEqual(first, self.evaluate("--output", output, "--html", html))
        self.assertTrue(html.read_text(encoding="utf-8").startswith("<!doctype html>"))
        shown = self.invoke("evaluation", "show", "--report", output, text=True)
        self.assertIn("verdict: pass", shown)
        self.assertIn("suite_id: classification.smoke", shown)

    def test_measured_failure_and_missing_are_nonzero_completed_verdicts(self) -> None:
        self.write("input.json", measured(0.7))
        self.assertEqual(self.evaluate(expected=1)["verdict"], "fail")
        value = measured()
        value["observations"] = []
        self.write("input.json", value)
        self.assertEqual(self.evaluate(expected=1)["verdict"], "incomplete")

    def test_unknown_semantics_and_duplicate_json_are_rejected(self) -> None:
        value = measured()
        value["observations"][0]["definition"]["unit"] = 6
        self.write("input.json", value)
        result = self.invoke("evaluation", "run", "--suite", self.suite, "--input", self.input, expected=2)
        self.assertEqual(result["error"]["name"], "compatibility")
        self.input.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        result = self.invoke("evaluation", "run", "--suite", self.suite, "--input", self.input, expected=2)
        self.assertEqual(result["error"]["name"], "serialization")

    def test_compare_direction_and_provenance_mismatch(self) -> None:
        baseline, candidate = self.root / "baseline.json", self.root / "candidate.json"
        self.evaluate("--output", baseline)
        value = measured(0.87)
        value["identity"]["evaluation_id"] = "eval-2"
        value["identity"]["model_digest"] = digest("d")
        self.write("input.json", value)
        self.evaluate("--output", candidate)
        result = self.invoke("evaluation", "compare", "--baseline", baseline, "--candidate", candidate, expected=1)["data"]
        self.assertEqual(result["verdict"], "fail")
        self.assertAlmostEqual(result["results"][0]["regression"], 0.03)
        changed = suite()
        changed["environment_id"] = "different-workload"
        self.write("suite.json", changed)
        candidate2 = self.root / "candidate2.json"
        self.evaluate("--output", candidate2)
        result = self.invoke("evaluation", "compare", "--baseline", baseline, "--candidate", candidate2, expected=2)
        self.assertEqual(result["error"]["name"], "compatibility")

    def test_forged_report_and_existing_output_are_preserved(self) -> None:
        output = self.root / "quality.json"
        self.evaluate("--output", output)
        original = output.read_bytes()
        self.write("input.json", measured(0.7))
        self.invoke("evaluation", "run", "--suite", self.suite, "--input", self.input,
                    "--output", output, expected=3)
        self.assertEqual(output.read_bytes(), original)
        forged = json.loads(original)
        forged["verdict"] = 2
        self.write("forged.json", forged)
        result = self.invoke("evaluation", "show", "--report", self.root / "forged.json", expected=2)
        self.assertEqual(result["status"], "error")

    def test_report_retained_in_cas_and_idempotent_relationship(self) -> None:
        storage = self.root / "runs"
        metadata = self.write("run.json", {"schema_version": 1, "run_id": "run-1", "project_id": "project-1",
            "experiment_id": "experiment-1", "display_name": "Local run", "created_at_unix_ms": 10})
        self.invoke("run", "start", "--root", storage, "--metadata", metadata)
        first = self.evaluate("--root", storage, "--reference-id", "eval-1.report", "--attached-ms", 20)
        second = self.evaluate("--root", storage, "--reference-id", "eval-1.report", "--attached-ms", 20)
        self.assertEqual(first, second)
        references = self.invoke("run", "show", "--root", storage, "--id", "run-1")["data"]["references"]
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0]["artifact"]["sha256"], first["report_digest"]["hex"])
        self.assertTrue(self.invoke("run", "verify", "--root", storage)["data"]["valid"])
        with sqlite3.connect(storage / "runs.sqlite3") as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 2)

    def test_data_checks_leakage_conflicting_label_without_sample_disclosure(self) -> None:
        row = {"sample_id": "private-sample", "split": "train", "content_digest": digest("a"),
               "fields": {"feature": 0.5}, "label": "private-label"}
        data = {"schema_version": 1, "schema": {"feature": {"required": True, "minimum": 0, "maximum": 1}},
                "rows": [row]}
        path = self.write("data.json", data)
        self.assertEqual(self.invoke("data", "inspect", "--input", path)["data"]["verdict"], "pass")
        other = copy.deepcopy(row)
        other.update(split="validation", label="different-private-label", fields={"feature": 2})
        data["rows"].append(other)
        self.write("data.json", data)
        result = self.invoke("data", "verify", "--input", path, expected=1)["data"]
        self.assertEqual(result["cross_split_leaks"], 1)
        self.assertEqual(result["conflicting_labels"], 1)
        self.assertEqual(result["out_of_range_values"], 1)
        self.assertNotIn("private", json.dumps(result))

    def test_config_precedence_and_secret_redaction(self) -> None:
        schema = self.write("config-schema.json", {"schema_version": 1, "schema_id": "cli.test", "version": 1,
            "root": {"kind": "dictionary", "fields": {"rate": {"kind": "floating"}, "token": {"kind": "string"}}},
            "defaults": {"rate": 1, "token": "private-token-123"}, "secret_paths": ["/token"]})
        result = self.invoke("config", "render", "--schema", schema, "--set", "/rate=0.25")["data"]
        self.assertEqual(result["config"]["rate"], 0.25)
        self.assertNotIn("private-token-123", json.dumps(result))
        explanation = self.invoke("config", "explain", "--schema", schema, "--path", "/token")["data"]
        self.assertNotIn("private-token-123", json.dumps(explanation))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    args = parser.parse_args()
    CLI = args.cli.resolve(strict=True)
    unittest.main(argv=[__file__], verbosity=2)
