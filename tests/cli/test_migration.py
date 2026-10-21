# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Product CLI migration contract; run once at the chapter validation boundary.

The user-owned example schema below is a fixture with a real directed
v1 -> v2 -> v3 and explicit reverse chain, not a claimed historic global format.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def schema(version: int, fields: dict, secret: str | None) -> dict:
    result = {
        "schema_version": 1, "schema_id": "example.training", "version": version,
        "root": {"kind": "dictionary", "fields": fields},
    }
    if secret is not None:
        result["secret_paths"] = [secret]
    return result


def migration_plan() -> dict:
    integer = {"kind": "integer"}
    string = {"kind": "string"}
    nested = lambda fields: {"kind": "dictionary", "fields": fields}
    return {
        "schema_version": 1, "plan_id": "example-training-v1-v3",
        "feature_id": "CONFIG-MIGRATION", "schema_id": "example.training",
        "schemas": [
            schema(1, {"timeout_ms": integer, "api_key": string, "retries": integer}, "/api_key"),
            schema(2, {"timeout_seconds": integer, "token": string, "retries": integer}, "/token"),
            schema(3, {"runtime": nested({"timeout_seconds": integer, "token": string}),
                       "retry": integer}, "/runtime/token"),
        ],
        "steps": [
            {"from_version": 1, "to_version": 2, "breaking_change": "timeout-units-token-name",
             "operations": [
                 {"kind": "rename", "from": "/timeout_ms", "path": "/timeout_seconds"},
                 {"kind": "scale-int", "path": "/timeout_seconds", "numerator": 1, "denominator": 1000},
                 {"kind": "rename", "from": "/api_key", "path": "/token"},
             ]},
            {"from_version": 2, "to_version": 3, "breaking_change": "runtime-nesting",
             "operations": [
                 {"kind": "set-default", "path": "/runtime", "value": {}},
                 {"kind": "rename", "from": "/timeout_seconds", "path": "/runtime/timeout_seconds"},
                 {"kind": "rename", "from": "/token", "path": "/runtime/token"},
                 {"kind": "rename", "from": "/retries", "path": "/retry"},
             ]},
            {"from_version": 3, "to_version": 2, "breaking_change": "runtime-unnesting",
             "operations": [
                 {"kind": "rename", "from": "/runtime/timeout_seconds", "path": "/timeout_seconds"},
                 {"kind": "rename", "from": "/runtime/token", "path": "/token"},
                 {"kind": "rename", "from": "/retry", "path": "/retries"},
                 {"kind": "remove-checked", "path": "/runtime", "expected": {}},
             ]},
            {"from_version": 2, "to_version": 1, "breaking_change": "timeout-units-token-restore",
             "operations": [
                 {"kind": "scale-int", "path": "/timeout_seconds", "numerator": 1000, "denominator": 1},
                 {"kind": "rename", "from": "/timeout_seconds", "path": "/timeout_ms"},
                 {"kind": "rename", "from": "/token", "path": "/api_key"},
             ]},
        ],
    }


class MigrationContract(unittest.TestCase):
    cli: Path

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="jyppx-migration-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / "input.json"
        self.plan_path = self.root / "plan.json"
        self.output = self.root / "migration"
        self.plan = migration_plan()
        self.document = {
            "schema_version": 1, "schema_id": "example.training", "version": 1,
            "values": {"timeout_ms": 5000, "api_key": "secret://vault/team/token?version=v1", "retries": 3},
        }
        self.write_inputs()

    def write_inputs(self) -> None:
        self.input.write_text(json.dumps(self.document, indent=2) + "\n", encoding="utf-8")
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")

    def invoke(self, *args: object, expected: int = 0) -> dict:
        process = subprocess.run([str(self.cli), *(str(value) for value in args), "--format", "json"],
                                 capture_output=True, encoding="utf-8", timeout=30, check=False)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        self.assertNotIn("secret://", process.stdout + process.stderr)
        self.assertNotIn("plain-secret-fixture", process.stdout + process.stderr)
        envelope = json.loads(process.stdout)
        return envelope.get("data", envelope)

    def migrate(self, *, mode: str = "preview", source: Path | None = None,
                target: int = 3, output: Path | None = None, expected: int = 0) -> dict:
        args = ["config", "migrate", "--mode", mode, "--input", source or self.input,
                "--plan", self.plan_path, "--to-version", target]
        if output is not None:
            args += ["--output", output]
        return self.invoke(*args, expected=expected)

    def test_preview_apply_idempotence_and_explicit_reverse_round_trip(self) -> None:
        original = self.input.read_bytes()
        preview = self.migrate()
        self.assertEqual(preview["mode"], "preview")
        self.assertEqual(preview["from_version"], 1)
        self.assertEqual(preview["to_version"], 3)
        self.assertEqual(len(preview["changes"]), 7)
        self.assertFalse(self.output.exists())
        result = self.migrate(mode="apply", output=self.output)
        self.assertTrue(result["source_untouched"])
        self.assertEqual(self.input.read_bytes(), original)
        self.assertEqual((self.output / "original.json").read_bytes(), original)
        migrated = json.loads((self.output / "migrated.json").read_bytes())
        self.assertEqual(migrated["values"], {
            "runtime": {"timeout_seconds": 5, "token": self.document["values"]["api_key"]}, "retry": 3,
        })
        repeated = self.migrate(source=self.output / "migrated.json")
        self.assertTrue(repeated["already_at_target"])
        self.assertEqual(repeated["changes"], [])
        reverse_output = self.root / "reverse"
        self.migrate(mode="apply", source=self.output / "migrated.json", target=1, output=reverse_output)
        self.assertEqual(json.loads((reverse_output / "migrated.json").read_bytes()), self.document)
        self.migrate(mode="apply", output=self.output, expected=3)

    def test_rollback_validates_exact_receipt_plan_and_original_bytes(self) -> None:
        self.migrate(mode="apply", output=self.output)
        rollback = self.root / "rollback"
        result = self.invoke("config", "migrate", "--mode", "rollback", "--receipt", self.output,
                             "--plan", self.plan_path, "--output", rollback)
        self.assertTrue(result["restored"])
        self.assertEqual((rollback / "restored.json").read_bytes(), self.input.read_bytes())
        (self.output / "original.json").write_bytes(b"tampered")
        self.invoke("config", "migrate", "--mode", "rollback", "--receipt", self.output,
                    "--plan", self.plan_path, "--output", self.root / "invalid", expected=2)

    def test_refuses_missing_reverse_future_version_unknown_fields_and_duplicate_keys(self) -> None:
        self.migrate(mode="apply", output=self.output)
        self.plan["steps"] = [step for step in self.plan["steps"] if step["from_version"] < step["to_version"]]
        self.write_inputs()
        self.migrate(source=self.output / "migrated.json", target=1, expected=2)
        self.document["version"] = 9
        self.write_inputs()
        self.migrate(expected=2)
        self.document["version"] = 1
        self.document["values"]["unknown"] = True
        self.write_inputs()
        self.migrate(expected=2)
        self.input.write_text('{"schema_version":1,' + self.input.read_text()[1:], encoding="utf-8")
        self.migrate(expected=2)

    def test_intermediate_schema_precondition_and_secret_failures_preserve_source(self) -> None:
        for mutation in ("intermediate", "target", "collision", "secret", "precondition"):
            self.plan = migration_plan()
            self.document["values"] = {"timeout_ms": 5000, "api_key": "secret://vault/team/token", "retries": 3}
            if mutation == "intermediate":
                self.plan["schemas"][1]["root"]["fields"]["timeout_seconds"] = {"kind": "boolean"}
            elif mutation == "target":
                self.plan["schemas"][2]["root"]["fields"]["retry"] = {"kind": "string"}
            elif mutation == "collision":
                self.plan["steps"][0]["operations"][0]["path"] = "/api_key"
            elif mutation == "secret":
                self.document["values"]["api_key"] = "plain-secret-fixture"
            else:
                self.plan["steps"][0]["operations"].insert(0, {
                    "kind": "replace", "path": "/retries", "expected": 99, "value": 3,
                })
            self.write_inputs()
            before = self.input.read_bytes()
            self.migrate(mode="apply", output=self.output, expected=2)
            self.assertEqual(self.input.read_bytes(), before)
            self.assertFalse(self.output.exists())

    def test_exact_integer_conversion_rejects_rounding_and_overflow(self) -> None:
        self.document["values"]["timeout_ms"] = 5001
        self.write_inputs()
        self.migrate(expected=2)
        self.document["values"]["timeout_ms"] = 2**63 - 1
        self.plan["steps"][0]["operations"][1] = {
            "kind": "scale-int", "path": "/timeout_seconds", "numerator": 2, "denominator": 1,
        }
        self.write_inputs()
        self.migrate(expected=3)
        self.document["values"]["timeout_ms"] = -(2**63)
        self.plan["steps"][0]["operations"][1]["numerator"] = 1
        self.write_inputs()
        self.migrate(mode="apply", output=self.output)
        migrated = json.loads((self.output / "migrated.json").read_bytes())
        self.assertEqual(migrated["values"]["runtime"]["timeout_seconds"], -(2**63))

    def test_user_defined_copy_defaults_and_checked_replacement(self) -> None:
        self.plan["schemas"][1]["root"]["fields"]["copied_retries"] = {"kind": "integer"}
        self.plan["steps"][0]["operations"] += [
            {"kind": "copy", "from": "/retries", "path": "/copied_retries"},
            {"kind": "set-default", "path": "/retries", "value": 99},
            {"kind": "replace", "path": "/copied_retries", "expected": 3, "value": 4},
        ]
        self.write_inputs()
        self.migrate(mode="apply", target=2, output=self.output)
        migrated = json.loads((self.output / "migrated.json").read_bytes())
        self.assertEqual(migrated["values"]["retries"], 3)
        self.assertEqual(migrated["values"]["copied_retries"], 4)

    def test_post_operation_expansion_budget_rejects_before_any_publication(self) -> None:
        payload = {"field" + str(index): index for index in range(1800)}
        fields = {key: {"kind": "integer"} for key in payload}
        dictionary = {"kind": "dictionary", "fields": fields}
        self.plan["schemas"][0]["root"]["fields"]["payload"] = dictionary
        self.document["values"]["payload"] = payload
        self.plan["steps"][0]["operations"].insert(
            0, {"kind": "copy", "from": "/payload", "path": "/extra_payload"})
        self.write_inputs()
        self.migrate(mode="apply", output=self.output, expected=3)
        self.assertFalse(self.output.exists())

    def test_link_traversal_oversize_and_overlap_are_rejected(self) -> None:
        alias = self.root / "alias.json"
        os.link(self.input, alias)
        self.migrate(expected=2)
        alias.unlink()
        self.input.write_bytes(b" " * (1024 * 1024 + 1))
        self.migrate(expected=3)
        self.write_inputs()
        self.migrate(source=self.root / "nested" / ".." / "input.json", expected=2)
        self.plan["steps"][0]["operations"][0]["path"] = "/timeout_ms/child"
        self.write_inputs()
        self.migrate(expected=2)
        try:
            alias.symlink_to(self.input)
        except OSError:
            return
        self.migrate(source=alias, expected=2)

    def test_api_scan_is_lexical_bounded_read_only_and_has_actionable_findings(self) -> None:
        tree = self.root / "source"
        tree.mkdir()
        source = tree / "main.cpp"
        source.write_text('#include <jyppx/version.hpp>\n// jyppx/optim/sgd.hpp\n', encoding="utf-8")
        (tree / "CMakeLists.txt").write_text("find_package(jyppx_training_framework CONFIG)\n", encoding="utf-8")
        ignored = tree / "third_party"
        ignored.mkdir()
        (ignored / "fixture.cpp").write_text("jyppx::utils", encoding="utf-8")
        before = source.read_bytes()
        result = self.invoke("api", "scan", "--root", tree, expected=1)
        self.assertEqual(result["scanner"], "bounded-lexical-v1")
        self.assertEqual(result["files_scanned"], 2)
        self.assertEqual(len(result["findings"]), 3)
        self.assertTrue(all(not item["automatic_change"] for item in result["findings"]))
        self.assertEqual(source.read_bytes(), before)
        self.invoke("api", "scan", "--root", tree, "--max-findings", 1, expected=3)
        source.write_text("int main() {}\n", encoding="utf-8")
        (tree / "CMakeLists.txt").write_text("find_package(jyppx CONFIG)\n", encoding="utf-8")
        self.assertEqual(self.invoke("api", "scan", "--root", tree)["findings"], [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    MigrationContract.cli = arguments.cli.resolve()
    unittest.main(argv=[__file__, *remaining])
