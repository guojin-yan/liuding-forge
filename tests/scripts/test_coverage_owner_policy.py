"""Pure policy/owner fixtures; no gcovr, build, product or real coverage qualification."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/validation/coverage_gate.py"
SPEC = importlib.util.spec_from_file_location("coverage_gate", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class CoverageOwnerPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="coverage-owner-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.umbrella = "include/jyppx/liuding/liuding.hpp"
        self.write(self.umbrella, b"#ifndef FIXTURE_HPP\n#define FIXTURE_HPP\n#include <fixture.hpp>\n#endif\n")
        for name in ("stable", "new"):
            self.write(f"src/jyppx/{name}/unit.cpp", b"// Synthetic source; never compiled or instrumented.\n")
        self.policy = {
            "schema_version": 2, "kind": "jyppx-target-coverage-policy",
            "baseline": {"commit": "fixture-only", "environment": "fixture-only"},
            "targets": {
                "stable": {"path_prefixes": ["src/jyppx/stable/"], "metrics": {
                    metric: {"baseline_covered": 1, "baseline_total": 1, "minimum_percent": 100}
                    for metric in GATE.METRIC_NAMES}},
                "new": {"path_prefixes": ["src/jyppx/new/"], "baseline_status": "pending-observation"},
            },
            "observations": {},
            "non_executable_files": [{"path": self.umbrella, "sha256": hashlib.sha256((self.root / self.umbrella).read_bytes()).hexdigest(), "reason": "Synthetic include-only fixture"}],
            "profile": {"id": "synthetic-fixture-not-release-coverage", "cmake_options": {"JYPPX_FIXTURE_COMPONENT": "ON"}},
        }
        self.report = {"files": []}
        for name in ("stable", "new"):
            row = {"filename": f"src/jyppx/{name}/unit.cpp"}
            for metric in GATE.METRIC_NAMES:
                row.update({metric + "_covered": 1, metric + "_total": 1})
            self.report["files"].append(row)
        for metric in GATE.METRIC_NAMES:
            self.report.update({metric + "_covered": 2, metric + "_total": 2})
        self.cache = self.root / "fixture-CMakeCache.txt"
        self.cache.write_text("JYPPX_FIXTURE_COMPONENT:BOOL=ON\n", encoding="utf-8")

    def write(self, relative: str, raw: bytes) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def audit(self, policy: dict | None = None) -> dict:
        policy = self.policy if policy is None else policy
        return GATE._verify_source_ownership(self.root, GATE._read_policy_targets(policy), GATE._read_non_executable_files(policy))

    def cli(self, *, audit: bool = False, report: dict | None = None) -> subprocess.CompletedProcess[str]:
        policy_path = self.root / "fixture-policy.json"
        policy_path.write_text(json.dumps(self.policy), encoding="utf-8")
        command = [sys.executable, "-I", "-B", str(SCRIPT)]
        if not audit:
            report_path = self.root / "fixture-report.json"
            report_path.write_text(json.dumps(self.report if report is None else report), encoding="utf-8")
            command.append(str(report_path))
        command.extend([str(policy_path), "--source-root", str(self.root), "--output", str(self.root / "fixture-output.json")])
        command.extend(["--audit-owners"] if audit else ["--cmake-cache", str(self.cache)])
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_legacy_fixture_passes_and_three_ratchets_remain(self) -> None:
        fixture = ROOT / "tests/fixtures/coverage"
        policy = GATE._read_object(fixture / "target-policy.json", "fixture")
        targets = GATE._read_policy_targets(policy)
        for filename, fails in (("target-summary.json", False), ("target-summary-regressed.json", True), ("target-summary-shrunk.json", True)):
            with self.subTest(filename=filename):
                report = GATE._read_object(fixture / filename, "fixture")
                observed = GATE._aggregate(GATE._read_report_files(report), targets)
                GATE._verify_report_totals(report, observed)
                self.assertEqual(bool(GATE._evaluate(observed, targets)), fails)

    def test_unmeasured_owner_never_passes_even_at_fixture_full_coverage(self) -> None:
        targets = GATE._read_policy_targets(self.policy)
        observed = GATE._aggregate(GATE._read_report_files(self.report), targets)
        self.assertEqual(GATE._evaluate(observed, targets), ["new baseline pending observation and review"])

    def test_pending_owner_rejects_metrics_including_zero_or_null(self) -> None:
        for metrics in ({}, None, {"line": {"baseline_total": 0}}):
            policy = copy.deepcopy(self.policy)
            policy["targets"]["new"]["metrics"] = metrics
            with self.subTest(metrics=metrics), self.assertRaisesRegex(ValueError, "pending baseline"):
                GATE._read_policy_targets(policy)

    def test_v1_cannot_silently_ignore_pending_owner(self) -> None:
        self.policy["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "requires v2"):
            GATE._read_policy_targets(self.policy)

    def test_unique_ownership_and_exact_non_executable_exception(self) -> None:
        result = self.audit()
        self.assertEqual(result["source_file_count"], 3)
        self.assertEqual(result["non_executable_files"], [self.umbrella])
        self.assertFalse(result["coverage_evaluated"])
        self.assertFalse(result["release_qualified"])

    def test_unobserved_new_source_is_still_rejected_when_unowned(self) -> None:
        self.write("src/jyppx/unknown/unit.cpp", b"// Not present in any synthetic report.\n")
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            self.audit()

    def test_executable_inc_is_owned_and_counted_in_report(self) -> None:
        filename = "src/jyppx/new/detail/adapter.inc"
        self.write(filename, b"int included_implementation() { return 1; }\n")
        audit = self.audit()
        self.assertEqual(audit["file_owners"].get(filename), "new")
        self.assertEqual(audit["source_file_count"], 4)
        row = copy.deepcopy(self.report["files"][1])
        row["filename"] = filename
        self.report["files"].append(row)
        for metric in GATE.METRIC_NAMES:
            self.report[metric + "_covered"] += 1
            self.report[metric + "_total"] += 1
        result = self.cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("pending observation and review", result.stderr)
        output = json.loads((self.root / "fixture-output.json").read_bytes())
        self.assertIn(filename, output["targets"]["new"]["files"])
        self.assertEqual(output["targets"]["new"]["metrics"]["line"]["total"], 2)

    def test_unowned_inc_cannot_disappear_from_inventory(self) -> None:
        self.write("src/jyppx/unknown/detail/adapter.inc",
                   b"int unowned_implementation() { return 1; }\n")
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            self.audit()

    def test_parent_and_child_prefixes_cannot_hide_double_ownership(self) -> None:
        self.policy["targets"]["new"]["path_prefixes"] = ["src/jyppx/"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            GATE._read_policy_targets(self.policy)

    def test_exception_rejects_changed_bytes_and_executable_code_even_with_new_hash(self) -> None:
        path = self.root / self.umbrella
        path.write_bytes(path.read_bytes() + b"// changed bytes\n")
        with self.assertRaisesRegex(ValueError, "changed or emits code"):
            self.audit()
        raw = b"#ifndef FIXTURE_HPP\n#define FIXTURE_HPP\ninline int hidden() { return 1; }\n#endif\n"
        path.write_bytes(raw)
        self.policy["non_executable_files"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "changed or emits code"):
            self.audit()

    def test_lf_fingerprint_accepts_lf_and_crlf_without_granting_coverage(self) -> None:
        raw = (self.root / self.umbrella).read_bytes()
        self.policy["non_executable_files"][0]["normalization"] = "LF"
        for header in (raw, raw.replace(b"\n", b"\r\n")):
            with self.subTest(header=header):
                self.write(self.umbrella, header)
                result = self.audit()
                self.assertEqual(result["non_executable_files"], [self.umbrella])
                self.assertEqual(result["source_file_count"], 3)
                self.assertFalse(result["coverage_evaluated"])
                self.assertFalse(result["release_qualified"])

    def test_undeclared_normalization_keeps_exact_raw_fingerprint(self) -> None:
        raw = (self.root / self.umbrella).read_bytes()
        self.write(self.umbrella, raw.replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(ValueError, "changed or emits code"):
            self.audit()

    def test_lf_fingerprint_rejects_content_whitespace_and_comment_changes(self) -> None:
        raw = (self.root / self.umbrella).read_bytes()
        self.policy["non_executable_files"][0]["normalization"] = "LF"
        for changed in (raw.replace(b"<fixture.hpp>", b"<other.hpp>"),
                        raw.replace(b"#include ", b"#include  "),
                        raw + b"// changed comment\n"):
            with self.subTest(changed=changed):
                self.write(self.umbrella, changed.replace(b"\n", b"\r\n"))
                with self.assertRaisesRegex(ValueError, "changed or emits code"):
                    self.audit()

    def test_lf_fingerprint_rejects_executable_code_even_with_updated_pin(self) -> None:
        raw = b"#ifndef FIXTURE_HPP\n#define FIXTURE_HPP\ninline int hidden() { return 1; }\n#endif\n"
        self.policy["non_executable_files"][0].update(
            normalization="LF", sha256=hashlib.sha256(raw).hexdigest())
        self.write(self.umbrella, raw.replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(ValueError, "changed or emits code"):
            self.audit()

    def test_lf_fingerprint_rejects_bare_cr_even_with_updated_pin(self) -> None:
        raw = (self.root / self.umbrella).read_bytes().replace(b"\n", b"\r")
        self.policy["non_executable_files"][0].update(
            normalization="LF", sha256=hashlib.sha256(raw).hexdigest())
        self.write(self.umbrella, raw)
        with self.assertRaisesRegex(ValueError, "rejects bare CR"):
            self.audit()

    def test_normalization_descriptor_rejects_unknown_values_types_and_fields(self) -> None:
        for value in ("lf", "CRLF", "", None, True, 0, [], {}):
            policy = copy.deepcopy(self.policy)
            policy["non_executable_files"][0]["normalization"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "normalization must be LF"):
                self.audit(policy)
        policy = copy.deepcopy(self.policy)
        policy["non_executable_files"][0].update(normalization="LF", ignored=True)
        with self.assertRaisesRegex(ValueError, "invalid non-executable file descriptor"):
            self.audit(policy)

    def test_report_cannot_assign_nonzero_entities_to_umbrella(self) -> None:
        row = copy.deepcopy(self.report["files"][0])
        row["filename"] = self.umbrella
        files = GATE._read_report_files({"files": [row]})
        with self.assertRaisesRegex(ValueError, "coverage entities"):
            GATE._aggregate(files, GATE._read_policy_targets(self.policy), GATE._read_non_executable_files(self.policy))

    def test_actual_profile_cache_requires_exact_typed_flags(self) -> None:
        profile = GATE._read_profile(self.policy)
        GATE._verify_profile(self.cache, profile)
        for raw in ("", "JYPPX_FIXTURE_COMPONENT:BOOL=OFF\n", "JYPPX_FIXTURE_COMPONENT:STRING=ON\n", "JYPPX_FIXTURE_COMPONENT:BOOL=ON\nJYPPX_FIXTURE_COMPONENT:BOOL=ON\n"):
            self.cache.write_text(raw, encoding="utf-8")
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                GATE._verify_profile(self.cache, profile)
        with self.assertRaisesRegex(ValueError, "actual --cmake-cache"):
            GATE._verify_profile(None, profile)

    def test_profile_malformed_values_fail_as_validation_errors(self) -> None:
        for value in ([], True, "TRUE"):
            self.policy["profile"]["cmake_options"]["JYPPX_FIXTURE_COMPONENT"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                GATE._read_profile(self.policy)

    def test_owner_audit_cli_has_no_report_and_never_grants_coverage(self) -> None:
        result = self.cli(audit=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads((self.root / "fixture-output.json").read_bytes())
        self.assertFalse(output["coverage_evaluated"])
        self.assertFalse(output["release_qualified"])

    def test_fixture_observation_is_written_but_pending_gate_rejects(self) -> None:
        result = self.cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("pending observation and review", result.stderr)
        output = json.loads((self.root / "fixture-output.json").read_bytes())
        self.assertEqual(output["pending_baseline_targets"], ["new"])
        self.assertEqual(output["profile"], "synthetic-fixture-not-release-coverage")

    def test_nonexistent_report_file_cannot_pad_a_known_prefix(self) -> None:
        self.report["files"][1]["filename"] = "src/jyppx/new/missing.cpp"
        result = self.cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("outside the audited source inventory", result.stderr)

    def test_duplicate_policy_json_keys_and_boolean_schema_rejected(self) -> None:
        path = self.root / "duplicate.json"
        path.write_text('{"targets":{},"targets":{}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "repeats JSON key"):
            GATE._read_object(path, "fixture")
        self.policy["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "schema_version"):
            GATE._read_policy_targets(self.policy)


if __name__ == "__main__":
    unittest.main()
