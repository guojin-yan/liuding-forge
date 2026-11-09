# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Check real CLI-test discovery and grouping without launching a product CLI."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "cli/test_local_attempt.py"


class LocalAttemptRegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="liuding-attempt-registration-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Discovery must succeed with a nonexistent CLI and create no artifacts.
        self.cli = self.root / "missing-product-cli"

    def invoke(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), "--cli", str(self.cli),
                               "--list-contracts", *arguments],
                              capture_output=True, text=True, timeout=10, check=False)

    def names(self, group: str = "all") -> set[str]:
        result = self.invoke("--contract-group", group)
        self.assertEqual(result.returncode, 0, result.stderr)
        names = json.loads(result.stdout)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(list(self.root.iterdir()), [])
        return set(names)

    def test_default_discovery_matches_the_actual_test_case(self) -> None:
        spec = importlib.util.spec_from_file_location("local_attempt_contract", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = {"LocalAttemptTest." + name for name in
                    unittest.defaultTestLoader.getTestCaseNames(module.LocalAttemptTest)}
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(json.loads(result.stdout)), expected)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_groups_are_nonempty_disjoint_and_exhaustive(self) -> None:
        union = set()
        for group in ("core", "resources", "environment", "runtime_files", "host"):
            selected = self.names(group)
            self.assertTrue(selected, group)
            self.assertFalse(union & selected, group)
            union |= selected
        self.assertEqual(union, self.names())

    def test_representative_contracts_remain_in_their_groups(self) -> None:
        for group, method in (
                ("core", "test_success_literal_argv_raw_cas_and_terminal_replay"),
                ("resources", "test_os_memory_boundary_is_persisted_and_replayed_without_relaunch"),
                ("environment", "test_environment_reference_digest_fences_retry_drift_and_preserves_confirmed_prefix"),
                ("runtime_files", "test_runtime_files_fence_retry_drift_and_preserve_confirmed_prefix"),
                ("host", "test_host_disk_requirement_preserves_retry_prefix_until_space_restored")):
            with self.subTest(group=group):
                self.assertIn("LocalAttemptTest." + method, self.names(group))

    def test_invalid_group_and_discovery_filters_fail_before_launch(self) -> None:
        for arguments in (("--contract-group", "missing"), ("-k", "test_success")):
            with self.subTest(arguments=arguments):
                result = self.invoke(*arguments)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_unittest_pattern_filters_are_preserved(self) -> None:
        # Python 3.14 gives an empty suite its own exit code; preserve the
        # interpreter's unittest behavior instead of manufacturing a pass.
        reference = subprocess.run([sys.executable, "-I", "-B", "-c", "import unittest; unittest.main()"],
            capture_output=True, text=True, timeout=10, check=False)
        self.assertIn("Ran 0 tests", reference.stderr)
        for group in ("all", "host"):
            with self.subTest(group=group):
                result = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT),
                    "--cli", str(self.cli), "--contract-group", group,
                    "-k", "__liuding_no_matching_contract__"],
                    capture_output=True, text=True, timeout=10, check=False)
                self.assertEqual(result.returncode, reference.returncode, result.stderr)
                self.assertIn("Ran 0 tests", result.stderr)
                self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
