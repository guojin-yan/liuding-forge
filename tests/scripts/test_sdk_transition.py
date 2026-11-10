#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0.

"""Receipt replay must be read-only and bind the original native invocation."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("sdk_transition", ROOT / "scripts/release/check_sdk_transition.py")
transition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transition)


class SdkTransition(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_tree_proof_detects_membership_content_and_mtime_changes(self):
        path = self.root / "state.bin"
        path.write_bytes(b"original")
        proof = transition.tree_state(self.root)
        path.write_bytes(b"changed!")
        self.assertNotEqual(transition.tree_state(self.root), proof)
        path.write_bytes(b"original")
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
        self.assertNotEqual(transition.tree_state(self.root), proof)
        path.unlink()
        self.assertNotEqual(transition.tree_state(self.root), proof)

    def saved_call(self):
        cli = self.root / "native-cli"
        output = b'{"schema_version":1,"status":"ok","data":{"retained":true}}\n'
        (self.root / "probe.stdout.json").write_bytes(output)
        (self.root / "probe.stderr.log").write_bytes(b"")
        call = {"name": "probe", "command": [str(cli), "--version", "--format", "json"],
                "returncode": 0, "seconds": .1,
                "stdout_sha256": hashlib.sha256(output).hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest()}
        transition.write_json(self.root / "probe.call.json", call)
        return cli, call

    def test_saved_verification_never_starts_a_process_or_changes_receipts(self):
        cli, _ = self.saved_call()
        proof = transition.tree_state(self.root)
        replay = transition.Rehearsal(self.root, cli, cli, 30, saved=True)
        with mock.patch.object(transition.subprocess, "run", side_effect=AssertionError("must remain offline")):
            result = replay.invoke(cli, "probe", "--version")
        self.assertTrue(result["data"]["retained"])
        self.assertEqual(transition.tree_state(self.root), proof)

    def test_saved_verification_rejects_a_different_command(self):
        cli, _ = self.saved_call()
        replay = transition.Rehearsal(self.root, cli, cli, 30, saved=True)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            replay.invoke(cli, "probe", "recipe", "resume")

    def test_saved_verification_rejects_changed_native_output(self):
        cli, _ = self.saved_call()
        (self.root / "probe.stdout.json").write_bytes(b'{}\n')
        replay = transition.Rehearsal(self.root, cli, cli, 30, saved=True)
        with self.assertRaisesRegex(ValueError, "output differs"):
            replay.invoke(cli, "probe", "--version")

    def test_result_publication_preserves_an_existing_failure(self):
        path = self.root / "transition.json"
        transition.write_json(path, {"passed": False})
        with self.assertRaises(FileExistsError):
            transition.write_json(path, {"passed": True})
        self.assertEqual(json.loads(path.read_text()), {"passed": False})


if __name__ == "__main__":
    unittest.main()
