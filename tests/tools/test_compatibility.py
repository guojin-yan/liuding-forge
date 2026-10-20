# Copyright 2026 JYPPX Training Framework Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the canonical LibTorch compatibility source validator."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import compatibility  # noqa: E402


class CompatibilityValidatorTest(unittest.TestCase):
    """Covers matrix identity, structure, and evidence invariants."""

    def setUp(self) -> None:
        """Load a fresh copy of the repository compatibility source."""
        self.document = compatibility.load_document(
            REPOSITORY_ROOT / "compatibility.json"
        )

    def test_repository_source_is_valid(self) -> None:
        """Accept the checked-in compatibility source."""
        compatibility.validate(self.document)

    def test_rejects_duplicate_verified_lane_id_across_versions(self) -> None:
        """Reject an identity reused by the primary and candidate matrices."""
        changed = copy.deepcopy(self.document)
        primary_id = changed["libtorch"]["primary"]["verified_lanes"][0]["id"]
        changed["libtorch"]["compatibility_candidate"]["verified_lanes"][0][
            "id"
        ] = primary_id
        with self.assertRaisesRegex(
            compatibility.CompatibilityError, "duplicate verified lane id"
        ):
            compatibility.validate(changed)

    def test_rejects_missing_environment_field(self) -> None:
        """Reject a lane that cannot identify its compiler environment."""
        changed = copy.deepcopy(self.document)
        del changed["libtorch"]["compatibility_candidate"]["verified_lanes"][0][
            "compiler"
        ]
        with self.assertRaisesRegex(
            compatibility.CompatibilityError, "compiler must be a non-empty string"
        ):
            compatibility.validate(changed)

    def test_rejects_empty_quality_evidence(self) -> None:
        """Reject a claimed verification lane without auditable evidence."""
        changed = copy.deepcopy(self.document)
        changed["libtorch"]["compatibility_candidate"]["verified_lanes"][0][
            "quality_evidence"
        ] = {}
        with self.assertRaisesRegex(
            compatibility.CompatibilityError, "quality_evidence must not be empty"
        ):
            compatibility.validate(changed)

    def test_rejects_malformed_commit(self) -> None:
        """Reject a lane that is not bound to a Git commit identity."""
        changed = copy.deepcopy(self.document)
        changed["libtorch"]["compatibility_candidate"]["verified_lanes"][0][
            "commit"
        ] = "not-a-commit"
        with self.assertRaisesRegex(
            compatibility.CompatibilityError, "hexadecimal Git commit"
        ):
            compatibility.validate(changed)

    def test_rejects_unknown_lane_status(self) -> None:
        """Reject an ambiguous support status."""
        changed = copy.deepcopy(self.document)
        changed["libtorch"]["compatibility_candidate"]["status"] = "maybe"
        with self.assertRaisesRegex(
            compatibility.CompatibilityError, "status must be one of"
        ):
            compatibility.validate(changed)


if __name__ == "__main__":
    unittest.main()
