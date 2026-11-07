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
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "validation"))

import compatibility  # noqa: E402


class CompatibilityValidatorTest(unittest.TestCase):
    """Covers matrix identity, structure, and evidence invariants."""

    def setUp(self) -> None:
        """Load a fresh copy of the repository compatibility source."""
        self.document = compatibility.load_document(
            REPOSITORY_ROOT / "compatibility.json"
        )
        self.pending_document = copy.deepcopy(self.document)
        self.pending_document["libtorch"]["primary"]["pending_lanes"] = [
            {
                "id": "test-unverified-cpu-profile",
                "distribution": "cpu",
                "os": "ubuntu-22.04",
                "architecture": "x86_64",
                "compiler": "GCC 11.4.0",
                "accelerator": "CPU",
                "build_type": "Debug",
                "instrumentation": "Unverified test fixture for pending-lane rules",
                "status": "candidate",
                "verification_status": "unverified",
                "required_evidence": [
                    "configure", "build", "ctest", "installed-consumer"
                ],
            }
        ]

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


    def test_legacy_source_without_pending_lanes_remains_valid(self) -> None:
        """Keep existing version and evidence declarations backward compatible."""
        changed = copy.deepcopy(self.pending_document)
        for lane in changed["libtorch"].values():
            lane.pop("pending_lanes", None)
        compatibility.validate(changed)

    def test_pending_candidate_cannot_claim_verification_or_support(self) -> None:
        """Reject either support or verification promotion of a pending row."""
        for key, value, message in (
            ("status", "supported", "status must be candidate"),
            ("verification_status", "verified", "verification_status must be unverified"),
        ):
            with self.subTest(field=key):
                changed = copy.deepcopy(self.pending_document)
                changed["libtorch"]["primary"]["pending_lanes"][0][key] = value
                with self.assertRaisesRegex(compatibility.CompatibilityError, message):
                    compatibility.validate(changed)

    def test_pending_candidate_cannot_carry_verified_evidence(self) -> None:
        """Keep passed counts, evidence and validation commits out of pending rows."""
        for key in ("tests", "quality_evidence", "commit"):
            with self.subTest(field=key):
                changed = copy.deepcopy(self.pending_document)
                changed["libtorch"]["primary"]["pending_lanes"][0][key] = "claimed"
                with self.assertRaisesRegex(
                    compatibility.CompatibilityError, "cannot contain verified evidence"
                ):
                    compatibility.validate(changed)

    def test_pending_candidate_requires_explicit_distribution(self) -> None:
        """A CPU candidate must not inherit a CUDA distribution implicitly."""
        changed = copy.deepcopy(self.pending_document)
        del changed["libtorch"]["primary"]["pending_lanes"][0]["distribution"]
        with self.assertRaisesRegex(
            compatibility.CompatibilityError, "distribution must be a non-empty string"
        ):
            compatibility.validate(changed)

    def test_rejects_pending_id_collision_with_later_verified_lane(self) -> None:
        """Reject cross-version aliases regardless of declaration order."""
        changed = copy.deepcopy(self.pending_document)
        changed["libtorch"]["primary"]["pending_lanes"][0]["id"] = changed[
            "libtorch"
        ]["compatibility_candidate"]["verified_lanes"][0]["id"]
        with self.assertRaisesRegex(
            compatibility.CompatibilityError, "duplicate declared lane id"
        ):
            compatibility.validate(changed)

    def test_pending_candidate_cannot_establish_supported_group(self) -> None:
        """A pending declaration alone never satisfies the green-lane requirement."""
        changed = copy.deepcopy(self.pending_document)
        changed["libtorch"]["primary"]["verified_lanes"] = []
        changed["libtorch"]["primary"]["status"] = "supported"
        with self.assertRaisesRegex(
            compatibility.CompatibilityError, "cannot be supported without verified lanes"
        ):
            compatibility.validate(changed)

    def test_pending_candidate_requires_evidence_plan(self) -> None:
        """Reject a candidate that omits every required validation gate."""
        changed = copy.deepcopy(self.pending_document)
        changed["libtorch"]["primary"]["pending_lanes"][0]["required_evidence"] = []
        with self.assertRaisesRegex(
            compatibility.CompatibilityError, "required_evidence must be a non-empty"
        ):
            compatibility.validate(changed)


if __name__ == "__main__":
    unittest.main()
