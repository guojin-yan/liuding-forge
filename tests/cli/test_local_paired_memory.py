# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

"""Qualify the complete native paired task with actual OS memory boundaries.

The unchanged local/search fixtures own all training and quality assertions.
This maintenance-only layer combines their full task with host admission,
logical reservations and delegated cgroup v2 limits. It never retries training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import unittest

import test_paired_recipe as paired
import test_local_search_recipe as search


LIMIT = 2 << 30
GPU_RESERVATION = 1 << 30
CGROUP_PARENT: Path
EVIDENCE: Path | None = None
OBSERVED: dict[tuple[str, int], dict] = {}


def policy(device: str) -> dict:
    gpu = {"id": "cuda0", "memory_bytes": GPU_RESERVATION} if device == "cuda" else None
    return {
        "os_memory_limit": {"bytes": LIMIT, "cgroup_parent": str(CGROUP_PARENT)},
        "host_requirements": {
            "minimum_cpu_threads": 1, "minimum_total_memory_bytes": LIMIT,
            "minimum_available_memory_bytes": 256 << 20,
            "minimum_disk_free_bytes": 256 << 20,
        },
        "resources": {
            "capacity": {"cpu_slots": 1, "memory_bytes": LIMIT,
                         "gpus": {"cuda0": GPU_RESERVATION} if gpu else {}},
            "request": {"cpu_slots": 1, "memory_bytes": LIMIT, "gpu": gpu},
        },
    }


def attach(arguments: tuple, data: dict, device: str) -> None:
    if arguments[:2] != ("recipe", "local-spec") or "specification" not in data:
        return
    data["specification"].update(policy(device))
    path = Path(arguments[arguments.index("--spec") + 1])
    path.write_text(json.dumps(data["specification"]), encoding="utf-8")


def boundaries(data: dict) -> None:
    for attempt in data.get("attempts", []):
        outcome = attempt.get("result")
        if not outcome or not outcome.get("process_id"):
            continue
        boundary = outcome.get("memory_boundary")
        if not boundary or boundary["source"] != "linux_cgroup_v2" or boundary["limit_bytes"] != LIMIT:
            raise AssertionError(("Missing actual delegated memory boundary", attempt))
        if not outcome["process_tree_cleanup_confirmed"]:
            raise AssertionError(("Native process tree cleanup was not confirmed", attempt))
        if not attempt["resource_reserved"] or not attempt["resource_released"]:
            raise AssertionError(("Native logical reservation was not released", attempt))
        if type(boundary["peak_bytes"]) is not int or boundary["peak_bytes"] <= 0 or boundary["oom_kills"] != 0:
            raise AssertionError(("Missing actual kernel peak or unexpected OOM", boundary))
        key = (attempt["attempt_id"], outcome["process_id"])
        row = {"attempt_id": key[0], "process_id": key[1], "memory_boundary": boundary,
               "cleanup_confirmed": True, "resource_reserved": True, "resource_released": True}
        if key in OBSERVED and OBSERVED[key] != row:
            raise AssertionError("Confirmed memory boundary changed during replay")
        OBSERVED[key] = row


class BoundedPairedLocal(paired.LocalPairedRecipe):
    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        envelope = super().invoke(*arguments, expected=expected)
        data = envelope.get("data", envelope)
        attach(arguments, data, cls.device)
        boundaries(data)
        return envelope

    def test_05_complete_checkpoint_and_frozen_resource_policy(self) -> None:
        self.train()
        expected = policy(self.device)
        for name, value in expected.items():
            self.assertEqual(self.plan()["specification"][name], value)
        self.assertEqual(self.file_sha(self.work / "final.jyck"),
                         self.file_sha(self.baseline_root / "final.jyck"))
        type(self).reference_model_sha256 = self.file_sha(self.baseline_root / "model.safetensors")
        self.assertEqual([row["result"]["exit_code"] for row in self.completed["attempts"]], [75, 0])
        boundaries(self.completed)
        self.assertFalse(any(path.is_dir() for path in CGROUP_PARENT.iterdir()))


class BoundedPairedSearch(search.PairedSearchRecipe):
    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        data = super().invoke(*arguments, expected=expected)
        attach(arguments, data, cls.device)
        boundaries(data)
        return data

    def test_05_resource_policy_and_full_task_match_local_reference(self) -> None:
        self.train()
        for name, value in policy(self.device).items():
            self.assertEqual(self.admission["execution"][name], value)
        for field in paired.RESULT_FIELDS:
            self.assertEqual(self.result[field], BoundedPairedLocal.baseline[field], field)
        self.assertEqual(paired.LocalPairedRecipe.file_sha(self.recipe / "model.safetensors"),
                         BoundedPairedLocal.reference_model_sha256)
        boundaries(self.local)
        self.assertFalse(any(path.is_dir() for path in CGROUP_PARENT.iterdir()))


def load_tests(loader, _standard, _pattern):
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(BoundedPairedLocal))
    suite.addTests(loader.loadTestsFromTestCase(BoundedPairedSearch))
    return suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--evidence-dir", type=Path,
                        default=os.environ.get("JYPPX_LIUDING_PAIRED_MEMORY_EVIDENCE_DIR"))
    args, remaining = parser.parse_known_args()
    if sys.platform != "linux" or not os.environ.get("JYPPX_LIUDING_TEST_CGROUP_PARENT"):
        print("A delegated Linux cgroup v2 parent is required", file=sys.stderr)
        raise SystemExit(77)
    CGROUP_PARENT = Path(os.environ["JYPPX_LIUDING_TEST_CGROUP_PARENT"])
    if not CGROUP_PARENT.is_absolute() or not (CGROUP_PARENT / "cgroup.procs").is_file():
        parser.error("A writable, absolute delegated cgroup v2 parent is required")
    if args.evidence_dir:
        EVIDENCE = args.evidence_dir.resolve()
        EVIDENCE.mkdir(parents=True, exist_ok=False)
        os.environ["JYPPX_LIUDING_RECIPE_EVIDENCE_DIR"] = str(EVIDENCE / "local")
    for cls in (BoundedPairedLocal, BoundedPairedSearch):
        cls.cli, cls.device = args.cli.resolve(), args.device
    paired.ROOT = args.source_root.resolve()
    BoundedPairedSearch.source_root = args.source_root.resolve()
    BoundedPairedSearch.evidence_dir = EVIDENCE / "search" if EVIDENCE else None
    program = unittest.main(argv=[sys.argv[0], *remaining], verbosity=2, failfast=True, exit=False)
    if EVIDENCE:
        (EVIDENCE / "boundaries.json").write_text(json.dumps({
            "device": args.device, "policy": policy(args.device),
            "tests_run": program.result.testsRun, "passed": program.result.wasSuccessful(),
            "skipped": len(program.result.skipped), "observed": list(OBSERVED.values()),
            "VRAM_hard_limit": False, "paired_performance_claim": False,
        }, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
