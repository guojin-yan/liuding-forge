# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run existing real C++ recipe contracts inside delegated OS memory limits.

The original fixtures own training, quality, exact checkpoint and CAS checks.
This maintenance-only layer adds a policy to their generated execution specs
and independently requires the native boundary receipt. It never trains a
Python model or retries a failed quality experiment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import unittest

import test_local_recipe as recipe
import test_local_recipe_failure_lifecycle as failure
import test_local_recipe_lifecycle as lifecycle
import test_local_search_image_recipe as search


LIMIT = 2 << 30
OOM_LIMIT = 128 << 20
CGROUP_PARENT: Path
EVIDENCE: Path | None = None


def evidence_for(case: str) -> None:
    if EVIDENCE:
        os.environ["JYPPX_LIUDING_RECIPE_EVIDENCE_DIR"] = str(EVIDENCE / case)
    else:
        os.environ.pop("JYPPX_LIUDING_RECIPE_EVIDENCE_DIR", None)


def attach_policy(arguments: tuple, data: dict, limit: int = LIMIT) -> dict:
    if arguments[:2] == ("recipe", "local-spec") and "specification" in data:
        specification = data["specification"]
        specification["os_memory_limit"] = {
            "bytes": limit, "cgroup_parent": str(CGROUP_PARENT),
        }
        # The native planner has already created and frozen the recipe inputs.
        # Modifying the newly generated execution policy is a caller action.
        path = Path(arguments[arguments.index("--spec") + 1])
        path.write_text(json.dumps(specification), encoding="utf-8")
    return data


def require_boundaries(data: dict, limit: int = LIMIT) -> None:
    for attempt in data.get("attempts", []):
        result = attempt.get("result")
        if not result or not result.get("process_id"):
            continue
        boundary = result.get("memory_boundary")
        if EVIDENCE:
            with (EVIDENCE / "observed-boundaries.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"attempt_id": attempt["attempt_id"],
                                         "status": result["status"],
                                         "process_tree_cleanup_confirmed":
                                             result["process_tree_cleanup_confirmed"],
                                         "memory_boundary": boundary}) + "\n")
        if not boundary or boundary["source"] != "linux_cgroup_v2" or boundary["limit_bytes"] != limit:
            raise AssertionError(("Missing or changed OS memory boundary", attempt))
        if not result["process_tree_cleanup_confirmed"]:
            raise AssertionError(("OS memory boundary retained a native process", attempt))
        if type(boundary["peak_bytes"]) is not int or boundary["peak_bytes"] <= 0:
            raise AssertionError(("Missing actual kernel peak", boundary))
        # Kernels can report brief charge overshoot; do not equate memory.peak
        # with a sampled RSS ceiling or discard the raw value.
        if type(boundary["oom_kills"]) is not int or boundary["oom_kills"] < 0:
            raise AssertionError(("Missing actual kernel OOM counter", boundary))
        if limit == LIMIT and boundary["oom_kills"]:
            raise AssertionError(("Positive recipe unexpectedly exhausted its limit", boundary))


class BoundedRecipe(recipe.LocalRecipe):
    @classmethod
    def setUpClass(cls) -> None:
        evidence_for("direct")
        cls.training_attempted = False
        super().setUpClass()

    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        data = attach_policy(arguments, super().invoke(*arguments, expected=expected))
        require_boundaries(data)
        return data

    @classmethod
    def train(cls) -> None:
        if not hasattr(cls, "completed"):
            if cls.training_attempted:
                raise AssertionError("The original real task failed; do not retrain")
            cls.training_attempted = True
        super().train()


class BoundedCancellation(lifecycle.LocalRecipeLifecycle):
    def setUp(self) -> None:
        evidence_for("cancel")

    def invoke(self, *arguments: object, expected: int = 0) -> dict:
        data = attach_policy(arguments, super().invoke(*arguments, expected=expected))
        require_boundaries(data)
        return data


class BoundedTimeout(failure.LocalRecipeFailureLifecycle):
    def setUp(self) -> None:
        evidence_for("timeout")
        super().setUp()

    def invoke(self, *arguments: object, expected: int = 0) -> dict:
        data = attach_policy(arguments, super().invoke(*arguments, expected=expected))
        require_boundaries(data)
        return data

    def collect_execution(self, expected: int) -> dict:
        data = super().collect_execution(expected)
        require_boundaries(data)
        return data


class BoundedSearch(search.ImageSearchRecipe):
    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        data = attach_policy(arguments, super().invoke(*arguments, expected=expected))
        require_boundaries(data)
        return data

    @classmethod
    def train(cls) -> None:
        super().train()
        expected = {"bytes": LIMIT, "cgroup_parent": str(CGROUP_PARENT)}
        if cls.admission["execution"].get("os_memory_limit") != expected:
            raise AssertionError("Search admission changed the native memory policy")
        require_boundaries(cls.local)


class BoundedOOM(recipe.LocalRecipe):
    @classmethod
    def setUpClass(cls) -> None:
        evidence_for("oom")
        super().setUpClass()

    @classmethod
    def invoke(cls, *arguments: object, expected: int = 0) -> dict:
        data = attach_policy(arguments, super().invoke(*arguments, expected=expected), OOM_LIMIT)
        require_boundaries(data, OOM_LIMIT)
        return data

    def test_real_native_recipe_oom_releases_lease_without_retry_or_outputs(self) -> None:
        specification = json.loads(self.spec_path.read_bytes())
        specification["resources"] = {
            "capacity": {"cpu_slots": 1, "memory_bytes": LIMIT, "gpus": {}},
            "request": {"cpu_slots": 1, "memory_bytes": LIMIT, "gpu": None},
        }
        self.spec_path.write_text(json.dumps(specification), encoding="utf-8")
        result = self.invoke("local", "execute", "--root", self.store,
                             "--id", "real-recipe-oom", "--spec", self.spec_path, expected=1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["attempts"]), 1)
        attempt = result["attempts"][0]
        outcome = attempt["result"]
        self.assertEqual((outcome["status"], outcome["signal"]), ("signalled", signal.SIGKILL))
        self.assertGreaterEqual(outcome["memory_boundary"]["oom_kills"], 1)
        self.assertTrue(attempt["resource_reserved"])
        self.assertTrue(attempt["resource_released"])
        self.assertFalse(outcome.get("training_outputs_confirmed", False))
        references = self.refs(attempt["attempt_id"])
        self.assertTrue({"checkpoint.last", "model.weights", "training.result",
                         "local.training.outputs"}.isdisjoint(references))
        job_refs = self.refs("real-recipe-oom")
        release = json.loads(self.blob(job_refs["local.resource.release.1"]))
        for role in ("intent", "owner", "result", "stdout", "stderr"):
            self.assertEqual(release[role + "_sha256"],
                             references["local." + role]["artifact"]["sha256"])
        self.assertNotIn("local.resource.lease.2", job_refs)
        for name in ("final.jyck", "result.json"):
            self.assertFalse((self.work / name).exists())
        before = {str(p.relative_to(self.work)): (recipe.sha(p), p.stat().st_mtime_ns)
                  for p in self.work.rglob("*") if p.is_file()}
        for action in ("recover", "execute"):
            options = ("--spec", self.spec_path) if action == "execute" else ()
            self.assertEqual(self.invoke("local", action, "--root", self.store,
                                         "--id", "real-recipe-oom", *options, expected=1), result)
        after = {str(p.relative_to(self.work)): (recipe.sha(p), p.stat().st_mtime_ns)
                 for p in self.work.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.invoke("run", "verify", "--root", self.store)
        if EVIDENCE:
            (EVIDENCE / "oom-result.json").write_text(json.dumps({
                "specification": specification, "local": result, "release": release,
                "workspace_replay_equal": before == after,
                "checkpoint_observed": (self.work / "latest.jyck").is_file(),
                "exact_updates_before_oom_known": False,
            }, indent=2) + "\n", encoding="utf-8")


def load_tests(loader, _standard, _pattern):
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(BoundedRecipe))
    suite.addTests(loader.loadTestsFromTestCase(BoundedCancellation))
    # Unknown-worker recovery has a separate conservative fence contract.
    suite.addTest(BoundedTimeout(
        "test_timeout_after_checkpoint_cleans_native_process_and_releases_lease"))
    suite.addTests(loader.loadTestsFromTestCase(BoundedSearch))
    suite.addTest(BoundedOOM("test_real_native_recipe_oom_releases_lease_without_retry_or_outputs"))
    return suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--evidence-dir", type=Path,
                        default=os.environ.get("JYPPX_LIUDING_RECIPE_MEMORY_EVIDENCE_DIR"))
    parser.add_argument("--temp-parent", type=Path)
    options, remaining = parser.parse_known_args()
    if sys.platform != "linux" or not os.environ.get("JYPPX_LIUDING_TEST_CGROUP_PARENT"):
        print("A delegated Linux cgroup v2 parent is required", file=sys.stderr)
        raise SystemExit(77)
    CGROUP_PARENT = Path(os.environ["JYPPX_LIUDING_TEST_CGROUP_PARENT"])
    if not CGROUP_PARENT.is_absolute() or not (CGROUP_PARENT / "cgroup.procs").is_file():
        parser.error("JYPPX_LIUDING_TEST_CGROUP_PARENT must be an existing absolute cgroup v2 path")
    if options.evidence_dir:
        EVIDENCE = options.evidence_dir.resolve()
        EVIDENCE.mkdir(parents=True, exist_ok=False)
    for cls in (BoundedRecipe, BoundedCancellation, BoundedTimeout, BoundedSearch, BoundedOOM):
        cls.cli, cls.device = options.cli.resolve(), "cpu"
    BoundedTimeout.attempt_timeout_ms = 60000
    BoundedSearch.source_root = options.source_root.resolve()
    BoundedSearch.evidence_dir = EVIDENCE / "search" if EVIDENCE else None
    BoundedSearch.temp_parent = options.temp_parent.resolve() if options.temp_parent else None
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
