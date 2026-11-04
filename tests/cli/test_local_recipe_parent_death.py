# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Linux owner death stops its real trainer while the unknown lease stays held.

Reuse the adjacent native recipe failure fixture and run only its worker-kill
case. Observe the pinned trainer's termination before fixture cleanup can kill
it. This supervises the native CLI; Python does not implement model training.
"""
from __future__ import annotations

import argparse
import errno
import os
from pathlib import Path
import stat
import sys
import time
import unittest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-ms", type=int, default=60000)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This regression requires Linux pidfd/process-parent semantics")
    if not 60000 <= args.timeout_ms <= 300000:
        parser.error("--timeout-ms must be between 60000 and 300000")
    import test_local_recipe_failure_lifecycle as lifecycle

    class LocalRecipeParentDeath(lifecycle.LocalRecipeFailureLifecycle):
        def prepare(self, case: str) -> None:
            self.assertEqual(case, "worker-kill")
            # Linux can clear the signal on privileged exec. Qualify the
            # actual trusted same-credential executable, not a privileged one.
            self.assertFalse(self.cli.stat().st_mode & (stat.S_ISUID | stat.S_ISGID))
            try:
                capabilities = os.getxattr(self.cli, "security.capability")
            except OSError as error:
                if error.errno not in (errno.ENODATA, errno.ENOTSUP):
                    raise
            else:
                self.assertFalse(capabilities, "File capabilities are outside this regression")
            # Keep evidence distinct when the failure fixture and this test
            # share one evidence root in the same CTest batch.
            super().prepare("parent-death")

        def collect_execution(self, expected: int) -> dict:
            result = super().collect_execution(expected)
            self.assertEqual(expected, 7)
            self.assertEqual(result["status"], "recovery_required")
            self.assertEqual(len(result["attempts"]), 1)
            self.assertIsNone(result["attempts"][0]["result"])
            self.assertFalse(result["attempts"][0]["resource_released"])
            deadline = time.monotonic() + 10
            while self.trainer.running() and time.monotonic() < deadline:
                time.sleep(0.01)
            dead = not self.trainer.running()
            self.observation["direct_trainer_dead_before_fixture_cleanup"] = dead
            self.observation["parent_death_regression_wait_seconds"] = 10
            self.save("parent-death-before-fixture-cleanup", {
                "direct_trainer_dead": dead,
                "worker_dead": not self.worker.running(),
                "fixture_killed_processes": list(self.observation["supervisor_killed_processes"]),
                "native_result_still_unknown": True,
                "logical_lease_released": False,
            })
            self.assertFalse(self.worker.running())
            self.assertEqual(self.observation["supervisor_killed_processes"], [])
            self.assertTrue(dead, "Worker died but its pinned direct trainer remained alive; fixture cleanup follows")
            return result

        def test_worker_death_stops_trainer_and_preserves_unknown_lease(self) -> None:
            super().test_killed_worker_remains_unknown_and_retains_lease_after_os_cleanup()

    LocalRecipeParentDeath.cli = args.cli.resolve()
    LocalRecipeParentDeath.device = args.device
    LocalRecipeParentDeath.attempt_timeout_ms = args.timeout_ms
    # Selecting exactly this method avoids repeating the separate timeout
    # experiment inherited from the shared fixture.
    suite = unittest.TestSuite([LocalRecipeParentDeath(
        "test_worker_death_stops_trainer_and_preserves_unknown_lease")])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
