# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Exercise normal process exit after parallel CPU work and exception unwinding."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100:
        parser.error("iterations must be between 1 and 100")
    executable = args.harness.resolve(strict=True)
    rows = []
    for index in range(args.iterations):
        for threads in (2, 4, 8):
            for mode in ("normal", "unwind"):
                started = time.monotonic()
                # Preserve and then reconfigure nontrivial application parallelism.
                env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
                try:
                    result = subprocess.run([str(executable), str(threads), mode],
                                            env=env, capture_output=True, timeout=10)
                except subprocess.TimeoutExpired as error:
                    raise AssertionError((index, threads, mode, "process exit timed out",
                                          error.stdout, error.stderr)) from error
                if result.returncode or result.stderr:
                    raise AssertionError((index, threads, mode, result.returncode,
                                          result.stdout, result.stderr))
                row = json.loads(result.stdout.decode("utf-8"))
                if row != dict(body_completed=True, threads=threads, gradient_checked=True,
                               unwound=mode == "unwind"):
                    raise AssertionError((index, threads, mode, row))
                rows.append(dict(index=index, mode=mode, **row,
                                 seconds=time.monotonic() - started, exit_code=result.returncode))
                if args.evidence:
                    args.evidence.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, processes=len(rows), threads=[2, 4, 8],
                          modes=["normal", "unwind"], gradient_checked=True)))


if __name__ == "__main__":
    main()
