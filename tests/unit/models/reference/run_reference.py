#!/usr/bin/env python3
"""Require all five C++ differentials against the reviewed existing fixtures.

This stdlib-only runner supports the same fixture set on Ubuntu without a
second Python Torch installation. Fixed SHA-256 values come from the reviewed
generator source; runtime evidence JSON cannot authorize different bytes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re

from generate_reference import FROZEN_FILES, digest, run_cpp_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--cpp-test-executable", required=True, type=Path)
    parser.add_argument("--jyppx-commit", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.jyppx_commit) is None:
        raise ValueError("--jyppx-commit must identify the actual complete C++ source revision")
    output = args.fixture_dir.resolve(strict=True)
    report = output / "models-cpp-reference-evidence.json"
    report.unlink(missing_ok=True)
    result = run_cpp_gate(args.cpp_test_executable, output)
    evidence = {
        "schema_id": "jyppx.models.cpp-reference", "schema_version": 1,
        "jyppx_commit": args.jyppx_commit, "platform": platform.platform(),
        "runner_sha256": digest(Path(__file__)),
        "fixture_contract": "official-cpu-torch-2.14.1+cpu-torchvision-0.29.1+cpu",
        "files": FROZEN_FILES, "cpp_test": result,
        "configured_intraop_threads": 2, "configured_interop_threads": 1,
    }
    report.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "cases": 5, "evidence": str(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
