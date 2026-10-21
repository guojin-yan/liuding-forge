#!/usr/bin/env python3
"""Chapter gate: official dependencies and real collector are mandatory.

Run only when this chapter's unified Windows/Ubuntu validation begins. This
entry point checks the installed runtime adapter through the product CLI.
"""

import importlib.metadata
import os
from pathlib import Path
import sys
import unittest

SIDECAR_ROOT = Path(__file__).resolve().parents[3] / "tools" / "telemetry_sidecar"
sys.path.insert(0, str(SIDECAR_ROOT))
import jyppx_telemetry_sidecar as sidecar


def main() -> int:
    try:
        sidecar.check_dependencies("tensorboard")
        sidecar.check_dependencies("opentelemetry")
        collector = Path(os.environ["JYPPX_TEST_OTELCOL"]).resolve(strict=True)
        if not collector.is_file():
            raise ValueError("collector configuration")
        cli = Path(os.environ["JYPPX_TEST_CLI"]).resolve(strict=True)
        if not cli.is_file():
            raise ValueError("product CLI configuration")
        metadata = sidecar.strict_json((SIDECAR_ROOT / "dependency-metadata.json").read_bytes(),
                                        maximum=256 * 1024)
        for package in metadata:
            if importlib.metadata.version(package["name"]) != package["version"]:
                raise ValueError("pinned dependency version")
    except Exception:
        sys.stderr.write("Telemetry acceptance prerequisites are missing or differ from the locked environment.\n")
        return 2
    tests = Path(__file__).parent
    suite = unittest.defaultTestLoader.discover(str(tests), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    # A skipped collector or missing optional dependency must never turn a
    # formal telemetry chapter gate green.
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
