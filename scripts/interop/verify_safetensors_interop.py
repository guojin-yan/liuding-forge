#!/usr/bin/env python3
"""Run official safetensors 0.6.2 -> C++ -> official Python byte-exact checks.

Draft integration tool. It neither installs packages nor compiles the C++ test.
Use a pinned test-only Python environment containing safetensors==0.6.2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import struct
import subprocess
import sys
from typing import Any


PINNED_VERSION = "0.6.2"
CPP_TEST_NAME = "SafeTensorsTest.OfficialPythonBidirectionalInterchange"
DTYPES = (
    ("BOOL", 1), ("U8", 1), ("I8", 1), ("U16", 2), ("I16", 2),
    ("U32", 4), ("I32", 4), ("U64", 8), ("I64", 8), ("F16", 2),
    ("BF16", 2), ("F32", 4), ("F64", 8),
)


def matrix() -> dict[str, dict[str, Any]]:
    """Construct raw bits independently of a tensor runtime or NumPy dtype."""
    result: dict[str, dict[str, Any]] = {}
    for index, (dtype, width) in enumerate(DTYPES):
        data = bytes(
            offset % 2 if dtype == "BOOL" else (index * 19 + offset * 7) % 256
            for offset in range(4 * width)
        )
        result[f"tensor.{dtype}"] = {
            "dtype": dtype, "shape": [2, 2], "data": data
        }
    result["scalar"] = {"dtype": "F32", "shape": [], "data": struct.pack("<f", 1.0)}
    result["empty"] = {"dtype": "BF16", "shape": [2, 0, 3], "data": b""}
    return result


def check_official_decode(
    safetensors: Any, encoded: bytes, expected: dict[str, dict[str, Any]]
) -> None:
    """Ask the official Rust-backed Python API to validate and decode all bytes."""
    decoded = dict(safetensors.deserialize(encoded))
    if set(decoded) != set(expected):
        raise ValueError("official reader returned a different tensor directory")
    for name, reference in expected.items():
        actual = decoded[name]
        if actual["dtype"] != reference["dtype"]:
            raise ValueError(f"dtype mismatch for {name}")
        if list(actual["shape"]) != reference["shape"]:
            raise ValueError(f"shape mismatch for {name}")
        if bytes(actual["data"]) != reference["data"]:
            raise ValueError(f"raw-bit mismatch for {name}")


def metadata(encoded: bytes) -> dict[str, str]:
    """Read text metadata after the official parser has accepted the artifact."""
    size = struct.unpack_from("<Q", encoded)[0]
    header = json.loads(encoded[8:8 + size].decode("utf-8"))
    return header.get("__metadata__", {})


def main() -> int:
    """Generate official fixtures, invoke the compiled adapter, and record evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-test-executable", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--jyppx-commit", required=True)
    args = parser.parse_args()
    version = importlib.metadata.version("safetensors")
    if version != PINNED_VERSION:
        raise ValueError(f"requires safetensors=={PINNED_VERSION}; found {version}")
    import safetensors

    executable = args.cpp_test_executable.resolve(strict=True)
    directory = args.work_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    expected = matrix()
    expected_metadata = {"producer": "official-python", "description": "中文"}
    python_bytes = safetensors.serialize(expected, metadata=expected_metadata)
    check_official_decode(safetensors, python_bytes, expected)
    (directory / "python.safetensors").write_bytes(python_bytes)

    # The macro is deliberately opt-in for a distinct interoperability target.
    # Missing this test is a failure, never an unreported successful no-op.
    listed = subprocess.run(
        [str(executable), "--gtest_list_tests"], check=True, capture_output=True,
        text=True, timeout=30,
    )
    if "OfficialPythonBidirectionalInterchange" not in listed.stdout:
        raise ValueError("C++ target lacks JYPPX_SAFETENSORS_PYTHON_INTEROP=1")
    environment = dict(os.environ)
    environment["JYPPX_SAFETENSORS_INTEROP_DIR"] = str(directory)
    cpp_path = directory / "cpp.safetensors"
    # A prior run's artifact must never make a zero-test invocation appear valid.
    cpp_path.unlink(missing_ok=True)
    completed = subprocess.run(
        [str(executable), f"--gtest_filter={CPP_TEST_NAME}"], env=environment,
        check=False, capture_output=True, text=True, timeout=120,
    )
    (directory / "cpp-test.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    completed.check_returncode()
    if cpp_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("unexpectedly large C++ interchange output")
    cpp_bytes = cpp_path.read_bytes()
    check_official_decode(safetensors, cpp_bytes, expected)
    if metadata(cpp_bytes) != expected_metadata:
        raise ValueError("C++ failed to preserve official string metadata")
    report = {
        "evidence_schema": 1,
        "jyppx_commit": args.jyppx_commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "safetensors": version,
        "official_api": "safetensors.serialize/deserialize",
        "official_source": "https://github.com/huggingface/safetensors/tree/v0.6.2",
        "directions": ["official-python-to-cpp", "cpp-to-official-python"],
        "dtypes": [dtype for dtype, _ in DTYPES],
        "tensor_count": len(expected),
        "checks": ["dtype", "shape", "raw-bit-equality", "scalar", "empty", "metadata"],
        "python_artifact": {
            "size": len(python_bytes), "sha256": hashlib.sha256(python_bytes).hexdigest()
        },
        "cpp_artifact": {
            "size": len(cpp_bytes), "sha256": hashlib.sha256(cpp_bytes).hexdigest()
        },
    }
    (directory / "interop-report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"verified {len(expected)} tensors across both official/C++ directions")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError, OSError, ImportError, subprocess.SubprocessError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        print(f"safetensors interchange failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
