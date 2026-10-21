#!/usr/bin/env python3
"""Verify an official Python-scripted archive with the real C++ JIT runtime.

Use a separately provisioned, locked Python/PyTorch environment. This script
neither installs dependencies nor compiles the C++ test. It refuses missing,
skipped or zero-test C++ invocations; evidence is written only after success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
from typing import Tuple
import xml.etree.ElementTree as ET


TEST_NAME = (
    "TorchScriptCompatibilityOfficial.PythonScriptArchiveReallyLoadsAndRuns"
)


def main() -> int:
    """Produce and independently exercise one trusted compatibility fixture."""
    # TorchScript resolves postponed annotations in the method's module globals.
    global torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-test-executable", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--expected-torch-version", required=True)
    parser.add_argument("--expected-python-version", required=True)
    parser.add_argument("--jyppx-commit", required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if platform.python_version() != args.expected_python_version:
        raise ValueError("Python version differs from the declared exact lock")
    if re.fullmatch(r"[0-9a-f]{7,40}", args.jyppx_commit) is None:
        raise ValueError("--jyppx-commit must identify the actual source revision")

    import torch

    if torch.__version__ != args.expected_torch_version:
        raise ValueError("PyTorch version differs from the declared exact lock")
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("CUDA evidence requires an available real CUDA device")
    executable = args.cpp_test_executable.resolve(strict=True)
    directory = args.work_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "torchscript-compatibility-evidence.json"
    report_path.unlink(missing_ok=True)
    xml_path = directory / "cpp-test.xml"
    xml_path.unlink(missing_ok=True)

    class Fixture(torch.nn.Module):
        """Small official scripted graph with tuple outputs and a real default."""

        def forward(
            self, x: torch.Tensor, y: torch.Tensor, scale: float = 1.5
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            total = (x + y) * scale
            return total, total.sum(1)

    native = Fixture().eval()
    scripted = torch.jit.script(native)
    fixture_path = directory / "official-python.pt"
    torch.jit.save(scripted, str(fixture_path))
    encoded = fixture_path.read_bytes()
    if not encoded or len(encoded) > 512 * 1024 * 1024:
        raise ValueError("official fixture exceeds the compatibility boundary")
    digest = hashlib.sha256(encoded).hexdigest()
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    y = torch.full((2, 3), 2.0, dtype=torch.float32)
    cases = []
    with torch.inference_mode():
        loaded = torch.jit.load(str(fixture_path), map_location="cpu").eval()
        for scale in (None, 2.0):
            reference = native(x, y) if scale is None else native(x, y, scale)
            actual = loaded(x, y) if scale is None else loaded(x, y, scale=scale)
            for expected, observed in zip(reference, actual, strict=True):
                torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
            cases.append({"scale": scale, "outputs": [v.tolist() for v in reference]})
        if args.cuda:
            cuda_model = torch.jit.load(str(fixture_path), map_location="cuda:0").eval()
            for scale in (1.5, 2.0):
                outputs = cuda_model(x.cuda(0), y.cuda(0), scale=scale)
                for expected, observed in zip(native(x, y, scale), outputs, strict=True):
                    torch.testing.assert_close(observed.cpu(), expected, rtol=1e-5, atol=1e-6)
            torch.cuda.synchronize(0)

    listed = subprocess.run(
        [str(executable), "--gtest_list_tests"], check=True,
        capture_output=True, text=True, timeout=30,
    )
    if "PythonScriptArchiveReallyLoadsAndRuns" not in listed.stdout:
        raise ValueError("C++ target lacks JYPPX_TORCHSCRIPT_PYTHON_INTEROP=1")
    environment = dict(os.environ)
    environment["JYPPX_TORCHSCRIPT_FIXTURE"] = str(fixture_path)
    environment["JYPPX_TORCHSCRIPT_EXPECT_DIGEST"] = digest
    environment["JYPPX_TORCHSCRIPT_EXPECT_LIBTORCH_VERSION"] = torch.__version__.split("+")[0]
    if args.cuda:
        environment["JYPPX_TORCHSCRIPT_EXPECT_CUDA"] = "1"
    else:
        environment.pop("JYPPX_TORCHSCRIPT_EXPECT_CUDA", None)
    log_path = directory / "cpp-test.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(executable), f"--gtest_filter={TEST_NAME}",
             f"--gtest_output=xml:{xml_path}"], env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=False, timeout=120,
        )
    if completed.returncode != 0:
        raise ValueError(f"real C++ compatibility test failed; see {log_path}")
    root = ET.parse(xml_path).getroot()
    cases_run = root.findall(".//testcase")
    if (len(cases_run) != 1 or root.get("tests") != "1"
            or cases_run[0].get("name") != "PythonScriptArchiveReallyLoadsAndRuns"
            or cases_run[0].get("status") != "run"
            or cases_run[0].find("skipped") is not None
            or cases_run[0].find("failure") is not None):
        raise ValueError("C++ evidence must contain exactly one real successful test")
    if fixture_path.read_bytes() != encoded:
        raise ValueError("fixture changed during cross-runtime verification")
    report = {
        "schema_id": "jyppx.torchscript.compatibility.evidence",
        "schema_version": 1,
        "jyppx_commit": args.jyppx_commit,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "artifact": {"sha256": digest, "size_bytes": len(encoded)},
        "cpp_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "devices": ["cpu", "cuda:0"] if args.cuda else ["cpu"],
        "dtype": "float32",
        "cases": cases,
        "cpp_tests_run": 1,
        "cpp_tests_skipped": 0,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-5,
        "compatibility_only": True,
        "deprecated": True,
        "claim": "official scripted archive load and numerical forward; no exporter, trace, or sandbox",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"TorchScript compatibility passed: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, ET.ParseError) as error:
        print(f"TorchScript compatibility failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
