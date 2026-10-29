#!/usr/bin/env python3
"""Probe a pinned official ONNX environment using real CPU export/checker/ORT.

Future-chapter draft; not executed or wired to CMake. Run only inside the
process owner's bounded, isolated environment. The owner enforces CPU/memory/
wall-time/output quotas and imports evidence.json into its ArtifactStore.
This script is not a sandbox, installer, model exporter, or request preflight.
Its probe can establish environment-smoke evidence only; model support remains
conditional until an actual model/request-bound operator/schema probe exists.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import sys
from typing import Any


REQUIRED = {"python", "torch", "onnx", "onnxscript", "onnxruntime", "safetensors", "numpy"}
VERSION = re.compile(r"[0-9][A-Za-z0-9._+-]{0,255}\Z")
COMMIT = re.compile(r"[0-9a-f]{7,40}\Z")
MAX_LOCK_BYTES = 65536
MAX_SMOKE_ARTIFACT_BYTES = 16 * 1024 * 1024


@contextlib.contextmanager
def private_diagnostics(log: Any) -> Any:
    """Capture Python and native library writes before they reach control I/O."""
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            yield
    finally:
        log.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate decoded keys rather than silently accepting the last."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate environment lock field")
        result[key] = value
    return result


def lock_bytes(path: pathlib.Path) -> tuple[bytes, dict[str, str]]:
    """Read the host-approved exact-version lock without package installation."""
    with path.open("rb") as source:
        encoded = source.read(MAX_LOCK_BYTES + 1)
    if len(encoded) > MAX_LOCK_BYTES:
        raise ValueError("environment lock exceeds its byte limit")
    lock = json.loads(encoded.decode("utf-8"), object_pairs_hook=unique_object)
    if not isinstance(lock, dict) or set(lock) != {"schema_version", "packages"}:
        raise ValueError("environment lock requires exact schema fields")
    if type(lock["schema_version"]) is not int or lock["schema_version"] != 1:
        raise ValueError("unsupported environment lock version")
    packages = lock["packages"]
    if not isinstance(packages, dict) or set(packages) != REQUIRED:
        raise ValueError("environment lock requires all seven exact package pins")
    if any(not isinstance(value, str) or VERSION.fullmatch(value) is None
           for value in packages.values()):
        raise ValueError("environment lock versions must be exact pins")
    return encoded, packages


def observed_versions() -> tuple[dict[str, str], list[str]]:
    """Collect actual distribution observations, recording missing packages."""
    versions = {"python": platform.python_version()}
    missing = []
    for name in sorted(REQUIRED - {"python"}):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    return versions, missing


def official_smoke(root: pathlib.Path) -> dict[str, Any]:
    """Use official APIs only; verify fixed and dynamic batches with real ORT."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import onnxscript  # noqa: F401 -- import is part of the real dependency probe.
    import safetensors  # noqa: F401 -- import is part of the real dependency probe.
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    class Affine(torch.nn.Module):
        def forward(self, x: Any) -> Any:
            return x * 2.0 + 1.0

    model = Affine().eval()
    fixed = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    dynamic = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    exported = torch.export.export(
        model, (fixed,),
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=8)}},
        strict=True,
    )
    output = root / "official-smoke.onnx"
    # No fallback to legacy tracing or handmade graph/wire generation.
    torch.onnx.export(
        exported, args=(), f=str(output), dynamo=True,
        opset_version=18, external_data=False,
    )
    if not output.is_file() or not 0 < output.stat().st_size <= MAX_SMOKE_ARTIFACT_BYTES:
        raise ValueError("official exporter did not produce a bounded smoke artifact")
    graph = onnx.load(str(output), load_external_data=False)
    onnx.checker.check_model(graph, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(graph, check_type=True, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("official smoke has unexpected input/output arity")
    cases = []
    for kind, tensor in (("fixed", fixed), ("dynamic", dynamic)):
        with torch.no_grad():
            expected = model(tensor).cpu().numpy()
        actual = session.run(None, {session.get_inputs()[0].name: tensor.numpy()})[0]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError("official smoke runtime schema mismatch")
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4,
                                   equal_nan=False)
        cases.append({"kind": kind, "shape": list(actual.shape), "passed": True})
    payload = output.read_bytes()
    return {
        "official_onnx_exporter": True, "onnx_checker": True,
        "onnx_runtime": True, "shape_inference_passed": True,
        "dynamic_shapes_verified": True, "cases": cases,
        "smoke_artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "smoke_artifact_bytes": len(payload),
        "torch_runtime_version": str(torch.__version__),
        "ort_providers_observed": session.get_providers(),
        "devices_verified": ["cpu"], "dtypes_verified": ["float32"],
        "opset": 18, "absolute_tolerance": 1e-5, "relative_tolerance": 1e-4,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-lock", required=True, type=pathlib.Path)
    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--jyppx-commit", required=True)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if VERSION.fullmatch(args.backend_version) is None:
        parser.error("backend version must be an exact numeric version")
    if COMMIT.fullmatch(args.jyppx_commit) is None:
        parser.error("JYPPX commit must be the exact tested source revision")
    encoded_lock, pins = lock_bytes(args.environment_lock)
    environment_lock = {"algorithm": "sha256", "hex": hashlib.sha256(encoded_lock).hexdigest()}
    # A new explicit directory prevents stale evidence/artifacts from proving a
    # later failed probe and prevents overwriting any existing user output.
    args.work_dir.mkdir(parents=False, exist_ok=False)
    versions, missing = observed_versions()
    mismatches = sorted(name for name, value in versions.items() if pins[name] != value)
    smoke: dict[str, Any] = {}
    failure_kind: str | None = None
    if not missing and not mismatches:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        # Third-party diagnostics stay in an isolated log artifact. Ordinary
        # control stdout never contains paths, stack traces, or sample values.
        with (args.work_dir / "smoke.log").open("w", encoding="utf-8") as log:
            try:
                with private_diagnostics(log):
                    smoke = official_smoke(args.work_dir)
            except Exception as error:  # typed category only, never its message
                failure_kind = type(error).__name__
    complete = bool(smoke) and not missing and not mismatches and failure_kind is None
    evidence = {
        "schema_version": 1, "phase": "environment-smoke", "complete": complete,
        "backend_id": "onnx.sidecar", "backend_version": args.backend_version,
        "jyppx_commit": args.jyppx_commit,
        "environment_lock": environment_lock, "platform": platform.platform(),
        "versions_observed": versions, "packages_missing": missing,
        "version_mismatches": mismatches, "failure_category": failure_kind,
        "request_model_preflight_performed": False, "model_support_claimed": False,
        "smoke": smoke,
    }
    evidence_bytes = (json.dumps(evidence, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")) + "\n").encode("utf-8")
    (args.work_dir / "evidence.json").write_bytes(evidence_bytes)
    evidence_ref = {
        "digest": {"algorithm": "sha256", "hex": hashlib.sha256(evidence_bytes).hexdigest()},
        "size_bytes": len(evidence_bytes), "logical_name": "onnx-environment-probe-evidence",
        "media_type": "application/json", "producer": "onnx.sidecar.environment-probe",
    }
    probe = {
        "schema_version": 1, "backend_id": "onnx.sidecar",
        "backend_version": args.backend_version, "protocol_version": 1,
        "environment_lock": environment_lock,
        "dependencies": [{"name": name, "version": version}
                         for name, version in sorted(versions.items())],
        "operations": ["export_model"] if complete else [],
        "devices": ["cpu"] if complete else [],
        "dtypes": ["float32"] if complete else [],
        "dynamic_shapes_verified": complete,
        "official_onnx_exporter": complete, "onnx_checker": complete,
        "onnx_runtime": complete, "request_digest": None,
        "model_preflight_passed": False, "unsupported_operators": [],
        "evidence": evidence_ref,
    }
    control = {"protocol": "jyppx.export.sidecar", "protocol_version": 1,
               "kind": "probe.result", "payload": probe}
    # This is environment evidence, never an export.candidate/success message.
    sys.stdout.write(json.dumps(control, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")) + "\n")
    return 0 if complete else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # The process owner records full private diagnostics when appropriate;
        # this control boundary intentionally exposes only an exception class.
        sys.stderr.write(f"environment probe failed: {type(error).__name__}\n")
        raise SystemExit(2) from None
