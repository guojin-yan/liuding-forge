#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Generate and verify fixed timm ConvNeXt Tiny/Small random references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


ARCHITECTURES = ("convnext_tiny", "convnext_small")
TIMM_COMMIT = "af97f99efef62bd9bd7eaa908fffe5b7a2d98e2c"
SOURCE_HASHES = {
    "timm/models/convnext.py":
        "cd153d288d0b69d4af402212605d97cb7cd237a861f39da4b49cb1b55560f6d8",
    "timm/layers/norm.py":
        "afe3abf01a4379065c1c11bf6b6f947a1a743aff7f5e51849b353697b0db557d",
    "timm/layers/classifier.py":
        "688e8f77964fa00e24a682c69c3a63c6522bfa1919116bcbcbc515548061e2d5",
    "timm/layers/mlp.py":
        "c024128a1c7e2a55ebc5e33a7c94c585ae1e960cd032b39b08445ea30f05c5a5",
    "timm/version.py":
        "abcf490ed112377cc1cf5f697fca8a48aadb89c81bb7b6033cfa8af593dbdf5c",
}
VERSIONS = {
    "python": "3.11.9",
    "torch": "2.14.1+cpu",
    "timm": "1.0.31.dev0",
    "safetensors": "0.6.2",
}
TESTS = (
    "ModelsPythonInterop.OfficialTimmConvNeXtTinyReferenceReallyRuns",
    "ModelsPythonInterop.OfficialTimmConvNeXtSmallReferenceReallyRuns",
)
FROZEN_FILES = {
    "convnext_small-input.safetensors": {
        "size_bytes": 53900,
        "sha256": "2e36ac17bf36b4ae4c8453c50a01a591d7765ca1e54ae58928f8a6e7ea0eeb7b",
    },
    "convnext_small-output.safetensors": {
        "size_bytes": 194848,
        "sha256": "abb1f5f1a04e8f065a908e6c26d78be019f8226a96b65f1b8e12dba7b835560b",
    },
    "convnext_small-state.safetensors": {
        "size_bytes": 200928944,
        "sha256": "9dddc15d47fb6d5415dac083ec8c7f5931a27961ec89887bf321121845de648e",
    },
    "convnext_tiny-input.safetensors": {
        "size_bytes": 53900,
        "sha256": "0061105c4f63d9cecf83b6c8fb7861955ea57bcf5ecd80e62f35657dfb35bc4a",
    },
    "convnext_tiny-output.safetensors": {
        "size_bytes": 194848,
        "sha256": "5fa834bf7d356b6b27ae43659c2c4d6dea225ef4fa20c4885fb4e7a287b5a5a5",
    },
    "convnext_tiny-state.safetensors": {
        "size_bytes": 114374272,
        "sha256": "2187fc91737b80a62a4369aa25da836f44a284829829b89eef65c990fbac4894",
    },
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_files(directory: Path) -> dict:
    if len(FROZEN_FILES) != 6:
        raise ValueError("All six reviewed reference identities are required")
    for name, identity in FROZEN_FILES.items():
        path = directory / name
        if path.stat().st_size != identity["size_bytes"] or digest(path) != identity["sha256"]:
            raise ValueError("Fixed reference identity mismatch: " + name)
    return FROZEN_FILES


def generate(directory: Path, timm_source: Path) -> dict:
    for name, identity in SOURCE_HASHES.items():
        if digest(timm_source / name) != identity:
            raise ValueError("Pinned timm source differs: " + name)
    sys.path.insert(0, str(timm_source))
    import torch
    import timm
    import safetensors
    from safetensors.torch import save_file

    if not Path(timm.__file__).resolve().is_relative_to(timm_source):
        raise ValueError("Imported timm is not the pinned source tree")
    observed = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "safetensors": safetensors.__version__,
    }
    if observed != VERSIONS:
        raise ValueError(f"Reference toolchain differs: {observed}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    models = {}
    for architecture in ARCHITECTURES:
        torch.manual_seed(1729)
        model = timm.create_model(architecture, pretrained=False).eval()
        image = torch.randn((1, 3, 65, 69), dtype=torch.float32)
        stages = {}
        handles = []
        for index in range(4):
            def capture(_module, _inputs, result, stage=index):
                stages[f"stage{stage}"] = result.detach().cpu().contiguous()
            handles.append(model.stages[index].register_forward_hook(capture))
        with torch.inference_mode():
            logits = model(image)
        for handle in handles:
            handle.remove()
        for role, tensors in (
            ("state", model.state_dict()),
            ("input", {"input": image}),
            ("output", {"output": logits, **stages}),
        ):
            save_file({key: value.detach().cpu().contiguous()
                       for key, value in tensors.items()},
                      str(directory / f"{architecture}-{role}.safetensors"))
        models[architecture] = {
            "parameters": sum(value.numel() for value in model.parameters()),
            "state_tensors": len(model.state_dict()),
            "depths": [len(stage.blocks) for stage in model.stages],
            "input_shape": list(image.shape),
            "logits_shape": list(logits.shape),
            "stage_shapes": {key: list(value.shape)
                             for key, value in stages.items()},
            "preprocessing_candidate": {
                key: model.default_cfg[key]
                for key in ("input_size", "interpolation", "crop_pct", "mean", "std")
            },
        }
        del model
    files = {path.name: {"size_bytes": path.stat().st_size,
                         "sha256": digest(path)}
             for architecture in ARCHITECTURES
             for path in directory.glob(architecture + "-*.safetensors")}
    if FROZEN_FILES and files != FROZEN_FILES:
        raise ValueError("Generated bytes differ from reviewed fixed identities")
    return {
        "schema_id": "jyppx.models.convnext-generation",
        "schema_version": 1,
        "generator_sha256": digest(Path(__file__)),
        "platform": platform.platform(),
        "versions": observed,
        "timm_commit": TIMM_COMMIT,
        "source_sha256": SOURCE_HASHES,
        "seed": 1729,
        "reference_source": "timm.create_model(name, pretrained=False)",
        "pretrained": False,
        "models": models,
        "files": files,
    }


def verify_cpp(executable: Path, directory: Path) -> dict:
    files = verify_files(directory)
    environment = dict(os.environ)
    environment["JYPPX_LIUDING_CONVNEXT_REFERENCE_DIR"] = str(directory)
    xml_path = directory / "convnext-cpp-test.xml"
    log_path = directory / "convnext-cpp-test.log"
    xml_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [str(executable.resolve(strict=True)),
             "--gtest_filter=" + ":".join(TESTS),
             "--gtest_output=xml:" + str(xml_path)],
            env=environment, stdout=stream, stderr=subprocess.STDOUT,
            check=False, timeout=600,
        )
    if result.returncode != 0:
        raise ValueError("C++ ConvNeXt differential failed: " + str(log_path))
    cases = ET.parse(xml_path).getroot().findall(".//testcase")
    observed = {case.get("classname", "") + "." + case.get("name", "")
                for case in cases if case.get("status") == "run" and
                not any(case.find(tag) is not None
                        for tag in ("skipped", "failure", "error"))}
    if observed != set(TESTS) or len(cases) != len(TESTS):
        raise ValueError("Both exact C++ ConvNeXt references must pass without skips")
    verify_files(directory)
    return {
        "schema_id": "jyppx.models.convnext-cpp-reference",
        "schema_version": 1,
        "platform": platform.platform(),
        "generator_sha256": digest(Path(__file__)),
        "files": files,
        "tests": TESTS,
        "status": "passed",
        "executable_sha256": digest(executable),
        "xml_sha256": digest(xml_path),
        "log_sha256": digest(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--timm-source", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--cpp-test-executable", type=Path)
    args = parser.parse_args()
    directory = args.fixture_dir.resolve()
    if args.verify_only:
        verify_files(directory)
    else:
        if args.timm_source is None:
            parser.error("--timm-source is required when generating")
        directory.mkdir(parents=True, exist_ok=True)
        report = generate(directory, args.timm_source.resolve(strict=True))
        (directory / "convnext-generation-evidence.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.cpp_test_executable:
        report = verify_cpp(args.cpp_test_executable, directory)
        (directory / "convnext-cpp-evidence.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "passed", "tests": TESTS}, sort_keys=True))
    else:
        print(json.dumps({"status": "fixtures-verified" if args.verify_only else "generated"},
                         sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
