#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Generate or verify fixed torchvision ResNet34/101 random references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET


ARCHITECTURES = ("resnet34", "resnet101")
TEST_NAMES = (
    "ModelsPythonInterop.OfficialTorchvisionResNet34ReferenceReallyRuns",
    "ModelsPythonInterop.OfficialTorchvisionResNet101ReferenceReallyRuns",
)
VERSIONS = {
    "python": "3.13.12",
    "torch": "2.9.1+cpu",
    "torchvision": "0.24.1+cpu",
    "safetensors": "0.8.0",
}

# Reviewed fixed bytes are independent of the runtime evidence report.
FROZEN_FILES = {
    "resnet101-input.safetensors": {
        "size_bytes": 49232,
        "sha256": "40d24319665f627fa301a607cf5a26ad785bded8e2d98bb1820904cf8fc48ae1",
    },
    "resnet101-output.safetensors": {
        "size_bytes": 495912,
        "sha256": "db6dd2f90fecc0ec720a03d8b11bb9fc58d18abb45ef23b645a9b08c033db210",
    },
    "resnet101-state.safetensors": {
        "size_bytes": 178675808,
        "sha256": "44fcfafdd8504224968ddc05afd6afbb8ef2c5c2e2cbaabe0a52e63aaf1aeec5",
    },
    "resnet34-input.safetensors": {
        "size_bytes": 49232,
        "sha256": "d94d095908ae36bc606680c08416da2880f06ff6c2dd881b7db33f961793716c",
    },
    "resnet34-output.safetensors": {
        "size_bytes": 127264,
        "sha256": "3da8bcbe56bd12267ee99df2133f74c904adcb3d0451b88f255807f0705c9b6b",
    },
    "resnet34-state.safetensors": {
        "size_bytes": 87278528,
        "sha256": "4dfa489a46f30668754e30e90befc3e381b12a3dbf68535dae56967cdb309e56",
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_files(directory: Path) -> None:
    for name, identity in FROZEN_FILES.items():
        path = directory / name
        if path.stat().st_size != identity["size_bytes"] or digest(path) != identity["sha256"]:
            raise ValueError("Reviewed ResNet reference file identity differs: " + name)


def generate(directory: Path) -> list[dict]:
    import torch
    import torchvision
    import safetensors
    from safetensors.torch import save_file

    observed = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "safetensors": safetensors.__version__,
    }
    if observed != VERSIONS:
        raise ValueError(f"Reference toolchain differs: {observed}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cases = []
    for name in ARCHITECTURES:
        torch.manual_seed(1729)
        model = getattr(torchvision.models, name)(weights=None).eval()
        input_image = torch.randn((1, 3, 64, 64), dtype=torch.float32)
        stages = {}
        handles = []
        for stage_name in ("layer1", "layer2", "layer3", "layer4"):
            def capture(_module, _args, result, key=stage_name):
                stages[key] = result.detach().cpu().contiguous()
            handles.append(getattr(model, stage_name).register_forward_hook(capture))
        with torch.inference_mode():
            logits = model(input_image)
        for handle in handles:
            handle.remove()
        for role, tensors in (
            ("state", model.state_dict()),
            ("input", {"input": input_image}),
            ("output", {"output": logits, **stages}),
        ):
            save_file({key: value.detach().cpu().contiguous()
                       for key, value in tensors.items()},
                      str(directory / f"{name}-{role}.safetensors"))
        cases.append({
            "architecture": name,
            "reference_source": f"torchvision.models.{name}(weights=None)",
            "parameters": sum(value.numel() for value in model.parameters()),
            "state_tensors": len(model.state_dict()),
            "input_shape": list(input_image.shape),
            "output_shape": list(logits.shape),
            "stage_shapes": {key: list(value.shape) for key, value in stages.items()},
            "pretrained": False,
        })
        del model, input_image, logits, stages
    return cases


def run_cpp_gate(executable: Path, directory: Path) -> dict:
    environment = dict(os.environ)
    environment["JYPPX_LIUDING_RESNET_EXTENDED_REFERENCE_DIR"] = str(directory)
    xml_path = directory / "resnet-extended-cpp-test.xml"
    log_path = directory / "resnet-extended-cpp-test.log"
    xml_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [str(executable.resolve(strict=True)),
             "--gtest_filter=" + ":".join(TEST_NAMES),
             "--gtest_output=xml:" + str(xml_path)],
            env=environment, stdout=stream, stderr=subprocess.STDOUT,
            check=False, timeout=300,
        )
    if result.returncode != 0:
        raise ValueError("ResNet C++ differential failed: " + str(log_path))
    cases = ET.parse(xml_path).getroot().findall(".//testcase")
    observed = {case.get("classname", "") + "." + case.get("name", "")
                for case in cases}
    if (len(cases) != len(TEST_NAMES) or observed != set(TEST_NAMES) or
            any(case.get("status") != "run" or case.find("skipped") is not None or
                case.find("failure") is not None or case.find("error") is not None
                for case in cases)):
        raise ValueError("Both ResNet C++ differences must pass without skips")
    verify_files(directory)
    return {"status": "passed", "tests": list(TEST_NAMES),
            "executable_sha256": digest(executable),
            "xml_sha256": digest(xml_path), "log_sha256": digest(log_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--cpp-test-executable", type=Path)
    args = parser.parse_args()
    directory = args.fixture_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cases = [] if args.verify_only else generate(directory)
    verify_files(directory)
    files = {name: {"size_bytes": (directory / name).stat().st_size,
                    "sha256": digest(directory / name)} for name in FROZEN_FILES}
    cpp = (run_cpp_gate(args.cpp_test_executable, directory)
           if args.cpp_test_executable else {"status": "not-run", "tests": list(TEST_NAMES)})
    evidence = {
        "schema_id": "jyppx.models.resnet-extended-reference",
        "schema_version": 1,
        "generator_sha256": digest(Path(__file__)),
        "platform": platform.platform(),
        "versions": VERSIONS,
        "random_seed_per_architecture": 1729,
        "pretrained": False,
        "cases": cases,
        "files": files,
        "cpp_test": cpp,
    }
    report = directory / "resnet-extended-reference-evidence.json"
    report.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"status": cpp["status"], "files": len(files),
                      "evidence": str(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
