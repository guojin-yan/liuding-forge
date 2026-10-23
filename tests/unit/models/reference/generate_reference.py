#!/usr/bin/env python3
"""Generate five official random architecture references and enforce C++ parity.

This workflow never acquires pretrained weights. A fixture-only run records
the C++ gate as pending; a supplied C++ executable must run all five named
tests without skips before a passed report can be produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


ARCHITECTURES = (
    "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
    "transformer_encoder",
)
TEST_NAMES = (
    "ModelsPythonInterop.OfficialTorchvisionResNet18ReferenceReallyRuns",
    "ModelsPythonInterop.OfficialTorchvisionResNet50ReferenceReallyRuns",
    "ModelsPythonInterop.OfficialTorchvisionMobileNetV3SmallReferenceReallyRuns",
    "ModelsPythonInterop.OfficialTorchvisionMobileNetV3LargeReferenceReallyRuns",
    "ModelsPythonInterop.OfficialTransformerEncoderReferenceReallyRuns",
)


# Reviewed bytes of the official CPU 2.14.1/torchvision 0.29.1 random fixtures.
FROZEN_FILES = {'mobilenet_v3_large-input.safetensors': {'sha256': 'c33bce57a419a3b79e14fbae93c9bcba4514e1d2211aeb88c2ca02e1446054b3',
                                          'size_bytes': 98384},
 'mobilenet_v3_large-output.safetensors': {'sha256': 'b1cfef0ea201bcf7472e4d3054dea3bb4702a572637cc8ab5bf8a4655eeaacfc',
                                           'size_bytes': 8080},
 'mobilenet_v3_large-state.safetensors': {'sha256': 'c28d8eb4d0437515d69501c03d4e5da19d482ea9ec1efb86a0b0f98d8c0f5404',
                                          'size_bytes': 22060512},
 'mobilenet_v3_small-input.safetensors': {'sha256': '56c82c0b3e6009ac07360ad32021b6e828cc02e39ba8e7ab79c47695a2dc60e6',
                                          'size_bytes': 98384},
 'mobilenet_v3_small-output.safetensors': {'sha256': 'cb8b1630ddc5ac331729390377c54707bc3caa05b562f394a583e73b144249cb',
                                           'size_bytes': 8080},
 'mobilenet_v3_small-state.safetensors': {'sha256': '5414156665cfec12dcafe53f60d12359e8b0c0dc15703ee32a7af53424dbdca1',
                                          'size_bytes': 10243416},
 'resnet18-input.safetensors': {'sha256': '08b128164c087413e272a25af748608e2162069beae422c491087a88f5c0d00c',
                                'size_bytes': 98384},
 'resnet18-output.safetensors': {'sha256': '4ceeeb2cc2e00405ebdec14ecdab4311635d2aa91c056dc3700d87fb5733d92d',
                                 'size_bytes': 8080},
 'resnet18-state.safetensors': {'sha256': '1d4ab66c412050ef8544eab4b3df51c38327008309aca10103f0fc1b1a61cd1a',
                                'size_bytes': 46807448},
 'resnet50-input.safetensors': {'sha256': '1195f7fc72b2f5a4fef921f08ff6c63debf5adfa0100d8c2bfdbeabab50af15f',
                                'size_bytes': 98384},
 'resnet50-output.safetensors': {'sha256': '46e6de1ef6c566128e72884781503295f53edaf6ff014a9b44f4c43a7db2c4de',
                                 'size_bytes': 8080},
 'resnet50-state.safetensors': {'sha256': '9db254eff7e63cee123e56c0bc858def81cfc099095fe86d72639c873ee805a4',
                                'size_bytes': 102469840},
 'transformer_encoder-input.safetensors': {'sha256': '48e5989ff42ba42d9ad2bdd996c27710758ca7dd8ba424161c4b0e28ab8f9455',
                                           'size_bytes': 20747},
 'transformer_encoder-output.safetensors': {'sha256': '9cb1e2e1c1ac0edc186897da83335415aab3669ede4157c847526f13e36e0223',
                                            'size_bytes': 20560},
 'transformer_encoder-state.safetensors': {'sha256': 'be4d41ef9fd15d7c3345a32d04c12d524a730a555cb3d329323bad490de461f2',
                                           'size_bytes': 75668848}}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def run_cpp_gate(executable: Path, output: Path) -> dict:
    """Run all exact C++ cases with independent fixed fixture identities."""
    executable = executable.resolve(strict=True)
    output = output.resolve(strict=True)
    for filename, identity in FROZEN_FILES.items():
        path = output / filename
        if path.stat().st_size != identity["size_bytes"] or digest(path) != identity["sha256"]:
            raise ValueError("Reference bytes differ from the reviewed fixed size/SHA-256")
    subprocess.run([str(executable), "--gtest_list_tests"], check=True,
                   capture_output=True, text=True, timeout=30)
    environment = dict(os.environ)
    environment["JYPPX_LIUDING_MODELS_REFERENCE_DIR"] = str(output)
    xml, log = output / "cpp-test.xml", output / "cpp-test.log"
    xml.unlink(missing_ok=True)
    log.unlink(missing_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            [str(executable), "--gtest_filter=" + ":".join(TEST_NAMES),
             "--gtest_output=xml:" + str(xml)], env=environment,
            stdout=stream, stderr=subprocess.STDOUT, check=False, timeout=300)
    if completed.returncode != 0:
        raise ValueError("C++ models differential failed; see " + str(log))
    result = ET.parse(xml).getroot()
    cases_xml = result.findall(".//testcase")
    observed = {case.get("classname", "") + "." + case.get("name", "") for case in cases_xml}
    if (len(cases_xml) != len(TEST_NAMES) or observed != set(TEST_NAMES) or
            result.get("tests") != str(len(TEST_NAMES)) or
            any(case.get("status") != "run" or case.find("skipped") is not None or
                case.find("failure") is not None or case.find("error") is not None
                for case in cases_xml)):
        raise ValueError("C++ differential XML must record all five tests passing without skips")
    for filename, identity in FROZEN_FILES.items():
        if digest(output / filename) != identity["sha256"]:
            raise ValueError("Reference bytes changed while executing C++")
    return {"executable": str(executable), "executable_sha256": digest(executable),
            "status": "passed", "required_tests": list(TEST_NAMES),
            "xml_sha256": digest(xml), "log_sha256": digest(log)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-python-version", required=True)
    parser.add_argument("--expected-torch-version", required=True)
    parser.add_argument("--expected-torchvision-version", required=True)
    parser.add_argument("--expected-safetensors-version", required=True)
    parser.add_argument("--jyppx-commit", required=True)
    parser.add_argument("--cpp-test-executable", type=Path)
    args = parser.parse_args()
    if platform.python_version() != args.expected_python_version:
        raise ValueError("Python version differs from the exact reference lock")
    if re.fullmatch(r"[0-9a-f]{40}", args.jyppx_commit) is None:
        raise ValueError("--jyppx-commit must be an actual complete source revision")

    import torch
    import torchvision
    import safetensors
    from safetensors.torch import save_file

    if (torch.__version__ != args.expected_torch_version or
            torchvision.__version__ != args.expected_torchvision_version or
            safetensors.__version__ != args.expected_safetensors_version):
        raise ValueError("Observed reference versions differ from the exact locks")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Clear stale success before any operation that may fail.
    for filename in ("models-reference-evidence.json", "cpp-test.xml", "cpp-test.log"):
        (output / filename).unlink(missing_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1729)
    cases = []
    files = {}
    for architecture in ARCHITECTURES:
        if architecture == "transformer_encoder":
            class FeatureEncoder(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    layer = torch.nn.TransformerEncoderLayer(
                        512, 8, 2048, dropout=0.0, activation="relu",
                        layer_norm_eps=1e-5, batch_first=False, norm_first=False)
                    self.encoder = torch.nn.TransformerEncoder(
                        layer, 6, norm=torch.nn.LayerNorm(512, eps=1e-5),
                        enable_nested_tensor=False)

                def forward(self, value, attention_mask, padding_mask):
                    return self.encoder(
                        value.transpose(0, 1), mask=attention_mask,
                        src_key_padding_mask=padding_mask).transpose(0, 1)

            reference = FeatureEncoder().eval()
            inputs = {
                "input": torch.randn((2, 5, 512), dtype=torch.float32),
                "attention_mask": torch.ones((5, 5), dtype=torch.bool).triu(1),
                "padding_mask": torch.tensor([[False] * 5,
                                               [False, False, False, False, True]]),
            }
            source = "torch.nn.TransformerEncoder(post-norm, sequence-first, explicit batch transpose)"
            options = {"model_dimension": 512, "attention_heads": 8,
                       "feedforward_dimension": 2048, "layers": 6,
                       "dropout": 0.0, "gelu": False, "final_norm": True}
        else:
            reference = getattr(torchvision.models, architecture)(weights=None).eval()
            inputs = {"input": torch.randn((2, 3, 64, 64), dtype=torch.float32)}
            source = "torchvision.models." + architecture + "(weights=None)"
            options = {"num_classes": 1000, "input_channels": 3}
        with torch.inference_mode():
            actual = reference(**({"value": inputs["input"],
                                   "attention_mask": inputs["attention_mask"],
                                   "padding_mask": inputs["padding_mask"]}
                                  if architecture == "transformer_encoder"
                                  else {"x": inputs["input"]}))
        if not torch.isfinite(actual).all():
            raise ValueError("Official reference returned a nonfinite output")
        maximum = actual.abs().max().item()
        if maximum <= 0:
            raise ValueError("Reference output has no nonzero differential scale")
        for kind, tensors in (
            ("state", reference.state_dict()), ("input", inputs), ("output", {"output": actual}),
        ):
            filename = architecture + "-" + kind + ".safetensors"
            path = output / filename
            save_file({key: value.detach().cpu().contiguous()
                       for key, value in tensors.items()}, str(path))
            files[filename] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
        cases.append({
            "architecture": architecture, "reference_source": source,
            "options": options,
            "parameters": sum(parameter.numel() for parameter in reference.parameters()),
            "state_tensors": len(reference.state_dict()),
            "input_shape": list(inputs["input"].shape), "output_shape": list(actual.shape),
            "output_max_abs": maximum, "keyword_inputs": sorted(set(inputs) - {"input"}),
            "comparison": "allclose(actual/reference_max_abs, reference/reference_max_abs)",
            "tolerance": {"rtol": 1e-4, "atol": 1e-5},
        })
        print("generated", architecture, cases[-1]["parameters"], "parameters", flush=True)
        del reference, actual, inputs

    if files != FROZEN_FILES:
        raise ValueError("Generated fixture bytes differ from the reviewed fixed hashes")

    evidence = {
        "schema_id": "jyppx.models.python-reference", "schema_version": 2,
        "jyppx_commit": args.jyppx_commit,
        "generator_sha256": digest(Path(__file__)),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torchvision_version": torchvision.__version__, "safetensors_version": safetensors.__version__,
        "device": "cpu", "intraop_threads": 2, "interop_threads": 1,
        "seed": 1729, "pretrained": False,
        "license": "NOASSERTION", "weights_status": "random-architecture-differential-only",
        "cases": cases, "files": files,
        "cpp_test": {"status": "not-run", "required_tests": list(TEST_NAMES)},
    }
    if args.cpp_test_executable is not None:
        evidence["cpp_test"] = run_cpp_gate(args.cpp_test_executable, output)
    (output / "models-reference-evidence.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["cpp_test"]["status"], "cases": len(cases),
                      "output_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
