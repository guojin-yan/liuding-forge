#!/usr/bin/env python3
"""Offline independent Small CPU differential; never used by product code.

Use the existing family-specific locked Python environment and checkout.
No fixture/state is generated from the C++ implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


PINS = {
    "vit_small_patch14_dinov2": {
        "revision": "af97f99efef62bd9bd7eaa908fffe5b7a2d98e2c",
        "python": ["3.11.9", "3.12.14"], "torch": "2.14.1+cpu", "safetensors": "0.6.2",
        "state": (88240512, "cd1aa20613acd6d68210f84d170e1c3b5c16d67968c10c8918881a682f2971a9"),
        "input": (1505368, "0ba7d144d17e1d8a0d97494357b80d8d16d7448339e7755b2b21d45d767a4632"),
        "output": (989336, "c8347532d3e7ae4b5328a889d4ce18f9a20f86c13197d96beb52e456bf46730a"),
        "sources": {"timm/models/vision_transformer.py": "e3435821723dffe0d0e58d86d72e8337ce1106c1041f7b87ed37ed685ea3cd79"},
        "patch": 14, "prefix": 1, "rtol": 1e-4, "atol": 1e-5,
    },
    "dinov3_vits16": {
        "revision": "6876159a11b4df116f30f667f8c9888617df0751",
        "python": ["3.13.12"], "torch": "2.9.1+cpu", "safetensors": "0.8.0",
        "state": (86477024, "74412baf39fda638532a3c1a8e469721f66763e3a0ca274854079ca643301e1d"),
        "input": (42080, "43ce9bd3c548f763a6e733f7f38e7d118d770feffb82781c6bbe812c559dcbf5"),
        "output": (37016, "27ffaf9b5c69c0f435b8e5a0e8103013338011f27eb881129173a0d5897e2e9d"),
        "sources": {
            "dinov3/hub/backbones.py": "821b0f6f034fd866862d1b355036321389436fd534766829251cd4e19da2ef45",
            "dinov3/models/vision_transformer.py": "1395c5a6297861101db35af44a04613a0c60a502c947fd3b54092852ddd682b5",
            "dinov3/layers/rope_position_encoding.py": "41f233d8a34f304e16a621508ff8040d58e4a99a445a6bf9e3b08b2e742b391c",
            "dinov3/layers/attention.py": "ac8677e640f6c365c63f60d684bc01c38a4018298e3cfc2238d44e5e359364f2",
            "dinov3/layers/block.py": "1a5639a0147ffa26d9afb89315121ecf1930774a443070ea5ee43f107ef95629",
            "dinov3/layers/ffn_layers.py": "62875b278c7045b6538581f8e5c19fc2b89505c028832bf95359ce4f3d8b3741",
            "dinov3/layers/patch_embed.py": "a9ec7336332ac0b0073a60ff79040a250d05e719f3ab85fcb6759a7d97dff6be",
        },
        "patch": 16, "prefix": 5, "rtol": 5e-4, "atol": 5e-5,
    },
}
BLOCKS = [8, 2]
VIEWS = ["cls", "patch_tokens", "patch_grid"]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=PINS, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--cpp-executable", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    pin = PINS[args.architecture]
    source = args.upstream_root.resolve(strict=True)
    fixtures = args.fixtures.resolve(strict=True)
    binary = args.cpp_executable.resolve(strict=True)
    if platform.python_version() not in pin["python"]:
        raise ValueError("Use the existing exact family-specific reference environment")
    for name, expected in pin["sources"].items():
        if digest(source / name) != expected:
            raise ValueError("Pinned independent upstream source changed: " + name)
    inputs = {}
    for role in ("state", "input", "output"):
        path = fixtures / (args.architecture + "-" + role + ".safetensors")
        size, expected = pin[role]
        if path.stat().st_size != size or digest(path) != expected:
            raise ValueError("Existing fixed reference fixture changed: " + role)
        inputs[role] = path
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(source))
    import safetensors
    import torch
    from safetensors.torch import load_file, save_file
    if torch.__version__ != pin["torch"] or safetensors.__version__ != pin["safetensors"]:
        raise ValueError("Reference package versions differ from the existing lock")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    v2 = pin["patch"] == 14
    if v2:
        import timm
        import torchvision
        if torchvision.__version__ != "0.29.1+cpu" or digest(Path(inspect.getsourcefile(timm.models.vision_transformer))) != pin["sources"]["timm/models/vision_transformer.py"]:
            raise ValueError("Loaded timm/torchvision differs from the fixed reference")
        model = timm.create_model(args.architecture, pretrained=False, num_classes=0, dynamic_img_size=True).cpu().eval()
    else:
        import numpy
        from dinov3.hub import backbones
        if numpy.__version__ != "2.5.4" or digest(Path(inspect.getsourcefile(backbones))) != pin["sources"]["dinov3/hub/backbones.py"]:
            raise ValueError("Loaded official DINOv3/numpy differs from the fixed reference")
        model = backbones.dinov3_vits16(pretrained=False).cpu().eval()
    model.load_state_dict(load_file(str(inputs["state"]), device="cpu"), strict=True)
    images = load_file(str(inputs["input"]), device="cpu")["input"].requires_grad_(True)
    captured = {}
    def capture(block):
        def hook(module, arguments, output):
            if isinstance(output, (list, tuple)) and len(output) == 1:
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise TypeError("Official block hook did not return one tensor")
            captured[block] = output
        return hook
    hooks = [model.blocks[index].register_forward_hook(capture(index)) for index in BLOCKS]
    try:
        full = model.forward_features(images)
    finally:
        for hook in hooks:
            hook.remove()
    full_tokens = full if v2 else torch.cat((full["x_norm_clstoken"].unsqueeze(1), full["x_storage_tokens"], full["x_norm_patchtokens"]), dim=1)
    old = load_file(str(inputs["output"]), device="cpu")["tokens"]
    torch.testing.assert_close(full_tokens.detach(), old, rtol=pin["rtol"], atol=pin["atol"])
    # Keep the captured graph through block 8, release the unused later graph.
    del full_tokens, full, old
    if set(captured) != set(BLOCKS):
        raise AssertionError("Independent official hooks did not execute every requested block")
    expected = {}
    height, width = images.shape[2] // pin["patch"], images.shape[3] // pin["patch"]
    normalized = []
    for block in BLOCKS:
        for mode in ("raw", "normalized"):
            tokens = captured[block] if mode == "raw" else model.norm(captured[block])
            patches = tokens[:, pin["prefix"]:]
            values = [tokens[:, 0], patches,
                      patches.view(images.shape[0], height, width, tokens.shape[-1]).permute(0, 3, 1, 2)]
            for view, value in zip(VIEWS, values):
                expected[f"{mode}.block_{block}_{view}"] = value
            if mode == "normalized":
                normalized.extend(values)
    loss = torch.zeros(())
    for index, value in enumerate(normalized):
        coefficients = torch.linspace(-0.3, 0.7, value.numel()).view(value.shape)
        loss = loss + (value * coefficients).mean() * (index + 1)
    patch = dict(model.named_parameters())["patch_embed.proj.weight"]
    before = patch.detach().clone()
    loss.backward()
    if patch.grad is None or images.grad is None or not patch.grad.abs().sum().item() > 0:
        raise AssertionError("Independent official reference has no true parameter/input gradient")
    if any(parameter.grad is not None for name, parameter in model.named_parameters()
           if name.startswith(("blocks.9.", "blocks.10.", "blocks.11."))):
        raise AssertionError("Unselected later blocks received a gradient")
    expected["input_gradient"] = images.grad.detach()
    expected["patch_gradient"] = patch.grad.detach().clone()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    expected["patch_delta"] = patch.detach() - before
    if not expected["patch_delta"].abs().sum().item() > 0:
        raise AssertionError("Independent official optimizer did not update patch weights")
    args.work.mkdir(parents=True, exist_ok=False)
    reference_path = args.work / "independent-output.safetensors"
    save_file({name: value.detach().contiguous().clone()
               for name, value in expected.items()}, str(reference_path))
    output_path = args.work / "cpp-output.safetensors"
    command = [str(binary), args.architecture, str(fixtures), str(output_path)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf8", timeout=300, check=False)
    (args.work / "cpp.stdout.log").write_text(result.stdout, encoding="utf8")
    (args.work / "cpp.stderr.log").write_text(result.stderr, encoding="utf8")
    if result.returncode != 0:
        raise RuntimeError({"native_exit_code": result.returncode, "stderr": result.stderr})
    native = json.loads(result.stdout)
    if native != {"actual_cpp_backward": True, "optimizer_steps": 1, "later_block_gradients": 0, "scope": "small_cpu_random_reference"}:
        raise AssertionError("Native gradient/update contract changed")
    actual = load_file(str(output_path), device="cpu")
    if set(actual) != set(expected):
        raise AssertionError("Native output names differ from the independent selection contract")
    errors = {}
    for name in sorted(expected):
        reference, value = expected[name].detach(), actual[name]
        if not torch.isfinite(reference).all() or not torch.isfinite(value).all():
            raise AssertionError("Nonfinite independent/native tensor: " + name)
        scale = reference.abs().max().item()
        if not scale > 0:
            raise AssertionError("Degenerate all-zero independent tensor: " + name)
        if name in {"input_gradient", "patch_gradient", "patch_delta"}:
            torch.testing.assert_close(value / scale, reference / scale, rtol=pin["rtol"], atol=pin["atol"])
        else:
            torch.testing.assert_close(value, reference, rtol=pin["rtol"], atol=pin["atol"])
        errors[name] = {"shape": list(reference.shape), "max_abs_error": (value - reference).abs().max().item(),
                        "reference_max_abs": scale, "max_normalized_abs_error": ((value - reference) / scale).abs().max().item()}
    report = {"schema": "liuding.dino-intermediate-independent-reference/v1", "passed": True,
        "architecture": args.architecture, "upstream_revision": pin["revision"],
        "fixture_identities": {role: {"sha256": digest(path), "bytes": path.stat().st_size} for role, path in inputs.items()},
        "python": platform.python_version(), "torch": torch.__version__, "safetensors": safetensors.__version__,
        "upstream_source_sha256": pin["sources"], "checker_sha256": digest(Path(__file__)),
        "torch_git_version": torch.version.git_version,
        "rtol": pin["rtol"], "atol": pin["atol"], "block_indices": BLOCKS, "views": VIEWS,
        "native_executable_sha256": digest(binary), "native_executable_bytes": binary.stat().st_size,
        "native_output_sha256": digest(output_path), "native_output_bytes": output_path.stat().st_size,
        "independent_output_sha256": digest(reference_path), "independent_output_bytes": reference_path.stat().st_size,
        "normalized_and_raw_features": True, "real_backward_and_sgd_update": True, "tensor_results": errors,
        "pretrained_task_quality_cuda_high_capacity_performance_or_full_models_qualified": False}
    (args.work / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
