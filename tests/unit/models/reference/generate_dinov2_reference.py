#!/usr/bin/env python3
"""Generate pinned timm DINOv2 random differential fixtures.

This workflow uses random architecture state only. It never downloads
pretrained weights; the complete state, input and outputs are saved as bounded
safetensors files for the C++ strict-import gate.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import re
import struct
import sys


ARCHITECTURE = "vit_small_patch14_dinov2"
TIMM_COMMIT = "af97f99efef62bd9bd7eaa908fffe5b7a2d98e2c"
TIMM_VIT_SOURCE_SHA256 = "e3435821723dffe0d0e58d86d72e8337ce1106c1041f7b87ed37ed685ea3cd79"
PYTHON_VERSIONS = ("3.12.14", "3.11.9")
TORCH_VERSION = "2.14.1+cpu"
TORCHVISION_VERSION = "0.29.1+cpu"
SAFETENSORS_VERSION = "0.6.2"
SEED = 20261025
INPUT_SHAPE = (2, 3, 224, 280)

# Filled from the first reviewed generation, then enforced for every run.
FROZEN_FILES: dict[str, dict[str, int | str]] = {
    "vit_small_patch14_dinov2-input.safetensors": {
        "sha256": "0ba7d144d17e1d8a0d97494357b80d8d16d7448339e7755b2b21d45d767a4632",
        "size_bytes": 1505368,
    },
    "vit_small_patch14_dinov2-output.safetensors": {
        "sha256": "c8347532d3e7ae4b5328a889d4ce18f9a20f86c13197d96beb52e456bf46730a",
        "size_bytes": 989336,
    },
    "vit_small_patch14_dinov2-state.safetensors": {
        "sha256": "cd1aa20613acd6d68210f84d170e1c3b5c16d67968c10c8918881a682f2971a9",
        "size_bytes": 88240512,
    },
}


CAPACITIES = {
    "small": (384, 22056192),
    "base": (768, 86579712),
    "large": (1024, 304367616),
    "giant": (1536, 1136479232),
}
# Original capacity identities also enforced by the native reference reader.
FROZEN_BY_CAPACITY = {
    "base": {
        "vit_base_patch14_dinov2-state.safetensors": {
            "sha256": "2941c91046ca4ae4cc3c1d2526b01863e9444383dbea1eb0a7d78b32e7d55e9b",
            "size_bytes": 346334872
        },
        "vit_base_patch14_dinov2-input.safetensors": {
            "sha256": "48b75b936e2b87cca3cfcf4e76f8e21427b2a7461dfa9c6e3c913286c9949242",
            "size_bytes": 1505368
        },
        "vit_base_patch14_dinov2-output.safetensors": {
            "sha256": "c9d11866fe58e00b42f3541f69f33af5aa22f968820c1879b7fd0b96901c9687",
            "size_bytes": 1978520
        }
    },
    "large": {
        "vit_large_patch14_dinov2-state.safetensors": {
            "sha256": "0de163cfe892194462c39ab65c4be7ebdcc85ec7dca9858c51b24a6cfc8557ff",
            "size_bytes": 1217502752
        },
        "vit_large_patch14_dinov2-input.safetensors": {
            "sha256": "dc67edfd1e67541e68115fd914b8fba35cdec0375d69b42f9ead25b3e26cb4fc",
            "size_bytes": 1505368
        },
        "vit_large_patch14_dinov2-output.safetensors": {
            "sha256": "1dfd3ae6fe886d27593820ee0fb5d2702ae61527a65b97ca5e55c7a1b6d51b00",
            "size_bytes": 2637976
        }
    },
    "giant": {
        "vit_giant_patch14_dinov2-state.safetensors": {
            "sha256": "b40769853a079205711aab6ecb7867ad45d9c3e35d5856e189cb321d21a55027",
            "size_bytes": 4545971216
        },
        "vit_giant_patch14_dinov2-input.safetensors": {
            "sha256": "2839e43d3b6f84207e66b08628e65a9b05e5d84c0be27f50a87f20ce691de698",
            "size_bytes": 1505368
        },
        "vit_giant_patch14_dinov2-output.safetensors": {
            "sha256": "7c9b742fe3556340c81a4f43437a2a971ca0fc53bb64d6d8e3e1bbb3d3ef1cd4",
            "size_bytes": 3956896
        }
    }
}
FROZEN_BY_CAPACITY["small"] = FROZEN_FILES


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_streamed_f32(tensors: dict, path: Path) -> None:
    """Write the original capacity byte layout without copying the full state."""
    import torch
    from safetensors import safe_open

    if sys.byteorder != "little":
        raise ValueError("Capacity fixtures require a little-endian CPU")
    header, offset = {}, 0
    for name, tensor in sorted(tensors.items()):
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("Capacity fixtures require contiguous CPU F32 tensors")
        size = tensor.numel() * tensor.element_size()
        header[name] = {"dtype": "F32", "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for _, tensor in sorted(tensors.items()):
            stream.write(memoryview(tensor.detach().numpy()).cast("B"))
    with safe_open(path, framework="pt", device="cpu") as view:
        if set(view.keys()) != set(tensors):
            raise ValueError("Independent safetensors reader rejects capacity keys")
        for name, tensor in tensors.items():
            if view.get_slice(name).get_shape() != list(tensor.shape):
                raise ValueError("Independent safetensors reader rejects capacity shape")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--jyppx-commit", required=True)
    parser.add_argument("--capacity", choices=tuple(CAPACITIES), default="small")
    parser.add_argument("--giant-gate", action="store_true",
                        help="Explicitly admit the serial high-memory Giant generation")
    args = parser.parse_args()
    if args.capacity == "giant" and not args.giant_gate:
        raise ValueError("Giant requires an explicit serial resource gate")
    if re.fullmatch(r"[0-9a-f]{40}", args.jyppx_commit) is None:
        raise ValueError("--jyppx-commit must identify the exact complete source revision")
    if platform.python_version() not in PYTHON_VERSIONS:
        raise ValueError("Python version differs from the reviewed DINOv2 reference lock")

    if args.capacity != "small" and platform.python_version() != "3.11.9":
        raise ValueError("Capacity references require the original Python 3.11.9 lock")

    import safetensors
    import timm
    import torch
    import torchvision
    from safetensors.torch import save_file

    if (torch.__version__ != TORCH_VERSION or
            torchvision.__version__ != TORCHVISION_VERSION or
            safetensors.__version__ != SAFETENSORS_VERSION):
        raise ValueError("Python model reference versions differ from the exact lock")
    vision_source = Path(inspect.getsourcefile(timm.models.vision_transformer)).resolve()
    if digest(vision_source) != TIMM_VIT_SOURCE_SHA256:
        raise ValueError("Installed timm VisionTransformer source differs from the pinned revision")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve previous fixtures; output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    architecture = f"vit_{args.capacity}_patch14_dinov2"
    dimension, expected_parameters = CAPACITIES[args.capacity]

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    model = timm.create_model(
        architecture, pretrained=False, num_classes=0, img_size=518,
        dynamic_img_size=True
    ).cpu().eval()
    if sum(parameter.numel() for parameter in model.parameters()) != expected_parameters:
        raise ValueError("Pinned timm returned an unexpected DINOv2 parameter count")
    inputs = torch.randn(INPUT_SHAPE, dtype=torch.float32)
    with torch.inference_mode():
        tokens = model.forward_features(inputs)
        output_cls = model(inputs)
    if tokens.shape != (2, 321, dimension) or output_cls.shape != (2, dimension):
        raise ValueError("Pinned timm returned an unexpected DINOv2 feature shape")
    if not torch.isfinite(tokens).all() or not torch.isfinite(output_cls).all():
        raise ValueError("Pinned timm returned nonfinite DINOv2 features")

    tensors = {
        f"{architecture}-state.safetensors": {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        },
        f"{architecture}-input.safetensors": {"input": inputs.contiguous()},
        f"{architecture}-output.safetensors": {
            "tokens": tokens.cpu().contiguous(),
            "output": output_cls.cpu().contiguous(),
        },
    }
    files: dict[str, dict[str, int | str]] = {}
    for filename, values in tensors.items():
        path = output / filename
        if args.capacity == "small":
            save_file(values, str(path))
        else:
            save_streamed_f32(values, path)
        files[filename] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
    if files != FROZEN_BY_CAPACITY[args.capacity]:
        raise ValueError("Generated bytes differ from the reviewed fixed DINOv2 fixture")

    evidence = {
        "schema_id": "jyppx.models.dinov2-reference",
        "schema_version": 1,
        "jyppx_commit": args.jyppx_commit,
        "architecture": architecture,
        "capacity": args.capacity,
        "serializer": "safetensors.torch.save_file" if args.capacity == "small" else "bounded-f32-stream-v1",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
        "timm_commit": TIMM_COMMIT,
        "timm_vision_transformer_sha256": TIMM_VIT_SOURCE_SHA256,
        "safetensors_version": safetensors.__version__,
        "device": "cpu",
        "intraop_threads": 2,
        "interop_threads": 1,
        "seed": SEED,
        "pretrained": False,
        "license": "NOASSERTION",
        "weights_status": "random-architecture-differential-only",
        "input_shape": list(INPUT_SHAPE),
        "token_shape": list(tokens.shape),
        "cls_shape": list(output_cls.shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "position_interpolation": "bicubic, align_corners=false, antialias=true",
        "files": files,
    }
    evidence_path = output / "dinov2-reference-evidence.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"status": "generated", "evidence": str(evidence_path),
                      "files": files}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
