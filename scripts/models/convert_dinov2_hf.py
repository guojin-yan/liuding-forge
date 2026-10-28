#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Convert the pinned official DINOv2-S/14 safetensors, offline and without torch.

Only the reviewed facebook/dinov2-small revision is accepted. Tensor payloads
are streamed unchanged; Q/K/V are concatenated in that order. The mask token
is recorded as omitted because this library's encoder accepts unmasked images.
Use DinoV2Options.position_antialias=false with the converted weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile


REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
SOURCE_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
SOURCE_SIZE = 88249960
OUTPUT_SHA256 = "f3ef1ae01fd4af3cd15926a0d5e631fa0caea50ca68b29db8a68c8c84c9810e5"
OUTPUT_SIZE = 88240512
MAPPING_VERSION = "jyppx.liuding.dinov2-hf-small.unmasked.v1"
SOURCE_URI = f"https://huggingface.co/facebook/dinov2-small/resolve/{REVISION}/model.safetensors"
LICENSE_URI = f"https://huggingface.co/facebook/dinov2-small/blob/{REVISION}/README.md"


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate safetensors header key")
        result[key] = value
    return result


def mapping():
    # (target, shape, ordered source keys); dim-0 concatenation preserves bits.
    result = [
        ("cls_token", [1, 1, 384], ["embeddings.cls_token"]),
        ("pos_embed", [1, 1370, 384], ["embeddings.position_embeddings"]),
        ("patch_embed.proj.weight", [384, 3, 14, 14], ["embeddings.patch_embeddings.projection.weight"]),
        ("patch_embed.proj.bias", [384], ["embeddings.patch_embeddings.projection.bias"]),
        ("norm.weight", [384], ["layernorm.weight"]),
        ("norm.bias", [384], ["layernorm.bias"]),
    ]
    for block in range(12):
        src, dst = f"encoder.layer.{block}.", f"blocks.{block}."
        for suffix, shape in [("weight", [1152, 384]), ("bias", [1152])]:
            result.append((dst + "attn.qkv." + suffix, shape,
                           [src + "attention.attention." + q + "." + suffix for q in ["query", "key", "value"]]))
        for target, source, shape in [
            ("attn.proj.weight", "attention.output.dense.weight", [384, 384]),
            ("attn.proj.bias", "attention.output.dense.bias", [384]),
            ("ls1.gamma", "layer_scale1.lambda1", [384]),
            ("ls2.gamma", "layer_scale2.lambda1", [384]),
            ("norm1.weight", "norm1.weight", [384]),
            ("norm1.bias", "norm1.bias", [384]),
            ("norm2.weight", "norm2.weight", [384]),
            ("norm2.bias", "norm2.bias", [384]),
            ("mlp.fc1.weight", "mlp.fc1.weight", [1536, 384]),
            ("mlp.fc1.bias", "mlp.fc1.bias", [1536]),
            ("mlp.fc2.weight", "mlp.fc2.weight", [384, 1536]),
            ("mlp.fc2.bias", "mlp.fc2.bias", [384]),
        ]:
            result.append((dst + target, shape, [src + source]))
    return sorted(result)


def convert(source: Path, destination: Path, report_path: Path) -> dict:
    source = source.resolve(strict=True)
    destination, report_path = destination.absolute(), report_path.absolute()
    if destination == report_path or destination.exists() or report_path.exists():
        raise ValueError("Output and report must be distinct new files")
    if source.stat().st_size != SOURCE_SIZE or file_digest(source) != SOURCE_SHA256:
        raise ValueError("Source must match the pinned official complete size and SHA-256")
    entries = mapping()
    with source.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size > (1 << 20):
            raise ValueError("Oversized safetensors header")
        header = json.loads(stream.read(header_size), object_pairs_hook=unique_object)
        payload_start = 8 + header_size
        descriptors = {k: v for k, v in header.items() if k != "__metadata__"}
        consumed = {"embeddings.mask_token"}
        mask = descriptors["embeddings.mask_token"]
        if mask["dtype"] != "F32" or mask["shape"] != [1, 384]:
            raise ValueError("Unexpected mask token")
        output_header, offset = {}, 0
        for target, shape, sources in entries:
            total = 0
            for key in sources:
                if key in consumed:
                    raise ValueError("Source key consumed more than once")
                consumed.add(key)
                desc = descriptors[key]
                expected_shape = [shape[0] // len(sources)] + shape[1:]
                size = 4
                for extent in expected_shape:
                    size *= extent
                start, end = desc["data_offsets"]
                if (desc["dtype"] != "F32" or desc["shape"] != expected_shape or
                        start < 0 or end - start != size or payload_start + end > SOURCE_SIZE):
                    raise ValueError("Unexpected tensor descriptor: " + key)
                total += size
            output_header[target] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + total]}
            offset += total
        if consumed != descriptors.keys() or len(entries) != 174:
            raise ValueError("Incomplete upstream state accounting")
        raw = json.dumps(output_header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        raw += b" " * ((-len(raw)) % 8)
        staging = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".liuding-weight-", delete=False) as out:
                staging = Path(out.name)
                out.write(struct.pack("<Q", len(raw)))
                out.write(raw)
                for _, _, sources in entries:
                    for key in sources:
                        start, end = descriptors[key]["data_offsets"]
                        stream.seek(payload_start + start)
                        remaining = end - start
                        while remaining:
                            chunk = stream.read(min(remaining, 1 << 20))
                            if not chunk:
                                raise ValueError("Truncated source tensor")
                            out.write(chunk)
                            remaining -= len(chunk)
                out.flush()
                os.fsync(out.fileno())
            # Verify the complete source again before publishing either output.
            if file_digest(source) != SOURCE_SHA256:
                raise ValueError("Source changed during conversion")
            if staging.stat().st_size != OUTPUT_SIZE or file_digest(staging) != OUTPUT_SHA256:
                raise ValueError("Converted artifact differs from the reviewed canonical identity")
            start, end = mask["data_offsets"]
            stream.seek(payload_start + start)
            mask_digest = hashlib.sha256(stream.read(end - start)).hexdigest()
            report = {
                "schema_id": "jyppx.liuding.weight-conversion", "schema_version": 1,
                "mapping_version": MAPPING_VERSION, "source_uri": SOURCE_URI,
                "source_revision": REVISION, "license_spdx": "Apache-2.0", "license_uri": LICENSE_URI,
                "source": {"sha256": SOURCE_SHA256, "size_bytes": SOURCE_SIZE, "tensors": len(descriptors)},
                "output": {"sha256": file_digest(staging), "size_bytes": staging.stat().st_size, "tensors": len(entries)},
                "required_model_options": {"dinov2.position_antialias": False, "dinov2.num_classes": 0},
                "mappings": [{"target": name, "sources": keys, "operation": "concat-dim0" if len(keys) > 1 else "identity"}
                             for name, _, keys in entries],
                "omitted": [{"key": "embeddings.mask_token", "reason": "unmasked-image-encoder-only", "payload_sha256": mask_digest}],
                "qualification": "data-layout conversion only; task accuracy and fine-tuning need separate validation",
            }
            # No replacement: concurrent writers or an existing output fail.
            os.link(staging, destination)
            try:
                with report_path.open("x", encoding="utf-8", newline="\n") as out:
                    out.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
            except BaseException:
                destination.unlink()
                raise
            return report
        finally:
            if staging is not None:
                staging.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = convert(args.source, args.destination, args.report)
    print(json.dumps({"mapping_version": report["mapping_version"], "output": report["output"]}, sort_keys=True))


if __name__ == "__main__":
    main()
