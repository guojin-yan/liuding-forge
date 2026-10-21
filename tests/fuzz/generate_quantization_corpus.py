# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regenerate only the small quantization wire-format fuzz fixtures.

These data-only fixtures mirror v1 encodings with exact unit-scale weights.
The C++ harness independently creates its seeds with the production APIs and
validates their successful roundtrips before applying structured mutations.
Existing corpora for all other fuzz targets are left untouched.
"""

from __future__ import annotations

from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "fuzz" / "quantization"
MAGIC = b"JYPPXQ1\0"
FORMAT = "jyppx.qint8.symmetric.per_tensor/v1"


def text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def observer(samples: int, values: int, finite: bool = False,
             minimum: float = 0.0, maximum: float = 0.0) -> bytes:
    return struct.pack("<4sIQQB7xff", b"JYQO", 1, samples, values,
                       finite, minimum, maximum)


def qparams() -> bytes:
    return struct.pack("<dibbH", 1.0, 0, -127, 127, 0)


def artifact(in_features: int, out_features: int, weights: bytes,
             bias: tuple[float, ...]) -> bytes:
    descriptor = (text(FORMAT) + struct.pack("<IBBBBBBHII", 1, 1, 1, 1,
                                             1, 0, 0, 0, 0, 1))
    lineage = (text("fuzz.calibration/v1") + b"\x01" + text("a" * 64) +
               text("identity-fp32/v1") + struct.pack("<Q", 20261021))
    sample_ids = ("sample-a", "sample.\u4e2d")
    calibration = (lineage + struct.pack("<Q", len(sample_ids)) +
                   b"".join(text(sample_id) for sample_id in sample_ids) +
                   observer(2, 5, True, -127.0, 127.0) + qparams())
    return (MAGIC + struct.pack("<I", 1) + descriptor +
            struct.pack("<QQ", in_features, out_features) + qparams() +
            struct.pack("<Q", len(weights)) + weights +
            struct.pack("<Q", len(bias)) +
            b"".join(struct.pack("<f", value) for value in bias) + calibration)


def fixtures() -> dict[str, bytes]:
    biased = artifact(3, 2, bytes((129, 192, 0, 64, 127, 32)), (0.0, 1.25))
    finite = observer(2, 5, True, -127.0, 127.0)
    invalid_flag = bytearray(finite)
    invalid_flag[24] = 2
    invalid_padding = bytearray(finite)
    invalid_padding[25] = 1
    nan_minimum = bytearray(finite)
    nan_minimum[32:36] = struct.pack("<I", 0x7FC00000)
    dimensions_offset = len(MAGIC) + 4 + len(text(FORMAT)) + 20
    huge_dimensions = bytearray(biased)
    huge_dimensions[dimensions_offset:dimensions_offset + 8] = b"\xff" * 8
    weight_count_offset = dimensions_offset + 16 + len(qparams())
    huge_weights = bytearray(biased)
    huge_weights[weight_count_offset:weight_count_offset + 8] = b"\xff" * 8
    nonfinite_bias = bytearray(biased)
    bias_offset = weight_count_offset + 8 + 6 + 8
    nonfinite_bias[bias_offset:bias_offset + 4] = struct.pack("<I", 0x7FC00000)
    return {
        "artifact-biased": biased,
        "artifact-zero-unbiased": artifact(1, 1, b"\0", ()),
        "artifact-truncated": MAGIC + struct.pack("<II", 1, 0xFFFFFFFF),
        "artifact-huge-dimensions": bytes(huge_dimensions),
        "artifact-huge-weight-count": bytes(huge_weights),
        "artifact-nonfinite-bias": bytes(nonfinite_bias),
        "observer-empty": observer(0, 0),
        "observer-finite": finite,
        "observer-nonfinite-samples": observer(1, 3),
        "observer-empty-sample": observer(1, 0),
        "observer-finite-zero": observer(1, 1, True),
        "observer-signed-zero": observer(1, 1, True, -0.0, -0.0),
        "observer-invalid-flag": bytes(invalid_flag),
        "observer-invalid-padding": bytes(invalid_padding),
        "observer-nan-minimum": bytes(nan_minimum),
        "observer-trailing": finite + b"\0",
    }


def main() -> None:
    corpus = ROOT / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for name, value in fixtures().items():
        (corpus / name).write_bytes(value)
    tokens = (MAGIC, b"JYQO", FORMAT.encode("ascii"),
              struct.pack("<I", 1), b"\xff" * 8, qparams(),
              struct.pack("<I", 0x80000000), struct.pack("<I", 0x7FC00000),
              struct.pack("<I", 0x7F800000), b"sample-a", "sample.\u4e2d".encode("utf-8"))
    lines = ["# Quantization v1 binary framing, bounds and numeric mutations"]
    for index, token in enumerate(tokens):
        escaped = "".join(f"\\x{byte:02x}" for byte in token)
        lines.append(f'field{index}="{escaped}"')
    (ROOT / "quantization.dict").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
