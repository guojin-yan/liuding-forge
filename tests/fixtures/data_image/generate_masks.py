# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generate tiny original label PNGs from the PNG specification, using stdlib only.

Includes all five row filters, packed label samples and Adam7. No dataset,
upstream implementation, third-party encoder or pretrained artifact is copied.
"""

from pathlib import Path
import struct
import zlib

ROOT = Path(__file__).parent / "masks"
SIGNATURE = b"\x89PNG\r\n\x1a\n"
RGB = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
       (17, 23, 31), (255, 255, 255), (1, 2, 3), (17, 23, 31), (0, 0, 0)]
GRAY16 = [0, 1, 255, 256, 1024, 32768, 65534, 65535, 42]
PASSES = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8),
          (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]


def chunk(kind, payload):
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload)))


def paeth(a, b, c):
    estimate = a + b - c
    return min((a, b, c), key=lambda value: abs(estimate - value))


def pack(values, depth):
    if depth == 16:
        return b"".join(struct.pack(">H", value) for value in values)
    bits = "".join(f"{value:0{depth}b}" for value in values)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def make_png(width, height, depth, color, interlaced=False, *, override=None,
             palette_size=None, extra=(), bad_filter=False, bad_adler=False,
             short_raster=False, split=False):
    raster = bytearray()
    for ox, oy, dx, dy in PASSES if interlaced else [(0, 0, 1, 1)]:
        if ox >= width or oy >= height:
            continue
        previous = b""
        for row_index, y in enumerate(range(oy, height, dy)):
            values = []
            for x in range(ox, width, dx):
                if override is not None:
                    values.extend(override(x, y))
                elif color in (2, 6):
                    values.extend(RGB[(x + y) % len(RGB)])
                    if color == 6:
                        values.append(255)
                else:
                    values.append(GRAY16[(x + y) % 9] if depth == 16
                                  else (x * 3 + y * 5) % (1 << depth))
            row = pack(values, depth)
            stride = max(1, ((3 if color == 2 else 4 if color == 6 else 1) * depth) // 8)
            mode = row_index % 5
            raster.append(5 if bad_filter else mode)
            for i, value in enumerate(row):
                a = row[i - stride] if i >= stride else 0
                b = previous[i] if previous else 0
                c = previous[i - stride] if previous and i >= stride else 0
                predictor = (0, a, b, (a + b) // 2, paeth(a, b, c))[mode]
                raster.append((value - predictor) % 256)
            previous = row
    if short_raster:
        raster.pop()
    encoded = zlib.compress(raster, level=9)
    if bad_adler:
        encoded = encoded[:-1] + bytes([encoded[-1] ^ 1])
    header = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, int(interlaced))
    chunks = [chunk(b"IHDR", header)]
    if color == 3:
        size = palette_size if palette_size is not None else 1 << depth
        # Duplicate display colors deliberately make RGB-to-index reconstruction impossible.
        palette = bytes(v for i in range(size)
                        for v in ((19, 23, 29) if i < 2 else (i, 255 - i, i // 2)))
        chunks += [chunk(b"PLTE", palette), chunk(b"tRNS", bytes(range(size)))]
    chunks += [chunk(kind, data) for kind, data in extra]
    if split:
        cuts = [encoded[:2], b"", encoded[2:-4], encoded[-4:]]
    else:
        cuts = [encoded]
    chunks += [chunk(b"IDAT", part) for part in cuts]
    chunks.append(chunk(b"IEND", b""))
    return SIGNATURE + b"".join(chunks)


def main():
    ROOT.mkdir(exist_ok=True)
    for color, name, depths in [(0, "gray", (1, 2, 4, 8, 16)), (3, "palette", (1, 2, 4, 8))]:
        for depth in depths:
            for interlaced in (False, True):
                suffix = "-adam7" if interlaced else ""
                (ROOT / f"{name}{depth}{suffix}.png").write_bytes(
                    make_png(9, 7, depth, color, interlaced))
    for interlaced in (False, True):
        (ROOT / ("rgb-adam7.png" if interlaced else "rgb.png")).write_bytes(
            make_png(9, 7, 8, 2, interlaced))
    for width, height in ((1, 1), (1, 9), (9, 1), (2, 3)):
        (ROOT / f"tiny-{width}x{height}.png").write_bytes(make_png(width, height, 4, 0, True))
    variants = {
        "split-idat": dict(split=True),
        "bad-filter": dict(bad_filter=True),
        "bad-adler": dict(bad_adler=True),
        "short-raster": dict(short_raster=True),
        "apng": dict(extra=[(b"acTL", struct.pack(">II", 1, 0))]),
        "critical": dict(extra=[(b"ABCD", b"")]),
        "gray-trns": dict(extra=[(b"tRNS", b"\0\0")]),
    }
    for name, kwargs in variants.items():
        (ROOT / f"{name}.png").write_bytes(make_png(9, 7, 8, 0, **kwargs))
    (ROOT / "bad-palette-index.png").write_bytes(make_png(9, 7, 4, 3, palette_size=2))
    (ROOT / "rgba.png").write_bytes(make_png(9, 7, 8, 6))
    # A two-pixel target for the existing paired-image dataset fixture.
    (ROOT / "pair16.png").write_bytes(
        make_png(2, 1, 16, 0, override=lambda x, y: [65535 if x else 1024]))
    print(f"Generated {len(list(ROOT.glob('*.png')))} original label PNG fixtures")


if __name__ == "__main__":
    main()
