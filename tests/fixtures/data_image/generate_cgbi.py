# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generate the owned 2x1 Apple CgBI rejection fixture with Python's stdlib."""
from pathlib import Path
import struct
import zlib


def chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack('>I', len(data)) + kind + data
            + struct.pack('>I', zlib.crc32(kind + data)))


# Filter byte, then two premultiplied BGRA pixels: red at alpha 128, opaque blue.
scanline = bytes([0, 0, 0, 128, 128, 255, 0, 0, 255])
encoder = zlib.compressobj(wbits=-15)
payload = encoder.compress(scanline) + encoder.flush()
png = (b'\x89PNG\r\n\x1a\n'
       + chunk(b'CgBI', bytes.fromhex('40000000'))
       + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 1, 8, 6, 0, 0, 0))
       + chunk(b'IDAT', payload) + chunk(b'IEND', b''))
Path(__file__).with_name('cgbi.png').write_bytes(png)
