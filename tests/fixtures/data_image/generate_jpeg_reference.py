# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Generate small synthetic JPEG/Pillow decode references, never training data."""
import hashlib
import json
from pathlib import Path
import PIL
from PIL import Image, features

root = Path(__file__).resolve().parent / 'jpeg_reference'
root.mkdir(exist_ok=True)
pixels = bytes((i * 37 + i // 11 * 13) % 256 for i in range(17 * 13 * 3))
rgb = Image.frombytes('RGB', (17, 13), pixels)
files = {}
for name, image, options in [
    ('baseline', rgb, dict(quality=91, subsampling=2)),
    ('progressive', rgb, dict(quality=88, subsampling=2, progressive=True)),
    ('gray', rgb.convert('L'), dict(quality=91)),
    ('cmyk', rgb.convert('CMYK'), dict(quality=91)),
]:
    path = root / (name + '.jpg')
    image.save(path, **options)
    if name != 'cmyk':
        with Image.open(path) as decoded:
            decoded.convert('RGB').tobytes()
            (root / (name + '.rgb')).write_bytes(decoded.convert('RGB').tobytes())
            (root / (name + '.gray')).write_bytes(decoded.convert('L').tobytes())
for path in sorted(root.iterdir()):
    if path.is_file() and path.suffix != '.json':
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'reference.json').write_text(json.dumps(dict(
    generator='synthetic deterministic RGB, Pillow JPEG encode/decode',
    pillow=PIL.__version__, jpeg=features.version('jpg'),
    libjpeg_turbo=features.check('libjpeg_turbo'), width=17, height=13,
    files=files), indent=2) + '\n', encoding='utf-8')
