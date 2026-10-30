#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Check that native calibration identity is independent of LF/CRLF files."""

import argparse
import array
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def floats(count):
    values = array.array('f', ((i % 31 - 15) / 16 for i in range(count)))
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    args = parser.parse_args()
    tool = args.tool.resolve(strict=True)
    digest = '1234567890abcdef' * 4
    ids = [f'official-train/{i:06d}' for i in range(160)]
    inputs = {'weights.f32': floats(784 * 10), 'bias.f32': floats(10),
              'calibration.f32': floats(784 * 160)}
    checked = 0
    with tempfile.TemporaryDirectory(prefix='liuding-int8-text-') as temporary:
        root = Path(temporary)
        for per_channel in (False, True):
            expected = None
            artifact_name = ('source.per-output-channel.qint8' if per_channel else 'source.qint8')
            cases = [
                ('lf', digest + '\n', '\n'.join(ids) + '\n', True),
                ('crlf', digest + '\r\n', '\r\n'.join(ids) + '\r\n', True),
                ('mixed-digest', digest + '\r\n', '\n'.join(ids) + '\n', True),
                ('mixed-ids', digest + '\n', '\r\n'.join(ids) + '\r\n', True),
                ('no-final-newline', digest, '\n'.join(ids), True),
                ('digest-space', digest + ' \n', '\n'.join(ids) + '\n', False),
                ('empty-id', digest + '\n', '\n'.join(['', *ids[1:]]) + '\n', False),
                ('oversize-id', digest + '\n', '\n'.join(['x' * 257, *ids[1:]]) + '\n', False),
                ('extra-id', digest + '\n', '\n'.join([*ids, 'unexpected/160']) + '\n', False),
            ]
            for label, dataset_text, sample_text, success in cases:
                directory = root / ('channel' if per_channel else 'tensor') / label
                directory.mkdir(parents=True)
                for name, data in inputs.items():
                    (directory / name).write_bytes(data)
                (directory / 'dataset.sha256').write_bytes(dataset_text.encode('ascii'))
                (directory / 'calibration.ids').write_bytes(sample_text.encode('ascii'))
                frozen = {name: (directory / name).read_bytes()
                          for name in [*inputs, 'dataset.sha256', 'calibration.ids']}
                command = [str(tool)]
                if per_channel:
                    command.append('--per-output-channel')
                result = subprocess.run([*command, str(directory)], capture_output=True, timeout=20)
                if (result.returncode == 0) != success:
                    raise AssertionError(f'{artifact_name}/{label}: exit={result.returncode}, '
                                         f'stderr={result.stderr.decode(errors="replace")}')
                assert all((directory / name).read_bytes() == data for name, data in frozen.items())
                artifact = directory / artifact_name
                if success:
                    report = json.loads(result.stdout)
                    assert report['calibration_samples'] == 160
                    if expected is None:
                        expected = artifact.read_bytes()
                    assert artifact.read_bytes() == expected, f'Calibration lineage differs for {label}'
                else:
                    assert not artifact.exists(), f'Invalid identity published an artifact: {label}'
                checked += 1
    print(json.dumps({'cases': checked, 'line_endings': ['LF', 'CRLF', 'mixed', 'no-final-newline'],
                      'original_input_bytes_preserved': True, 'exact_artifact_identity': True}))


if __name__ == '__main__':
    main()
