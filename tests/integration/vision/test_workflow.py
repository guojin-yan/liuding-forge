# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Exercise native vision training, exact resume, validation and bundle use."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib


def png(path, seed):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    pixels = bytearray()
    for y in range(64):
        pixels.append(0)
        for x in range(64):
            pixels.extend(((x * 3 + seed * 31) % 256,
                           (y * 5 + seed * 47) % 256,
                           (x + y * 2 + seed * 53) % 256))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' +
                     chunk(b'IHDR', struct.pack('>IIBBBBB', 64, 64, 8, 2, 0, 0, 0)) +
                     chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


def verify_bundle(probe, recipe, bundle, state, digest, env, root):
    if probe:
        result = subprocess.run([str(probe), str(recipe), str(bundle), str(state), digest],
                                cwd=root, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=180)
        output = result.stdout.decode('utf-8', errors='replace')
        print(output, flush=True)
        if result.returncode:
            raise AssertionError('Native/ORT numerical differential failed: ' + output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', default='full')
    parser.add_argument('--onnx', action='store_true')
    parser.add_argument('--probe', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='vision-workflow-') as directory:
        root = Path(directory)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')

        def write(name, value):
            path = root / name
            path.write_text(json.dumps(value), encoding='utf-8')
            return path

        def run(label, *arguments, succeeds=True):
            result = subprocess.run([str(args.cli), *map(str, arguments)], cwd=root, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            (root / (label + '.log')).write_bytes(result.stdout)
            if (result.returncode == 0) != succeeds:
                raise AssertionError(label + ': ' + result.stdout.decode('utf-8', errors='replace'))
            print(label, result.returncode, flush=True)
            return result.stdout

        for split, count, offset in [('train', 5, 0), ('validation', 3, 10)]:
            rows = []
            for i in range(count):
                image = f'{split}-{i}.png'
                png(root / image, offset + i)
                rows.append({'image': image, 'width': 64, 'height': 64,
                             'labels': [i % 2, (i + 1) % 2, -1]})
            write(split + '.json', {'schema': 'vision.dataset/v1', 'image_root': '.',
                                    'categories': ['a', 'b', 'unknown'], 'images': rows})
        recipe = {'schema_version': 2,
                  'model': {'architecture': 'resnet18', 'task': 'classify_multilabel',
                            'variant': 'p5', 'num_classes': 3},
                  'image_size': 64, 'batch_size': 2, 'accumulate_steps': 2,
                  'workers': 0, 'epochs': 2, 'seed': 19,
                  'device': args.device, 'precision': args.precision,
                  'initial_loss_scale': 1. if args.precision == 'cuda_float16' else 65536.,
                  'optimizer': {'name': 'sgd', 'learning_rate': .001, 'weight_decay': .0001},
                  'extension': {'training_manifest': 'train.json',
                                'validation_manifest': 'validation.json'},
                  'augmentation': {'horizontal_flip': .5, 'hsv_h': .05, 'hsv_s': .2},
                  'classification_augmentation': {},
                  'ema': {'enabled': True}, 'full_validation_metrics': True,
                  'verify_image_contents': True}
        recipe_path = write('recipe.json', recipe)
        run('inspect', 'inspect', '--recipe', recipe_path, '--decode', '--output', root / 'audit.json')
        run('continuous', 'train', '--recipe', recipe_path, '--output', root / 'continuous')
        run('paused', 'train', '--recipe', recipe_path, '--pause-after', 1, '--output', root / 'paused')
        run('resumed', 'resume', '--recipe', recipe_path, '--checkpoint', root / 'paused/last.checkpoint',
            '--output', root / 'resumed')
        read = lambda path: json.loads((root / path).read_text(encoding='utf-8'))
        continuous, resumed = read('continuous/training.json'), read('resumed/training.json')
        assert continuous['status'] == resumed['status'] == 1
        assert continuous['updates'] == resumed['updates'] == 4
        assert continuous['epochs'] == resumed['epochs'] == 2
        assert read('continuous/weights.json') == read('resumed/weights.json'), 'Raw/EMA resume differs'
        run('completed-replay', 'resume', '--recipe', recipe_path,
            '--checkpoint', root / 'resumed/last.checkpoint', '--output', root / 'replayed')
        assert read('replayed/training.json') == resumed
        assert read('replayed/weights.json') == read('resumed/weights.json')
        weights = read('resumed/weights.json')['raw']
        state = root / 'resumed' / weights['file']
        assert hashlib.sha256(state.read_bytes()).hexdigest() == weights['sha256']
        initialization = ['--weights', state, '--sha256', weights['sha256']]
        run('validation', 'val', '--recipe', recipe_path, *initialization,
            '--output', root / 'validation-result.json')
        run('calibration', 'calibrate', '--recipe', recipe_path, *initialization,
            '--output', root / 'thresholds.json')
        run('prediction', 'predict', '--recipe', recipe_path, *initialization,
            '--source', root / 'validation-0.png', '--output', root / 'prediction')
        if args.onnx:
            run('export', 'export', '--recipe', recipe_path, *initialization,
                '--output', root / 'bundle')
            import onnx
            onnx.checker.check_model(str(root / 'bundle/model.onnx'), full_check=True)
            verify_bundle(args.probe, recipe_path, root / 'bundle', state,
                          weights['sha256'], env, root)
            run('bundle-prediction', 'bundle-predict', '--bundle', root / 'bundle',
                '--source', root / 'validation-0.png', '--output', root / 'bundle-result.json')
            assert len(read('bundle-result.json')['results']) == 1
        png(root / 'train-0.png', 99)
        run('changed-data', 'resume', '--recipe', recipe_path,
            '--checkpoint', root / 'paused/last.checkpoint', '--output', root / 'rejected',
            succeeds=False)
        print('Native workflow and exact raw/EMA resume passed', args.device, args.precision)


if __name__ == '__main__':
    main()
