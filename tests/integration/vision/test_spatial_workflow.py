# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Verify canonical object imports and actual native spatial training recovery."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from test_workflow import png, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--architecture', default='yolov8n')
    parser.add_argument('--task', choices=['detect', 'segment', 'pose', 'obb'], default='detect')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', default='full')
    parser.add_argument('--onnx', action='store_true')
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='vision-spatial-') as directory:
        root = Path(directory)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')

        def write(name, value):
            path = root / name
            path.write_text(json.dumps(value), encoding='utf-8')
            return path

        def read(name):
            return json.loads((root / name).read_text(encoding='utf-8'))

        def run(label, *arguments, succeeds=True):
            result = subprocess.run([str(args.cli), *map(str, arguments)], cwd=root, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            (root / (label + '.log')).write_bytes(result.stdout)
            if (result.returncode == 0) != succeeds:
                raise AssertionError(label + ': ' + result.stdout.decode('utf-8', errors='replace'))
            print(label, result.returncode, flush=True)
            return result.stdout

        xml_root = root / 'voc'
        xml_root.mkdir()
        for split, count, offset in [('train', 5, 0), ('validation', 3, 10)]:
            image_dir, label_dir = root / 'images' / split, root / 'labels' / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for index in range(count):
                png(image_dir / f'{index}.png', offset + index)
                category = index % 2
                if index == count - 1:
                    label = ''  # An explicit empty label is a valid negative image.
                elif args.task == 'detect':
                    label = f'{category} 0.3125 0.3125 0.375 0.375\n'
                elif args.task in ['segment', 'obb']:
                    label = f'{category} 0.125 0.125 0.5 0.125 0.5 0.5 0.125 0.5\n'
                else:
                    label = f'{category} 0.3125 0.3125 0.375 0.375 0.25 0.25 2 0.375 0.375 2\n'
                (label_dir / f'{index}.txt').write_text(label, encoding='utf-8')
                if split == 'train' and args.extended:
                    objects = (f'<object><name>{["a", "b"][category]}</name><difficult>0</difficult>'
                               '<bndbox><xmin>9</xmin><ymin>9</ymin><xmax>32</xmax><ymax>32</ymax>'
                               '</bndbox></object>') if label else ''
                    (xml_root / f'{index}.xml').write_text(
                        f'<annotation><filename>images/train/{index}.png</filename>'
                        '<size><width>64</width><height>64</height><depth>3</depth></size>'
                        + objects + '</annotation>', encoding='utf-8')
                    shapes = [{'label': ['a', 'b'][category], 'shape_type': 'rectangle',
                               'points': [[8, 8], [32, 32]], 'flags': {}}] if label else []
                    write(f'images/train/{index}.json', {'imagePath': f'{index}.png',
                          'imageWidth': 64, 'imageHeight': 64, 'shapes': shapes})
        (root / 'dataset.yaml').write_text(
            'path: .\ntrain: images/train\nval: images/validation\nnames: [a, b]\n'
            + ('kpt_shape: [2, 3]\nflip_idx: [1, 0]\n' if args.task == 'pose' else ''),
            encoding='utf-8')
        recipe = {
            'schema_version': 1,
            'model': {'architecture': args.architecture, 'task': args.task,
                      'variant': 'p5', 'num_classes': 2, 'keypoints': 2},
            'image_size': 64, 'batch_size': 2, 'workers': 2, 'prefetch_batches': 2,
            'epochs': 2, 'accumulate_steps': 2, 'seed': 23,
            'device': args.device, 'precision': args.precision,
            'initial_loss_scale': 1. if args.precision == 'cuda_float16' else 65536.,
            'optimizer': {'name': 'sgd', 'learning_rate': .001, 'weight_decay': .0001},
            'scheduler': {'warmup_updates': 1, 'final_factor': .1},
            'ema': {'enabled': True}, 'full_validation_metrics': True,
            'dataset': {'format': 'yolo', 'source': 'dataset.yaml'},
            'validation': {'format': 'yolo', 'source': 'dataset.yaml', 'split': 'validation'},
            'cache': {'ram_bytes': 1 << 20, 'disk_bytes': 1 << 20, 'directory': 'decode-cache'},
            'augmentation': {'horizontal_flip': .5, 'hsv_h': .05, 'hsv_s': .2,
                             'mosaic': .5, 'mixup': .3, 'cutmix': .2, 'close_epochs': 1},
            'verify_image_contents': True}
        if args.task == 'pose':
            recipe['keypoint_sigmas'] = [.1, .1]
        if args.task in ['pose', 'obb']:
            recipe['augmentation']['cutmix'] = 0
        recipe_path = write('recipe.json', recipe)
        run('inspect', 'inspect', '--recipe', recipe_path, '--decode', '--output', root / 'inspection.json')
        run('convert-yolo', 'convert-data', '--recipe', recipe_path, '--output', root / 'canonical.json')
        canonical = read('canonical.json')
        assert len(canonical['images']) == 5 and len(canonical['annotations']) == 4
        assert all(item['bbox'] == [8, 8, 24, 24] for item in canonical['annotations'])
        if args.extended:
            def normalized(value):
                paths = {row['id']: row['file_name'] for row in value['images']}
                return sorted((paths[row['image_id']], row['category_id'], row['bbox'])
                              for row in value['annotations'])
            for name, source in [('coco', 'canonical.json'), ('voc', 'voc'), ('labelme', 'images/train')]:
                converted = dict(recipe, dataset={'format': name, 'source': source,
                                 'image_root': '.', 'categories': ['a', 'b']})
                path = write(name + '.recipe.json', converted)
                run('convert-' + name, 'convert-data', '--recipe', path,
                    '--output', root / (name + '.json'))
                assert normalized(read(name + '.json')) == normalized(canonical), name
            run('preview', 'preview', '--recipe', recipe_path, '--samples', 3,
                '--output', root / 'preview')
        run('continuous', 'train', '--recipe', recipe_path, '--output', root / 'continuous')
        run('paused', 'train', '--recipe', recipe_path, '--pause-after', 1, '--output', root / 'paused')
        run('resumed', 'resume', '--recipe', recipe_path, '--checkpoint', root / 'paused/last.checkpoint',
            '--output', root / 'resumed')
        continuous, resumed = read('continuous/training.json'), read('resumed/training.json')
        assert continuous['status'] == resumed['status'] == 1
        assert continuous['updates'] == resumed['updates'] == 4
        assert continuous['epochs'] == resumed['epochs'] == 2
        assert read('continuous/weights.json') == read('resumed/weights.json'), 'Raw/EMA resume differs'
        assert all(math.isfinite(float(value)) for value in resumed['validation'].values())
        weights = read('resumed/weights.json')['raw']
        state = root / 'resumed' / weights['file']
        assert hashlib.sha256(state.read_bytes()).hexdigest() == weights['sha256']
        initialization = ['--weights', state, '--sha256', weights['sha256']]
        run('validation', 'val', '--recipe', recipe_path, *initialization,
            '--output', root / 'validation-result.json')
        run('prediction', 'predict', '--recipe', recipe_path, *initialization,
            '--source', root / 'images/validation/0.png', '--output', root / 'prediction')
        if args.onnx:
            run('export', 'export', '--recipe', recipe_path, *initialization,
                '--output', root / 'bundle', '--proposals', 16)
            import onnx
            onnx.checker.check_model(str(root / 'bundle/model.onnx'), full_check=True)
            if args.architecture != 'faster_rcnn_resnet50_fpn':
                verify_bundle(args.probe, recipe_path, root / 'bundle', state,
                              weights['sha256'], env, root)
            run('bundle-prediction', 'bundle-predict', '--bundle', root / 'bundle',
                '--source', root / 'images/validation/0.png', '--output', root / 'bundle-result.json')
            assert len(read('bundle-result.json')['results']) == 1
        if args.extended:
            search = write('search.json', {'trials': 2, 'seed': 31, 'objective': 'validation.map',
                           'maximize': True, 'space': [{'field': 'learning_rate',
                           'distribution': 'uniform', 'lower': .0005, 'upper': .001}]})
            result = run('tune', 'tune', '--recipe', recipe_path, '--search', search,
                         '--execution', 'process', '--trial-timeout', 120, '--output', root / 'search')
            assert json.loads(result)['completed'] == 2
        png(root / 'images/train/0.png', 99)
        run('changed-data', 'resume', '--recipe', recipe_path,
            '--checkpoint', root / 'paused/last.checkpoint', '--output', root / 'rejected', succeeds=False)
        print('Canonical data and native spatial workflow passed', args.architecture, args.task,
              args.device, args.precision)


if __name__ == '__main__':
    main()
