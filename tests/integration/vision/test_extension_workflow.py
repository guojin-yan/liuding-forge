# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Exercise actual native schema-2 training, recovery and multi-input bundles."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib

from test_workflow import png, verify_bundle


CASES = {
    'semantic': ('fcn_resnet50', 'semantic_segment'),
    'depth': ('depth_resnet50_fpn', 'depth'),
    'reid': ('reid_resnet50', 'reid'),
    'simmim': ('simmim_vit_tiny', 'pretrain_simmim'),
    'fast_rcnn': ('fast_rcnn_resnet50_fpn', 'detect'),
    'rtdetr': ('rtdetr_resnet18_vd', 'detect'),
    'fcos': ('fcos_resnet50_bifpn', 'detect'),
    'ssd': ('ssd300_vgg16', 'detect'),
    'text': ('yoloe_v8s_text', 'detect'),
    'visual': ('yoloe_v8s_visual', 'detect'),
    'prompt_free': ('yoloe_v8s_prompt_free', 'detect'),
    'text_segment': ('yoloe_v8s_text_segment', 'segment'),
    'world': ('yoloworld_v8s_text', 'detect'),
    'distillation': ('resnet18', 'classify'),
}


def mask_png(path):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    rows = b''.join(b'\0' + bytes(255 if y == 0 else (x // 16 + y // 16) % 2
                                  for x in range(64)) for y in range(64))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' +
                     chunk(b'IHDR', struct.pack('>IIBBBBB', 64, 64, 8, 0, 0, 0, 0)) +
                     chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def depth_archive(path):
    size = 64 * 64
    header = {'depth': {'dtype': 'F32', 'shape': [1, 64, 64], 'data_offsets': [0, size * 4]},
              'valid': {'dtype': 'BOOL', 'shape': [1, 64, 64], 'data_offsets': [size * 4, size * 5]}}
    raw = json.dumps(header).encode('utf-8')
    raw += b' ' * (-len(raw) % 8)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw +
                     struct.pack('<' + 'f' * size, *(1 + (i % 64) / 32 for i in range(size))) +
                     bytes(i % 64 != 0 for i in range(size)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--onnx', action='store_true')
    parser.add_argument('--probe', type=Path)
    args = parser.parse_args()
    architecture, task = CASES[args.case]
    with tempfile.TemporaryDirectory(prefix='vision-extension-') as directory:
        root = Path(directory)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

        def write(name, value):
            path = root / name
            path.write_text(json.dumps(value), encoding='utf-8')
            return path

        def read(name):
            return json.loads((root / name).read_text(encoding='utf-8'))

        def run(label, *arguments, succeeds=True):
            result = subprocess.run([str(args.cli), *map(str, arguments)], cwd=root, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
            if (result.returncode == 0) != succeeds:
                raise AssertionError(label + ': ' + result.stdout.decode('utf-8', errors='replace'))
            print(label, result.returncode, flush=True)
            return result.stdout

        prompts = [[float(i == c) for i in range(512)] for c in range(2)]
        proposals = [[4 + i % 4, 4 + i // 4, 48, 48] for i in range(16)]
        visual = [[0, 8, 8, 24, 24], [1, 32, 32, 48, 48]]
        context = {'schema': 'vision.input_context/v1', 'label_space_id': 'test-classes/v1',
                   'text_encoder_id': 'test-fixed-embeddings/v1', 'prompt_embeddings': prompts,
                   'visual_prompts': visual, 'proposals': proposals,
                   'proposal_generator': 'fixture-grid/v1', 'proposal_inventory': 'fixed-16/v1'}
        context_path = write('context.json', context)
        for split, offset in [('train', 0), ('validation', 20)]:
            rows = []
            for i in range(4):
                name = f'{split}-{i}.png'
                png(root / name, offset + i)
                row = {'image': name, 'width': 64, 'height': 64}
                if task == 'semantic_segment':
                    mask_png(root / ('mask-' + name))
                    row['semantic_mask'] = 'mask-' + name
                elif task == 'depth':
                    depth_archive(root / (name + '.safetensors'))
                    row['depth'] = name + '.safetensors'
                elif task in ['reid', 'classify']:
                    row.update(identity=i // 2, camera_id=str(i % 2), sequence_id=split)
                elif task != 'pretrain_simmim':
                    row['boxes'] = [[i % 2, 8, 8, 40, 40]]
                    if args.case == 'fast_rcnn':
                        row['proposals'] = proposals
                    if args.case == 'visual':
                        row['visual_prompts'] = visual
                    if task == 'segment':
                        row['polygons'] = [[[[8, 8], [40, 8], [40, 40], [8, 40]]]]
                rows.append(row)
            classes = [{'source_id': i + (100 if split == 'validation' else 0), 'name': str(i)}
                       for i in range(2)] if task == 'reid' else ['a', 'b']
            doc = {'schema': 'vision.dataset/v1', 'image_root': '.', 'categories': classes,
                   'images': rows, 'label_space_id': context['label_space_id'],
                   'text_encoder_id': context['text_encoder_id']}
            if args.case in ['text', 'text_segment', 'world']:
                doc.update(prompt_embeddings=prompts, prompt_dimension=512)
            if args.case == 'fast_rcnn':
                doc.update(proposal_generator=context['proposal_generator'],
                           proposal_inventory=context['proposal_inventory'])
            write(split + '.json', doc)
        recipe = {'schema_version': 2,
                  'model': {'architecture': architecture, 'task': task, 'num_classes': 2},
                  'image_size': 300 if args.case == 'ssd' else 256 if args.case == 'rtdetr' else 64,
                  'batch_size': 4 if task == 'reid' else 2, 'workers': 1, 'epochs': 2, 'seed': 23,
                  'optimizer': {'name': 'sgd', 'learning_rate': .0001, 'weight_decay': .0001},
                  'extension': {'training_manifest': 'train.json',
                                'validation_manifest': 'validation.json'},
                  'augmentation': {'horizontal_flip': .5}, 'classification_augmentation': {},
                  'ema': {'enabled': task != 'pretrain_simmim'},
                  'full_validation_metrics': True, 'verify_image_contents': True}
        if task == 'pretrain_simmim':
            recipe['extension'].update(model_patch_size=16, mask_patch_size=32,
                                       mask_ratio=.6, activation_recomputation=True)
        if args.case == 'distillation':
            teacher_recipe = write('teacher.json', recipe)
            run('teacher', 'train', '--recipe', teacher_recipe, '--output', root / 'teacher')
            teacher = read('teacher/weights.json')['raw']
            recipe['extension'].update(teacher_architecture='resnet18',
                                       teacher_weights='teacher/' + teacher['file'],
                                       teacher_sha256=teacher['sha256'], feature_outputs=True,
                                       feature_projection=512, feature_distillation_weight=.5)
        recipe_path = write('recipe.json', recipe)
        run('continuous', 'train', '--recipe', recipe_path, '--output', root / 'continuous')
        run('paused', 'train', '--recipe', recipe_path, '--pause-after', 1, '--output', root / 'paused')
        run('resumed', 'resume', '--recipe', recipe_path, '--checkpoint', root / 'paused/last.checkpoint',
            '--output', root / 'resumed')
        continuous, resumed = read('continuous/training.json'), read('resumed/training.json')
        assert continuous == resumed, 'Resumed result differs'
        assert resumed['status'] == 1 and resumed['epochs'] == 2
        assert resumed['updates'] == (2 if task == 'reid' else 4)
        assert all(math.isfinite(v) for v in resumed['training'].values())
        assert read('continuous/weights.json') == read('resumed/weights.json'), 'Raw/EMA differs'
        weights = read('resumed/weights.json')['raw']
        state = root / 'resumed' / weights['file']
        assert hashlib.sha256(state.read_bytes()).hexdigest() == weights['sha256']
        initialization = ['--weights', state, '--sha256', weights['sha256']]
        run('validation', 'val', '--recipe', recipe_path, *initialization,
            '--output', root / 'validation-result.json')
        run('prediction', 'predict', '--recipe', recipe_path, *initialization,
            '--source', root / 'validation-0.png', '--sample-data', context_path,
            '--output', root / 'prediction')
        if task == 'pretrain_simmim':
            run('encoder', 'extract-encoder', '--recipe', recipe_path, *initialization,
                '--output', root / 'encoder.safetensors')
            assert (root / 'encoder.safetensors').stat().st_size > 1000
        if args.onnx:
            run('export', 'export', '--recipe', recipe_path, *initialization,
                '--output', root / 'bundle', '--proposals', 16)
            import onnx
            onnx.checker.check_model(str(root / 'bundle/model.onnx'), full_check=True)
            verify_bundle(args.probe, recipe_path, root / 'bundle', state,
                          weights['sha256'], env, root)
            run('bundle-prediction', 'bundle-predict', '--bundle', root / 'bundle',
                '--source', root / 'validation-0.png', '--sample-data', context_path,
                '--output', root / 'bundle-result.json')
            assert len(read('bundle-result.json')['results']) == 1
        png(root / 'train-0.png', 99)
        run('changed-data', 'resume', '--recipe', recipe_path,
            '--checkpoint', root / 'paused/last.checkpoint', '--output', root / 'rejected', succeeds=False)
        print('Native extension workflow passed', args.case)


if __name__ == '__main__':
    main()
