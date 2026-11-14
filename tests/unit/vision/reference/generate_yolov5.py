# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Generate development-only references from an exact original YOLOv5 checkout.

The original anchor buffer stores grid units and its head uses an additional
ModuleList name. The fixture explicitly translates that state representation;
it does not qualify automatic import of arbitrary official checkpoints.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--architecture', default='yolov5n')
    parser.add_argument('--variant', choices=['p5', 'p6'], default='p5')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.upstream_root.resolve()
    os.environ.update(YOLOv5_AUTOINSTALL='false',
                      YOLOv5_CONFIG_DIR=str(args.output / 'settings'),
                      MPLCONFIGDIR=str(args.output / 'matplotlib'))
    sys.path.insert(0, str(root))
    import torch
    from models.yolo import DetectionModel
    from safetensors.torch import save_file

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    config = root / 'models' / (('hub/' + args.architecture + '6.yaml')
                               if args.variant == 'p6' else args.architecture + '.yaml')
    model = DetectionModel(str(config), ch=3, nc=3)
    model.eval()
    head = model.model[-1]
    head.training = True
    head_prefix = f'model.{len(model.model) - 1}.'

    def name(source):
        return source.replace(head_prefix + 'm.', head_prefix)

    upstream_state = args.output / 'upstream-state.safetensors'
    state_path = args.output / 'state.safetensors'
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(upstream_state))
    translated = {name(k): v.detach().contiguous() for k, v in model.state_dict().items()
                  if k != head_prefix + 'anchors'}
    translated['anchors_pixels'] = (head.anchors * head.stride.view(-1, 1, 1)).detach().contiguous()
    save_file(translated, str(state_path))
    side = 128 if args.variant == 'p6' else 64
    image = torch.randn(1, 3, side, side, requires_grad=True)
    output = torch.cat([level.reshape(1, -1, 8) for level in model(image)], 1)

    def record(value):
        value = value.detach().cpu().float()
        return {'shape': list(value.shape), 'values': value.flatten().tolist()}

    fixture = {'schema': 'vision.independent-reference/v1', 'architecture': args.architecture,
               'task': 'detect', 'variant': args.variant, 'classes': 3,
               'upstream_revision': '915bbf294bb74c859f0b41f1c23bc395014ea679', 'seed': 17,
               'torch_version': str(torch.__version__), 'state': str(state_path.resolve()),
               'state_sha256': hashlib.sha256(state_path.read_bytes()).hexdigest(),
               'upstream_state_sha256': hashlib.sha256(upstream_state.read_bytes()).hexdigest(),
               'state_transforms': {'head_prefix': head_prefix + 'm. -> ' + head_prefix,
                                    'anchors_pixels': 'upstream anchors * stride'},
               'parameter_elements': sum(p.numel() for p in model.parameters()),
               'input': record(image), 'outputs': [record(output)]}
    output.square().mean().backward()
    fixture['input_gradient'] = record(image.grad)
    available = [(k, p) for k, p in model.named_parameters() if p.grad is not None]
    selected = dict([available[0], available[len(available) // 2], available[-1]])
    fixture['gradients'] = {name(k): record(p.grad) for k, p in selected.items()}
    torch.optim.SGD(model.parameters(), lr=.001).step()
    fixture['updated'] = {name(k): record(p) for k, p in selected.items()}
    fixture['source_sha256'] = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob('*.py'))}
    fixture['source_sha256'][config.relative_to(root).as_posix()] = hashlib.sha256(config.read_bytes()).hexdigest()
    (args.output / 'fixture.json').write_text(json.dumps(fixture) + '\n', encoding='utf-8')
    print('Original YOLOv5 fixture generated:', args.architecture, args.variant, flush=True)


if __name__ == '__main__':
    main()
