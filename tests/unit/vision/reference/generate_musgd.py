"""Generate fixed optimizer references from an unchanged official MuSGD module."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--separate', action='store_true',
                    help='Generate dense per-matrix BF16 references from the earlier official implementation.')
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('official_muon_reference', args.source)
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)

shapes = [(3, 2), (1, 2), (40, 2), (2, 2, 2, 2), (2, 1, 3, 1), (3,)]
if args.separate:
    shapes = [(7, 5), (9, 5), (17, 5), (5, 11), (4, 3, 3, 3), (3,)]
generator = torch.Generator().manual_seed(20261126)
initial = [(torch.arange(torch.tensor(s).prod().item(), dtype=torch.float32)
            .reshape(s) * .01 - .07) for s in shapes]
variants = {}
for layout in (['contiguous'] if args.separate else ['contiguous', 'channels_last']):
    parameters = [torch.nn.Parameter(x.clone().contiguous(memory_format=torch.channels_last)
                  if layout == 'channels_last' and x.ndim == 4 else x.clone()) for x in initial]
    optimizer = official.MuSGD([
        {'params': parameters[:-1], 'use_muon': True},
        {'params': parameters[-1:], 'use_muon': False},
    ], lr=.012, momentum=.83, weight_decay=.017, nesterov=True, muon=.528, sgd=.674)
    steps = []
    for step in range(3):
        gradients = []
        for i, p in enumerate(parameters):
            if not args.separate and step == 1 and i in (1, 5):
                p.grad = None
                gradients.append(None)
            else:
                g = torch.cos(torch.arange(p.numel(), dtype=torch.float32) * .17 + step + i)
                if args.separate:
                    g = torch.randint(-256, 257, (p.numel(),), generator=generator).float() / 256
                g = g.reshape(p.shape) * .03
                p.grad = g.contiguous(memory_format=torch.channels_last) if layout == 'channels_last' and g.ndim == 4 else g
                gradients.append(p.grad.flatten().tolist())
        optimizer.param_groups[0]['lr'] = .012 / (step + 1)
        optimizer.param_groups[1]['lr'] = .018 / (step + 1)
        optimizer.step()
        steps.append({'gradients': gradients,
                      'rates': [group['lr'] for group in optimizer.param_groups],
                      'parameters': [p.detach().flatten().tolist() for p in parameters]})
    variants[layout] = steps
payload = {'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
           'torch': torch.__version__, 'shapes': shapes,
           'initial': [p.flatten().tolist() for p in initial], 'variants': variants}
# Bound each literal below MSVC's per-literal size limit without changing JSON.
encoded = json.dumps(payload, separators=(',', ':'))
literals = '\n'.join('R"fixture(' + encoded[i:i+4096] + ')fixture"'
                     for i in range(0, len(encoded), 4096))
header = '''// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
// Independent Ultralytics 8.4.177 MuSGD CPU reference; do not edit values.
#ifndef JYPPX_LIUDING_VISION_MUSGD_REFERENCE_HPP
#define JYPPX_LIUDING_VISION_MUSGD_REFERENCE_HPP
#include <string_view>
namespace jyppx::liuding::vision::tests {
inline constexpr std::string_view musgd_reference =
%s;
}
#endif
''' % literals
args.output.write_text(header, encoding='utf-8')
if args.separate:
    header = header.replace('Ultralytics 8.4.177', 'Ultralytics cb13d5f9')
    header = header.replace('MUSGD_REFERENCE_HPP', 'MUSGD_SEPARATE_REFERENCE_HPP')
    header = header.replace('musgd_reference =', 'musgd_separate_reference =')
    args.output.write_text(header, encoding='utf-8')
print(json.dumps({'output': str(args.output), 'source': payload['source_sha256'],
                  'torch': payload['torch'], 'steps': len(steps)}))
