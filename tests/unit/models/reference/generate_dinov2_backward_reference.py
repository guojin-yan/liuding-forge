#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Generate pinned DINOv2-S forward/gradient/grouped-SGD references offline.

This synthetic numerical test grants no dataset accuracy or runtime performance
qualification. Python and Transformers are used only to generate test data.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--weights-dir', required=True, type=Path)
parser.add_argument('--output-dir', required=True, type=Path)
parser.add_argument('--reference-packages', type=Path)
args = parser.parse_args()
root = args.weights_dir.resolve(strict=True)
work = args.output_dir.resolve()
work.mkdir(parents=True, exist_ok=True)
if any((work / name).exists() for name in ('dinov2-backward-reference.safetensors', 'reference-generation.json')):
    raise ValueError('Reference outputs must be new files')
repo = Path(__file__).resolve().parents[4]
if args.reference_packages:
    sys.path.append(str(args.reference_packages.resolve(strict=True)))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import Dinov2Config
from transformers.models.dinov2 import modeling_dinov2


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


contract = {'reference': {
    'torch': '2.14.1+cpu', 'transformers': '4.56.2',
    'source_sha256': '81116a0fddfd3d96f9d175cc56a22ded11382eb21bc78748e79b29d34660f862',
    'facebook_weights_sha256': 'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1',
    'liuding_weights_sha256': 'f3ef1ae01fd4af3cd15926a0d5e631fa0caea50ca68b29db8a68c8c84c9810e5',
    'input_reference_sha256': '4736a2ef8f98bfd773700c6d06fe2614a0e5fb8d6d8c02c36b9f0c643d6250a6',
}}
assert torch.__version__ == contract['reference']['torch']
assert transformers.__version__ == contract['reference']['transformers']
source = Path(inspect.getsourcefile(modeling_dinov2))
assert sha(source) == contract['reference']['source_sha256']
facebook = root / 'dinov2-small.facebook.safetensors'
liuding = root / 'dinov2-small.liuding.safetensors'
inputs = root / 'reference/pretrained-reference.safetensors'
for path, key in ((facebook, 'facebook_weights_sha256'), (liuding, 'liuding_weights_sha256'),
                  (inputs, 'input_reference_sha256')):
    assert sha(path) == contract['reference'][key]
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

config = Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                      image_size=518, patch_size=14, layer_norm_eps=1e-6, use_swiglu_ffn=False)
config._attn_implementation = 'eager'
model = modeling_dinov2.Dinov2Model(config).cpu().train()
model.load_state_dict(load_file(str(facebook)), strict=True)
head = torch.nn.Linear(384, 10)
with torch.no_grad():
    head.weight.copy_((torch.arange(3840).reshape(10, 384).remainder(97).float() - 48) / 3000)
    head.bias.copy_((torch.arange(10).float() - 4.5) / 1000)
image = load_file(str(inputs))['dinov2_input']
images = torch.cat((image, image.roll((7, -5), (2, 3))), dim=0).contiguous()
labels = torch.tensor([0, 1], dtype=torch.long)
output = {'input1': images, 'input2': images.roll((1, 2), (2, 3)).contiguous(),
          'labels': labels.float(), 'head_weight': head.weight.detach().clone(),
          'head_bias': head.bias.detach().clone()}

# Use only the already reviewed name mapping, not the production architecture.
import importlib.util
mapping_file = repo / 'scripts/models/convert_dinov2_hf.py'
spec = importlib.util.spec_from_file_location('reviewed_dinov2_mapping', mapping_file)
mapping_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapping_module)
mapping = mapping_module.mapping()
parameters = dict(model.named_parameters())
assert {name for _, _, sources in mapping for name in sources} == set(parameters) - {'embeddings.mask_token'}
assert len(mapping) == 174

selected = {
    'cls_token', 'pos_embed', 'patch_embed.proj.weight', 'blocks.0.attn.qkv.weight',
    'blocks.11.mlp.fc2.weight', 'norm.weight',
}
optimizer = torch.optim.SGD([
    {'params': model.parameters(), 'lr': 0.0001},
    {'params': head.parameters(), 'lr': 0.01},
], momentum=.9, weight_decay=.0001)
inventory = []
started = time.monotonic()
for step in (1, 2):
    optimizer.zero_grad(set_to_none=True)
    sample = output['input' + str(step)].detach().clone().requires_grad_()
    features = model(pixel_values=sample).last_hidden_state[:, 0]
    logits = head(features)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    output[f'features{step}'] = features.detach().contiguous()
    output[f'logits{step}'] = logits.detach().contiguous()
    output[f'loss{step}'] = loss.detach().reshape(1)
    output[f'input_gradient{step}'] = sample.grad.detach().contiguous()
    assert parameters['embeddings.mask_token'].grad is None
    def canonical(gradients):
        result = {}
        for target, shape, sources in mapping:
            values = [parameters[name].grad if gradients else parameters[name].detach() for name in sources]
            assert all(value is not None for value in values)
            tensor = torch.cat(values, dim=0) if len(values) > 1 else values[0]
            assert list(tensor.shape) == shape and torch.isfinite(tensor).all()
            result['backbone.' + target] = tensor
        for name, parameter in head.named_parameters():
            result['head.' + name] = parameter.grad if gradients else parameter.detach()
        return result
    gradients = canonical(True)
    for name, tensor in gradients.items():
        flat = tensor.detach().flatten()
        double = flat.double()
        output[f'gradient_stats{step}.' + name] = torch.stack((double.norm(), double.abs().max(), double.mean())).float()
        output[f'gradient_sample{step}.' + name] = flat[:16].clone()
        if name.removeprefix('backbone.') in selected or name.startswith('head.'):
            output[f'gradient{step}.' + name] = tensor.detach().clone().contiguous()
        if step == 1:
            inventory.append({'name': name, 'shape': list(tensor.shape), 'elements': tensor.numel()})
    optimizer.step()
    for name, tensor in canonical(False).items():
        if name.removeprefix('backbone.') in selected or name.startswith('head.'):
            output[f'updated{step}.' + name] = tensor.detach().clone().contiguous()
    print(json.dumps({'phase': 'reference_step', 'step': step, 'loss': loss.item(), 'gradients': len(gradients)}), flush=True)
assert len(inventory) == 176
output_file = work / 'dinov2-backward-reference.safetensors'
assert not output_file.exists()
save_file({name: tensor.contiguous() for name, tensor in sorted(output.items())}, str(output_file))
if (output_file.stat().st_size, sha(output_file)) != (
        33568048, '4ae767b4f071bebb46a4e63ad7b0002dae6f25f1f8f7d0471ded57654ecc77e2'):
    raise ValueError('Generated fixture differs from the reviewed complete identity')
report = {
    'status': 'reference_generated', 'python': sys.version, 'torch': torch.__version__,
    'transformers': transformers.__version__, 'source_sha256': sha(source),
    'mapping_source_sha256': sha(mapping_file), 'generator_sha256': sha(__file__),
    'reference_sha256': sha(output_file), 'reference_size_bytes': output_file.stat().st_size,
    'reference_tensors': len(output), 'parameter_inventory': inventory,
    'full_gradient_and_parameter_targets': sorted(selected | {'head.weight', 'head.bias'}),
    'seconds': time.monotonic() - started,
    'task_quality_qualified': False,
}
(work / 'reference-generation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: value for key, value in report.items() if key != 'parameter_inventory'}, indent=2), flush=True)
