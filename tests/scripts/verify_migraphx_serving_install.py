#!/usr/bin/env python3
"""Offline installed native GPU bridge and serving-without-SDK contracts."""
from pathlib import Path
import argparse
import array
import hashlib
import json
import math
import re
import subprocess
import time
import xml.etree.ElementTree as ET

p = argparse.ArgumentParser()
for name in ('cmake', 'ctest', 'source-root', 'parent-build', 'fixtures'):
    p.add_argument('--' + name, required=True)
p.add_argument('--model', choices=('linear', 'resnet18'), default='linear')
p.add_argument('--case', choices=('0', '1'), default='0')
a = p.parse_args()
parent, source = Path(a.parent_build), Path(a.source_root)
out = parent / 'evidence/migraphx-serving-installed' / str(time.time_ns())
out.mkdir(parents=True, exist_ok=False)
receipt = dict(status='preparing', model=a.model, case=int(a.case), processes=[])


def save():
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


def run(label, args, expect=True, timeout=300):
    command = list(map(str, args))
    started = time.monotonic()
    row = dict(label=label, command=command, timeout_seconds=timeout)
    receipt['processes'].append(row)
    save()
    try:
        with (out / (label + '.log')).open('wb') as log:
            result = subprocess.run(command, stdout=log, stderr=log, timeout=timeout)
        row['returncode'] = result.returncode
        if expect and result.returncode:
            raise RuntimeError(label + ' failed; ' + str(out / (label + '.log')))
        if not expect and not result.returncode:
            raise RuntimeError(label + ' unexpectedly succeeded')
    finally:
        row['elapsed_seconds'] = time.monotonic() - started
        save()


def initial_cache(path, flags):
    assert all(']====]' not in str(value) for value in flags.values())
    path.write_text(''.join(f'set({key} [====[{value}]====] CACHE STRING "" FORCE)\n'
                            for key, value in flags.items()), encoding='utf-8')


def audit(child):
    references = {
        ('linear', '0'): '3b106db38a6dca7aafe95d1af546237ca38ac4d74a084a8f1b0cea5a361c709a',
        ('linear', '1'): '509b6a760633711b6655feca75f3d873d5184fc1076cc55213773a8b91be5529',
        ('resnet18', '0'): '4c8b23c87357417e3c647d9bc7d8a3db1cc8f1a8897bda5088865910b56b489f',
        ('resnet18', '1'): 'a80ff115fbbddc54314cdebace0f302765ba4fc311e2f52328952ec23c653165',
    }
    expected_bytes = (Path(a.fixtures) / a.model / f'case-{a.case}-output.f32').read_bytes()
    expected_sha = hashlib.sha256(expected_bytes).hexdigest()
    assert expected_sha == references[a.model, a.case]
    actual_bytes = (child / 'actual.f32').read_bytes()
    actual_sha = hashlib.sha256(actual_bytes).hexdigest()
    actual, expected = array.array('f'), array.array('f')
    actual.frombytes(actual_bytes)
    expected.frombytes(expected_bytes)
    batch = 5 if a.case == '1' else 2
    columns = 7 if a.model == 'resnet18' else 3
    assert len(actual) == len(expected) == batch * columns
    violations = sum(not math.isfinite(x) or abs(x - y) > 1e-5 + 1e-4 * abs(y)
                     for x, y in zip(actual, expected))
    assert violations == 0
    if (a.model, a.case) == ('linear', '0'):
        assert actual_sha == expected_sha
    lock_path = child / 'environment-lock.json'
    lock = json.loads(lock_path.read_bytes())
    assert lock['format'] == 'jyppx.liuding.migraphx_runtime_lock' and lock['schema_version'] == 1
    assert lock['program'] == dict(target='gpu', dtype='float32', offload_copy=True,
        fast_math=False, exhaustive_tune=False, input_name='input',
        input_shape=[batch, 3, 224, 224] if a.model == 'resnet18' else [batch, 8],
        output_shape=[batch, columns])
    assert (child / 'modules.txt').stat().st_size > 0
    receipt['output_audit'] = dict(batch=batch, elements=len(actual), violations=violations,
        max_abs=max(abs(x - y) for x, y in zip(actual, expected)), output_sha256=actual_sha,
        reference_sha256=expected_sha, lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest())


try:
    cache = dict((m[1], m[2]) for line in (parent / 'CMakeCache.txt').read_text(encoding='utf-8').splitlines()
                 if (m := re.match(r'^([^/#][^:]*):[^=]+=(.*)$', line)))
    for name in ('FETCHCONTENT_FULLY_DISCONNECTED', 'FETCHCONTENT_UPDATES_DISCONNECTED'):
        assert cache[name] == 'ON'
    install = out / 'install'
    run('install', [a.cmake, '--install', parent, '--config', 'Release', '--prefix', install])
    assert (install / 'include/jyppx/liuding/deployment/migraphx_serving/backend.hpp').is_file()
    assert not list((install / 'bin').glob('migraphx*.dll'))
    assert not list((install / 'lib').glob('libmigraphx*.so*'))
    flags = {key: cache[key] for key in ('CMAKE_CXX_COMPILER', 'CMAKE_MAKE_PROGRAM',
        'CMAKE_TOOLCHAIN_FILE', 'CMAKE_PREFIX_PATH', 'Torch_DIR', 'HIP_DIR', 'hip_DIR',
        'CMAKE_HIP_COMPILER') if cache.get(key)}
    flags.update(CMAKE_BUILD_TYPE='Release', CMAKE_MSVC_RUNTIME_LIBRARY='MultiThreadedDLL',
        FETCHCONTENT_FULLY_DISCONNECTED='ON', FETCHCONTENT_UPDATES_DISCONNECTED='ON',
        jyppx_DIR=str(install / 'lib/cmake/jyppx'),
        JYPPX_LIUDING_MIGRAPHX_ROOT=cache['JYPPX_LIUDING_MIGRAPHX_ROOT'],
        JYPPX_LIUDING_MIGRAPHX_HIP_ROOT=cache['JYPPX_LIUDING_MIGRAPHX_HIP_ROOT'],
        JYPPX_MIGRAPHX_CONSUMER_FIXTURES=a.fixtures,
        JYPPX_MIGRAPHX_CONSUMER_MODEL=a.model, JYPPX_MIGRAPHX_CONSUMER_CASE=a.case)
    resnet = a.model == 'resnet18'
    for mode in (('runtime',) if resnet else ('runtime', 'serving', 'missing-sdk')):
        selected = dict(flags)
        if mode != 'runtime':
            selected['JYPPX_LIUDING_MIGRAPHX_ROOT'] = ''
            selected['JYPPX_LIUDING_MIGRAPHX_HIP_ROOT'] = ''
        if mode == 'serving':
            selected['JYPPX_MIGRAPHX_SERVING_ONLY'] = 'ON'
        initial = out / (mode + '.cmake')
        initial_cache(initial, selected)
        child = out / mode
        run(mode + '-configure', [a.cmake, '-S', source / 'tests/install/migraphx_serving_consumer',
            '-B', child, '-G', cache['CMAKE_GENERATOR'], '-C', initial], expect=mode != 'missing-sdk')
        if mode == 'missing-sdk':
            assert 'MIGraphX runtime requires explicit existing JYPPX_LIUDING_MIGRAPHX_ROOT' in (out / (mode + '-configure.log')).read_text(encoding='utf-8')
            continue
        run(mode + '-build', [a.cmake, '--build', child, '--config', 'Release', '--parallel', '2'])
        report = out / (mode + '.xml')
        run(mode + '-run', [a.ctest, '--test-dir', child, '-C', 'Release', '--output-on-failure',
            '--no-tests=error', '-j', '1', '--output-junit', report], timeout=630)
        nodes = ET.parse(report).getroot().findall('.//testcase')
        assert len(nodes) == 1 and nodes[0].get('name') == 'Runs'
        assert all(nodes[0].find(tag) is None for tag in ('skipped', 'failure', 'error'))
        if mode == 'runtime':
            audit(child)
    for missing in (() if resnet else ('serving', 'migraphx_runtime')):
        selected = dict(flags)
        selected.update(JYPPX_LIUDING_BUILD_MIGRAPHX_SERVING='ON',
            JYPPX_LIUDING_BUILD_SERVING='ON', JYPPX_LIUDING_BUILD_MIGRAPHX_RUNTIME='ON')
        selected['JYPPX_LIUDING_BUILD_' + missing.upper()] = 'OFF'
        label = missing + '-disabled'
        initial = out / (label + '.cmake')
        initial_cache(initial, selected)
        run(label, [a.cmake, '-S', source, '-B', out / label, '-G', cache['CMAKE_GENERATOR'],
            '-C', initial], expect=False)
        assert 'MIGRAPHX_SERVING=ON requires SERVING and MIGRAPHX_RUNTIME' in ' '.join((out / (label + '.log')).read_text(encoding='utf-8').split())
    receipt.update(status='passed', installed_gpu_bridge=1, inner_tests=1 if resnet else 2,
        serving_without_migraphx_sdk=0 if resnet else 1,
        missing_external_sdk_rejected=0 if resnet else 1,
        disabled_prerequisites_rejected=0 if resnet else 2)
except BaseException as error:
    receipt.update(status='failed', error=repr(error))
    raise
finally:
    save()
print('MIGraphX Serving install evidence=' + str(out), flush=True)
