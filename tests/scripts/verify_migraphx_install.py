#!/usr/bin/env python3
"""Offline install/independent consumer and explicit SDK closure contracts."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET

p=argparse.ArgumentParser()
for name in ('cmake','ctest','source-root','parent-build','fixtures'):p.add_argument('--'+name,required=True)
a=p.parse_args()
parent=Path(a.parent_build);source=Path(a.source_root)
out=parent/'evidence/migraphx-installed'/str(time.time_ns())
out.mkdir(parents=True,exist_ok=False)
receipt=dict(status='preparing',processes=[])
def save():(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
def run(label,args,expect=True,timeout=300):
    with (out/(label+'.log')).open('wb') as log:
        proc=subprocess.run(list(map(str,args)),stdout=log,stderr=log,timeout=timeout)
    receipt['processes'].append(dict(label=label,command=list(map(str,args)),returncode=proc.returncode))
    save()
    if expect and proc.returncode:raise RuntimeError(label+' failed; '+str(out/(label+'.log')))
    if not expect and not proc.returncode:raise RuntimeError(label+' unexpectedly succeeded')
try:
    cache=dict((match[1],match[2]) for line in (parent/'CMakeCache.txt').read_text(encoding='utf-8').splitlines() if (match:=re.match(r'^([^/#][^:]*):[^=]+=(.*)$',line)))
    install=out/'install'
    run('install',[a.cmake,'--install',parent,'--config','Release','--prefix',install])
    assert (install/'include/jyppx/liuding/deployment/migraphx_runtime/session.hpp').is_file()
    assert (install/'lib/cmake/jyppx/MIGraphXRuntimeDependency.cmake').is_file()
    assert not list((install/'bin').glob('migraphx*.dll')) and not list((install/'lib').glob('libmigraphx*.so*'))
    child_source=source/'tests/install/migraphx_consumer'
    flags={key:cache[key] for key in ('CMAKE_CXX_COMPILER','CMAKE_MAKE_PROGRAM','CMAKE_TOOLCHAIN_FILE','CMAKE_PREFIX_PATH','Torch_DIR','HIP_DIR','hip_DIR','CMAKE_HIP_COMPILER') if cache.get(key)}
    flags.update(CMAKE_BUILD_TYPE='Release',CMAKE_MSVC_RUNTIME_LIBRARY='MultiThreadedDLL',CMAKE_BUILD_PARALLEL_LEVEL='2',
        FETCHCONTENT_FULLY_DISCONNECTED='ON',FETCHCONTENT_UPDATES_DISCONNECTED='ON',jyppx_DIR=str(install/'lib/cmake/jyppx'),
        JYPPX_LIUDING_MIGRAPHX_ROOT=cache['JYPPX_LIUDING_MIGRAPHX_ROOT'],JYPPX_LIUDING_MIGRAPHX_HIP_ROOT=cache['JYPPX_LIUDING_MIGRAPHX_HIP_ROOT'],JYPPX_MIGRAPHX_CONSUMER_FIXTURES=a.fixtures)
    for mode in ('runtime','foundation','missing-sdk'):
        selected=dict(flags)
        if mode!='runtime':selected['JYPPX_LIUDING_MIGRAPHX_ROOT']='';selected['JYPPX_LIUDING_MIGRAPHX_HIP_ROOT']=''
        if mode=='foundation':selected['JYPPX_MIGRAPHX_CONSUMER_FOUNDATION']='ON'
        initial=out/(mode+'.cmake')
        assert all(']====]' not in str(value) for value in selected.values())
        initial.write_text(''.join(f'set({key} [====[{value}]====] CACHE STRING "" FORCE)\n' for key,value in selected.items()),encoding='utf-8')
        build=out/mode
        run(mode+'-configure',[a.cmake,'-S',child_source,'-B',build,'-G',cache['CMAKE_GENERATOR'],'-C',initial],expect=mode!='missing-sdk')
        if mode=='missing-sdk':
            assert 'MIGraphX runtime requires explicit existing JYPPX_LIUDING_MIGRAPHX_ROOT' in (out/(mode+'-configure.log')).read_text(encoding='utf-8')
            continue
        run(mode+'-build',[a.cmake,'--build',build,'--config','Release','--parallel','2'])
        report=out/(mode+'.xml')
        run(mode+'-run',[a.ctest,'--test-dir',build,'-C','Release','--output-on-failure','--no-tests=error','-j','1','--output-junit',report],timeout=630)
        cases=ET.parse(report).getroot().findall('.//testcase')
        assert len(cases)==1 and cases[0].get('name')=='Runs' and all(cases[0].find(t) is None for t in ('skipped','error','failure'))
        if mode=='runtime':
            assert hashlib.sha256((build/'actual.f32').read_bytes()).hexdigest()=='3b106db38a6dca7aafe95d1af546237ca38ac4d74a084a8f1b0cea5a361c709a'
            trace=(build/'dispatch.txt').read_text().splitlines()
            assert any(n.startswith(('gpu::hip_gemm','gpu::code_object')) for n in trace)
            assert any(n.startswith('hip::sync_stream') for n in trace)
            assert not any(n.startswith(('ref::','cpu::')) for n in trace)
    receipt.update(status='passed',installed_runtime_consumer=1,foundation_without_migraphx_sdk=1,missing_external_sdk_rejected=1)
except BaseException as error:receipt.update(status='failed',error=repr(error));raise
finally:save()
print('MIGraphX install evidence='+str(out),flush=True)
