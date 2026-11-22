#!/usr/bin/env python3
"""Offline installed native GPU bridge and serving-without-SDK contracts."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import time
import xml.etree.ElementTree as ET

p=argparse.ArgumentParser()
for name in ('cmake','ctest','source-root','parent-build','fixtures'):p.add_argument('--'+name,required=True)
a=p.parse_args();parent=Path(a.parent_build);source=Path(a.source_root)
out=parent/'evidence/migraphx-serving-installed'/str(time.time_ns());out.mkdir(parents=True,exist_ok=False)
receipt=dict(status='preparing',processes=[])
def save():(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
def run(label,args,expect=True,timeout=300):
    with (out/(label+'.log')).open('wb') as log:
        result=subprocess.run(list(map(str,args)),stdout=log,stderr=log,timeout=timeout)
    receipt['processes'].append(dict(label=label,command=list(map(str,args)),returncode=result.returncode));save()
    if expect and result.returncode:raise RuntimeError(label+' failed; '+str(out/(label+'.log')))
    if not expect and not result.returncode:raise RuntimeError(label+' unexpectedly succeeded')
try:
    cache=dict((m[1],m[2]) for line in (parent/'CMakeCache.txt').read_text(encoding='utf-8').splitlines() if (m:=re.match(r'^([^/#][^:]*):[^=]+=(.*)$',line)))
    install=out/'install';run('install',[a.cmake,'--install',parent,'--config','Release','--prefix',install])
    assert (install/'include/jyppx/liuding/deployment/migraphx_serving/backend.hpp').is_file()
    assert not list((install/'bin').glob('migraphx*.dll')) and not list((install/'lib').glob('libmigraphx*.so*'))
    flags={key:cache[key] for key in ('CMAKE_CXX_COMPILER','CMAKE_MAKE_PROGRAM','CMAKE_TOOLCHAIN_FILE','CMAKE_PREFIX_PATH','Torch_DIR','HIP_DIR','hip_DIR','CMAKE_HIP_COMPILER') if cache.get(key)}
    flags.update(CMAKE_BUILD_TYPE='Release',CMAKE_MSVC_RUNTIME_LIBRARY='MultiThreadedDLL',FETCHCONTENT_FULLY_DISCONNECTED='ON',FETCHCONTENT_UPDATES_DISCONNECTED='ON',
        jyppx_DIR=str(install/'lib/cmake/jyppx'),JYPPX_LIUDING_MIGRAPHX_ROOT=cache['JYPPX_LIUDING_MIGRAPHX_ROOT'],JYPPX_LIUDING_MIGRAPHX_HIP_ROOT=cache['JYPPX_LIUDING_MIGRAPHX_HIP_ROOT'],JYPPX_MIGRAPHX_CONSUMER_FIXTURES=a.fixtures)
    for mode in ('runtime','serving','missing-sdk'):
        selected=dict(flags)
        if mode!='runtime':selected['JYPPX_LIUDING_MIGRAPHX_ROOT']='';selected['JYPPX_LIUDING_MIGRAPHX_HIP_ROOT']=''
        if mode=='serving':selected['JYPPX_MIGRAPHX_SERVING_ONLY']='ON'
        initial=out/(mode+'.cmake');assert all(']====]' not in str(v) for v in selected.values())
        initial.write_text(''.join(f'set({k} [====[{v}]====] CACHE STRING "" FORCE)\n' for k,v in selected.items()),encoding='utf-8')
        child=out/mode
        run(mode+'-configure',[a.cmake,'-S',source/'tests/install/migraphx_serving_consumer','-B',child,'-G',cache['CMAKE_GENERATOR'],'-C',initial],expect=mode!='missing-sdk')
        if mode=='missing-sdk':
            assert 'MIGraphX runtime requires explicit existing JYPPX_LIUDING_MIGRAPHX_ROOT' in (out/(mode+'-configure.log')).read_text(encoding='utf-8');continue
        run(mode+'-build',[a.cmake,'--build',child,'--config','Release','--parallel','2'])
        report=out/(mode+'.xml');run(mode+'-run',[a.ctest,'--test-dir',child,'-C','Release','--output-on-failure','--no-tests=error','-j','1','--output-junit',report],timeout=630)
        nodes=ET.parse(report).getroot().findall('.//testcase');assert len(nodes)==1 and nodes[0].get('name')=='Runs' and all(nodes[0].find(t) is None for t in ('skipped','failure','error'))
        if mode=='runtime':
            assert hashlib.sha256((child/'actual.f32').read_bytes()).hexdigest()=='3b106db38a6dca7aafe95d1af546237ca38ac4d74a084a8f1b0cea5a361c709a'
            lock=json.loads((child/'environment-lock.json').read_bytes());assert lock['program']['input_shape']==[2,8] and lock['program']['output_shape']==[2,3]
    for missing in ('serving','migraphx_runtime'):
        selected=dict(flags)
        selected.update(JYPPX_LIUDING_BUILD_MIGRAPHX_SERVING='ON',JYPPX_LIUDING_BUILD_SERVING='ON',JYPPX_LIUDING_BUILD_MIGRAPHX_RUNTIME='ON')
        selected['JYPPX_LIUDING_BUILD_'+missing.upper()]='OFF'
        initial=out/(missing+'-disabled.cmake')
        initial.write_text(''.join(f'set({k} [====[{v}]====] CACHE STRING "" FORCE)\n' for k,v in selected.items()),encoding='utf-8')
        label=missing+'-disabled'
        run(label,[a.cmake,'-S',source,'-B',out/label,'-G',cache['CMAKE_GENERATOR'],'-C',initial],expect=False)
        assert 'MIGRAPHX_SERVING=ON requires SERVING and MIGRAPHX_RUNTIME' in ' '.join((out/(label+'.log')).read_text(encoding='utf-8').split())
    receipt.update(status='passed',installed_gpu_bridge=1,serving_without_migraphx_sdk=1,missing_external_sdk_rejected=1,disabled_prerequisites_rejected=2)
except BaseException as error:receipt.update(status='failed',error=repr(error));raise
finally:save()
print('MIGraphX Serving install evidence='+str(out),flush=True)
