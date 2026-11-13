# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
from pathlib import Path
import contextlib
import hashlib
import importlib.metadata
import io
import json
import sys

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

root, output = map(Path, sys.argv[1:3])
definition = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
rows = [r for r in definition['rows'] if r['split']=='test']
raw = json.loads((output/'predictions.json').read_text())
result = json.loads((output/'results.json').read_text())
assert raw['ids']==[r['id'] for r in rows]
assert len(set(raw['ids']))==25
images = np.fromfile(output/'images.f32', dtype='<f4').reshape(25,3,128,128)
masks = np.fromfile(output/'instances.i64', dtype='<i8').reshape(25,128,128)
logits = np.fromfile(output/'logits.f32', dtype='<f4').reshape(25,2,128,128)
assert np.isfinite(logits).all() and np.isfinite(images).all()
assert -2e-6<=images.min()<=images.max()<=1+2e-6  # FP32 filter-weight rounding.
predictions = logits.argmax(axis=1)
shifted = logits.astype(np.float64) - logits.max(axis=1,keepdims=True)
probabilities = np.exp(shifted)
probabilities /= probabilities.sum(axis=1,keepdims=True)
image_error = 0.
decoded = {}
for row in definition['rows']:
    for key in ['image','mask']:
        assert hashlib.sha256((root/row[key]).read_bytes()).hexdigest()==row[key+'_sha256']
    photo = np.array(Image.open(root/row['image']).convert('RGB'))
    signature = hashlib.sha256(photo.tobytes()).hexdigest()
    assert signature not in decoded, ('exact decoded duplicate',row['id'],decoded.get(signature))
    decoded[signature] = row['id']
for i,row in enumerate(rows):
    expected = np.array(Image.open(root/row['mask']).resize((128,128),Image.Resampling.NEAREST),dtype=np.int64)
    assert np.array_equal(expected,masks[i]),row['id']
    original = np.array(Image.open(root/row['image']).convert('RGB'),dtype=np.float32)/255.
    expected_photo = np.stack([np.array(Image.fromarray(original[:,:,c]).resize((128,128),Image.Resampling.BILINEAR)) for c in range(3)])
    error = float(np.max(np.abs(expected_photo-images[i])))
    image_error=max(image_error,error)
    assert error<2e-6,(row['id'],error)

def semantic(selected, score):
    gt=masks[selected]>0
    pred=predictions[selected]
    counts=[]
    values=[]
    for c in range(2):
        intersection=int(((gt==c)&(pred==c)).sum())
        target=int((gt==c).sum()); predicted=int((pred==c).sum())
        counts.append([intersection,target,predicted])
        values.append(intersection/(target+predicted-intersection) if score=='iou' else 2*intersection/(target+predicted))
    return {'counts':counts,'per_class':values,'overall':float(np.mean(values))}

def close(actual,expected,location):
    if isinstance(expected,dict):
        for k,v in expected.items():close(actual[k],v,location+'.'+k)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),location
        for i,(a,b) in enumerate(zip(actual,expected)):close(a,b,location+'.'+str(i))
    elif isinstance(expected,int):assert actual==expected,(location,actual,expected)
    else:assert abs(actual-expected)<2e-12,(location,actual,expected)

reference={}
sources=np.array(raw['sources'])
for score in ['iou','dice']:
    expected=semantic(np.ones(25,dtype=bool),score)
    expected['by_source']=[semantic(sources==i,score) for i in range(2)]
    close(result[score],expected,score)
    reference[score]=expected

def connected(index):
    foreground=predictions[index]==1
    seen=np.zeros((128,128),dtype=bool)
    components=[]
    for y,x in zip(*np.where(foreground)):
        if seen[y,x]:continue
        seen[y,x]=True; pending=[(int(y),int(x))]; pixels=[]
        while pending:
            yy,xx=pending.pop(); pixels.append((yy,xx))
            for y1,x1 in [(yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)]:
                if 0<=y1<128 and 0<=x1<128 and foreground[y1,x1] and not seen[y1,x1]:
                    seen[y1,x1]=True;pending.append((y1,x1))
        if len(pixels)<16:continue
        yy,xx=np.array(pixels).T
        components.append({'box':[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1],
                           'score':float(probabilities[index,1,yy,xx].mean())})
    return components

annotations=[]; detected=[]
for i in range(25):
    expected=connected(i)
    actual=raw['proposals'][i]
    assert len(actual)==len(expected),(i,len(actual),len(expected))
    for a,b in zip(actual,expected):
        assert a['box']==b['box'],(i,a,b)
        assert abs(a['score']-b['score'])<2e-7,(i,a,b)
        x1,y1,x2,y2=a['box']
        detected.append({'image_id':i,'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'score':a['score']})
    for instance in sorted(set(masks[i].ravel())-{0}):
        yy,xx=np.where(masks[i]==instance)
        x1,y1,x2,y2=int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
        annotations.append({'id':len(annotations)+1,'image_id':i,'category_id':1,
                            'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0})

def ap(selected):
    with contextlib.redirect_stdout(io.StringIO()):
        truth=COCO()
        truth.dataset={'info':{},'images':[{'id':int(i),'height':128,'width':128} for i in selected],
                       'categories':[{'id':1,'name':'person'}],
                       'annotations':[a for a in annotations if a['image_id'] in selected]}
        truth.createIndex()
        candidates=[a for a in detected if a['image_id'] in selected]
        # COCO.loadRes needs a first row; every frozen selection must predict.
        assert candidates
        dt=truth.loadRes(candidates)
        evaluation=COCOeval(truth,dt,'bbox')
        evaluation.params.imgIds=[int(i) for i in selected]
        evaluation.params.catIds=[1]
        evaluation.params.iouThrs=np.array([.5,.75])
        evaluation.params.areaRng=[[0,1e10]]
        evaluation.params.areaRngLbl=['all']
        evaluation.params.maxDets=[1,10,100]
        evaluation.evaluate();evaluation.accumulate()
    precision=evaluation.eval['precision'][:,:,0,0,-1]
    scores=precision.mean(axis=1)
    # Per-image maxDet is applied by COCO before retained-prediction counts.
    retained=sum(min(100,sum(a['image_id']==i for a in candidates)) for i in selected)
    return {'ap50':float(scores[0]),'ap75':float(scores[1]),'map':float(scores.mean()),
            'gt':len(truth.dataset['annotations']),'predictions':retained}

expected=ap(list(range(25)))
expected['by_source']=[ap(np.flatnonzero(sources==i).tolist()) for i in range(2)]
close(result['detection'],expected,'detection')
reference['detection']=expected
assert reference['iou']['overall']>=.60
assert reference['iou']['per_class'][1]>=.35
assert reference['detection']['ap50']>=.10
reference['validation']={'images':25,'all_data_sha256_verified':170,
    'exact_decoded_image_duplicates':0,'mask_nearest_exact_equal':True,
    'photo_max_absolute_error':image_error,'metric_reference':'pycocotools 2.0.11 bbox and independent NumPy pixel counts',
    'pillow':importlib.metadata.version('Pillow'),'numpy':np.__version__,
    'pycocotools':importlib.metadata.version('pycocotools'),
    'qualifications':'Small held-out real dataset; hash split not official or scene-disjoint. Segmentation-derived proposals, no independent detector or full detection recipe.'}
target=output/'independent-reference.json'
assert not target.exists()
target.write_text(json.dumps(reference,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'reference':'passed','output':str(output),'foreground_iou':reference['iou']['per_class'][1],
                  'mean_iou':reference['iou']['overall'],'ap50':reference['detection']['ap50'],
                  'photo_max_error':image_error}),flush=True)
