"""Prepare read-only AREA scoring of the untrained, original public Edge weights.

Only add the common edge/ variable scope. Check original-author prediction
parity before saving; preserve every model and BN value, raw BGR, and flow units.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import time
import numpy as np
import tensorflow as tf
from data import read_sample, digest
from initialization import checkpoint_sha, restore_model
from model import graph, ROOT
from verify_adaptation import author_reference, session_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','data','manifests','reference-audit','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--code-commit',required=True)
    p.add_argument('--quantize',action='store_true',help='Enable the same fixed64 FC2 PTQ audit; no adaptation')
    a=p.parse_args()
    if a.out.exists() or not re.fullmatch('[0-9a-f]{40}',a.code_commit):
        raise ValueError('A new destination and verified code commit are required')
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('GPU author-parity check required')
    source=checkpoint_sha(a.checkpoint)
    reference=json.loads((a.reference_audit/'summary.json').read_text())
    protocol=json.loads((a.reference_audit/'protocol.json').read_text())
    if reference['status']!='passed' or digest(a.reference_audit/'sintel_full.json')!=protocol['full_manifest_sha256']:
        raise ValueError('Reference population not verified')
    if digest(a.reference_audit/'calibration.json')!=protocol['calibration_sha256']:
        raise ValueError('Reference calibration changed')
    cases,expected,_=author_reference(a.checkpoint,a.data,a.manifests,False)
    g=graph('edge',bn_mode='frozen',edge_public=True)
    a.out.mkdir(parents=True)
    (a.out/'source').mkdir()
    prefix=a.out/'source/model'
    differences=[]
    with tf.compat.v1.Session(config=session_config()) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        restored=restore_model(sess,g,a.checkpoint,public_edge=True)
        before=sess.run(g['weights'])
        for (row,sintel),original in zip(cases,expected):
            x,_,_=read_sample(a.data,row,sintel,images='raw')
            actual=sess.run(g['preds']+[g['prediction']],{g['x']:x[None],g['training']:False})
            for left,right in zip(actual,original):
                np.testing.assert_allclose(left,right,rtol=1e-5,atol=1e-4)
                differences.append(float(np.max(np.abs(left-right))))
        if not all(np.array_equal(v,w) for v,w in zip(before,sess.run(g['weights']))):
            raise AssertionError('Inference changed weights')
        g['weight_saver'].save(sess,str(prefix),write_meta_graph=False)
        reader=tf.train.load_checkpoint(str(prefix))
        if not all(np.array_equal(v,reader.get_tensor(w.op.name)) for v,w in zip(before,g['weights'])):
            raise AssertionError('Scope-only checkpoint save changed tensor values')
    if source!=checkpoint_sha(a.checkpoint):
        raise AssertionError('Original checkpoint changed')
    case=dict(id='public-edge-208',model='edge',checkpoint=str(prefix),hw=[160,208],
              geometry='public-original',edge_public=True,quantize=a.quantize,
              expected_monitor=None,expected_full=None)
    for name in ('sintel_full.json','sintel_monitor.json','calibration.json'):
        shutil.copy2(a.reference_audit/name,a.out/name)
    protocol.update(created=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),cases=[case],
        phase='public-original',native_input='BGR float32 0..255, AREA whole-frame resize',
        code_commit=a.code_commit,script_sha256=digest(Path(__file__).with_name('audit_deployment.py')),
        selection='Original author public checkpoint; no training or selection',
        source_checkpoint=str(a.checkpoint),source_checkpoint_sha256=source,
        preparation_script_sha256=digest(__file__))
    proof=dict(passed=True,no_training=True,original_weights_unchanged=True,
        original_checkpoint_sha256=source,mapped_checkpoint_sha256=checkpoint_sha(prefix),
        model_bn_values_exact=True,original_author_prediction_parity=True,
        parity_pairs=[dict(paths=row,sintel=sintel) for row,sintel in cases],
        max_prediction_abs_difference=max(differences),all_three_heads_and_accumulated_flow=True,
        restore=restored,source_code_sha256={str(f.relative_to(ROOT)):digest(f) for f in (
            ROOT/'EdgeFlowNet/code/network/BaseLayers.py',ROOT/'EdgeFlowNet/code/network/MultiScaleResNet.py',
            ROOT/'EdgeFlowNet/code/misc/utils.py')})
    for name,value in (('cases.json',[case]),('protocol.json',protocol),('author-parity.json',proof)):
        (a.out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(prepared=str(a.out),passed=True,parity_max_abs=max(differences))),flush=True)


if __name__=='__main__':main()
