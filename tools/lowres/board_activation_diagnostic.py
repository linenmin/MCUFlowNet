"""Expose existing INT8 intermediate tensors, then check complete board dumps.

Keeps every original constant and quantization table unchanged. A prefix with
extra outputs can change Vela scheduling; always regress the complete model.
Activation differences are quantized feature codes, never optical-flow EPE.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import tempfile
from types import SimpleNamespace
import zlib
import flatbuffers

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema
from board_slice_diagnostic import tensor_io
from check_board_reference import decode_board_outputs

SCOPE = 'Internal activation prefix diagnosis only; no benchmark EPE, calibration or weights'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def interpreter(data, preserve=False):
    it = tf.lite.Interpreter(model_content=bytes(data), num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        experimental_preserve_all_tensors=preserve)
    it.allocate_tensors()
    return it


def prepare(a):
    if a.out.exists():
        raise FileExistsError(a.out)
    parent = json.loads((a.export/'export.json').read_text())
    raw = (a.export/'model_int8.tflite').read_bytes()
    old = json.loads((a.fixtures/'fixtures.json').read_text())
    old_blob = (a.fixtures/'fixtures.bin').read_bytes()
    assert parent['status'] == old['status'] == 'passed'
    assert sha(raw) == parent['exports']['int8']['sha256'] == old['source_sha256']
    assert sha(old_blob) == old['blob_sha256']
    assert f'{zlib.crc32(old_blob):08x}' == old['blob_crc32']
    assert len(a.tensors) == len(a.labels) == 2 and len(set(a.tensors)) == 2
    assert all(re.fullmatch('[a-z0-9_-]+', name) for name in a.labels)
    original = schema.Model.GetRootAsModel(raw, 0); og = original.Subgraphs(0)
    assert og.InputsLength() == og.OutputsLength() == 1
    assert 0 <= a.through_op < og.OperatorsLength()
    produced = {int(t) for i in range(a.through_op+1)
                for t in og.Operators(i).OutputsAsNumpy()}
    assert set(a.tensors) <= produced
    # Append a two-element output vector, pointing at existing tensor tables.
    # Only output-vector pointer and operator-count metadata are changed.
    modified = bytearray(raw)
    g = schema.Model.GetRootAsModel(modified, 0).Subgraphs(0)
    output_field = g._tab.Pos + g._tab.Offset(8)
    operator_vector = g._tab.Vector(g._tab.Offset(10))
    modified.extend(b'\0' * (-len(modified) % 4))
    output_vector = len(modified)
    modified.extend(struct.pack('<Iii', 2, *a.tensors))
    struct.pack_into('<I', modified, output_field, output_vector-output_field)
    struct.pack_into('<I', modified, operator_vector-4, a.through_op+1)
    model = schema.Model.GetRootAsModel(modified, 0); pg = model.Subgraphs(0)
    assert pg.OperatorsLength() == a.through_op+1
    assert pg.OutputsAsNumpy().tolist() == a.tensors
    assert model.BuffersLength() == original.BuffersLength()
    for i in range(model.BuffersLength()):
        before, after = original.Buffers(i), model.Buffers(i)
        assert before.DataLength() == after.DataLength()
        if before.DataLength():
            assert np.array_equal(before.DataAsNumpy(), after.DataAsNumpy())
    input_io = tensor_io(pg, pg.Inputs(0))
    assert input_io == tensor_io(og, og.Inputs(0))
    outputs = [dict(tensor_io(pg, index), label=label, original_tensor_index=index,
                    original_tensor_name=og.Tensors(index).Name().decode())
               for index, label in zip(a.tensors, a.labels)]
    assert all(len(t['shape']) == 4 and t['shape'][0] == 1 for t in outputs)
    full, prefix = interpreter(raw, True), interpreter(modified)
    pi = prefix.get_input_details()[0]; po = prefix.get_output_details()
    assert [x['index'] for x in po] == a.tensors
    sizes = [int(np.prod(t['shape'])) for t in outputs]
    meta = dict(old, source_sha256=sha(modified), parent_source_sha256=sha(raw),
        diagnostic_only=True, scope=SCOPE, output_units='quantized internal activation codes',
        reference='CPU TFLite INT8 BUILTIN_REF; original full-model intermediate tensors',
        reference_kernel='BUILTIN_REF', outputs=outputs, output_bytes=sum(sizes),
        performance=dict(enabled=False, scope='Activation diagnosis only; no timing'), pairs=[])
    # Old flow-output quantization no longer describes these feature tensors.
    for key in ('output_scale', 'output_zero_point', 'output_multiplier'):
        meta.pop(key, None)
    blob = bytearray(); checks = []
    for pair in old['pairs']:
        start = pair['offset']; payload = old_blob[start:start+old['input_bytes']]
        assert sha(payload) == pair['input_sha256']
        value = np.frombuffer(payload, np.int8).reshape(input_io['shape'])
        full.set_tensor(full.get_input_details()[0]['index'], value); full.invoke()
        prefix.set_tensor(pi['index'], value); prefix.invoke()
        values = []
        for index, info in zip(a.tensors, outputs):
            actual, expected = prefix.get_tensor(index), full.get_tensor(index)
            assert actual.shape == tuple(info['shape'])
            assert np.array_equal(actual, expected), 'Prefix changed an original activation'
            values.append(actual.tobytes())
            checks.append(dict(pair_index=pair['index'], label=info['label'], byte_exact=True))
        result = b''.join(values)
        meta['pairs'].append(dict(pair, offset=len(blob), output_sha256=sha(result),
            output_crc32=f'{zlib.crc32(result):08x}',
            output_tensor_sha256=[sha(x) for x in values]))
        blob.extend(payload); blob.extend(result)
    meta.update(blob_sha256=sha(blob), blob_crc32=f'{zlib.crc32(blob):08x}', blob_bytes=len(blob))
    report = dict(status='passed', diagnostic_only=True, model=parent['model'],
        edge_public=parent.get('edge_public',False), input_hw=parent['input_hw'],
        scope=SCOPE, output_convention='Internal quantized activations; no optical-flow displacement units',
        reference_kernel='BUILTIN_REF', parent_source_sha256=sha(raw),
        constant_buffers_byte_exact=True, cpu_checks=checks,
        operators_before=og.OperatorsLength(), operators_after=pg.OperatorsLength(),
        modified_original_byte_positions=[i for i,(x,y) in enumerate(zip(raw,modified)) if x!=y],
        appended_metadata_bytes=len(modified)-len(raw),
        exports=dict(int8=dict(sha256=sha(modified), input=input_io, outputs=outputs)))
    a.out.mkdir(parents=True); (a.out/'fixtures').mkdir()
    (a.out/'model_int8.tflite').write_bytes(modified)
    (a.out/'export.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'fixtures/fixtures.bin').write_bytes(blob)
    (a.out/'fixtures/fixtures.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(report),flush=True)


def check(a):
    if a.out.exists():
        raise FileExistsError(a.out)
    report = json.loads((a.export/'export.json').read_text())
    meta = json.loads((a.export/'fixtures/fixtures.json').read_text())
    source = (a.export/'model_int8.tflite').read_bytes(); compiled = a.compiled.read_bytes()
    blob = (a.export/'fixtures/fixtures.bin').read_bytes()
    assert report['status'] == 'passed' and report['scope'] == SCOPE
    assert meta['reference_kernel'] == 'BUILTIN_REF'
    assert sha(source) == report['exports']['int8']['sha256'] == meta['source_sha256']
    assert sha(blob) == meta['blob_sha256'] and f'{zlib.crc32(blob):08x}' == meta['blob_crc32']
    sg = schema.Model.GetRootAsModel(source,0).Subgraphs(0)
    cg = schema.Model.GetRootAsModel(compiled,0).Subgraphs(0)
    assert cg.OutputsLength() == sg.OutputsLength() == len(meta['outputs']) == 2
    assert tensor_io(cg,cg.Inputs(0)) == tensor_io(sg,sg.Inputs(0))
    for i in range(2):
        assert tensor_io(cg,cg.Outputs(i)) == tensor_io(sg,sg.Outputs(i))
    text = a.board_log.read_bytes().decode('ascii',errors='replace')
    text = text.rsplit('Optical camera oflow app start',1)[-1]
    crc = f'{zlib.crc32(compiled):08x}'
    assert re.search(r'FLOW_MODEL [^\r\n]+ crc='+crc+r' PASS',text)
    for i,info in enumerate(meta['outputs']):
        marker = f'FLOW_PROBE_IO tensor={i} label={info["label"]} shape='+\
                 ','.join(map(str,info['shape']))+f' bytes={int(np.prod(info["shape"]))}'
        assert marker in text, 'Board output boundary metadata missing'
    inputs = dict(re.findall(r'FLOW_CACHE index=(\d+) input_crc=([0-9a-f]{8})',text))
    board = decode_board_outputs(a.board_log,meta['output_bytes'])
    assert set(board) == set(range(len(meta['pairs'])))
    results = []
    for n,pair in enumerate(meta['pairs']):
        start = pair['offset']; end = start+meta['input_bytes']
        assert inputs[str(n)] == f'{zlib.crc32(blob[start:end]):08x}'
        expected = blob[end:end+meta['output_bytes']]; offset = 0
        for info in meta['outputs']:
            size = int(np.prod(info['shape']))
            actual = np.frombuffer(board[n][offset:offset+size],np.int8).reshape(info['shape'])
            golden = np.frombuffer(expected[offset:offset+size],np.int8).reshape(info['shape'])
            delta = np.abs(actual.astype(np.int16)-golden.astype(np.int16))
            results.append(dict(fixture=n,pair_index=pair['index'],label=info['label'],
                shape=info['shape'],max_abs_q=int(delta.max()),mean_abs_q=float(delta.mean()),
                changed=int(np.count_nonzero(delta)),components=int(delta.size),
                actual_range=[int(actual.min()),int(actual.max())],
                reference_range=[int(golden.min()),int(golden.max())],
                byte_exact=actual.tobytes()==golden.tobytes(),
                strict_code_gate_passed=int(delta.max())<=2 and float(delta.mean())<=0.05))
            offset += size
        assert offset == meta['output_bytes']
    result = dict(scope=SCOPE,units='quantized internal activation codes, not flow pixels or EPE',
        source_sha256=sha(source),compiled_sha256=sha(compiled),model_crc32_verified=crc,
        log_sha256=sha(a.board_log.read_bytes()),input_output_crc_passed=True,
        comparisons=results,all_byte_exact=all(x['byte_exact'] for x in results),
        limits='Extra outputs and a shorter graph change Vela scheduling; regress the complete model.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def self_check(a):
    """Exercise real artifact boundaries using synthetic, explicitly non-board logs."""
    if a.out.exists():
        raise FileExistsError(a.out)
    meta = json.loads((a.export/'fixtures/fixtures.json').read_text())
    blob = (a.export/'fixtures/fixtures.bin').read_bytes()
    model_crc = f'{zlib.crc32(a.compiled.read_bytes()):08x}'
    def log_bytes(perturb=False):
        lines = ['Optical camera oflow app start', f'FLOW_MODEL synthetic crc={model_crc} PASS']
        for k,t in enumerate(meta['outputs']):
            lines.append(f'FLOW_PROBE_IO tensor={k} label={t["label"]} shape='+
                ','.join(map(str,t['shape']))+f' bytes={int(np.prod(t["shape"]))}')
        for n,pair in enumerate(meta['pairs']):
            start=pair['offset']; end=start+meta['input_bytes']
            value=bytearray(blob[end:end+meta['output_bytes']])
            if n==0 and perturb:
                value[int(np.prod(meta['outputs'][0]['shape']))] ^= 1
            lines += [f'FLOW_CACHE index={n} input_crc={zlib.crc32(blob[start:end]):08x}',
                f'FLOW_Q_BEGIN index={n} bytes={len(value)} crc={zlib.crc32(value):08x}']
            for offset in range(0,len(value),64):
                lines.append(f'FLOW_Q index={n} offset={offset} data={value[offset:offset+64].hex()}')
            lines.append(f'FLOW_Q_END index={n}')
        return ('\n'.join(lines)+'\n').encode()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='activation-host-test-',dir=a.out.parent) as tmp:
        root=Path(tmp)
        for name,perturb in [('exact',False),('second_boundary',True)]:
            log=root/f'{name}.bin'; log.write_bytes(log_bytes(perturb))
            out=root/f'{name}.json'
            with contextlib.redirect_stdout(io.StringIO()):
                check(SimpleNamespace(export=a.export,compiled=a.compiled,board_log=log,out=out))
            result=json.loads(out.read_text())
            if not perturb:
                assert result['all_byte_exact'] and len(result['comparisons'])==6
            else:
                changed=[x for x in result['comparisons'] if not x['byte_exact']]
                assert len(changed)==1 and changed[0]['fixture']==0
                assert changed[0]['label']==meta['outputs'][1]['label']
                assert changed[0]['changed']==changed[0]['max_abs_q']==1
        bad=root/'missing.bin'
        bad.write_bytes(log_bytes().replace(next(x for x in log_bytes().splitlines(keepends=True)
            if x.startswith(b'FLOW_Q index=0 offset=64 ')),b'',1))
        try:
            decode_board_outputs(bad,meta['output_bytes'])
        except ValueError:
            pass
        else:
            raise AssertionError('Missing output chunk was accepted')
    result=dict(synthetic_only=True,board_measured=False,status='passed',
        checks=['all six known feature outputs exact','one-byte change belongs only to second boundary',
                'missing serial chunk rejected','temporary fake logs removed'],
        compiled_sha256=sha(a.compiled.read_bytes()))
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def rewrite_sigmoid_reshape(a):
    """Move an unchanged pointwise sigmoid after its following reshape.

    The temporary tensor now stores reshaped logits, so it inherits the
    original logits' quantization. All other tensor quantization and every
    constant buffer stay unchanged. Keep the original optimized-CPU goldens.
    """
    if a.out.exists():
        raise FileExistsError(a.out)
    parent = json.loads((a.export/'export.json').read_text())
    raw = (a.export/'model_int8.tflite').read_bytes()
    meta = json.loads((a.fixtures/'fixtures.json').read_text())
    blob = (a.fixtures/'fixtures.bin').read_bytes()
    assert parent['status'] == meta['status'] == 'passed'
    assert sha(raw) == parent['exports']['int8']['sha256'] == meta['source_sha256']
    assert sha(blob) == meta['blob_sha256'] and f'{zlib.crc32(blob):08x}' == meta['blob_crc32']
    original = schema.Model.GetRootAsModel(raw,0); og = original.Subgraphs(0)
    assert a.reshape_op == a.sigmoid_op+1 and og.InputsLength() == og.OutputsLength() == 1
    obj = schema.ModelT.InitFromObj(original); g = obj.subgraphs[0]
    sig, resh = g.operators[a.sigmoid_op], g.operators[a.reshape_op]
    assert original.OperatorCodes(sig.opcodeIndex).BuiltinCode() == schema.BuiltinOperator.LOGISTIC
    assert original.OperatorCodes(resh.opcodeIndex).BuiltinCode() == schema.BuiltinOperator.RESHAPE
    assert len(sig.inputs) == len(sig.outputs) == len(resh.outputs) == 1
    logits, temporary, output = map(int,(sig.inputs[0],sig.outputs[0],resh.outputs[0]))
    assert len({logits,temporary,output}) == 3 and int(resh.inputs[0]) == temporary
    assert temporary not in list(g.inputs)+list(g.outputs)
    assert all(temporary not in op.inputs for i,op in enumerate(g.operators) if i != a.reshape_op)
    before, after = tensor_io(og,temporary), tensor_io(og,output)
    assert np.prod(before['shape']) == np.prod(after['shape'])
    assert {k:v for k,v in before.items() if k!='shape'} == {k:v for k,v in after.items() if k!='shape'}
    new_reshape, new_sigmoid = copy.deepcopy(resh), copy.deepcopy(sig)
    new_reshape.inputs[0] = logits; new_reshape.outputs[0] = temporary
    new_sigmoid.inputs[0] = temporary; new_sigmoid.outputs[0] = output
    g.operators[a.sigmoid_op],g.operators[a.reshape_op] = new_reshape,new_sigmoid
    g.tensors[temporary].shape = copy.deepcopy(g.tensors[output].shape)
    g.tensors[temporary].quantization = copy.deepcopy(g.tensors[logits].quantization)
    builder = flatbuffers.Builder(len(raw)); offset = obj.Pack(builder)
    builder.Finish(offset,file_identifier=b'TFL3'); rewritten = bytes(builder.Output())
    model = schema.Model.GetRootAsModel(rewritten,0); ng = model.Subgraphs(0)
    assert ng.OperatorsLength() == og.OperatorsLength() and ng.TensorsLength() == og.TensorsLength()
    assert model.BuffersLength() == original.BuffersLength()
    for i in range(model.BuffersLength()):
        x,y = original.Buffers(i),model.Buffers(i)
        assert x.DataLength() == y.DataLength()
        if x.DataLength():
            assert np.array_equal(x.DataAsNumpy(),y.DataAsNumpy())
    for i in range(ng.TensorsLength()):
        x,y = og.Tensors(i),ng.Tensors(i)
        assert x.Type() == y.Type() and x.Buffer() == y.Buffer()
        if i != temporary:
            assert np.array_equal(x.ShapeAsNumpy(),y.ShapeAsNumpy())
            for name in ('ScaleAsNumpy','ZeroPointAsNumpy','MinAsNumpy','MaxAsNumpy'):
                assert np.array_equal(getattr(x.Quantization(),name)(),getattr(y.Quantization(),name)())
        else:
            assert tensor_io(ng,i) == dict(tensor_io(og,logits),shape=after['shape'])
    for i in range(ng.OperatorsLength()):
        if i not in (a.sigmoid_op,a.reshape_op):
            x,y = og.Operators(i),ng.Operators(i)
            assert x.OpcodeIndex() == y.OpcodeIndex()
            assert np.array_equal(x.InputsAsNumpy(),y.InputsAsNumpy())
            assert np.array_equal(x.OutputsAsNumpy(),y.OutputsAsNumpy())
    ii,oo = tensor_io(og,og.Inputs(0)),tensor_io(og,og.Outputs(0))
    assert ii == tensor_io(ng,ng.Inputs(0)) and oo == tensor_io(ng,ng.Outputs(0))
    inputs = []
    for pair in meta['pairs']:
        value = blob[pair['offset']:pair['offset']+meta['input_bytes']]
        assert sha(value) == pair['input_sha256']
        inputs.append((f'FC2 pair {pair["index"]}',np.frombuffer(value,np.int8).reshape(ii['shape'])))
    # Edge cases complement real fixtures; these are not benchmark samples.
    inputs += [(f'synthetic constant {v}',np.full(ii['shape'],v,np.int8)) for v in (-128,-1,127)]
    inputs.append(('synthetic random seed42',np.random.default_rng(42).integers(-128,128,ii['shape'],dtype=np.int8)))
    checks = []
    for name,resolver in [('BUILTIN_REF',tf.lite.experimental.OpResolverType.BUILTIN_REF),
                         ('BUILTIN_WITHOUT_DEFAULT_DELEGATES',tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)]:
        its = [tf.lite.Interpreter(model_content=data,num_threads=1,experimental_op_resolver_type=resolver)
               for data in (raw,rewritten)]
        for it in its: it.allocate_tensors()
        for label,value in inputs:
            values = []
            for it in its:
                it.set_tensor(it.get_input_details()[0]['index'],value);it.invoke()
                values.append(it.get_tensor(it.get_output_details()[0]['index']))
            assert np.array_equal(*values), f'Full CPU output changed: {name}, {label}'
            checks.append(dict(resolver=name,input=label,byte_exact=True))
    report = dict(status='passed',diagnostic_only=True,model=parent['model'],
        edge_public=parent.get('edge_public',False),input_hw=parent['input_hw'],
        scope='Equivalent sigmoid/reshape order diagnostic; no training, calibration or benchmark replacement',
        output_convention=parent['output_convention'],parent_source_sha256=sha(raw),
        constant_buffers_byte_exact=True,unchanged_quantization_except_reshaped_logits_tensor=temporary,
        operators_before=og.OperatorsLength(),operators_after=ng.OperatorsLength(),
        reordered_operator_positions=[a.sigmoid_op,a.reshape_op],cpu_checks=checks,
        original_fixture_blob_unchanged=True,acceptance_limits_unchanged=True,
        exports=dict(int8=dict(sha256=sha(rewritten),input=ii,output=oo)))
    meta.update(source_sha256=sha(rewritten),parent_source_sha256=sha(raw),diagnostic_only=True,
                original_fixture_blob_unchanged=True,performance=dict(meta['performance'],enabled=False))
    a.out.mkdir(parents=True);(a.out/'fixtures').mkdir()
    (a.out/'model_int8.tflite').write_bytes(rewritten)
    (a.out/'export.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'fixtures/fixtures.bin').write_bytes(blob)
    (a.out/'fixtures/fixtures.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(report),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action',required=True)
    q = sub.add_parser('prepare')
    for name in ('export','fixtures','out'):
        q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--through-op',type=int,required=True)
    q.add_argument('--tensors',type=int,nargs=2,required=True)
    q.add_argument('--labels',nargs=2,required=True)
    q = sub.add_parser('check')
    for name in ('export','compiled','board-log','out'):
        q.add_argument('--'+name,type=Path,required=True)
    q = sub.add_parser('self-check',help='Validate feature readback with synthetic logs, not board results')
    for name in ('export','compiled','out'):
        q.add_argument('--'+name,type=Path,required=True)
    q = sub.add_parser('reshape-before-sigmoid',help='Equivalent full-model order diagnostic; preserve original goldens')
    for name in ('export','fixtures','out'):
        q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--sigmoid-op',type=int,required=True)
    q.add_argument('--reshape-op',type=int,required=True)
    a = p.parse_args()
    {'prepare':prepare,'check':check,'self-check':self_check,
     'reshape-before-sigmoid':rewrite_sigmoid_reshape}[a.action](a)


if __name__ == '__main__':
    main()
