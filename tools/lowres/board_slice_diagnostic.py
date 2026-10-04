"""Remove only the final u/v slice from a copied INT8 graph for board diagnosis.

No training or recalibration. Every constant buffer stays byte-identical, and
the first two CPU output channels must equal all three original board fixtures.
The four-channel graph is a diagnostic artifact, not a benchmark replacement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import zlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


def sha(value):
    return hashlib.sha256(value).hexdigest()


def tensor_io(g, index):
    t = g.Tensors(index); q = t.Quantization()
    assert t.Type() == schema.TensorType.INT8 and q.ScaleLength() == q.ZeroPointLength() == 1
    return dict(shape=t.ShapeAsNumpy().tolist(), dtype='int8',
                scale=float(q.Scale(0)), zero_point=int(q.ZeroPoint(0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('export', 'fixtures', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    parent = json.loads((a.export/'export.json').read_text())
    raw = (a.export/'model_int8.tflite').read_bytes()
    old_meta = json.loads((a.fixtures/'fixtures.json').read_text())
    old_blob = (a.fixtures/'fixtures.bin').read_bytes()
    assert parent['status'] == 'passed'
    assert sha(raw) == parent['exports']['int8']['sha256'] == old_meta['source_sha256']
    assert sha(old_blob) == old_meta['blob_sha256']
    modified = bytearray(raw)
    original = schema.Model.GetRootAsModel(raw, 0)
    model = schema.Model.GetRootAsModel(modified, 0); g = model.Subgraphs(0)
    assert g.InputsLength() == g.OutputsLength() == 1
    tail = g.Operators(g.OperatorsLength()-1)
    assert model.OperatorCodes(tail.OpcodeIndex()).BuiltinCode() == schema.BuiltinOperator.STRIDED_SLICE
    assert tail.OutputsLength() == 1 and tail.Outputs(0) == g.Outputs(0)
    before = tensor_io(g, tail.Inputs(0)); after = tensor_io(g, tail.Outputs(0))
    assert before['shape'] == after['shape'][:-1]+[4] and after['shape'][-1] == 2
    assert before['scale'] == after['scale'] and before['zero_point'] == after['zero_point']
    old_count = g.OperatorsLength()
    g.OutputsAsNumpy()[0] = tail.Inputs(0)
    # SubGraph field 3 is the operator vector. Shorten it; other tables stay put.
    struct.pack_into('<I', modified, g._tab.Vector(g._tab.Offset(10))-4, old_count-1)
    check = schema.Model.GetRootAsModel(modified, 0)
    assert check.Subgraphs(0).OperatorsLength() == old_count-1
    assert check.BuffersLength() == original.BuffersLength()
    for n in range(check.BuffersLength()):
        b1, b2 = original.Buffers(n), check.Buffers(n)
        assert b1.DataLength() == b2.DataLength()
        if b1.DataLength():
            assert np.array_equal(b1.DataAsNumpy(), b2.DataAsNumpy())
    assert tensor_io(check.Subgraphs(0), check.Subgraphs(0).Inputs(0)) == tensor_io(original.Subgraphs(0), original.Subgraphs(0).Inputs(0))
    it = tf.lite.Interpreter(model_content=bytes(modified), num_threads=1,
         experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors(); ii = it.get_input_details()[0]; oo = it.get_output_details()[0]
    assert oo['shape'].tolist() == before['shape']
    meta = dict(old_meta, source_sha256=sha(modified), output_bytes=int(np.prod(oo['shape'])),
                reference='CPU TFLite INT8 BUILTIN_WITHOUT_DEFAULT_DELEGATES; all four original accumulated channels',
                diagnostic_only=True, parent_source_sha256=sha(raw),
                performance=dict(enabled=False, scope='Numerical diagnosis only; no timing'), pairs=[])
    blob = bytearray(); checks = []
    for pair in old_meta['pairs']:
        start = pair['offset']; end = start+old_meta['input_bytes']
        input_bytes = old_blob[start:end]
        expected = np.frombuffer(old_blob[end:end+old_meta['output_bytes']], np.int8).reshape(after['shape'])
        assert sha(input_bytes) == pair['input_sha256']
        it.set_tensor(ii['index'], np.frombuffer(input_bytes, np.int8).reshape(ii['shape']))
        it.invoke(); output = it.get_tensor(oo['index'])
        assert np.array_equal(output[..., :2], expected), 'Removing slice changed CPU u/v'
        result = output.tobytes()
        meta['pairs'].append(dict(pair, offset=len(blob), output_sha256=sha(result),
                                  output_crc32=f'{zlib.crc32(result):08x}'))
        blob.extend(input_bytes); blob.extend(result)
        checks.append(dict(pair_index=pair['index'], first_two_channels_byte_exact=True))
    meta.update(blob_sha256=sha(blob), blob_crc32=f'{zlib.crc32(blob):08x}', blob_bytes=len(blob))
    report = dict(status='passed', model=parent['model'], edge_public=parent.get('edge_public',False),
        input_hw=parent['input_hw'], output_convention=parent['output_convention'],
        scope='Final slice removal diagnostic only; no new benchmark EPE, calibration or weights',
        auxiliary_channels='Original remaining two accumulated channels; not used for flow EPE',
        parent_source_sha256=sha(raw), constant_buffers_byte_exact=True,
        cpu_checks=checks, modified_byte_positions=[i for i,(x,y) in enumerate(zip(raw,modified)) if x!=y],
        operators_before=old_count, operators_after=old_count-1,
        exports=dict(int8=dict(sha256=sha(modified),input=tensor_io(g,g.Inputs(0)),output=before)))
    a.out.mkdir(parents=True); (a.out/'fixtures').mkdir()
    (a.out/'model_int8.tflite').write_bytes(modified)
    (a.out/'fixtures/fixtures.bin').write_bytes(blob)
    (a.out/'fixtures/fixtures.json').write_text(json.dumps(meta,indent=2)+'\n')
    (a.out/'export.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
