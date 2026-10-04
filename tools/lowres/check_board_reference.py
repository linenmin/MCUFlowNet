"""Independently check saved INT8 board fixtures without changing their bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


def sha(value):
    return hashlib.sha256(value).hexdigest()


def io_metadata(path):
    m = schema.Model.GetRootAsModel(path.read_bytes(), 0)
    g = m.Subgraphs(0)
    result = {}
    for key, index in [('input', g.Inputs(0)), ('output', g.Outputs(0))]:
        t = g.Tensors(index); q = t.Quantization()
        result[key] = dict(shape=t.ShapeAsNumpy().tolist(), type=int(t.Type()),
                           scale=float(q.Scale(0)), zero_point=int(q.ZeroPoint(0)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'compiled', 'fixtures', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Use a new result path; keep earlier failures')
    blob = (a.fixtures/'fixtures.bin').read_bytes()
    meta = json.loads((a.fixtures/'fixtures.json').read_text())
    assert sha(a.model.read_bytes()) == meta['source_sha256']
    assert sha(blob) == meta['blob_sha256']
    assert f'{zlib.crc32(blob):08x}' == meta['blob_crc32']
    source_io = io_metadata(a.model); compiled_io = io_metadata(a.compiled)
    assert source_io == compiled_io, (source_io, compiled_io)
    result = dict(source_sha256=meta['source_sha256'],
                  compiled_sha256=sha(a.compiled.read_bytes()), io=source_io,
                  fixture_sha256=sha(blob), tensorflow=tf.__version__, comparisons=[])
    resolvers = [('builtin_no_delegate', tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES),
                 ('reference_no_delegate', tf.lite.experimental.OpResolverType.BUILTIN_REF)]
    for name, resolver in resolvers:
        it = tf.lite.Interpreter(model_path=str(a.model), num_threads=1,
                                experimental_op_resolver_type=resolver)
        it.allocate_tensors()
        ii = it.get_input_details()[0]; oo = it.get_output_details()[0]
        for item in meta['pairs']:
            start = item['offset']; end = start + meta['input_bytes']
            value = np.frombuffer(blob[start:end], dtype=np.int8).reshape(ii['shape'])
            expected = np.frombuffer(blob[end:end+meta['output_bytes']], dtype=np.int8).reshape(oo['shape'])
            assert sha(value.tobytes()) == item['input_sha256']
            assert sha(expected.tobytes()) == item['output_sha256']
            it.set_tensor(ii['index'], value); it.invoke(); actual = it.get_tensor(oo['index'])
            difference = np.abs(actual.astype(np.int16)-expected.astype(np.int16))
            result['comparisons'].append(dict(resolver=name, pair_index=item['index'],
                max_abs_q=int(difference.max()), mean_abs_q=float(difference.mean()),
                changed=int(np.count_nonzero(difference)),
                reference_crc32=f'{zlib.crc32(expected.tobytes()):08x}',
                actual_crc32=f'{zlib.crc32(actual.tobytes()):08x}',
                reference_first16=expected.reshape(-1)[:16].tolist()))
    result['builtin_fixture_exact'] = all(x['max_abs_q'] == 0 for x in result['comparisons']
                                         if x['resolver'] == 'builtin_no_delegate')
    result['reference_kernel_fixture_exact'] = all(x['max_abs_q'] == 0 for x in result['comparisons']
                                                  if x['resolver'] == 'reference_no_delegate')
    result['limits'] = ('This checks saved fixtures and CPU kernel differences, not NPU execution. '
                        'A reference-kernel difference does not by itself invalidate the measured CPU baseline.')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
