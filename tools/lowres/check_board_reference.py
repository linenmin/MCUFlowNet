"""Independently check saved INT8 board fixtures without changing their bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
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


def decode_board_outputs(path, expected_bytes):
    """Require complete, ordered hex dumps and CRCs from the latest boot."""
    log = path.read_bytes().decode('ascii', errors='replace')
    log = log.rsplit('Optical camera oflow app start', 1)[-1]
    output = {}; active = None; data = None; crc = None
    for line in log.splitlines():
        begin = re.fullmatch(r'FLOW_Q_BEGIN index=(\d+) bytes=(\d+) crc=([0-9a-f]{8})', line.strip())
        chunk = re.fullmatch(r'FLOW_Q index=(\d+) offset=(\d+) data=([0-9a-f]+)', line.strip())
        end = re.fullmatch(r'FLOW_Q_END index=(\d+)', line.strip())
        if begin:
            if active is not None:
                raise ValueError('Interrupted board output dump')
            active, size, crc = int(begin[1]), int(begin[2]), begin[3]
            if size != expected_bytes or active in output:
                raise ValueError('Unexpected or duplicate board dump')
            data = bytearray()
        elif chunk:
            if active != int(chunk[1]) or len(data) != int(chunk[2]):
                raise ValueError('Missing, unordered or duplicated board bytes')
            data.extend(bytes.fromhex(chunk[3]))
            if len(data) > expected_bytes:
                raise ValueError('Board output exceeds declared size')
        elif end:
            if active != int(end[1]) or len(data) != expected_bytes:
                raise ValueError('Incomplete board output dump')
            if f'{zlib.crc32(data):08x}' != crc:
                raise ValueError('Board output CRC differs from received bytes')
            output[active] = bytes(data); active = None
    if active is not None or not output:
        raise ValueError('No complete board outputs')
    return output


def compare_board_output(actual, expected, scale):
    def errors(value):
        delta = value.astype(np.int16) - expected.astype(np.int16)
        return dict(max_abs_q=int(np.abs(delta).max()),
                    mean_abs_q=float(np.abs(delta).mean()),
                    changed=int(np.count_nonzero(delta)))
    channels = actual.shape[-1]
    if channels not in (2, 4):
        raise ValueError('Only u/v or an explicit four-channel diagnostic supported')
    delta = (actual[..., :2].astype(np.float64) - expected[..., :2].astype(np.float64)) * scale
    flat_actual = actual.reshape(-1, channels); flat_expected = expected.reshape(-1, channels)
    correlations = []
    for channel in range(channels):
        a, e = flat_actual[:, channel], flat_expected[:, channel]
        correlations.append(float(np.corrcoef(a, e)[0, 1])
                            if np.std(a) and np.std(e) else None)
    return dict(**errors(actual),
                mean_vector_difference_input_pixels=float(np.linalg.norm(delta, axis=-1).mean()),
                first16=actual.reshape(-1)[:16].tolist(),
                actual_range=[int(actual.min()), int(actual.max())],
                reference_range=[int(expected.min()), int(expected.max())],
                mean_abs_q_per_channel=np.abs(flat_actual.astype(np.int16)-flat_expected).mean(axis=0).tolist(),
                channel_correlations=correlations,
                layout_hypotheses=dict(
                    swapped_uv=errors(actual[..., [1, 0]+list(range(2, channels))]),
                    channels_first=errors(actual.reshape(1, channels, *actual.shape[1:3]).transpose(0, 2, 3, 1))),
                limits='Layout variants diagnose possible ordering errors; they are not accepted model outputs or EPE scores.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'compiled', 'fixtures', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--board-log', type=Path,
                   help='Optional complete fixed-output hex dumps from the same compiled model')
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
    if a.board_log:
        log = a.board_log.read_bytes().decode('ascii', errors='replace')
        log = log.rsplit('Optical camera oflow app start', 1)[-1]
        model_crc = f'{zlib.crc32(a.compiled.read_bytes()):08x}'
        if not re.search(r'FLOW_MODEL [^\r\n]+ crc=' + model_crc + r' PASS', log):
            raise ValueError('Board model CRC was not verified in this boot')
        input_crcs = dict(re.findall(r'FLOW_CACHE index=(\d+) input_crc=([0-9a-f]{8})', log))
        data = decode_board_outputs(a.board_log, meta['output_bytes'])
        if set(data) != set(range(len(meta['pairs']))):
            raise ValueError('Board log must contain all saved fixtures')
        result['board'] = dict(log=str(a.board_log.resolve()),
                               log_sha256=sha(a.board_log.read_bytes()),
                               model_crc32=model_crc, input_crc_verified=True, comparisons=[])
        a.out.parent.mkdir(parents=True, exist_ok=True)
        for position, item in enumerate(meta['pairs']):
            end = item['offset'] + meta['input_bytes']
            if input_crcs.get(str(position)) != f'{zlib.crc32(blob[item["offset"]:end]):08x}':
                raise ValueError('Board copied input CRC differs from fixture')
            expected = np.frombuffer(blob[end:end+meta['output_bytes']], dtype=np.int8).reshape(compiled_io['output']['shape'])
            actual = np.frombuffer(data[position], dtype=np.int8).reshape(expected.shape)
            result['board']['comparisons'].append(dict(fixture=position, pair_index=item['index'],
                crc32=f'{zlib.crc32(data[position]):08x}',
                **compare_board_output(actual, expected, compiled_io['output']['scale'])))
            (a.out.parent/f'board-output-{position}.bin').write_bytes(data[position])
        result['board']['limits'] = ('The firmware model CRC must be checked separately. '
                                    'These are three fixed FC2 inputs, not a full Sintel board EPE measurement.')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
