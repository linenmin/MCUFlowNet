"""Validate and summarize completed full-Sintel deployment scores on CPU only.

This script never imports TensorFlow, restores weights, or runs inference. All
comparisons are paired by the immutable 1,041 flow paths. The remaining 196
pairs have been scored before: they are coverage checks, not a blind test.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


GROUPS = ('full', 'monitor', 'other196')
MOTION = ('below10', '10to40', 'over40')
MODELS = ('edge', 'S', 'L')
PIXELS = 416 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def finite(value, label, nonnegative=False):
    require(isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value), 'Nonfinite/nonnumeric ' + label)
    if nonnegative:
        require(value >= 0, 'Negative ' + label)
    return float(value)


def checkpoint_hashes(prefix):
    files = sorted(prefix.parent.glob(prefix.name + '.*'))
    require(prefix.with_name(prefix.name + '.index').is_file()
            and any('.data-' in path.name for path in files), 'Incomplete checkpoint: ' + str(prefix))
    return {path.name: sha(path) for path in files}


def reduce_records(records):
    require(bool(records), 'Empty scoring group')
    count = len(records)
    denominator = count * PIXELS
    bins = {}
    for name in MOTION:
        pixels = sum(row['motion_bins'][name]['pixels'] for row in records)
        errors = math.fsum(row['motion_bins'][name]['error_sum'] for row in records)
        bins[name] = dict(pixels=pixels, error_sum=errors,
                          epe=errors / pixels if pixels else None,
                          pixel_fraction=pixels / denominator,
                          contribution_to_total_epe=errors / denominator)
    require(sum(item['pixels'] for item in bins.values()) == denominator, 'Motion pixel total differs')
    epe = math.fsum(row['original_epe'] for row in records) / count
    require(abs(math.fsum(item['error_sum'] for item in bins.values()) / denominator - epe) < 1e-7,
            'Motion sums and per-pair EPE differ')
    scenes = {}
    for name in sorted({row['scene'] for row in records}):
        values = [row['original_epe'] for row in records if row['scene'] == name]
        scenes[name] = dict(pairs=len(values), epe=math.fsum(values) / len(values))
    return dict(pairs=count, epe=epe,
                small_epe=math.fsum(row['small_epe'] for row in records) / count,
                motion_bins=bins, scenes=scenes)


def agree_tree(actual, expected, label):
    """Reports must agree with independent aggregation of the saved pairs."""
    require(type(actual) is type(expected) or
            (isinstance(actual, (float, int)) and isinstance(expected, (float, int))),
            'Type differs: ' + label)
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Fields differ: ' + label)
        for key, value in expected.items():
            agree_tree(actual[key], value, label + '.' + key)
    elif isinstance(expected, float):
        require(math.isfinite(actual) and abs(actual - expected) <= max(1e-7, abs(expected) * 1e-12),
                'Value differs: ' + label)
    else:
        require(actual == expected, 'Value differs: ' + label)


def subset(records, group):
    if group == 'full':
        return records
    return [row for row in records if row['monitor'] == (group == 'monitor')]


def paired(reference, other, bootstrap=0, seed=20261004):
    """Positive delta means the second member has higher (worse) EPE."""
    require([row['sample'] for row in reference] == [row['sample'] for row in other],
            'Paired sample order differs')
    count = len(reference)
    scenes = []
    for name in sorted({row['scene'] for row in reference}):
        indices = [i for i, row in enumerate(reference) if row['scene'] == name]
        ref = math.fsum(reference[i]['original_epe'] for i in indices) / len(indices)
        test = math.fsum(other[i]['original_epe'] for i in indices) / len(indices)
        scenes.append(dict(scene=name, pairs=len(indices), reference_epe=ref,
                           other_epe=test, delta=test - ref,
                           contribution_to_total_epe=(test - ref) * len(indices) / count))
    r, o = reduce_records(reference), reduce_records(other)
    bins = {}
    for name in MOTION:
        first, second = r['motion_bins'][name], o['motion_bins'][name]
        require(first['pixels'] == second['pixels'], 'Paired ground-truth motion bins differ')
        bins[name] = dict(pixels=first['pixels'], pixel_fraction=first['pixel_fraction'],
                          reference_epe=first['epe'], other_epe=second['epe'],
                          delta=(second['epe'] - first['epe']) if first['pixels'] else None,
                          contribution_to_total_epe=(second['error_sum'] - first['error_sum']) / (count * PIXELS))
    result = dict(pairs=count, reference_epe=r['epe'], other_epe=o['epe'],
                  delta=o['epe'] - r['epe'], other_pairs_better=sum(
                      b['original_epe'] < a['original_epe'] for a, b in zip(reference, other)),
                  other_scenes_better=sum(row['delta'] < 0 for row in scenes),
                  total_scenes=len(scenes), scene_balanced_delta=math.fsum(
                      row['delta'] for row in scenes) / len(scenes),
                  scenes=sorted(scenes, key=lambda row: row['contribution_to_total_epe'], reverse=True),
                  motion_bins=bins)
    require(abs(math.fsum(row['contribution_to_total_epe'] for row in scenes) - result['delta']) < 1e-7,
            'Scene contributions do not sum to paired delta')
    require(abs(math.fsum(row['contribution_to_total_epe'] for row in bins.values()) - result['delta']) < 1e-7,
            'Motion contributions do not sum to paired delta')
    if bootstrap:
        rng = np.random.default_rng(seed)
        deltas = np.array([row['delta'] for row in scenes], dtype=np.float64)
        counts = np.array([row['pairs'] for row in scenes], dtype=np.float64)
        indices = rng.integers(0, len(scenes), size=(bootstrap, len(scenes)))
        sample_deltas, sample_counts = deltas[indices], counts[indices]
        pair_weighted = np.sum(sample_deltas * sample_counts, axis=1) / np.sum(sample_counts, axis=1)
        balanced = np.mean(sample_deltas, axis=1)
        result['scene_bootstrap'] = dict(repetitions=bootstrap, seed=seed,
            resampling='Whole scenes with replacement; paired models and all scene frames retained',
            pair_weighted_percentile_95=[float(v) for v in np.quantile(pair_weighted, [.025, .975])],
            scene_balanced_percentile_95=[float(v) for v in np.quantile(balanced, [.025, .975])],
            scope='Descriptive scene-sampling uncertainty for these fixed, already selected weights; '
                  'not independent training-seed variance, a blind test, or architecture significance')
    return result


def markdown(summary):
    scores = summary['scores']
    text = ['# 新权重完整评分与 INT8 验收', '',
            'EPE 越低越好。所有分数都按 Sintel Final 原图像素计算，未截断；输入为整图缩放。',
            '1041 对是完整评测，845 对是持续训练监控，另外 196 对也曾经评分，不能称为盲测。', '',
            '## 原生 FP32', '', '| 权重与输入 | 完整 1041 对 | 监控 845 对 | 其余 196 对 |',
            '|---|---:|---:|---:|']
    for key, row in scores.items():
        if row['kind'] == 'native':
            text.append(f"| {row['case']['id']} | {row['full']['epe']:.4f} | {row['monitor']['epe']:.4f} | {row['other196']['epe']:.4f} |")
    text.extend(['', '## 转换与量化', '',
                 '先逐图比较原生 FP32 与 TFLite FP32：全部 1041 对的 EPE 差均须小于 0.001，逐图绝对差的均值须小于 0.0001，防止均值相互抵消。', '',
                 '| 权重与输入 | TFLite FP32 EPE | INT8 EPE | 量化增量 | 最大逐图转换差 | 输出边界值占比 |',
                 '|---|---:|---:|---:|---:|---:|'])
    for case, check in summary['conversion_acceptance'].items():
        f, q = scores[case + '-float'], scores[case + '-int8']
        text.append(f"| {case} | {f['full']['epe']:.4f} | {q['full']['epe']:.4f} | {q['full']['epe']-f['full']['epe']:+.4f} | {check['max_per_pair_epe_abs_difference']:.8f} | {q['output_saturated_fraction']:.3%} |")
    text.extend(['', '输出边界值占比是 INT8 输出恰好等于 -128 或 127 的比例。它不能证明真实浮点输出超出量化范围的比例。', '',
                 '## 随机取图带来的变化', '',
                 '| 模型 | 完整 1041 对的改善 | 监控 845 对的改善 | 其余 196 对的改善 |',
                 '|---|---:|---:|---:|'])
    for model, groups in summary['geometry'].items():
        # Paired delta is random minus whole: flip for an improvement column.
        text.append('| ' + model + ' | ' + ' | '.join(f"{-groups[g]['delta']:+.4f}" for g in GROUPS) + ' |')
    text.extend(['', '## 与 Edge 的配对差值', '',
                 '以下为 MCU 减 Edge，正数表示 MCU 误差更大。每一行都用相同图像配对；',
                 '`max224-vs-edge208` 使用 MCU 的 224×160 与 Edge 的 208×160，其余条目输入尺寸相同。', '',
                 '| 条件 | MCU | 完整 1041 对差值 | 845 对差值 | 196 对差值 |',
                 '|---|---|---:|---:|---:|'])
    for item in summary['model_comparisons']:
        text.append('| ' + item['condition'] + ' | ' + item['model'] + ' | ' +
                    ' | '.join(f"{item['groups'][g]['delta']:+.4f}" for g in GROUPS) + ' |')
    text.extend(['', '## 同一权重从 208×160 改为 224×160', '',
                 '这里只改变推理尺寸，没有重新训练。差值为 224 输入减 208 输入，负数表示改善。', '',
                 '| MCU | 推理类型 | 完整 1041 对差值 | 845 对差值 | 196 对差值 |',
                 '|---|---|---:|---:|---:|'])
    for item in summary['resolution']:
        text.append('| ' + item['model'] + ' | ' + item['kind'] + ' | ' +
                    ' | '.join(f"{item['groups'][g]['delta']:+.4f}" for g in GROUPS) + ' |')
    text.extend(['', '## 误差主要来自哪些运动', '',
                 '下面仅列随机组、208×160 原生 FP32。误差贡献等于该组误差总和除以全部像素数，各组相加得到总 EPE。', '',
                 '| 模型 | 原图运动范围 | 像素占比 | 组内 EPE | 对总 EPE 的贡献 |',
                 '|---|---|---:|---:|---:|'])
    labels = {'below10': '<10 px', '10to40': '10–40 px', 'over40': '≥40 px'}
    for model in MODELS:
        for name, item in scores[f'random-{model}-208-native']['full']['motion_bins'].items():
            text.append(f"| {model} | {labels[name]} | {item['pixel_fraction']:.2%} | {item['epe']:.4f} | {item['contribution_to_total_epe']:.4f} |")
    if summary.get('vela'):
        text.extend(['', '## 同一量化文件的 Vela 编译', '',
                     'Ethos-U55-64、400 MHz、Size，沿用已核验的 Grove 配置。FPS 是编译器估计，不是实机测速。', '',
                     '| 配置 | SRAM 峰值 KiB | CPU 算子 | Vela 估计 FPS |',
                     '|---|---:|---:|---:|'])
        for item in summary['vela']['results']:
            text.append(f"| {item['case']['id']} | {item['sram_peak_kib']:.0f} | {item['cpu_operators']} | {item['estimated_fps']:.3f} |")
        text.extend(['', '五份编译输入的 SHA 与本轮 INT8 评分文件相同，编译产物和日志已核对。',
                     '峰值符合编译配置预算；预算不等于芯片总 SRAM 或固件可用 arena，仍须用新模型上板验收。'])
    text.extend(['', '## 怎样使用这些结果', '',
                 '- 原生、转换与量化评分均已逐样本验收；源 checkpoint 在汇总时重新计算 SHA，与各评分及导出记录一致。',
                 '- 场景明细、运动分组、PTQ 增量、尺寸增量和来源 SHA 均保存在 `summary.json`。',
                 '- 场景 bootstrap 只描述这几份固定权重在场景组成上的不确定性，不代替独立初始化种子实验。',
                 '- 本轮只有一个训练初始化种子，不据此声称架构具有统计显著优势，也不配用旧权重的板端 FPS。',
                 '- INT8 EPE 来自 Vela 编译前的 TFLite CPU 推理；编译通过也不代替板端数值、内存和实测速度。', '',
                 '核验时间：' + summary['verified_at_utc'], ''])
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=5000,
                        help='Whole-scene diagnostic repetitions; 0 disables (default 5000)')
    args = parser.parse_args()
    require(args.bootstrap >= 0, 'Bootstrap repetitions must be nonnegative')
    audit = args.audit.resolve()
    sources = {}

    def read(path):
        sources[str(path)] = sha(path)
        return json.loads(path.read_text(encoding='utf-8-sig'))

    script_dir = Path(__file__).resolve().parent
    scripts = {name: sha(script_dir / name) for name in (
        'audit_deployment.py', 'export_deployment.py', 'run_deployment_audit.py',
        'summarize_deployment.py', 'compile_deployment.py', 'model.py', 'data.py')}
    cases, protocol = read(audit / 'cases.json'), read(audit / 'protocol.json')
    full, monitor, calibration = (read(audit / filename) for filename in
                                  ('sintel_full.json', 'sintel_monitor.json', 'calibration.json'))
    expected_ids = {f'{geometry}-{model}-208' for geometry in ('whole', 'random') for model in MODELS}
    expected_ids |= {'random-S-224', 'random-L-224'}
    require(len(cases) == 8 and {case['id'] for case in cases} == expected_ids, 'Expected exactly eight native cases')
    require(protocol['cases'] == cases, 'Protocol and cases differ')
    require(protocol['samples'] == 1041 and protocol['monitor_samples'] == 845
            and protocol['independent_initialization_seeds'] == 1, 'Unexpected protocol counts')
    # Preparation preceded additions to the scoring reducer. Preserve its
    # original SHA and verify that the current script produces identical data.
    if protocol['script_sha256'] != scripts['audit_deployment.py']:
        recheck = audit / 'control/preparation-recheck'
        for name in ('cases.json', 'sintel_full.json', 'sintel_monitor.json', 'calibration.json'):
            require(sha(audit / name) == sha(recheck / name), 'Preparation regeneration differs: ' + name)
        require(read(recheck / 'protocol.json')['script_sha256'] == scripts['audit_deployment.py'],
                'Preparation recheck code SHA differs')
    require(protocol['full_manifest_sha256'] == sha(audit / 'sintel_full.json')
            and protocol['calibration_sha256'] == sha(audit / 'calibration.json'), 'Manifest SHA differs')
    require(len(full) == len({tuple(row) for row in full}) == 1041
            and len(monitor) == len({tuple(row) for row in monitor}) == 845
            and set(map(tuple, monitor)) <= set(map(tuple, full)), 'Scoring manifest contract differs')
    require(all(len(row) == 3 for row in full + monitor + calibration), 'Manifest must contain triples')
    require(len(calibration) == len({tuple(row) for row in calibration}) == 64, 'Expected 64 distinct calibration pairs')
    require(all(not Path(p).is_absolute() and '..' not in Path(p).parts
                and Path(p).parts[:2] == ('FlyingChairs2', 'train') for row in calibration for p in row),
            'Calibration is not relative FC2 TRAIN paths')
    sample_ids = [row[2] for row in full]
    require(len(set(sample_ids)) == 1041, 'Duplicate flow identifiers')
    monitor_ids = {row[2] for row in monitor}
    source_snapshot, scores, records, exports, acceptance = {}, {}, {}, {}, {}
    calibration_hashes = None
    for case in cases:
        case_id = case['id']
        require(case_id == f"{case['geometry']}-{case['model']}-{case['hw'][1]}"
                and case['hw'] == [160, 224 if case_id.endswith('-224') else 208], 'Case model/dimension differs')
        prefix = Path(case['checkpoint'])
        current_hashes = checkpoint_hashes(prefix)
        source_snapshot[str(prefix)] = current_hashes
        kinds = ('native', 'float', 'int8') if case['geometry'] == 'random' else ('native',)
        if case['geometry'] == 'random':
            exported = read(audit / 'exports' / case_id / 'export.json')
            require(exported['status'] == 'passed' and exported['model'] == case['model']
                    and exported['input_hw'] == case['hw'] and exported['batch'] == 1
                    and Path(exported['checkpoint']) == prefix, 'Export case contract differs: ' + case_id)
            for flag in ('native_restore_exact', 'native_inference_weights_unchanged',
                         'training_switch_fixed_false', 'checkpoint_unchanged', 'convolution_gpu_observed'):
                require(exported.get(flag) is True, 'Export acceptance missing: ' + case_id + '/' + flag)
            require(exported['checkpoint_sha256'] == current_hashes, 'Export checkpoint SHA differs: ' + case_id)
            require(exported['script_sha256'] == scripts['export_deployment.py'], 'Export script changed')
            require(exported['calibration_pairs'] == 64
                    and exported['calibration_manifest_sha256'] == protocol['calibration_sha256']
                    and [row['paths'] for row in exported['calibration']] == calibration,
                    'Export calibration manifest differs')
            require('BGR' in exported['input_convention'] and 'AREA resize' in exported['input_convention']
                    and '/255*2-1' in exported['input_convention']
                    and 'no 12.5 multiplier' in exported['output_convention']
                    and 'no clipping' in exported['output_convention']
                    and 'inference fixed False' in exported['bn']
                    and 'no BN update' in exported['bn'], 'Export flow/normalization/BN contract differs')
            hashes = [row['sha256'] for row in exported['calibration']]
            require(all(set(value) == set(paths) and all(len(s) == 64 for s in value.values())
                        for value, paths in zip(hashes, calibration)), 'Incomplete calibration file SHA')
            if calibration_hashes is None:
                calibration_hashes = hashes
            require(hashes == calibration_hashes, 'Calibration source files differ between exports')
            graph_file = audit / 'exports' / case_id / 'inference.pb'
            sources[str(graph_file)] = sha(graph_file)
            require(exported['frozen_graph_sha256'] == sources[str(graph_file)], 'Frozen graph SHA differs')
            require(exported['native_output_shape'] == [1, *case['hw'], 2]
                    and finite(exported['frozen_native_max_abs'], 'Frozen/native max', True) <= 1e-4,
                    'Frozen/native acceptance differs')
            require(set(exported['exports']) == {'float', 'int8'}, 'Both TFLite models required')
            for kind, info in exported['exports'].items():
                file = audit / 'exports' / case_id / f'model_{kind}.tflite'
                sources[str(file)] = sha(file)
                require(info['sha256'] == sources[str(file)] and info['bytes'] == file.stat().st_size,
                        'TFLite SHA/size differs: ' + case_id + '/' + kind)
                for boundary, channels in [('input', 6), ('output', 2)]:
                    require(info[boundary]['shape'] == [1, *case['hw'], channels]
                            and info[boundary]['dtype'] == ('int8' if kind == 'int8' else 'float32'),
                            'TFLite shape/dtype differs: ' + case_id + '/' + kind)
                    if kind == 'int8':
                        require(info[boundary]['scale'] > 0, 'Missing quantization scale')
                if kind == 'int8':
                    require(not info['float_tensors'], 'Float tensors remain in INT8 export')
                differences = info['per_pair_differences']
                require([row['sample'] for row in differences] == [row[0] for row in calibration],
                        'Export calibration comparison pairs differ')
                maximum = max(finite(row['max_abs'], 'Calibration conversion max', True) for row in differences)
                require(maximum == info['max_abs_difference'], 'Calibration conversion max differs')
                if kind == 'float':
                    require(maximum <= 1e-3, 'FP32 calibration component difference exceeds tolerance')
                    require(max(row['mean_vector_difference'] for row in differences) <= 1e-4,
                            'FP32 calibration mean vector difference exceeds tolerance')
            exports[case_id] = exported
        for kind in kinds:
            key = case_id + '-' + kind
            result = read(audit / 'scores' / key / 'result.json')
            file = audit / 'scores' / key / 'per_pair.jsonl'
            sources[str(file)] = sha(file)
            rows = [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line.strip()]
            require(result['case'] == case and result['kind'] == kind, 'Score case differs: ' + key)
            require(result['script_sha256'] == scripts['audit_deployment.py'], 'Scoring script changed')
            require(result['weights_unchanged'] is True and result['checkpoint_sha256'] == current_hashes,
                    'Current checkpoint SHA differs from score: ' + key)
            require([row['sample'] for row in rows] == sample_ids, 'Expected all 1041 pairs in manifest order: ' + key)
            for row in rows:
                require(row['scene'] == Path(row['sample']).parent.name
                        and row['monitor'] is (row['sample'] in monitor_ids), 'Scene/monitor flag differs: ' + key)
                for name in ('original_epe', 'small_epe', 'abs_u', 'abs_v', 'output_saturated_fraction'):
                    finite(row[name], key + '/' + name, True)
                require(row['output_saturated_fraction'] <= 1, 'Saturation fraction exceeds 1')
                require(set(row['motion_bins']) == set(MOTION), 'Motion groups differ')
                for item in row['motion_bins'].values():
                    require(type(item['pixels']) is int and item['pixels'] >= 0, 'Invalid motion pixel count')
                    finite(item['error_sum'], key + '/motion error', True)
                require(sum(item['pixels'] for item in row['motion_bins'].values()) == PIXELS,
                        'Per-pair motion counts differ')
                require(abs(math.fsum(item['error_sum'] for item in row['motion_bins'].values()) / PIXELS
                            - row['original_epe']) < 1e-7, 'Per-pair motion error sums differ')
            for group, count in zip(GROUPS, (1041, 845, 196)):
                recalculated = reduce_records(subset(rows, group))
                require(recalculated['pairs'] == count, 'Score subset count differs')
                agree_tree(result[group], recalculated, key + '/' + group)
            require(abs(result['output_saturated_fraction'] - math.fsum(
                row['output_saturated_fraction'] for row in rows) / 1041) < 1e-12, 'Output saturation aggregate differs')
            if kind == 'native':
                require(result['restore_exact'] is True and result['inference_device'] == 'GPU'
                        and any('GPU' in row['device'] for row in result['gpu_convolutions']),
                        'Native GPU/restore acceptance missing: ' + key)
                require(result['tflite_sha256'] is None, 'Unexpected native TFLite SHA')
                if case['expected_monitor'] is not None:
                    difference = abs(result['monitor']['epe'] - case['expected_monitor'])
                    require(difference < 1e-5 and abs(result['monitor_reproduction_abs_difference'] - difference) < 1e-12,
                            'Training monitor reproduction differs: ' + key)
            else:
                require(result['tflite_sha256'] == exports[case_id]['exports'][kind]['sha256']
                        and result['inference_device'] == 'TFLite CPU', 'TFLite runtime SHA differs: ' + key)
            records[key], scores[key] = rows, result
        if case['geometry'] == 'random':
            native, converted = records[case_id + '-native'], records[case_id + '-float']
            differences = [abs(a['original_epe'] - b['original_epe']) for a, b in zip(native, converted)]
            index = int(np.argmax(differences))
            item = dict(pairs=1041, tolerance_original_pixels=1e-3,
                        max_per_pair_epe_abs_difference=differences[index],
                        max_difference_sample=native[index]['sample'],
                        mean_per_pair_epe_abs_difference=math.fsum(differences) / 1041,
                        aggregate_epe_abs_difference=abs(scores[case_id + '-native']['full']['epe'] -
                                                         scores[case_id + '-float']['full']['epe']),
                        mean_pair_epe_tolerance_original_pixels=1e-4,
                        passed=(max(differences) < 1e-3 and math.fsum(differences) / 1041 < 1e-4),
                        scope='All Sintel per-image scalar EPE paired, plus all 64 calibration predictions '
                              'checked by exporter; Sintel per-pixel prediction arrays were not saved')
            acceptance[case_id] = item
            require(item['passed'], 'Native/FP32 TFLite per-pair EPE exceeds tolerance: ' + repr(item))
    require(len(scores) == 18 and len(exports) == 5 and len(acceptance) == 5,
            'Expected 8 native, 5 FP32 TFLite and 5 INT8 scores')
    canonical_bins = [tuple(row['motion_bins'][name]['pixels'] for name in MOTION)
                      for row in records['whole-edge-208-native']]
    for key, rows in records.items():
        require([tuple(row['motion_bins'][name]['pixels'] for name in MOTION) for row in rows]
                == canonical_bins, 'Per-pair ground-truth motion counts differ: ' + key)
    # Snapshot the current source files again after reading all reports.
    require(all(checkpoint_hashes(Path(prefix)) == hashes for prefix, hashes in source_snapshot.items()),
            'Checkpoint changed during summary')
    summary = dict(status='passed', verified_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   scope='CPU aggregation of completed read-only inference; no new model inference or training',
                   protocol=protocol, scores=scores, conversion_acceptance=acceptance,
                   geometry={}, model_comparisons=[], quantization=[], resolution=[],
                   checkpoint_sha256_current_snapshot=source_snapshot,
                   calibration_source_file_sha256=calibration_hashes,
                   independent_initialization_seeds=1,
                   held_out_test=False,
                   output_saturated_fraction_definition='Fraction of INT8 output values equal to -128 or 127; '
                       'not the proportion of true FP32 values outside the quantized representable range')
    for model in MODELS:
        summary['geometry'][model] = {group: paired(
            subset(records[f'whole-{model}-208-native'], group),
            subset(records[f'random-{model}-208-native'], group), args.bootstrap)
            for group in GROUPS}
    conditions = [('whole-208-native', 'whole', 'native'), ('random-208-native', 'random', 'native'),
                  ('random-208-float', 'random', 'float'), ('random-208-int8', 'random', 'int8')]
    for condition, geometry, kind in conditions:
        for model in ('S', 'L'):
            summary['model_comparisons'].append(dict(condition=condition, model=model,
                reference_case=f'{geometry}-edge-208-{kind}', other_case=f'{geometry}-{model}-208-{kind}',
                same_input_resolution=True, groups={group: paired(
                    subset(records[f'{geometry}-edge-208-{kind}'], group),
                    subset(records[f'{geometry}-{model}-208-{kind}'], group), args.bootstrap)
                    for group in GROUPS}))
    for model in ('S', 'L'):
        for kind in ('native', 'float', 'int8'):
            first, second = f'random-{model}-208-{kind}', f'random-{model}-224-{kind}'
            require(scores[first]['checkpoint_sha256'] == scores[second]['checkpoint_sha256'],
                    'Resolution comparison source weights differ')
            summary['resolution'].append(dict(model=model, kind=kind,
                reference_case=first, other_case=second, change='224x160 minus 208x160, same weights',
                groups={group: paired(subset(records[first], group), subset(records[second], group), args.bootstrap)
                        for group in GROUPS}))
            summary['model_comparisons'].append(dict(condition='random-max224-vs-edge208-' + kind,
                model=model, reference_case=f'random-edge-208-{kind}', other_case=second,
                same_input_resolution=False,
                scope='Configured maximum input sizes verified using older weights; new INT8 board fitness '
                      'and speed are not measured by this pre-Vela audit',
                groups={group: paired(subset(records[f'random-edge-208-{kind}'], group),
                                      subset(records[second], group), args.bootstrap) for group in GROUPS}))
    for case in acceptance:
        summary['quantization'].append(dict(case=case, change='INT8 minus TFLite FP32',
            groups={group: paired(subset(records[case + '-float'], group),
                                  subset(records[case + '-int8'], group), args.bootstrap) for group in GROUPS},
            output_saturated_fraction=scores[case + '-int8']['output_saturated_fraction'],
            boundary=exports[case]['exports']['int8']['output']))
    vela_path = audit / 'vela/summary.json'
    if vela_path.is_file():
        vela = read(vela_path)
        require(vela['status'] == 'completed_not_board_validated' and vela['failed'] == 0
                and vela['accelerator'] == 'ethos-u55-64' and vela['optimization'] == 'Size',
                'Vela compilation is incomplete or uses another target')
        require(vela['script_sha256'] == scripts['compile_deployment.py'], 'Vela wrapper script changed')
        require(vela['configuration_sha256'] ==
                'a07260cb487d49de1f93c93295ec9959ede034d6e847bb5aa92fe6650131e2da'
                and int(vela['memory']['arena_cache_size']) == 1468006
                and float(vela['system']['core_clock']) == 400e6,
                'Vela Grove configuration differs')
        require(len(vela['results']) == 5 and
                {item['case']['id'] for item in vela['results']} == set(acceptance),
                'Vela compilation case coverage differs')
        for item in vela['results']:
            case_id = item['case']['id']
            require(item['case'] == scores[case_id + '-int8']['case']
                    and item['input_sha256'] == exports[case_id]['exports']['int8']['sha256']
                    and item['checkpoint_sha256'] == exports[case_id]['checkpoint_sha256'],
                    'Vela input identity or checkpoint SHA differs: ' + case_id)
            require(item['status'] == 'compiled_not_board_validated' and item['returncode'] == 0
                    and item['input_unchanged'] is True and item['cpu_operators'] == 0
                    and item['within_configured_cache_budget'] is True
                    and item['sram_peak_bytes'] <= 1468006, 'Vela acceptance missing: ' + case_id)
            require(item['export_report_sha256'] == sha(audit / 'exports' / case_id / 'export.json'),
                    'Vela export report SHA differs: ' + case_id)
            # Vela runs on Windows while this reducer may run in the Linux
            # container. Resolve its recorded basenames under this audit root.
            for field, hash_field in (('log', 'log_sha256'), ('summary_csv', 'summary_csv_sha256'),
                                      ('compiled_model', 'compiled_sha256')):
                name = Path(item[field].replace('\\', '/')).name
                path = audit / 'vela' / case_id / 'Size' / name
                sources[str(path)] = sha(path)
                require(sources[str(path)] == item[hash_field], 'Vela artifact SHA differs: ' + str(path))
        summary['vela'] = vela
    summary['source_sha256'] = sources
    summary['script_sha256_current_snapshot'] = scripts
    save(audit / 'summary.json', summary)
    (audit / 'summary.md').write_text(markdown(summary), encoding='utf-8')
    print(json.dumps(dict(status=summary['status'], verified_at_utc=summary['verified_at_utc'],
                         scores=len(scores), exports=len(exports), summary=str(audit / 'summary.json'))), flush=True)


if __name__ == '__main__':
    main()
