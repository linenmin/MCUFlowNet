"""Validate sample identities/counts and rebuild a compact benchmark summary."""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from evaluate import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--mcu-reference',type=Path,help='Existing published S/L CSV directory; audit, do not pretend to re-run')
    args=p.parse_args()
    registry=json.loads(Path(__file__).with_name('models.json').read_text(encoding='utf-8'))
    rows=[]
    samples_reference=None
    dataset=json.loads((args.runs/'dataset-manifest.json').read_text(encoding='utf-8'))
    expected=[p['flow'] for p in dataset['pairs']]
    for model in registry['models']:
        row=dict(model)
        path=args.runs/model['run']
        if (path/'results.json').exists():
            manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
            assert manifest['status']=='completed',path
            samples=list(csv.DictReader((path/'samples.csv').open()))
            keys=[x['sample'] for x in samples]
            assert len(keys)==1041 and len(set(keys))==1041,path
            assert keys==expected,path
            if samples_reference is None:
                samples_reference=keys
            assert keys==samples_reference,path
            for col in ('raw_epe','legacy_epe'):
                values=np.array([float(x[col]) for x in samples])
                assert np.isfinite(values).all() and (values>=0).all()
                result=float(values.mean())
                assert abs(result-manifest['results'][col])<1e-10
                row[col]=result
            row.update(samples=1041,measurement='completed',samples_sha256=sha(path/'samples.csv'),
                       evidence=str(path/'manifest.json'),code_commit=manifest['code_commit'],
                       environment=manifest['environment'],weights_sha256=manifest['weights_sha256'])
        else:
            row['measurement']='not_completed'
        rows.append(row)
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),protocol=registry['protocol_id'],models=rows,
                dataset_manifest_sha256=sha(args.runs/'dataset-manifest.json'),
                source_inventory_sha256=sha(args.runs/'inventory.json'))
    result['checks']={name:json.loads((args.runs/filename).read_text(encoding='utf-8'))
                      for name,filename in [('spynet_weights','spynet-weight-check.json'),('nano_export','nano-export-check.json')]
                      if (args.runs/filename).exists()}
    if args.mcu_reference:
        references=[]
        old_results=json.loads((args.mcu_reference/'results.json').read_text(encoding='utf-8'))
        for name in ['MCUFlowNet-S','MCUFlowNet-L']:
            path=args.mcu_reference/f'{name}.csv'
            samples=list(csv.DictReader(path.open()))
            assert ['training/'+x['sample'] for x in samples]==expected
            values={col:float(np.mean([float(x[col]) for x in samples])) for col in ('raw_epe','clipped_epe')}
            for col,value in values.items():
                assert abs(value-old_results[name][col])<1e-10
            references.append(dict(name=name,**values,samples=len(samples),
                                   evidence=str(path),csv_sha256=sha(path),
                                   note='Reused 2026-09-17 measurement, not re-inferred. Matching sample list and scoring crop; published sintel_best checkpoint selected using Sintel.'))
        result['previous_mcu_measurements']=references
    (args.runs/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Sintel Final公开权重复测', '',
           '统一1041对、416×1024中心裁剪、原始光流标签、全像素EPE。数值单位是源图像素，越小越好。不是436×1024全图标准评测，也不是EdgeFlowNet表III所有行的精确复现。', '',
           '## 未用Sintel作梯度微调的权重', '', '| 模型 | 非截断EPE | 权重训练条件 |', '| --- | ---: | --- |']
    def score(row):
        return f'{row["raw_epe"]:.5f}' if row.get('measurement')=='completed' else '未完成'
    for row in rows:
        if row['status']=='eligible':
            lines.append(f'| {row["name"]} | {score(row)} | {row["training"]} |')
    if 'previous_mcu_measurements' in result:
        lines+=['','已有MCUFlowNet参考（2026-09-17实测，本轮核对同一1041对样本；未重新推理）：','']
        for row in result['previous_mcu_measurements']:
            lines.append(f'- {row["name"]}：{row["raw_epe"]:.5f}。')
        lines+=['','未进行Sintel微调不等于没有用Sintel选模，MCUFlowNet发布权重是sintel_best。']
    lines+=['','## 已用Sintel微调，仅单列参考','','| 模型 | 非截断EPE |','| --- | ---: |']
    for row in rows:
        if row['status']=='in_sample_reference':
            lines.append(f'| {row["name"]} | {score(row)} |')
    lines+=['','## 尚不能用于正式名次','']
    for row in rows:
        if row['status'] in ('provisional','blocked'):
            lines.append(f'- {row["name"]}：{score(row)}。{row["note"]}')
    lines+=['','## 复现证据','',
            '- summary.json：完整条件、结果及权重指纹；各运行目录含manifest.json与samples.csv。',
            '- dataset-manifest.json：数据文件指纹；inventory.json：上游代码版本与文件清单。',
            '- spynet-weight-check.json：转换权重与作者Lua张量的逐个比较。',
            '- nano-export-check.json：H5与浮点TFLite的两对预测比较，不能消除物理单位疑问。',
            '- EdgeFlowNet四块为208×512非重叠拼接，不能混写成176×240部署块。',
            '- 本次没有测端侧FPS、功耗或INT8精度。', '', '核对时间（UTC）：'+result['checked_at']]
    (args.runs/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for r in rows:
        print(r['run'],r.get('raw_epe','pending'),r['status'])


if __name__=='__main__':
    main()
