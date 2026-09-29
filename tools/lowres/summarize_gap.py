"""Join historical and current evidence without retraining or changing scores."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    a = p.parse_args()
    base = a.runs/'LOWRES-BENCH-01/ft3d-lr20-20260929'
    out = a.runs/'LOWRES-BENCH-01/gap-audit-20260929'
    sources = {}
    def read(path, table=False):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f)) if table else json.load(f)
    manifest = read(base/'manifests/sintel_monitor.json')
    keys = [r[2].split('/flow/',1)[1] for r in manifest]
    assert len(keys) == len(set(keys)) == 845
    summary = dict(fc2=[], ft3d=[], historical_same_845=[], paired_scene_comparison=[])
    for model in ('edge','S','L'):
        state = read(base/f'source/{model}/fc2/current.json')
        summary['fc2'].append(dict(model=model, best=state['best'],
            last=state['history'][-1],
            evaluated_history=[h for h in state['history'] if 'sintel_epe_original_pixels' in h]))
        for schedule in ('constant','cosine'):
            state = read(base/f'{schedule}/seed42/{model}/ft3d/current.json')
            summary['ft3d'].append(dict(model=model,schedule=schedule,best=state['best'],
                last=state['history'][-1], history=state['history']))
        filename = ('SINTEL-BASE-01/edge-full/samples.csv' if model=='edge'
                    else f'20260917-sl-sintel01/MCUFlowNet-{model}.csv')
        rows = read(a.runs/filename,True)
        lookup = {r['sample'].split('flow/',1)[1]:float(r['raw_epe']) for r in rows}
        assert set(keys) <= lookup.keys()
        old = dict(model=model, full_1041=mean(lookup.values()),
                   same_845=mean(lookup[k] for k in keys), samples=len(lookup))
        filename = ('GROVE-INT8-01/epe/edge-160x208/samples.csv' if model=='edge'
                    else f'GROVE-INT8-01/epe/MCUFlowNet-{model}-160x224/samples.csv')
        resized = read(a.runs/filename,True)
        lookup = {r['sample'].split('flow/',1)[1]:float(r['float_epe']) for r in resized}
        assert set(keys) <= lookup.keys()
        old['resize_same_845'] = mean(lookup[k] for k in keys)
        old['resize_hw'] = [160,208 if model=='edge' else 224]
        summary['historical_same_845'].append(old)
    inference = {}
    for model in ('edge','S','L'):
        path = out/f'inference/{model}.json'
        if path.exists():
            inference[model] = read(path)
    if len(inference) == 3:
        edge = inference['edge']['per_pair']
        for model in ('S','L'):
            other = inference[model]['per_pair']
            assert [r['sample'] for r in edge] == [r['sample'] for r in other] == [r[2] for r in manifest]
            scenes = defaultdict(list)
            for x,y in zip(edge,other):
                scenes[x['scene']].append((x['original_epe'],y['original_epe']))
            scene_result = []
            for name,pairs in scenes.items():
                av,bv = map(mean,zip(*pairs))
                scene_result.append(dict(scene=name,pairs=len(pairs),edge_epe=av,
                    model_epe=bv,delta=bv-av, contribution_to_total=(bv-av)*len(pairs)/845))
            scene_result.sort(key=lambda r:r['contribution_to_total'],reverse=True)
            bins = {}
            for name,ref in inference['edge']['motion_bins'].items():
                item = inference[model]['motion_bins'][name]
                assert item['pixels'] == ref['pixels']
                bins[name] = dict(pixels=ref['pixels'],edge_epe=ref['error_sum']/ref['pixels'],
                    model_epe=item['error_sum']/item['pixels'],
                    contribution_to_total=(item['error_sum']-ref['error_sum'])/(845*416*1024))
            summary['paired_scene_comparison'].append(dict(model=model,
                pairs_won=sum(y['original_epe']<x['original_epe'] for x,y in zip(edge,other)),
                scenes_won=sum(r['delta']<0 for r in scene_result),total_scenes=len(scenes),
                scene_balanced_delta=mean(r['delta'] for r in scene_result),
                original_delta=inference[model]['original_epe']-inference['edge']['original_epe'],
                small_epe=inference[model]['small_epe'],edge_small_epe=inference['edge']['small_epe'],
                scenes=scene_result,motion_bins=bins))
    summary['source_sha256'] = sources
    out.mkdir(exist_ok=True,parents=True)
    (out/'evidence.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_sha256','fc2','ft3d')},indent=2))


if __name__ == '__main__':
    main()
