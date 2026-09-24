"""Sample unique FT3D labels and check paired geometry before a training campaign."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'EdgeFlowNAS'))
from efnas.data.dataloader_builder import build_ft3d_provider
from efnas.data.ft3d_dataset import _read_flow, _random_crop_triplet, _apply_scale_only, FT3DBatchProvider
from efnas.data.prefetch_provider import PrefetchBatchProvider
from efnas.engine.stage_state import save_rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=512)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    base = build_ft3d_provider(recipe['config'], 'train')
    paths = sorted({sample[2] for sample in base.samples})
    rng = np.random.RandomState(42)
    chosen = [paths[i] for i in rng.choice(len(paths), min(args.count,len(paths)), replace=False)]
    totals = {name: dict(pixels=0, component_over50=0, component_over625=0, magnitude_ge400=0)
              for name in ('original', 'crop', 'scale_crop')}
    for path in chosen:
        flow = _read_flow(path)
        if not np.isfinite(flow).all(): raise ValueError('Nonfinite original label: '+path)
        if flow.shape != (540,960,2): raise ValueError(flow.shape)
        seed = int(rng.randint(2**31-1))
        # Only geometry is needed here. Reuse the real transform on dummy RGB.
        image = np.zeros((540,960,3),np.float32)
        crop = _random_crop_triplet(image,image,flow,352,480,np.random.RandomState(seed))[2]
        scaled = _apply_scale_only(image,image,flow,352,480,np.random.RandomState(seed),
            recipe['variants']['s_raw_scale']['augment'])[2]
        for name, field in [('original',flow),('crop',crop),('scale_crop',scaled)]:
            row = totals[name]
            row['pixels'] += int(field.shape[0]*field.shape[1])
            row['component_over50'] += int(np.any(np.abs(field)>50,axis=-1).sum())
            row['component_over625'] += int(np.any(np.abs(field)>625,axis=-1).sum())
            row['magnitude_ge400'] += int((np.linalg.norm(field,axis=-1)>=400).sum())
    for row in totals.values():
        row['fractions'] = {k: v/row['pixels'] for k,v in list(row.items()) if k!='pixels'}
    def clone(aug):
        return PrefetchBatchProvider(FT3DBatchProvider(base.samples,352,480,seed=42,
            sampling_mode='shuffle_no_replacement',crop_mode='random',flow_divisor=1,
            label_clip=None,strict_loading=True,num_workers=16,augment_cfg=aug),1)
    plain=clone({'enabled':False})
    aug=recipe['variants']['s_raw_scale']['augment']
    off=clone(dict(aug,scale_probability=0))
    on=clone(aug)
    try:
        changed=0
        for _ in range(2):
            for p in (plain,off,on): p.start_epoch()
            for _ in range(4):
                a,b,c=[p.next_batch(4) for p in (plain,off,on)]
                for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
                changed+=int(not np.array_equal(a[3],c[3]))
            for p in (plain,off,on): p.pause()
            assert save_rng(plain.rng)==save_rng(off.rng)==save_rng(on.rng)
        assert changed>0
    finally:
        for p in (base,plain,off,on): p.close()
    result=dict(status='passed',training_pairs=len(base),unique_flow_pool=len(paths),
        sample_count=len(chosen),seed=42,sampling='unique TRAIN labels, not double-counting clean/final',
        sample_paths=chosen,sample_list_sha256=hashlib.sha256('\n'.join(chosen).encode()).hexdigest(),
        geometry=totals,pairing_checks='32 real pairs; scale-off equals baseline; sample RNG unchanged with prefetch',
        interpretation='sample estimate, not whole-dataset census; threshold after geometry in input pixel units')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sample_paths'}))


if __name__=='__main__': main()
