"""Check the real FT3D crop/label sequence with color-only augmentation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'EdgeFlowNAS'))
from efnas.data.dataloader_builder import build_ft3d_provider
from efnas.data.ft3d_dataset import FT3DBatchProvider
from efnas.data.prefetch_provider import PrefetchBatchProvider
from efnas.engine.stage_state import save_rng


def check(recipe):
    cfg=copy.deepcopy(recipe['config'])
    cfg['data']['ft3d_train_augment']={'enabled':False}
    baseline=build_ft3d_provider(cfg,'train')
    color=next(v['augment'] for v in recipe['variants'].values() if v['augment'].get('enabled'))
    def clone(aug):
        return PrefetchBatchProvider(FT3DBatchProvider(baseline.samples,
            baseline.crop_h,baseline.crop_w,seed=cfg['runtime']['seed'],
            sampling_mode=baseline.sampling_mode,crop_mode=baseline.crop_mode,
            flow_divisor=baseline.flow_divisor,label_clip=baseline.label_clip,
            strict_loading=True,num_workers=16,augment_cfg=aug),1)
    off=clone(dict(color,photometric_aug_prob=0));on=clone(color)
    digest=hashlib.sha256();changed=0
    try:
        for _ in range(2):
            for p in (baseline,off,on): p.start_epoch()
            for _ in range(4):
                a,b,c=[p.next_batch(4) for p in (baseline,off,on)]
                for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
                np.testing.assert_array_equal(a[3],c[3])
                digest.update(a[3].tobytes())
                changed+=int(not np.array_equal(a[0],c[0]))
            off.pause();on.pause()
            assert save_rng(baseline.rng)==save_rng(off.rng)==save_rng(on.rng)
        assert changed>0
        return dict(status='passed',pairs_checked=32,training_pairs=len(baseline),
            changed_batches=changed,labels_sha256=digest.hexdigest(),
            checks=['disabled_color_matches_all_inputs','enabled_color_matches_labels',
                    'sampling_rng_identical_after_prefetch_pause','colors_change'])
    finally:
        for p in (baseline,off,on): p.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=check(json.loads(args.recipe.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
