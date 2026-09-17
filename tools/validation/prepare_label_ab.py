"""Generate versioned FC2 comparison configs and Sintel scene lists, without reading GT."""
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'EdgeFlowNAS/configs/experiments/label_ab'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt'
    lines = sorted(source.read_text(encoding='utf-8').splitlines())
    groups = defaultdict(list)
    for line in lines:
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(line)
        scene = Path(fields[0]).parent.name
        # Keep related sequences (e.g. ambush_2/ambush_4) on the same side.
        groups[scene.rsplit('_', 1)[0]].append(line)
    ranked = sorted(groups, key=lambda x: hashlib.sha256(('mcuflownet-sintel-v1:' + x).encode()).hexdigest())
    held_groups = set(ranked[:max(1, round(len(ranked) * .25))])
    holdout = sorted(line for group in held_groups for line in groups[group])
    monitor = sorted(set(lines) - set(holdout))
    scenes = defaultdict(list)
    for line in monitor:
        scenes[Path(line.split()[0]).parent.name].append(line)
    quick = sorted({items[round(i * (len(items)-1) / 3)] for items in scenes.values() for i in range(4)})
    assert len(set(lines)) == len(lines) == 1041
    assert set(quick) <= set(monitor) and not set(monitor) & set(holdout)
    manifests = {}
    for name, items in [('monitor_quick', quick), ('monitor_all', monitor), ('holdout', holdout)]:
        path = OUT / (name + '.txt')
        path.write_text('\n'.join(items) + '\n', encoding='utf-8')
        manifests[name] = {'samples': len(items), 'scenes': sorted({Path(x.split()[0]).parent.name for x in items}),
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(OUT / 'split_manifest.json', {'rule': 'hash scene families; lowest rounded 25% held out; 4 evenly spaced pairs per monitor scene',
        'held_scene_families': sorted(held_groups), 'source': str(source.relative_to(ROOT)), 'splits': manifests,
        'limitation': 'Internal future holdout only: old checkpoints were already evaluated on all 1041 pairs. Not an official benchmark split.'})
    rel = OUT.relative_to(ROOT).as_posix()
    for model in ('s', 'l'):
        base = json.loads((OUT.parent / f'local_smoke_{model}.json').read_text())
        base['runtime'].pop('stop_after_epoch', None)
        base['runtime']['audit_initial_state'] = True
        base['train'].pop('smoke_steps_per_epoch', None)
        base['train'].update(num_epochs=50, batch_size=32, micro_batch_size=8)
        base['data'].update(fc2_eval_label_clip=None, fc2_strict_loading=True)
        base['eval']['eval_batches'] = 0
        base['eval']['sintel'].update(sintel_list=f'{rel}/monitor_quick.txt', max_samples=None)
        base['eval']['sintel_full_monitor'] = dict(base['eval']['sintel'], sintel_list=f'{rel}/monitor_all.txt', eval_every_epoch=5)
        for variant, clip in [('clip50', 50.0), ('raw', None)]:
            cfg = copy.deepcopy(base)
            cfg['runtime']['experiment_name'] = f'fc2-label-ab-v1-{model}-{variant}-seed42'
            cfg['data']['fc2_train_label_clip'] = clip
            write_json(OUT / f'{model}_{variant}.json', cfg)
            # Engineering runs stop and resume after 50 updates. Never candidate weights.
            smoke = copy.deepcopy(cfg)
            smoke['runtime'].update(experiment_name=f'20260917-label-prep-{model}-{variant}', stop_after_epoch=1)
            smoke['train'].update(num_epochs=2, batch_size=2, micro_batch_size=2, smoke_steps_per_epoch=50)
            smoke['eval']['eval_batches'] = 2
            smoke['eval']['sintel_full_monitor']['eval_every_epoch'] = 2
            write_json(OUT / f'smoke_{model}_{variant}.json', smoke)
            smoke['runtime']['stop_after_epoch'] = 2
            smoke['checkpoint']['load_checkpoint'] = True
            write_json(OUT / f'smoke_{model}_{variant}_resume.json', smoke)
    print(json.dumps(manifests, indent=2))


if __name__ == '__main__':
    main()
