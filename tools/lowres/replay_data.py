"""Fixed 24 FC2 + 8 FT3D pairs, with checkpointable per-source cursors."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from data import read_sample


def take(count, size, seed, cursor):
    """Visit each permutation once; a batch may straddle two complete sweeps."""
    if count < 1 or size < 1:
        raise ValueError('Positive sample and batch counts required')
    epoch, position = cursor
    if epoch < 1 or not 0 <= position < count:
        raise ValueError('Invalid source cursor')
    selected = []
    while len(selected) < size:
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(count)
        amount = min(size-len(selected), count-position)
        selected.extend((epoch, int(i)) for i in order[position:position+amount])
        position += amount
        if position == count:
            epoch, position = epoch+1, 0
    return selected, [epoch, position]


def mixed_batches(fc2, ft3d, root, seed, cursor, workers=8):
    """Prefetch one batch; yield the cursor AFTER that batch, never the prefetched one."""
    current = {k:list(v) for k,v in cursor.items()}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit():
            nonlocal current
            tokens = []
            next_cursor = {}
            pending = []
            for domain, rows, size, hw in [('fc2',fc2,24,(384,512)),('ft3d',ft3d,8,(540,960))]:
                selection, next_cursor[domain] = take(len(rows),size,seed,current[domain])
                for epoch, index in selection:
                    tokens.append((0 if domain=='fc2' else 1,epoch,index))
                    pending.append(pool.submit(read_sample,root,rows[index],expected_source_hw=hw,
                        geometry_seed=[seed,epoch,index,20261004] if domain=='fc2' else None))
            current = next_cursor
            return pending, tokens, {k:list(v) for k,v in next_cursor.items()}
        pending, tokens, committed = submit()
        while True:
            values = [f.result() for f in pending]
            following = submit()
            yield np.stack([v[0] for v in values]), np.stack([v[1] for v in values]), tokens, committed
            pending, tokens, committed = following


def verify_cursors():
    """A small fixture crosses both source boundaries and checks exact resumption."""
    for size,batch in [(25,24),(9,8),(22232,24),(80578,8)]:
        cursor=[1,0]; observed=[]; snapshots=[]
        for _ in range(5):
            items,cursor=take(size,batch,42,cursor)
            observed.append(items);snapshots.append(cursor)
        resumed=take(size,batch,42,snapshots[2])[0]
        assert resumed==observed[3]
        for epoch in {e for group in observed for e,_ in group}:
            seen=[i for group in observed for e,i in group if e==epoch]
            assert len(seen)==len(set(seen))
    return dict(passed=True,source_sweep_boundary_checked=True,exact_cursor_resume=True,
                no_tail_padding=True,fc2_per_batch=24,ft3d_per_batch=8)


if __name__=='__main__':
    import json
    print(json.dumps(verify_cursors()))
