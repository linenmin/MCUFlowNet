"""Shared raw-GT Sintel evaluation. Upstream sources stay untouched."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_flow(path):
    with path.open('rb') as f:
        assert f.read(4) == b'PIEH', path
        w, h = np.fromfile(f, '<i4', 2)
        flow = np.fromfile(f, '<f4')
    assert flow.size == h*w*2 and np.isfinite(flow).all(), path
    return flow.reshape(h, w, 2)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def torch_model(args):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    assert torch.cuda.is_available()
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    def tensor(image):
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().cuda()
    if args.model == 'raft':
        sys.path.insert(0, str(args.upstream / 'RAFT/core'))
        from raft import RAFT
        from utils.utils import InputPadder
        class Config(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
        model = RAFT(Config(small=False, mixed_precision=False, alternate_corr=False, dropout=0))
        model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        model.cuda().eval()
        def predict(a, b):
            a, b = tensor(a[:, :, ::-1])[None], tensor(b[:, :, ::-1])[None]
            padder = InputPadder(a.shape)
            a, b = padder.pad(a, b)
            flow = padder.unpad(model(a, b, iters=32, test_mode=True)[1])[0]
            return flow.permute(1, 2, 0).cpu().numpy()
    elif args.model == 'spynet':
        saved = sys.argv
        sys.argv = [saved[0]]
        try:
            module = load_module('upstream_spynet', args.upstream/'pytorch-spynet/run.py')
        finally:
            sys.argv = saved
        original = torch.hub.load_state_dict_from_url
        torch.hub.load_state_dict_from_url = lambda **kw: state
        try:
            model = module.Network().cuda().eval()
        finally:
            torch.hub.load_state_dict_from_url = original
        module.netNetwork = model
        def predict(a, b):
            return module.estimate(tensor(a)/255, tensor(b)/255).permute(1, 2, 0).numpy()
    elif args.model == 'pwc':
        from pwc_compat import load_pwc
        model = load_pwc(args.upstream/'PWC-Net/PyTorch/models/PWCNet.py', state).cuda().eval()
        def predict(a, b):
            h, w = a.shape[:2]
            hh, ww = math.ceil(h/64)*64, math.ceil(w/64)*64
            pair = torch.cat([tensor(cv2.resize(x, (ww, hh)))/255 for x in (a, b)], 0)[None]
            low = (model(pair)[0]*20).permute(1, 2, 0).cpu().numpy()
            flow = cv2.resize(low, (w, h))
            flow *= np.array([w/ww, h/hh], np.float32)
            return flow
    else:
        raise ValueError(args.model)
    return predict, dict(framework=torch.__version__, cuda=torch.version.cuda,
                        device=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['raft', 'spynet', 'pwc', 'edge', 'edge-chunks', 'nano'], required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--nano-native', action='store_true', help='NanoFlowNet 112x160 input; bilinear output to common grid; source-pixel vector units')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = sorted((args.dataset/'training/flow').glob('*/*.flo'))
    assert len(files) == 1041, len(files)
    if args.limit:
        files = files[:args.limit]
    manifest = dict(model=args.model, weights=str(args.weights), command=sys.argv,
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    status='running', start=time.time(), samples_expected=len(files),
                    git_status=subprocess.check_output(['git','status','--short'],text=True).strip(),
                    protocol='Sintel training Final; center crop rows 10:426, width1024; all pixels; raw GT primary; no prediction clipping',
                    weights_sha256={x.name: sha(x) for x in ([args.weights] if args.weights.is_file() else args.weights.parent.glob(args.weights.name+'.*'))})
    def save():
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    try:
        if args.model in ('raft','spynet','pwc'):
            predict, info = torch_model(args)
        else:
            from tf_adapters import tf_model
            predict, info = tf_model(args)
        manifest['environment'] = info
        save()
        rows = []
        with (args.output/'samples.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample','raw_epe','legacy_epe','seconds'])
            writer.writeheader()
            for i, path in enumerate(files):
                base = args.dataset/'training/final'/path.parent.name
                a = cv2.imread(str(base/(path.stem+'.png')))
                b = cv2.imread(str(base/f'frame_{int(path.stem.split("_")[1])+1:04d}.png'))
                gt = read_flow(path)
                assert a is not None and b is not None
                assert a.shape == b.shape == (436,1024,3) and gt.shape == (436,1024,2)
                a,b,gt = a[10:426],b[10:426],gt[10:426]
                start = time.perf_counter()
                pred = np.asarray(predict(a,b), dtype=np.float32)
                assert pred.shape == gt.shape and np.isfinite(pred).all(), (path,pred.shape)
                row = dict(sample=str(path.relative_to(args.dataset)).replace('\\','/'),
                           raw_epe=float(np.linalg.norm(pred-gt,axis=-1).mean(dtype=np.float64)),
                           legacy_epe=float(np.linalg.norm(pred-np.clip(gt,-50,50),axis=-1).mean(dtype=np.float64)),
                           seconds=time.perf_counter()-start)
                writer.writerow(row)
                f.flush()
                rows.append(row)
                if i == 0:
                    np.save(args.output/'first_prediction.npy',pred)
                if i == 0 or (i+1)%50 == 0:
                    print(f'{i+1}/{len(files)} raw={np.mean([r["raw_epe"] for r in rows]):.6f} elapsed={time.time()-manifest["start"]:.1f}s',flush=True)
        result = dict(samples=len(rows), raw_epe=float(np.mean([r['raw_epe'] for r in rows])),
                      legacy_epe=float(np.mean([r['legacy_epe'] for r in rows])))
        manifest.update(status='completed',results=result)
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(result,flush=True)
    except BaseException as e:
        manifest.update(status='failed',error=repr(e))
        raise
    finally:
        manifest['elapsed_seconds']=time.time()-manifest['start']
        save()


if __name__ == '__main__':
    main()
