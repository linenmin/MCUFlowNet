"""Adapters for author checkpoints; shared scoring remains in evaluate.py."""
import json
import math
import sys
import types
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F


class SpatialCorrelationSampler(torch.nn.Module):
    """Exact kernel=1, stride=1, dilation=1 sum correlation, dy-major."""
    def __init__(self, kernel_size, patch_size, stride, padding, dilation):
        super().__init__()
        assert (kernel_size, stride, padding, dilation) == (1, 1, 0, 1)
        self.radius = patch_size // 2

    def forward(self, a, b):
        r = self.radius
        h, w = a.shape[-2:]
        padded = F.pad(b, (r, r, r, r))
        values = [(a * padded[:, :, y:y+h, x:x+w]).sum(1)
                  for y in range(2*r+1) for x in range(2*r+1)]
        return torch.stack(values, 1).reshape(a.shape[0], 2*r+1, 2*r+1, h, w)


def extension_model(args):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    state = None if args.weights.suffix == '.safetensors' else torch.load(args.weights, map_location='cpu', weights_only=True)
    def tensor(a, rgb=True):
        if rgb:
            a = a[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float().cuda()[None]
    def array(flow):
        return flow[0].float().permute(1, 2, 0).cpu().numpy()
    config = {}
    if args.model == 'fastflow':
        module = types.ModuleType('spatial_correlation_sampler')
        module.SpatialCorrelationSampler = SpatialCorrelationSampler
        sys.modules[module.__name__] = module
        sys.path.insert(0, str(args.upstream/'FastFlowNet'))
        from models.FastFlowNet_v2 import FastFlowNet
        model = FastFlowNet().cuda().eval()
        model.load_state_dict(state, strict=True)
        def predict(a, b):
            a, b = tensor(a, False)/255, tensor(b, False)/255
            h, w = a.shape[-2:]
            mean = torch.cat((a,b), 2).flatten(2).mean(2)[:, :, None, None]
            size = (math.ceil(h/64)*64, math.ceil(w/64)*64)
            pair = [F.interpolate(x-mean, size=size, mode='bilinear', align_corners=False) for x in (a,b)]
            flow = 20*F.interpolate(model(torch.cat(pair,1)), size=size, mode='bilinear', align_corners=False)
            flow = F.interpolate(flow, size=(h,w), mode='bilinear', align_corners=False)
            flow[:,0] *= w/size[1]
            flow[:,1] *= h/size[0]
            return array(flow)
        config = dict(input='BGR /255, joint mean subtraction, resize multiple64', correlation='exact torch sum / channels', flow_scale=20)
    elif args.model == 'neuflow2':
        sys.path.insert(0, str(args.upstream/'NeuFlow_v2'))
        from NeuFlow.neuflow import NeuFlow
        model = NeuFlow().cuda().eval()
        model.load_state_dict(state['model'], strict=True)
        model.init_bhwd(1, 416, 1024, torch.device('cuda'), amp=False)
        def predict(a,b):
            return array(model(tensor(a),tensor(b))[-1])
        config = dict(input='RGB 0..255', iters_s16=1, iters_s8=8, precision='FP32; author validation uses AMP')
    elif args.model == 'gmflow':
        sys.path.insert(0, str(args.upstream/'unimatch'))
        from unimatch.unimatch import UniMatch
        model = UniMatch().cuda().eval()
        model.load_state_dict(state['model'], strict=True)
        def predict(a,b):
            return array(model(tensor(a),tensor(b), attn_type='swin', attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], task='flow')['flow_preds'][-1])
        config = dict(input='RGB 0..255', scales=1, attention_splits=[2], refinement=False)
    elif args.model == 'sea-raft':
        sys.path.insert(0, str(args.upstream/'SEA-RAFT/core'))
        from raft import RAFT
        from safetensors.torch import load_file
        config = json.loads((args.upstream/'SEA-RAFT/config/eval/sintel-S.json').read_text())
        # Complete checkpoint follows. Avoid an unnecessary ImageNet download at construction.
        import torchvision.models as tv
        original = tv.resnet18
        tv.resnet18 = lambda **kw: original(weights=None)
        try:
            model = RAFT(SimpleNamespace(**config)).cuda().eval()
        finally:
            tv.resnet18 = original
        # safetensors removes duplicate names for shared BN tensors in BasicBlock.
        state = load_file(str(args.weights))
        target = model.state_dict()
        for key, value in target.items():
            aliases = [k for k, v in target.items() if v.data_ptr() == value.data_ptr() and k in state]
            if key not in state:
                assert aliases, f'Unexplained missing weight: {key}'
                state[key] = state[aliases[0]]
            for alias in aliases:
                assert torch.equal(state[key], state[alias]), (key, alias)
        model.load_state_dict(state, strict=True)
        def predict(a,b):
            return array(model(tensor(a),tensor(b),test_mode=True)['final'])
    elif args.model == 'rapidflow':
        from ptlflow.models.rapidflow.rapidflow import RAPIDFlow
        model = RAPIDFlow().cuda().eval()
        model.load_state_dict(state['state_dict'], strict=True)
        def predict(a,b):
            images = torch.stack((tensor(a,False),tensor(b,False)),1)/255
            return array(model({'images':images})['flows'][:,0])
        config = dict(input='BGR /255 per PTLFlow dataset convention', iters=12, corr_mode='allpairs')
    else:
        raise ValueError(args.model)
    return predict, dict(framework=torch.__version__,cuda=torch.version.cuda,
                        device=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()), settings=config)
