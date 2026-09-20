"""Inference-only adaptation of NVlabs PyTorch PWC-Net to current PyTorch.

Correlation implements corr_cuda_kernel.cu: channels enumerate dy then dx,
zero padding, channel mean, displacement [-4,4], kernel and strides one.
grid_sample explicitly preserves the pre-1.3 align_corners=True convention.
No upstream file or learned parameter is changed.
"""
import types
import torch


class Correlation(torch.nn.Module):
    def __init__(self, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply):
        super().__init__()
        assert (pad_size,kernel_size,max_displacement,stride1,stride2,corr_multiply)==(4,1,4,1,1,1)

    def forward(self, a, b):
        h,w = a.shape[-2:]
        padded = torch.nn.functional.pad(b, [4,4,4,4])
        return torch.cat([(a*padded[:,:,y:y+h,x:x+w]).mean(1,keepdim=True)
                          for y in range(9) for x in range(9)],dim=1)


def load_pwc(path, state):
    source = path.read_text()
    source = source.replace('from correlation_package.modules.corr import Correlation', '')
    source = source.replace('nn.init.kaiming_normal(', 'nn.init.kaiming_normal_(')
    source = source.replace('nn.functional.grid_sample(x, vgrid)', 'nn.functional.grid_sample(x, vgrid, align_corners=True)')
    source = source.replace('nn.functional.grid_sample(mask, vgrid)', 'nn.functional.grid_sample(mask, vgrid, align_corners=True)')
    module = types.ModuleType('upstream_pwc_compat')
    module.__dict__['Correlation'] = Correlation
    exec(compile(source,str(path),'exec'),module.__dict__)
    model = module.PWCDCNet()
    model.load_state_dict(state.get('state_dict',state),strict=True)
    return model
