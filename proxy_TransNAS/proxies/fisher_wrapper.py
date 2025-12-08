import torch
import torch.nn as nn
import types
import torch.nn.functional as F

def fisher_forward_conv2d(self, x):
    x = F.conv2d(
        x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
    )
    self.act = self.dummy(x)
    return self.act


def fisher_forward_linear(self, x):
    x = F.linear(x, self.weight, self.bias)
    self.act = self.dummy(x)
    return self.act


def compute_fisher_score(model, train_batches, loss_fn, device, decoder_only: bool = False, split_data: int = 1):
    """
    计算 Fisher 信息，返回单标量（按所有参数通道求和）。
    decoder_only=True 时，仅累积层名包含 'decoder' 的 Conv/Linear。
    计算流程源自 NASLib 的 fisher 度量，保持计算逻辑不变。
    """
    model = model.to(device)
    model.train()

    # 注册 hook，覆写 forward
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if decoder_only and ("decoder" not in str(name)):
                continue
            layer.fisher = None
            layer.act = 0.0
            layer.dummy = nn.Identity()
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(fisher_forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(fisher_forward_linear, layer)

            def hook_factory(layer_ref):
                def hook(module, grad_input, grad_output):
                    act = layer_ref.act.detach()
                    grad = grad_output[0].detach()
                    if len(act.shape) > 2:
                        g_nk = torch.sum((act * grad), list(range(2, len(act.shape))))
                    else:
                        g_nk = act * grad
                    del_k = g_nk.pow(2).mean(0).mul(0.5)
                    if layer_ref.fisher is None:
                        layer_ref.fisher = del_k
                    else:
                        layer_ref.fisher += del_k
                    del layer_ref.act
                return hook

            layer.dummy.register_full_backward_hook(hook_factory(layer))

    # 取首个 batch（可按 split_data 分块）
    data, label = train_batches[0]
    data = data.to(device)
    label = label.to(device)
    N = data.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        model.zero_grad()
        outputs = model(data[st:en])
        loss = loss_fn(outputs, label[st:en])
        loss.backward()

    # 汇总 Fisher，返回总和
    total = 0.0
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if decoder_only and ("decoder" not in str(name)):
                continue
            if layer.fisher is not None:
                total += float(torch.abs(layer.fisher).sum().item())
    return total


