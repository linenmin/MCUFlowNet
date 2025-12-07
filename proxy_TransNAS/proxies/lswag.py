import numpy as np  # 数值计算
import torch  # 深度学习框架
import torch.nn as nn  # 模块类型判断

from .zico import getgrad_safe  # 复用安全梯度收集


def compute_lswag_score(model: torch.nn.Module, train_batches, loss_fn, device: str, decoder_only: bool = False):
    """
    逐层 ZiCo 与逐层 SWAP 相乘并求和，返回单个标量。
    - ZiCo: 同 zico.py 按层收集梯度，计算 log(NSR)
    - SWAP: 按层统计符号模式唯一行数 (neurons, batch)
    - 最终：对齐层名后逐层相乘，再对非零/非 NaN 项求和
    """
    model = model.to(device)
    model.train()
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)

    # 先收集 SWAP (逐层，参数层+其后激活配对；用参数层名做 key 对齐 ZiCo)
    swap_records = []

    def make_swap_hook(pname):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                x = out.view(out.size(0), -1)
                sign = torch.sign(x).cpu().t()  # (neurons, batch)
                unique_rows = torch.unique(sign, dim=0).shape[0]
                swap_records.append((pname, int(unique_rows)))  # 用参数层名作为 key
        return hook

    handles = []
    param_layer_name = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if decoder_only and ("decoder" not in str(name)):
                param_layer_name = None
                continue
            param_layer_name = name  # 记录最近的参数层
        elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.GELU, nn.SiLU)):
            if decoder_only and ("decoder" not in str(name)):
                continue
            if param_layer_name is not None:
                handles.append(module.register_forward_hook(make_swap_hook(param_layer_name)))
                param_layer_name = None  # 绑定一次后重置，避免跨 block

    # 前向一次（不算梯度，swap 只需激活）
    data, _ = train_batches[0]
    with torch.no_grad():
        _ = model(data.to(device))

    for h in handles:
        h.remove()

    # 收集 ZiCo (逐层)
    grad_dict = {}
    model.train()
    for i, batch in enumerate(train_batches):
        model.zero_grad()
        data, label = batch
        data = data.to(device)
        label = label.to(device)
        logits = model(data)
        loss = loss_fn(logits, label)
        loss.backward()
        grad_dict = getgrad_safe(model, grad_dict, i, decoder_only=decoder_only)
        model.zero_grad(set_to_none=True)

    zico_records = []
    for name, grads in grad_dict.items():
        arr = np.array(grads)
        grad_std = np.std(arr, axis=0)
        nz = np.nonzero(grad_std)[0]
        if len(nz) == 0:
            zico_records.append((name, np.nan))
            continue
        grad_mean_abs = np.mean(np.abs(arr), axis=0)
        # tmpsum = np.mean(grad_mean_abs[nz] / grad_std[nz])
        tmpsum = np.sum(1 / grad_std[nz])
        zico_records.append((name, float(tmpsum) if tmpsum > 0 else np.nan))

    # 对齐层名并逐层相乘
    swap_dict = dict(swap_records)
    zico_dict = dict(zico_records)
    prod_vals = []
    final_score = 0.0
    for name, zval in zico_dict.items():
        sval = swap_dict.get(name, None)
        if sval is None:
            continue
        if zval is None or np.isnan(zval):
            continue
        product = zval * float(sval)
        if product <= 0:
            continue
        term = np.log(product)
        if np.isfinite(term):
            prod_vals.append(term)
            final_score += term

    # 求和作为单个 proxy 值
    if len(prod_vals) == 0:
        return 0.0
    return float(final_score)


