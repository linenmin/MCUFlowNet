import torch  # 深度学习框架
import numpy as np  # 数值计算


class SegmentationLoss(torch.nn.Module):  # 语义分割损失
    def __init__(self):  # 初始化
        super().__init__()  # 调用父类
        self.ce = torch.nn.CrossEntropyLoss()  # 交叉熵

    def forward(self, logits, label):  # 前向计算
        if label.dtype != torch.long:  # 确保类型
            label = label.long()  # 转为 long
        if label.ndim == 4 and label.shape[1] == 1:  # 处理通道维
            label = label.squeeze(1)  # 去掉多余维度
        if logits.ndim == 4 and label.ndim == 3:  # 常规分割
            return self.ce(logits, label)  # 直接计算
        if logits.ndim > 2:  # 其他高维情况
            num_classes = logits.shape[1]  # 类别数
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)  # 展平 logits
            label_flat = label.reshape(-1)  # 展平标签
            return self.ce(logits_flat, label_flat)  # 计算损失
        return self.ce(logits, label)  # 默认返回


def get_loss_fn(task: str):  # 选择损失函数
    if task in ["autoencoder", "normal"]:  # AE 或法线
        return torch.nn.L1Loss()  # L1 损失
    if task == "segmentsemantic":  # 语义分割
        return SegmentationLoss()  # 分割损失
    return torch.nn.CrossEntropyLoss()  # 其他任务


def getgrad_safe(model: torch.nn.Module, grad_dict: dict, step_iter=0):  # 安全取梯度
    if step_iter == 0:  # 首个 batch
        for name, mod in model.named_modules():  # 遍历模块
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):  # 目标层
                if mod.weight.grad is not None:  # 有梯度
                    grad_dict[name] = [mod.weight.grad.data.cpu().reshape(-1).numpy()]  # 记录
    else:  # 后续 batch
        for name, mod in model.named_modules():  # 遍历模块
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):  # 目标层
                if mod.weight.grad is not None:  # 有梯度
                    if name in grad_dict:  # 已存在
                        grad_dict[name].append(mod.weight.grad.data.cpu().reshape(-1).numpy())  # 追加
    return grad_dict  # 返回字典


def caculate_zico_safe(grad_dict):  # 计算 ZiCo
    if not grad_dict:  # 空字典
        return 0.0  # 返回零
    for modname in list(grad_dict.keys()):  # 遍历层
        grad_dict[modname] = np.array(grad_dict[modname])  # 转数组
    nsr_mean_sum_abs = 0  # 累积和
    valid_layer_count = 0  # 有效层计数
    for modname in grad_dict.keys():  # 遍历层
        nsr_std = np.std(grad_dict[modname], axis=0)  # 标准差
        if np.sum(nsr_std) == 0:  # 全零
            continue  # 跳过
        nonzero_idx = np.nonzero(nsr_std)[0]  # 非零索引
        if len(nonzero_idx) == 0:  # 无非零
            continue  # 跳过
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)  # 均值
        tmpsum = np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])  # 比值均值
        if tmpsum != 0:  # 非零
            nsr_mean_sum_abs += np.log(tmpsum)  # 累加对数
            valid_layer_count += 1  # 计数
    if valid_layer_count > 0:  # 有有效层
        return nsr_mean_sum_abs / valid_layer_count  # 返回均值
    return 0.0  # 否则零


def compute_zico_score(model, train_batches, loss_fn, device: torch.device):  # 计算 ZiCo 分数
    grad_dict = {}  # 梯度存储
    model.train()  # 训练模式
    model.to(device)  # 移动设备
    if isinstance(loss_fn, torch.nn.Module):  # 若损失为模块
        loss_fn.to(device)  # 移动损失
    for i, batch in enumerate(train_batches):  # 遍历 batch
        model.zero_grad()  # 清梯度
        data, label = batch  # 拆数据
        data = data.to(device)  # 数据上设备
        label = label.to(device)  # 标签上设备
        logits = model(data)  # 前向
        loss = loss_fn(logits, label)  # 计算损失
        loss.backward()  # 反向
        grad_dict = getgrad_safe(model, grad_dict, i)  # 记录梯度
        model.zero_grad(set_to_none=True)  # 清理
    res = caculate_zico_safe(grad_dict)  # 计算分数
    del grad_dict  # 释放字典
    return res  # 返回分数

