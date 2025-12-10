import copy  # 用于深拷贝模型，避免修改原模型
import torch  # 深度学习框架


def compute_loss_score(model, train_batches, loss_fn, device, decoder_only: bool = False):
    """在模型拷贝上跑少量步微训练，返回负的平均 loss 作为 proxy（越大越好）。"""
    if len(train_batches) == 0:
        return 0.0  # 无数据则返回 0

    # 微训练超参（可按需调整）
    max_steps = 10  # 最多训练步数
    lr = 0.05  # 学习率
    weight_decay = 0.0  # 权重衰减

    # 复制模型，防止对原模型造成副作用
    work_model = copy.deepcopy(model).to(device)
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)

    work_model.train()  # 训练模式
    optimizer = torch.optim.SGD(work_model.parameters(), lr=lr, weight_decay=weight_decay)  # 简单 SGD

    losses = []  # 记录每步 loss
    steps = min(max_steps, len(train_batches))  # 实际步数不超过可用 batch

    for i in range(steps):
        data, label = train_batches[i]  # 取第 i 个 batch
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad(set_to_none=True)  # 清梯度
        logits = work_model(data)  # 前向
        loss = loss_fn(logits, label)  # 计算 loss
        loss.backward()  # 反向
        optimizer.step()  # 更新一步

        losses.append(float(loss.item()))  # 记录当前 loss

    # 释放拷贝模型
    del work_model

    if len(losses) == 0:
        return 0.0  # 理论上不会走到这里
    # score = float(sum(losses) / len(losses))  # 平均 loss
    score = losses[-1]
    return -score  # 取负，使“越大越好”


