import copy
import torch


def compute_loss_score(model, train_batches, loss_fn, device, decoder_only: bool = False):
    """
    小步数“微训”后的平均 loss 作为 proxy：
      - 复制模型，避免影响后续 proxy
      - 默认使用 SGD，学习率/步数写死在函数内部，适合少量 batch（由上游 maxbatch 控制）
    """
    if len(train_batches) == 0:
        return 0.0

    # 超参：步数/学习率可按需调整
    max_steps = 10
    lr = 0.05
    weight_decay = 0.0

    # 深拷贝模型，避免对原模型产生副作用
    work_model = copy.deepcopy(model).to(device)
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)

    work_model.train()
    optimizer = torch.optim.SGD(work_model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    steps = min(max_steps, len(train_batches))

    for i in range(steps):
        data, label = train_batches[i]
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = work_model(data)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    # 清理
    del work_model

    if len(losses) == 0:
        return 0.0
    score = float(sum(losses) / len(losses))
    return -score


