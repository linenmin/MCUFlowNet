################################################################################
# FILE: compute_myscore.py                                                     #
# PURPOSE: Implement the C-SWAG zero-shot proxy (gradient stability +          #
#          expressivity) for TransNAS-Bench-101 models.                        #
#          [FAST VERSION]: Computes stability across mini-batches.             #
#                                                                              #
# NOTE: This file is designed to be API-compatible with compute_zico_score in  #
#       run_zico_transnas.py, so it can be dropped in as an alternative proxy. #
#                                                                              #
# AUTHOR: Your Name / (fill in your affiliation)                               #
# CREATION DATE: November 30, 2025                                             #
# VERSION: 2.1 (Top-K Channel Expressivity)                                    #
################################################################################

import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
import numpy as np  # 导入 NumPy 做统计计算
import math  # 导入数学库（主要用 log）


def compute_myscore_score(model, train_batches, loss_fn, device):
    """C-SWAG 代理分数计算入口函数（快速版：基于 Batch 间稳定性 + Top-K 通道表达能力）。"""  # 函数说明

    # === 超参数配置 ===
    ALPHA_THRESHOLD = 0.5  # 关键层筛选阈值中的超参数 alpha
    TOP_K_PERCENT = 0.30  # Top-K 截断比例（层内保留前 30% 稳定性分数/通道）
    EPSILON = 1e-8  # 数值稳定项，避免除零

    # === 模型准备 ===
    model = model.to(device)  # 把模型移动到指定设备
    model.train()  # 保持训练模式
    model.zero_grad()  # 清空梯度

    # ------------------------------------------------------------------
    # 第 1 步：遍历 Batch，收集每个 Batch 的平均梯度 和 特征图
    # ------------------------------------------------------------------
    raw_grads = {}  # 存放每层的梯度列表 {层名: [Tensor(P), ...]}
    batch_features = {} # 存放每层的特征图列表 {层名: [Tensor(B, C, H, W), ...]}

    # 注册 Forward Hook 用于捕获特征图
    features_dict = {}
    def get_feature_hook(name):
        def hook(module, inputs_hook, output_hook):
            features_dict[name] = output_hook.detach() # 捕获特征图，detached
        return hook

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            hooks.append(mod.register_forward_hook(get_feature_hook(name)))

    # 遍历 Mini-Batches
    for batch_idx, (data, label) in enumerate(train_batches):
        model.zero_grad()
        data = data.to(device)
        label = label.to(device)

        # 前向传播
        try:
            logits = model(data)
            loss = loss_fn(logits, label) # 计算 Batch Mean Loss
        except Exception as e:
            print(f"[Compute MyScore] Forward/Loss Error: {e}")
            for h in hooks: h.remove()
            return 0.0

        # 反向传播 (计算 Batch 平均梯度)
        loss.backward() 

        # 1.1 收集梯度 (用于计算 Stability)
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)) and mod.weight.grad is not None:
                g = mod.weight.grad.detach().reshape(-1) # 展平梯度
                if name not in raw_grads:
                    raw_grads[name] = []
                raw_grads[name].append(g.cpu()) # 移至 CPU 节省显存

        # 1.2 收集特征图 (暂不计算 Psi，留到后面根据 Top-K 索引来算)
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)) and name in features_dict:
                fmap = features_dict[name] # (B, C, H, W) 或 (B, C)
                if name not in batch_features:
                    batch_features[name] = []
                # 为了节省内存，可以先把 fmap 转到 CPU
                batch_features[name].append(fmap.cpu())

    # 移除 Hooks
    for h in hooks: h.remove()

    # 检查是否有梯度
    if not raw_grads:
        return 0.0
    
    # 检查 Batch 数量 (至少需要 2 个来算 std)
    num_batches_collected = len(next(iter(raw_grads.values())))
    if num_batches_collected < 2:
        return 0.0

    # ------------------------------------------------------------------
    # 第 2 步：计算统计量 (Batch 间梯度方差) & 筛选关键层
    # ------------------------------------------------------------------
    layer_stats = {}
    mean_sigmas = []

    for name, grads_list in raw_grads.items():
        # Stack -> (Num_Batches, P)
        grads = torch.stack(grads_list) 
        
        # 2.1 梯度方差 (沿 Batch 维度) -> Sigma_w
        sigma_w = torch.std(grads, dim=0) + EPSILON
        
        # 2.2 稳定性分数 S_w = 1 / Sigma_w
        s_w = 1.0 / sigma_w
        
        # 2.3 层级混乱度 MeanSigma_l
        mean_sigma_l = torch.mean(sigma_w).item()
        
        layer_stats[name] = {
            's_w': s_w,
            'mean_sigma': mean_sigma_l
        }
        mean_sigmas.append(mean_sigma_l)

    if not mean_sigmas:
        return 0.0

    # ------------------------------------------------------------------
    # 第 3 步：纵向筛选 (Critical Layers)
    # ------------------------------------------------------------------
    mus = np.array(mean_sigmas)
    mu_global = np.mean(mus)
    delta_global = np.std(mus)
    threshold = mu_global + ALPHA_THRESHOLD * delta_global

    critical_layers = [
        name for name, stats in layer_stats.items() if stats['mean_sigma'] > threshold
    ]

    if not critical_layers:
        return 1e-6

    # ------------------------------------------------------------------
    # 第 4 步：Top-K 聚合 & Top-K 通道表达能力计算
    # ------------------------------------------------------------------
    c_swag_score = 0.0

    for name in critical_layers:
        stats = layer_stats[name]
        s_w = stats['s_w'].to(device) # (P,) 展平后的参数稳定性分数
        
        # 4.1 Top-K 稳定性聚合
        num_params = s_w.numel()
        k_count = max(1, int(num_params * TOP_K_PERCENT))
        
        # 获取 Top-K 的值和索引
        sorted_s, sorted_indices = torch.sort(s_w, descending=True)
        top_k_s = sorted_s[:k_count]
        top_k_indices = sorted_indices[:k_count]
        
        base_score_l = torch.sum(top_k_s).item() # 基础稳定性得分

        # 4.2 计算 Top-K 通道的表达能力 (Psi)
        # 注意：参数索引是展平后的 (Out, In, K, K)。
        # 我们需要将其映射回 (Out_Channel) 维度，因为特征图是 (B, Out_Channel, H, W)。
        # 策略：如果一个 Out_Channel 对应的卷积核参数中有任意一个入选 Top-K，
        #       则认为该通道是“稳定通道”，纳入表达能力计算。
        
        psi_l = 1.0
        if name in batch_features:
            # 取出该层所有 Batch 的特征图
            fmaps = batch_features[name] # list of Tensors
            
            # 确定哪些 Channel 是 Top-K 相关的

            
            # 优化：在第一步收集梯度时，顺便记录 weight shape。
            # 这里为了不改动太大，我们用一个辅助查找：
            module = dict(model.named_modules())[name]
            weight_shape = module.weight.shape
            out_channels = weight_shape[0]
            params_per_channel = num_params // out_channels # 每个 filter 的参数量
            
            # 将 Top-K 的 parameter indices 转换为 channel indices
            # index // params_per_channel 即为 channel index
            top_k_channel_indices = torch.unique(top_k_indices // params_per_channel)
            
            # 计算这些 Top-K Channel 的平均表达能力
            psi_sum = 0.0
            valid_batches = 0
            
            for fmap in fmaps: # 遍历每个 Batch
                fmap = fmap.to(device) # (B, C, H, W)
                
                # 只选出 Top-K Channels
                # fmap index select
                selected_fmap = torch.index_select(fmap, 1, top_k_channel_indices)
                
                if selected_fmap.dim() == 4:  # 只有 Conv2d 特征图才做空间池化
                    H, W = selected_fmap.shape[2], selected_fmap.shape[3]
                    if H > 16 or W > 16:  # 只有当尺寸大于 16x16 时才缩放
                        selected_fmap = torch.nn.functional.adaptive_avg_pool2d(selected_fmap, (16, 16))
                

                # 二值化
                bin_map = (selected_fmap > 0).float()
                
                # 展平计算唯一编码
                B = bin_map.size(0)
                flat_codes = bin_map.view(B, -1)
                unique_codes = torch.unique(flat_codes, dim=0)
                psi_sum += float(unique_codes.size(0))
                valid_batches += 1
            
            if valid_batches > 0:
                psi_l = psi_sum / valid_batches

        # 4.3 最终层得分
        final_layer_score_l = base_score_l * psi_l

        if final_layer_score_l > 1e-9:
            c_swag_score += math.log(final_layer_score_l)

    return c_swag_score
