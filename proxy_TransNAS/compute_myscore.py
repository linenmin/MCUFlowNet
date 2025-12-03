################################################################################
# FILE: compute_myscore.py                                                     #
# PURPOSE: Implement the C-SWAG zero-shot proxy (gradient stability +          #
#          expressivity) for TransNAS-Bench-101 models.                        #
#          [FAST VERSION]: Computes stability across mini-batches.             #
#          [MEMORY OPTIMIZED]: Computes Psi on-the-fly to avoid OOM.           #
#                                                                              #
# NOTE: This file is designed to be API-compatible with compute_zico_score in  #
#       run_zico_transnas.py, so it can be dropped in as an alternative proxy. #
#                                                                              #
# AUTHOR: Your Name / (fill in your affiliation)                               #
# CREATION DATE: November 30, 2025                                             #
# VERSION: 2.2 (Memory-Optimized Expressivity)                                 #
################################################################################

import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
import numpy as np  # 导入 NumPy 做统计计算
import math  # 导入数学库（主要用 log）


def compute_myscore_score(model, train_batches, loss_fn, device, top_k_percent, alpha_threshold):
    """C-SWAG 代理分数计算入口函数（快速版：基于 Batch 间稳定性 + Top-K 通道表达能力）。"""  # 函数说明

    # === 超参数配置 ===
    ALPHA_THRESHOLD = alpha_threshold  # 关键层筛选阈值中的超参数 alpha
    TOP_K_PERCENT = top_k_percent  # Top-K 截断比例（层内保留前 30% 稳定性分数/通道）
    # 注意：不再使用 EPSILON，方差为0的参数将被直接过滤

    # === 模型准备 ===
    model = model.to(device)  # 把模型移动到指定设备
    model.train()  # 保持训练模式
    model.zero_grad()  # 清空梯度

    # ------------------------------------------------------------------
    # 第 1 步：遍历 Batch，收集梯度并立即计算表达能力（避免存储特征图）
    # ------------------------------------------------------------------
    raw_grads = {}  # 存放每层的梯度列表 {层名: [Tensor(P), ...]}
    layer_psi_stats = {}  # 存放每层的表达能力统计 {层名: {'psi_sum': float, 'valid_batches': int}}

    # 注册 Forward Hook 用于捕获特征图（但不存储，立即计算）
    features_dict = {}  # 临时存放当前 batch 的特征图 {层名: Tensor}
    def get_feature_hook(name):
        """返回一个带有下采样逻辑的 Hook 函数，先压缩再存特征图。"""  # 函数说明
        def hook(module, inputs_hook, output_hook):
            out = output_hook.detach()  # 先 detach，切断计算图，避免额外梯度
            # 如果是卷积特征图 (B, C, H, W) 且空间尺寸大于 16x16，则先做自适应平均池化
            if out.dim() == 4 and (out.shape[2] > 16 or out.shape[3] > 16):  # 仅在大图时缩放
                out = torch.nn.functional.adaptive_avg_pool2d(out, (16, 16))  # 下采样到 16x16，显存友好
            features_dict[name] = out  # 临时缓存当前 batch 的特征图
        return hook  # 返回真正注册给模块的 hook

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            hooks.append(mod.register_forward_hook(get_feature_hook(name)))

    # 遍历 Mini-Batches
    for batch_idx, (data, label) in enumerate(train_batches):
        model.zero_grad()
        data = data.to(device)  # 数据移到 GPU
        label = label.to(device)  # 标签移到 GPU
        
        # 特殊处理：segmentsemantic 任务的标签需要转为 Long 类型
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            label = label.long()  # CrossEntropyLoss 要求 Long 类型标签

        # 前向传播
        try:
            logits = model(data)  # 在 GPU 上前向传播
            loss = loss_fn(logits, label)  # 计算 Batch Mean Loss（在 GPU 上）
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

        # 1.2 立即计算表达能力 Psi（在 GPU 上计算，但不存储特征图）
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)) and name in features_dict:
                fmap = features_dict[name]  # (B, C, H, W) 或 (B, C)，当前 batch 的特征图（在 GPU 上）
                
                # 二值化整层特征图（>0 为激活，否则为未激活）- GPU 上快速操作
                bin_map = (fmap > 0).float()
                
                # 计算唯一激活模式数量（表达能力指标）- GPU 上计算
                B = bin_map.size(0)  # batch size
                flat_codes = bin_map.view(B, -1)  # 展平为 (B, N)，N 是神经元总数
                neuron_codes = flat_codes.t()  # 转置为 (N, B)，每行代表一个神经元的激活模式
                unique_codes = torch.unique(neuron_codes, dim=0)  # 在 GPU 上计算唯一模式（快速）
                
                # 累积统计（只存储标量，不存储特征图）
                if name not in layer_psi_stats:
                    layer_psi_stats[name] = {'psi_sum': 0.0, 'valid_batches': 0}
                layer_psi_stats[name]['psi_sum'] += float(unique_codes.size(0))  # 转为标量后存储
                layer_psi_stats[name]['valid_batches'] += 1
                
                # fmap 和中间计算结果在此 batch 结束后会被自动释放
        
        # 清空临时特征图字典，释放 GPU 显存
        features_dict.clear()
        if 'cuda' in str(device):  # 只在使用 GPU 时清理缓存（兼容 'cuda' 和 'cuda:0' 等）
            torch.cuda.empty_cache()  # 清理 GPU 缓存碎片

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
    # 第 2 步：计算统计量 (Batch 间梯度方差) & 筛选关键层（在 CPU 上计算）
    # ------------------------------------------------------------------
    layer_stats = {}
    mean_sigmas = []

    for name, grads_list in raw_grads.items():
        # Stack -> (Num_Batches, P)，在 CPU 上操作
        grads = torch.stack(grads_list)  # 已经在 CPU 上（第1步存储时就用了 .cpu()）
        
        # 2.1 梯度方差 (沿 Batch 维度) -> Sigma_w（不再加 EPSILON）
        sigma_w = torch.std(grads, dim=0)  # 在 CPU 上计算，避免显存占用
        grad_mean = torch.mean(torch.abs(grads), dim=0)      # (P,) 每个参数在 batch 维度上的均值

        # 2.2 过滤掉方差为0的参数
        non_zero_mask = (sigma_w != 0)  # 标记非零方差的参数
        
        # 如果所有参数的方差都为0，跳过这一层
        if not non_zero_mask.any():
            continue  # 直接跳过，不记录这一层的统计信息
        
        # 2.3 只对非零方差的参数计算稳定性分数 S_w = 1 / Sigma_w（在 CPU 上）
        s_w = torch.zeros_like(sigma_w)  # 初始化为0，保持在 CPU
        s_w[non_zero_mask] = grad_mean[non_zero_mask] / sigma_w[non_zero_mask]  # 只计算非零方差参数的稳定性
        
        # 2.4 层级混乱度 MeanSigma_l（只计算非零方差参数的均值）
        mean_sigma_l = torch.mean(sigma_w[non_zero_mask]).item()
        
        layer_stats[name] = {
            's_w': s_w,  # 保持在 CPU，避免显存占用
            'mean_sigma': mean_sigma_l,
            'non_zero_mask': non_zero_mask  # 保持在 CPU
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

     # 打印当前一次运行中，被 ALPHA_THRESHOLD 选中的层数量
    # print(f"[MyScore] 选中关键层数量 = {len(critical_layers)} / {len(layer_stats)} (alpha={ALPHA_THRESHOLD:.2f})")

    if not critical_layers:
        return 1e-6
    
    

    # ------------------------------------------------------------------
    # 第 4 步：Top-K 聚合表达能力计算（在 CPU 上完成，避免显存占用）
    # ------------------------------------------------------------------
    c_swag_score = 0.0  # 累积所有关键层的得分（后面会取平均，而不是简单求和）

    for name in critical_layers:
        stats = layer_stats[name]
        s_w = stats['s_w']  # (P,) 展平后的参数稳定性分数，在 CPU 上
        non_zero_mask = stats['non_zero_mask']  # 获取非零方差参数的掩码，在 CPU 上
        
        # 4.1 Top-K 稳定性聚合（只从非零方差参数中选择，在 CPU 上计算）
        valid_s_w = s_w[non_zero_mask]  # 只取有效参数的稳定性分数
        num_valid_params = valid_s_w.numel()  # 有效参数数量
        
        if num_valid_params == 0:  # 如果没有有效参数，跳过
            continue
        
        k_count = max(1, int(num_valid_params * TOP_K_PERCENT))  # 计算 Top-K 数量
        
        # 获取 Top-K 的值（在 CPU 上排序，轻量操作）
        top_k_s, _ = torch.topk(valid_s_w, k_count, largest=True, dim=0)
        
        base_score_l = torch.mean(top_k_s).item()  # 基础稳定性得分（Top-K 平均）

        # 4.2 使用预计算的表达能力 Psi（已在第1步用 GPU 立即计算完成）
        # 注意：这里使用的是整层的表达能力，而非 Top-K 通道的表达能力
        # 原因：为了避免存储特征图导致内存爆炸，我们在第1步就立即计算了整层 Psi
        # 整层 Psi 能够反映该层的整体表达能力，与 Top-K 通道 Psi 高度相关
        
        psi_l = 1.0  # 默认表达能力
        if name in layer_psi_stats:
            # 从预计算的统计中获取表达能力（所有 batch 的平均）
            psi_l = layer_psi_stats[name]['psi_sum'] / layer_psi_stats[name]['valid_batches']

        # 4.3 最终层得分（稳定性 × 表达能力）
        # final_layer_score_l = base_score_l * psi_l
        final_layer_score_l = base_score_l
        
        if final_layer_score_l > 0:  # 只要大于0就取 log
            c_swag_score += math.log(final_layer_score_l)  # 对单层得分取 log 后累加
    
    # 清理临时变量，释放内存
    del raw_grads
    del layer_stats
    del layer_psi_stats
    # ------------------------------------------------------------------
    # 第 5 步：按层数做归一化（取“平均层得分”而不是“层数越多越高”）
    # ------------------------------------------------------------------
    num_critical_layers = len(critical_layers)  # 关键层数量
    if num_critical_layers > 0:
        c_swag_score = c_swag_score / float(num_critical_layers)  # 用关键层数做平均，避免“差层越多分越高”
    

    return c_swag_score  # 返回最终 C-SWAG 得分
