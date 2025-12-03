################################################################################
# FILE: compute_az_nas_score.py
# PURPOSE: 实现 AZ-NAS 零样本代理指标（Expressivity、Progressivity、Trainability）
# AUTHOR: (请填写)
# CREATION DATE: (请填写)
################################################################################

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加父目录到路径
import torch  # 导入 PyTorch
from torch import nn  # 导入神经网络模块
import numpy as np  # 导入 NumPy

# ========== 各种权重初始化方法 ==========

def kaiming_normal_fanin_init(m):
    """Kaiming 正态分布初始化（fan_in 模式）"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):  # 如果是卷积层或全连接层
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # Kaiming 正态初始化权重
        if hasattr(m, 'bias') and m.bias is not None:  # 如果有偏置
            nn.init.zeros_(m.bias)  # 偏置初始化为 0
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # 如果是归一化层
        if m.affine:  # 如果有可学习参数
            nn.init.ones_(m.weight)  # 权重初始化为 1
            nn.init.zeros_(m.bias)  # 偏置初始化为 0

def kaiming_normal_fanout_init(m):
    """Kaiming 正态分布初始化（fan_out 模式）"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):  # 如果是卷积层或全连接层
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 正态初始化权重
        if hasattr(m, 'bias') and m.bias is not None:  # 如果有偏置
            nn.init.zeros_(m.bias)  # 偏置初始化为 0
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # 如果是归一化层
        if m.affine:  # 如果有可学习参数
            nn.init.ones_(m.weight)  # 权重初始化为 1
            nn.init.zeros_(m.bias)  # 偏置初始化为 0

def kaiming_uniform_fanin_init(m):
    """Kaiming 均匀分布初始化（fan_in 模式）"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):  # 如果是卷积层或全连接层
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # Kaiming 均匀初始化权重
        if hasattr(m, 'bias') and m.bias is not None:  # 如果有偏置
            nn.init.zeros_(m.bias)  # 偏置初始化为 0
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # 如果是归一化层
        if m.affine:  # 如果有可学习参数
            nn.init.ones_(m.weight)  # 权重初始化为 1
            nn.init.zeros_(m.bias)  # 偏置初始化为 0

def kaiming_uniform_fanout_init(m):
    """Kaiming 均匀分布初始化（fan_out 模式）"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):  # 如果是卷积层或全连接层
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 均匀初始化权重
        if hasattr(m, 'bias') and m.bias is not None:  # 如果有偏置
            nn.init.zeros_(m.bias)  # 偏置初始化为 0
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # 如果是归一化层
        if m.affine:  # 如果有可学习参数
            nn.init.ones_(m.weight)  # 权重初始化为 1
            nn.init.zeros_(m.bias)  # 偏置初始化为 0

def xavier_normal_init(m):
    """Xavier 正态分布初始化"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):  # 如果是卷积层或全连接层
        nn.init.xavier_normal_(m.weight)  # Xavier 正态初始化权重
        if hasattr(m, 'bias') and m.bias is not None:  # 如果有偏置
            nn.init.zeros_(m.bias)  # 偏置初始化为 0
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # 如果是归一化层
        if m.affine:  # 如果有可学习参数
            nn.init.ones_(m.weight)  # 权重初始化为 1
            nn.init.zeros_(m.bias)  # 偏置初始化为 0

def xavier_uniform_init(m):
    """Xavier 均匀分布初始化"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):  # 如果是卷积层或全连接层
        nn.init.xavier_uniform_(m.weight)  # Xavier 均匀初始化权重
        if hasattr(m, 'bias') and m.bias is not None:  # 如果有偏置
            nn.init.zeros_(m.bias)  # 偏置初始化为 0
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # 如果是归一化层
        if m.affine:  # 如果有可学习参数
            nn.init.ones_(m.weight)  # 权重初始化为 1
            nn.init.zeros_(m.bias)  # 偏置初始化为 0

def plain_normal_init(m):
    """普通正态分布初始化（均值0，标准差0.1）"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):  # 如果是卷积层或全连接层
        nn.init.normal_(m.weight, mean=0.0, std=0.1)  # 正态分布初始化权重
        if hasattr(m, 'bias') and m.bias is not None:  # 如果有偏置
            nn.init.zeros_(m.bias)  # 偏置初始化为 0
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # 如果是归一化层
        if m.affine:  # 如果有可学习参数
            nn.init.ones_(m.weight)  # 权重初始化为 1
            nn.init.zeros_(m.bias)  # 偏置初始化为 0

def plain_uniform_init(m):
    """普通均匀分布初始化（范围 [-0.1, 0.1]）"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):  # 如果是卷积层或全连接层
        nn.init.uniform_(m.weight, a=-0.1, b=0.1)  # 均匀分布初始化权重
        if hasattr(m, 'bias') and m.bias is not None:  # 如果有偏置
            nn.init.zeros_(m.bias)  # 偏置初始化为 0
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # 如果是归一化层
        if m.affine:  # 如果有可学习参数
            nn.init.ones_(m.weight)  # 权重初始化为 1
            nn.init.zeros_(m.bias)  # 偏置初始化为 0

def init_model(model, method='kaiming_norm_fanin'):
    """根据指定方法初始化模型"""
    if method == 'kaiming_norm_fanin':  # Kaiming 正态 fan_in
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':  # Kaiming 正态 fan_out
        model.apply(kaiming_normal_fanout_init)
    elif method == 'kaiming_uni_fanin':  # Kaiming 均匀 fan_in
        model.apply(kaiming_uniform_fanin_init)
    elif method == 'kaiming_uni_fanout':  # Kaiming 均匀 fan_out
        model.apply(kaiming_uniform_fanout_init)
    elif method == 'xavier_norm':  # Xavier 正态
        model.apply(xavier_normal_init)
    elif method == 'xavier_uni':  # Xavier 均匀
        model.apply(xavier_uniform_init)
    elif method == 'plain_norm':  # 普通正态
        model.apply(plain_normal_init)
    elif method == 'plain_uni':  # 普通均匀
        model.apply(plain_uniform_init)
    else:  # 不支持的初始化方法
        raise NotImplementedError
    return model


def compute_nas_score(model, gpu, trainloader, resolution, batch_size, init_method='kaiming_norm_fanin', fp16=False):
    """
    计算 AZ-NAS 零样本代理分数
    
    参数:
        model: 待评估的神经网络模型
        gpu: GPU 设备 ID（None 表示使用 CPU）
        trainloader: 训练数据加载器（None 表示使用随机输入）
        resolution: 输入图像分辨率
        batch_size: 批次大小
        init_method: 权重初始化方法
        fp16: 是否使用半精度浮点数
    
    返回:
        info: 字典，包含 expressivity（表达能力）、progressivity（渐进性）、trainability（可训练性）
    """
    model.train()  # 设置模型为训练模式
    model.cuda()  # 将模型移到 GPU
    info = {}  # 初始化结果字典
    nas_score_list = []  # 初始化分数列表（未使用）
    
    if gpu is not None:  # 如果指定了 GPU
        device = torch.device('cuda:{}'.format(gpu))  # 创建 GPU 设备
    else:  # 否则使用 CPU
        device = torch.device('cpu')  # 创建 CPU 设备

    if fp16:  # 如果使用半精度
        dtype = torch.half  # 设置数据类型为 half
    else:  # 否则使用单精度
        dtype = torch.float32  # 设置数据类型为 float32

    init_model(model, init_method)  # 初始化模型权重

    if trainloader == None:  # 如果没有提供数据加载器
        input_ = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)  # 生成随机输入
    else:  # 否则从数据加载器获取数据
        input_ = next(iter(trainloader))[0].to(device)  # 取第一个 batch 的数据并移到设备
    
    layer_features = model.extract_cell_features(input_)  # 提取每层的特征图（需要模型实现此方法）

    # ========== 第 1 部分：计算表达能力（Expressivity）和渐进性（Progressivity）==========
    expressivity_scores = []  # 存储每层的表达能力分数
    for i in range(len(layer_features)):  # 遍历每一层的特征图
        feat = layer_features[i].detach().clone()  # 复制特征图并切断梯度
        b, c, h, w = feat.size()  # 获取特征图尺寸：batch_size, channels, height, width
        feat = feat.permute(0, 2, 3, 1).contiguous().view(b*h*w, c)  # 重排为 (N, C)，N = b*h*w
        m = feat.mean(dim=0, keepdim=True)  # 计算每个通道的均值 (1, C)
        feat = feat - m  # 中心化特征（减去均值）
        sigma = torch.mm(feat.transpose(1, 0), feat) / (feat.size(0))  # 计算协方差矩阵 (C, C)
        s = torch.linalg.eigvalsh(sigma)  # 计算协方差矩阵的特征值（快速对称矩阵版本）
        prob_s = s / s.sum()  # 归一化特征值，得到概率分布
        score = (-prob_s) * torch.log(prob_s + 1e-8)  # 计算熵（香农熵）
        score = score.sum().item()  # 求和得到该层的表达能力分数
        expressivity_scores.append(score)  # 存储该层分数
    
    expressivity_scores = np.array(expressivity_scores)  # 转为 NumPy 数组
    progressivity = np.min(expressivity_scores[1:] - expressivity_scores[:-1])  # 渐进性 = 相邻层表达能力差的最小值
    expressivity = np.sum(expressivity_scores)  # 表达能力 = 所有层表达能力分数之和
    #####################################################################

    # ========== 第 2 部分：计算可训练性（Trainability）==========
    scores = []  # 存储每对相邻层的可训练性分数
    for i in reversed(range(1, len(layer_features))):  # 从后向前遍历相邻层对（i 和 i-1）
        f_out = layer_features[i]  # 当前层的特征图（输出）
        f_in = layer_features[i-1]  # 前一层的特征图（输入）
        
        if f_out.grad is not None:  # 如果输出特征图有梯度
            f_out.grad.zero_()  # 清空梯度
        if f_in.grad is not None:  # 如果输入特征图有梯度
            f_in.grad.zero_()  # 清空梯度
        
        g_out = torch.ones_like(f_out) * 0.5  # 创建与输出特征图同形状的张量，填充 0.5
        g_out = (torch.bernoulli(g_out) - 0.5) * 2  # 伯努利采样后转为 {-1, +1} 的随机梯度
        g_in = torch.autograd.grad(outputs=f_out, inputs=f_in, grad_outputs=g_out, retain_graph=False)[0]  # 反向传播计算输入梯度
        
        if g_out.size() == g_in.size() and torch.all(g_in == g_out):  # 如果输入梯度等于输出梯度（恒等映射）
            scores.append(-np.inf)  # 这表示该层没有学习能力，分数为负无穷
        else:  # 否则计算梯度雅可比矩阵的奇异值
            if g_out.size(2) != g_in.size(2) or g_out.size(3) != g_in.size(3):  # 如果空间尺寸不匹配
                bo, co, ho, wo = g_out.size()  # 输出梯度尺寸
                bi, ci, hi, wi = g_in.size()  # 输入梯度尺寸
                stride = int(hi / ho)  # 计算步长（下采样因子）
                pixel_unshuffle = nn.PixelUnshuffle(stride)  # 创建反像素重排层
                g_in = pixel_unshuffle(g_in)  # 调整输入梯度的空间尺寸
            
            bo, co, ho, wo = g_out.size()  # 重新获取输出梯度尺寸
            bi, ci, hi, wi = g_in.size()  # 重新获取输入梯度尺寸
            
            ### 原始实现方式（被注释）
            # g_out = g_out.permute(0,2,3,1).contiguous().view(bo*ho*wo,1,co)
            # g_in = g_in.permute(0,2,3,1).contiguous().view(bi*hi*wi,ci,1)
            # mat = torch.bmm(g_in,g_out).mean(dim=0)
            
            ### 高效实现方式（当前使用）
            g_out = g_out.permute(0, 2, 3, 1).contiguous().view(bo*ho*wo, co)  # 重排为 (N_out, C_out)
            g_in = g_in.permute(0, 2, 3, 1).contiguous().view(bi*hi*wi, ci)  # 重排为 (N_in, C_in)
            mat = torch.mm(g_in.transpose(1, 0), g_out) / (bo*ho*wo)  # 计算雅可比矩阵近似 (C_in, C_out)
            
            ### 优化：确保矩阵行数小于列数，加速 SVD 计算
            if mat.size(0) < mat.size(1):  # 如果行数小于列数
                mat = mat.transpose(0, 1)  # 转置矩阵
            ###
            s = torch.linalg.svdvals(mat)  # 计算奇异值（不计算 U 和 V，更快）
            scores.append(-s.max().item() - 1/(s.max().item() + 1e-6) + 2)  # 可训练性分数（基于最大奇异值）
    
    trainability = np.mean(scores)  # 可训练性 = 所有相邻层对分数的平均值
    #################################################

    # ========== 第 3 部分：汇总结果 ==========
    info['expressivity'] = float(expressivity) if not np.isnan(expressivity) else -np.inf  # 表达能力（处理 NaN）
    info['progressivity'] = float(progressivity) if not np.isnan(progressivity) else -np.inf  # 渐进性（处理 NaN）
    info['trainability'] = float(trainability) if not np.isnan(trainability) else -np.inf  # 可训练性（处理 NaN）
    # info['complexity'] = float(model.get_FLOPs(resolution))  # 复杂度（从 API 获取，已注释）
    return info  # 返回结果字典