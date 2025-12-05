import torch  # 张量计算
import torch.nn as nn  # 网络模块
import numpy as np  # 数值计算


def _count_parameters(model: nn.Module):  # 统计参数量
    return sum(p.numel() for p in model.parameters())  # 总参数个数


def _cal_regular_factor(model: nn.Module, mu: float, sigma: float):  # 计算正则因子
    params = torch.as_tensor(_count_parameters(model), dtype=torch.float32)  # 参数量
    return torch.exp(-((params - mu) ** 2) / sigma)  # 高斯型权重


class _SampleWiseActivationPatterns:  # 按样本收集激活模式
    def __init__(self, device):  # 初始化
        self.device = device  # 设备
        self.activations = None  # 激活缓存
        self.swap = -1  # SWAP 数值

    @torch.no_grad()
    def collect(self, activations: torch.Tensor):  # 收集激活
        n_sample, n_neuron = activations.shape  # 取维度
        if self.activations is None:  # 首次创建
            self.activations = torch.zeros(n_sample, n_neuron, device=self.device)  # 分配缓存
        self.activations = torch.sign(activations)  # 转为符号（-1/0/1）

    @torch.no_grad()
    def compute(self, regular_factor: float):  # 计算 SWAP
        self.activations = self.activations.t()  # 转置为 (neurons, samples)
        self.swap = torch.unique(self.activations, dim=0).size(0)  # 统计唯一行数
        del self.activations  # 释放
        self.activations = None  # 置空
        torch.cuda.empty_cache()  # 清空显存碎片
        return float(self.swap * regular_factor)  # 返回加权 SWAP


class SWAP:  # SWAP 主逻辑
    def __init__(self, model: nn.Module, inputs: torch.Tensor, device: str = "cuda", seed: int = 0,
                 regular: bool = False, mu: float = None, sigma: float = None):  # 初始化
        self.model = model  # 模型
        self.inputs = inputs  # 输入批次
        self.device = device  # 设备
        self.seed = seed  # 随机种子
        self.inter_features = []  # 中间特征缓存
        self.swap_helper = _SampleWiseActivationPatterns(device)  # 激活收集器
        self.regular_factor = 1.0  # 正则系数
        if regular and mu is not None and sigma is not None:  # 需要正则
            self.regular_factor = _cal_regular_factor(self.model, mu, sigma).item()  # 计算因子
        self._reinit(model, seed)  # 设置钩子与种子

    def _reinit(self, model: nn.Module = None, seed: int = None):  # 重置
        if model is not None:  # 重新挂钩
            self.model = model
            self._register_hook(self.model)
            self.swap_helper = _SampleWiseActivationPatterns(self.device)
        if seed is not None and seed != self.seed:  # 更新种子
            self.seed = seed
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        self.inter_features = []  # 清空特征
        torch.cuda.empty_cache()  # 清显存

    def _register_hook(self, model: nn.Module):  # 注册 ReLU 钩子
        for _, m in model.named_modules():  # 遍历模块
            if isinstance(m, nn.ReLU):  # 仅处理 ReLU
                m.register_forward_hook(self._hook_in_forward)  # 前向钩子

    def _hook_in_forward(self, module, inp, out):  # 前向钩子函数
        if isinstance(inp, tuple) and len(inp[0].size()) == 4:  # 仅记录 4D 特征
            self.inter_features.append(out.detach())  # 缓存输出

    @torch.no_grad()
    def forward(self):  # 计算 SWAP
        self.inter_features = []  # 清空缓存
        self.model.eval()  # 评估模式
        self.model.to(self.device)  # 模型上设备
        _ = self.model(self.inputs.to(self.device))  # 前向一次
        if len(self.inter_features) == 0:  # 无特征直接返回
            return 0.0
        # 拼接所有激活 (B, C*H*W)
        activations = torch.cat([f.view(self.inputs.size(0), -1) for f in self.inter_features], dim=1)  # 展平拼接
        self.swap_helper.collect(activations)  # 收集激活
        score = self.swap_helper.compute(self.regular_factor)  # 计算 SWAP
        return score  # 返回结果


def compute_swap_score(model: nn.Module, train_batches, device: str, regular=False, mu=None, sigma=None):  # 入口函数
    if len(train_batches) == 0:  # 无数据
        return 0.0  # 返回零
    data, _ = train_batches[0]  # 取首个 batch
    inputs = data.to(device)  # 数据上设备
    swap = SWAP(model=model, inputs=inputs, device=device, seed=0, regular=regular, mu=mu, sigma=sigma)  # 创建 SWAP
    return swap.forward()  # 计算并返回分数

