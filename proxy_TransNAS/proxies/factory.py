import numpy as np  # 数值计算
from .zico import compute_zico_score, get_loss_fn  # 导入 ZiCo
from .naswot import compute_naswot_score  # 导入 NASWOT
from .flops import compute_flops  # 导入 FLOPs
from .swap import compute_swap_score  # 导入 SWAP


def compute_proxy_score(model, proxy, train_batches, loss_fn, device, decoder_only: bool = False):  # 统一计算入口
    if proxy == "zico":  # ZiCo
        model.train()  # 训练模式
        return compute_zico_score(model, train_batches, loss_fn, device, decoder_only=decoder_only)  # 返回分数
    if proxy == "naswot":  # NASWOT
        model.eval()  # 评估模式
        return compute_naswot_score(model, train_batches, device, decoder_only=decoder_only)  # 返回分数
    if proxy == "flops":  # FLOPs
        model.eval()  # 评估模式
        score = compute_flops(model, train_batches, device, decoder_only=decoder_only)  # 计算 FLOPs
        return float(np.log(score)) if score > 0 else 0.0  # 转为对数
    if proxy == "swap":  # SWAP
        model.eval()  # 评估模式
        return compute_swap_score(model, train_batches, device, decoder_only=decoder_only)  # 计算 SWAP
    return 0.0  # 未知 proxy

