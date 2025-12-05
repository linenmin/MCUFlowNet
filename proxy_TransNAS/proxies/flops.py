import warnings  # 警告控制
import torch  # 深度学习框架
from fvcore.nn import FlopCountAnalysis  # FLOPs 工具


def compute_flops(model, train_batches, device):  # 计算 FLOPs
    model.eval()  # 评估模式
    model.to(device)  # 移动设备
    data, _ = train_batches[0]  # 取首个 batch
    input_tensor = data[:1].to(device)  # 只用一个样本
    # 注意：不在此捕获 OOM，让上层统一处理并跳过架构，避免返回 0 污染排名
    with torch.no_grad():  # 关闭梯度
        with warnings.catch_warnings():  # 捕获警告
            warnings.filterwarnings("ignore", message=".*unsupported.*")  # 忽略警告
            warnings.filterwarnings("ignore", message=".*Unsupported.*")  # 忽略警告
            warnings.filterwarnings("ignore", message=".*were never called.*")  # 忽略警告
            flop_analyzer = FlopCountAnalysis(model, input_tensor)  # 建立分析器
            flop_analyzer.unsupported_ops_warnings(False)  # 关闭不支持警告
            flop_analyzer.uncalled_modules_warnings(False)  # 关闭未调用模块警告
            total_flops = flop_analyzer.total()  # 获取总 FLOPs
    return float(total_flops)  # 返回浮点

