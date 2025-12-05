import warnings  # 警告控制
import torch  # 深度学习框架
from fvcore.nn import FlopCountAnalysis  # FLOPs 工具


def compute_flops(model, train_batches, device):  # 计算 FLOPs
    model.eval()  # 评估模式
    model.to(device)  # 移动设备
    data, _ = train_batches[0]  # 取首个 batch
    input_tensor = data[:1].to(device)  # 只用一个样本
    try:  # 捕获异常
        with torch.no_grad():  # 关闭梯度
            with warnings.catch_warnings():  # 捕获警告
                warnings.filterwarnings("ignore", message=".*unsupported.*")  # 忽略警告
                warnings.filterwarnings("ignore", message=".*Unsupported.*")  # 忽略警告
                warnings.filterwarnings("ignore", message=".*were never called.*")  # 忽略警告
                flop_analyzer = FlopCountAnalysis(model, input_tensor)  # 建立分析器
                flop_analyzer.unsupported_ops_warnings(False)  # 关闭不支持警告
                flop_analyzer.uncalled_modules_warnings(False)  # 关闭未调用警告
                total_flops = flop_analyzer.total()  # 获取总 FLOPs
        return float(total_flops)  # 返回浮点
    except Exception as e:  # 异常处理
        print(f"!!! FLOPs 计算失败: {e}")  # 打印错误
        return 0.0  # 返回零

