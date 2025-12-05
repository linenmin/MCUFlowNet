import numpy as np  # 数值计算
import torch  # 深度学习框架


def network_weight_gaussian_init(net: torch.nn.Module):  # 高斯初始化
    with torch.no_grad():  # 关闭梯度
        for m in net.modules():  # 遍历模块
            if isinstance(m, torch.nn.Conv2d):  # 卷积层
                if hasattr(m, "weight") and m.weight is not None:  # 有权重
                    torch.nn.init.normal_(m.weight)  # 正态初始化
                if hasattr(m, "bias") and m.bias is not None:  # 有偏置
                    torch.nn.init.zeros_(m.bias)  # 零初始化
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):  # 归一化
                if hasattr(m, "weight") and m.weight is not None:  # 有权重
                    torch.nn.init.ones_(m.weight)  # 权重置一
                if hasattr(m, "bias") and m.bias is not None:  # 有偏置
                    torch.nn.init.zeros_(m.bias)  # 偏置置零
            elif isinstance(m, torch.nn.Linear):  # 全连接
                if hasattr(m, "weight") and m.weight is not None:  # 有权重
                    torch.nn.init.normal_(m.weight)  # 正态初始化
                if hasattr(m, "bias") and m.bias is not None:  # 有偏置
                    torch.nn.init.zeros_(m.bias)  # 偏置置零
    return net  # 返回网络


def logdet(K):  # 计算对数行列式
    _, ld = np.linalg.slogdet(K)  # slogdet 计算
    return ld  # 返回对数行列式


def compute_naswot_score(model, train_batches, device):  # 计算 NASWOT 分数
    model = model.to(device)  # 模型上设备
    network_weight_gaussian_init(model)  # 初始化权重
    data, _ = train_batches[0]  # 取首个 batch
    input_tensor = data.to(device)  # 数据上设备
    batch_size = input_tensor.size(0)  # batch 大小
    model.K = np.zeros((batch_size, batch_size))  # 初始化核矩阵

    def counting_forward_hook(module, inp, out):  # 前向钩子
        try:  # 捕获异常
            if not module.visited_backwards:  # 未标记
                return  # 直接返回
            if isinstance(inp, tuple):  # 若输入是元组
                inp = inp[0]  # 取首项
            inp = inp.view(inp.size(0), -1)  # 展平输入
            x = (inp > 0).float()  # 二值化
            K = x @ x.t()  # 正激活核
            K2 = (1.0 - x) @ (1.0 - x.t())  # 负激活核
            model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()  # 累加核
        except Exception as err:  # 异常处理
            print("---- NASWOT 计算错误：")  # 提示
            print(model)  # 打印模型
            raise err  # 抛出异常

    for _, module in model.named_modules():  # 遍历模块
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.PReLU)):  # 激活层
            module.visited_backwards = True  # 标记
            module.register_forward_hook(counting_forward_hook)  # 注册钩子

    with torch.no_grad():  # 关闭梯度
        _ = model(input_tensor)  # 前向一次

    score = logdet(model.K)  # 计算分数
    return float(score)  # 返回浮点

