"""早停逻辑实现。"""  # 定义模块用途

from dataclasses import dataclass  # 导入数据类工具


@dataclass  # 使用数据类简化状态定义
class EarlyStopState:  # 定义早停状态数据类
    """保存早停过程状态。"""  # 说明数据类用途
    best_metric: float = float("inf")  # 记录当前最佳指标
    bad_epochs: int = 0  # 记录未改善轮数


def update_early_stop(state: EarlyStopState, metric: float, min_delta: float) -> bool:  # 定义早停更新函数
    """更新早停状态并返回是否触发改善。"""  # 说明函数用途
    if metric < (state.best_metric - min_delta):  # 判断指标是否显著改善
        state.best_metric = metric  # 更新最佳指标
        state.bad_epochs = 0  # 清零未改善计数
        return True  # 返回本轮有改善
    state.bad_epochs += 1  # 增加未改善计数
    return False  # 返回本轮无改善
