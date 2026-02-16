"""优化器构建占位实现。"""  # 定义模块用途

from typing import Dict  # 导入类型注解


def build_optimizer(train_cfg: Dict) -> Dict:  # 定义优化器构建函数
    """根据训练配置返回优化器描述。"""  # 说明函数用途
    return {  # 返回占位优化器对象
        "name": str(train_cfg.get("optimizer", "adam")).lower(),  # 返回优化器名称
        "lr": float(train_cfg.get("lr", 1e-4)),  # 返回学习率
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),  # 返回权重衰减
    }

