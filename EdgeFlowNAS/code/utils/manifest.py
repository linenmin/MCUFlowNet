"""训练清单生成工具。"""  # 定义模块用途

import hashlib  # 导入哈希模块
import json  # 导入JSON模块
from datetime import datetime  # 导入时间工具
from typing import Any, Dict  # 导入类型注解


def _hash_config(config: Dict[str, Any]) -> str:  # 定义配置哈希函数
    """计算配置字典的稳定哈希。"""  # 说明函数用途
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")  # 生成稳定字节串
    return hashlib.sha256(payload).hexdigest()  # 返回SHA256哈希


def build_manifest(config: Dict[str, Any], git_commit: str) -> Dict[str, Any]:  # 定义清单构建函数
    """根据配置构建可复现清单。"""  # 说明函数用途
    train_cfg = config.get("train", {})  # 读取训练配置
    runtime_cfg = config.get("runtime", {})  # 读取运行配置
    data_cfg = config.get("data", {})  # 读取数据配置
    return {  # 返回清单字典
        "created_at": datetime.utcnow().isoformat() + "Z",  # 写入UTC创建时间
        "experiment_name": runtime_cfg.get("experiment_name", "unknown"),  # 写入实验名称
        "seed": runtime_cfg.get("seed", 0),  # 写入随机种子
        "git_commit": git_commit,  # 写入提交哈希
        "config_hash": _hash_config(config),  # 写入配置哈希
        "input_shape": [data_cfg.get("input_height", 0), data_cfg.get("input_width", 0)],  # 写入输入尺寸
        "optimizer": train_cfg.get("optimizer", "adam"),  # 写入优化器
        "lr_schedule": train_cfg.get("lr_schedule", "cosine"),  # 写入学习率策略
        "wd": train_cfg.get("weight_decay", 0.0),  # 写入权重衰减
        "grad_clip": train_cfg.get("grad_clip_global_norm", 0.0),  # 写入梯度裁剪
    }

