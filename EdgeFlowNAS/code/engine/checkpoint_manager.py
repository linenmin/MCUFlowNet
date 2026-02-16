"""Checkpoint 管理工具。"""  # 定义模块用途
from pathlib import Path  # 导入路径工具
from typing import Any, Dict, Optional  # 导入类型注解

import tensorflow as tf  # 导入TensorFlow模块

from code.utils.json_io import read_json, write_json  # 导入JSON读写工具


def build_checkpoint_paths(experiment_dir: str) -> Dict[str, Path]:  # 定义checkpoint路径构建函数
    """构建best和last两个checkpoint前缀路径。"""  # 说明函数用途
    root = Path(experiment_dir) / "checkpoints"  # 计算checkpoint目录路径
    root.mkdir(parents=True, exist_ok=True)  # 确保checkpoint目录存在
    return {"root": root, "best": root / "supernet_best.ckpt", "last": root / "supernet_last.ckpt"}  # 返回路径字典


def _meta_path(path_prefix: Path) -> Path:  # 定义checkpoint元信息路径函数
    """根据checkpoint前缀构建meta文件路径。"""  # 说明函数用途
    return Path(str(path_prefix) + ".meta.json")  # 返回meta文件路径


def checkpoint_exists(path_prefix: Path) -> bool:  # 定义checkpoint存在性检查函数
    """检查checkpoint索引文件是否存在。"""  # 说明函数用途
    return Path(str(path_prefix) + ".index").exists()  # 返回checkpoint存在状态


def find_existing_checkpoint(path_prefix: Path) -> Optional[Path]:  # 定义checkpoint查找函数
    """优先查找指定前缀，其次回退到目录内最新checkpoint。"""  # 说明函数用途
    if checkpoint_exists(path_prefix=path_prefix):  # 判断指定前缀是否存在
        return path_prefix  # 返回指定前缀路径
    latest = tf.train.latest_checkpoint(str(path_prefix.parent))  # 查找目录内最新checkpoint
    if latest:  # 判断是否找到最新checkpoint
        return Path(latest)  # 返回最新checkpoint路径
    return None  # 未找到时返回空值


def load_checkpoint_meta(path_prefix: Path) -> Dict[str, Any]:  # 定义checkpoint元信息读取函数
    """读取checkpoint旁路meta文件。"""  # 说明函数用途
    meta_path = _meta_path(path_prefix=path_prefix)  # 计算meta文件路径
    if not meta_path.exists():  # 判断meta文件是否存在
        return {}  # 不存在时返回空字典
    payload = read_json(str(meta_path))  # 读取meta文件内容
    if isinstance(payload, dict):  # 判断读取结果是否为字典
        return payload  # 返回字典结果
    return {}  # 非字典结果时返回空字典


def save_checkpoint(  # 定义checkpoint保存函数
    sess,  # 定义TensorFlow会话参数
    saver: tf.compat.v1.train.Saver,  # 定义Saver参数
    path_prefix: Path,  # 定义checkpoint前缀路径参数
    epoch: int,  # 定义轮数参数
    metric: float,  # 定义指标参数
    global_step: int,  # 定义全局步数参数
    best_metric: float,  # 定义当前最佳指标参数
    bad_epochs: int,  # 定义连续未提升轮数参数
    fairness_counts: Dict[str, Dict[str, int]],  # 定义公平计数字典参数
    extra_payload: Optional[Dict[str, Any]] = None,  # 定义附加元信息参数
) -> Path:  # 定义保存后路径返回类型
    """保存TensorFlow checkpoint并写入同名前缀meta信息。"""  # 说明函数用途
    save_path = saver.save(sess, str(path_prefix))  # 保存checkpoint权重文件
    payload: Dict[str, Any] = {  # 构建meta载荷字典
        "checkpoint_path": str(save_path),  # 记录checkpoint路径
        "epoch": int(epoch),  # 记录当前轮数
        "metric": float(metric),  # 记录当前指标
        "global_step": int(global_step),  # 记录全局步数
        "best_metric": float(best_metric),  # 记录全程最佳指标
        "bad_epochs": int(bad_epochs),  # 记录连续未提升轮数
        "fairness_counts": fairness_counts,  # 记录公平计数字典
    }
    if extra_payload:  # 判断是否提供附加元信息
        payload["extra"] = extra_payload  # 写入附加元信息字段
    write_json(path_like=str(_meta_path(path_prefix=path_prefix)), payload=payload)  # 写入meta文件
    return Path(save_path)  # 返回实际保存路径


def restore_checkpoint(  # 定义checkpoint恢复函数
    sess,  # 定义TensorFlow会话参数
    saver: tf.compat.v1.train.Saver,  # 定义Saver参数
    path_prefix: Path,  # 定义checkpoint前缀路径参数
) -> Dict[str, Any]:  # 定义恢复信息返回类型
    """恢复TensorFlow checkpoint并读取meta信息。"""  # 说明函数用途
    saver.restore(sess, str(path_prefix))  # 恢复checkpoint权重
    meta = load_checkpoint_meta(path_prefix=path_prefix)  # 读取checkpoint元信息
    return {"checkpoint_path": str(path_prefix), "meta": meta}  # 返回恢复信息字典


def write_checkpoint_placeholder(path: Path, payload: Dict[str, Any]) -> None:  # 定义占位写入兼容函数
    """兼容旧接口，写入轻量meta文件。"""  # 说明函数用途
    write_json(path_like=str(path), payload=payload)  # 写入占位JSON文件
