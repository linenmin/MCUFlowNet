"""Dataloader 占位构建器。"""  # 定义模块用途

from typing import Dict  # 导入类型注解


def build_dataloader(dataset: Dict, batch_size: int) -> Dict:  # 定义数据加载器构建函数
    """构建占位dataloader对象。"""  # 说明函数用途
    return {"dataset": dataset, "batch_size": int(batch_size)}  # 返回占位加载器

