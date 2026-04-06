"""Supernet V2 训练应用编排层。"""  # 定义模块用途

from typing import Any, Dict  # 导入类型注解

from efnas.app.train_supernet_app import _load_yaml, _merge_overrides  # 复用YAML加载与覆写合并


def run_supernet_app_v2(config_path: str, overrides: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:  # 定义应用入口函数
    """执行Supernet V2训练应用流程。"""  # 说明函数用途
    base_config = _load_yaml(config_path)  # 读取基础配置
    final_config = _merge_overrides(base_config, overrides)  # 合并命令行覆写
    if dry_run:  # 判断是否为干跑模式
        return {"exit_code": 0, "config": final_config}  # 返回配置预览结果
    from efnas.engine.supernet_trainer_v2 import train_supernet  # 延迟导入V2训练执行函数
    exit_code = train_supernet(final_config)  # 调用V2训练执行层
    return {"exit_code": int(exit_code), "config": final_config}  # 返回执行结果
