"""FC2 数据加载器构建器。"""  # 定义模块用途

from typing import Dict  # 导入类型注解

from code.data.fc2_dataset import FC2BatchProvider, resolve_fc2_samples  # 导入FC2采样器与样本解析函数


def build_fc2_provider(config: Dict, split_file_name: str, seed_offset: int = 0) -> FC2BatchProvider:  # 定义FC2加载器构建函数
    """按配置构建FC2批采样器。"""  # 说明函数用途
    data_cfg = config.get("data", {})  # 读取数据配置字典
    runtime_cfg = config.get("runtime", {})  # 读取运行配置字典
    sample_paths = resolve_fc2_samples(  # 解析FC2样本路径列表
        data_list_dir=str(data_cfg.get("data_list", "")),  # 传入数据列表目录
        split_file_name=str(split_file_name),  # 传入划分列表文件名
        base_path=data_cfg.get("base_path", None),  # 传入基础数据路径
    )
    provider = FC2BatchProvider(  # 创建FC2批采样器实例
        samples=sample_paths,  # 传入样本路径列表
        crop_h=int(data_cfg.get("input_height", 180)),  # 传入裁剪高度
        crop_w=int(data_cfg.get("input_width", 240)),  # 传入裁剪宽度
        seed=int(runtime_cfg.get("seed", 42)) + int(seed_offset),  # 传入随机种子偏移
        allow_synthetic=bool(data_cfg.get("allow_synthetic_fallback", True)),  # 传入合成兜底开关
    )
    return provider  # 返回批采样器实例

