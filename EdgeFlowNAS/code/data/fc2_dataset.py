"""FC2 数据集占位定义。"""  # 定义模块用途

from typing import Dict, List  # 导入类型注解


def load_fc2_paths(data_list_dir: str, list_name: str) -> List[str]:  # 定义路径列表加载函数
    """读取FC2样本路径列表。"""  # 说明函数用途
    return [f"{data_list_dir}/{list_name}"]  # 返回占位路径列表


def build_fc2_dataset(data_list_dir: str, list_name: str) -> Dict[str, List[str]]:  # 定义数据集构建函数
    """构建FC2数据集占位对象。"""  # 说明函数用途
    paths = load_fc2_paths(data_list_dir=data_list_dir, list_name=list_name)  # 加载路径列表
    return {"paths": paths}  # 返回占位数据集对象

