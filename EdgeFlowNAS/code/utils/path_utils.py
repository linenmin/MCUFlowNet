"""路径工具函数集合。"""  # 定义模块用途

from pathlib import Path  # 导入路径工具


def project_root() -> Path:  # 定义项目根目录函数
    """返回 EdgeFlowNAS 项目根目录。"""  # 说明函数用途
    return Path(__file__).resolve().parents[2]  # 返回根目录路径


def ensure_directory(path_like: str) -> Path:  # 定义目录确保函数
    """确保目标目录存在并返回路径。"""  # 说明函数用途
    path = Path(path_like)  # 构造路径对象
    path.mkdir(parents=True, exist_ok=True)  # 递归创建目录
    return path  # 返回目录路径

