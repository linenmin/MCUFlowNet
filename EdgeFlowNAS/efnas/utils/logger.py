"""日志工具函数集合。"""  # 定义模块用途

import logging  # 导入日志模块
from pathlib import Path  # 导入路径工具


def build_logger(name: str, log_file: str) -> logging.Logger:  # 定义日志构建函数
    """构建文件+控制台双输出日志器。"""  # 说明函数用途
    logger = logging.getLogger(name)  # 获取命名日志器
    logger.setLevel(logging.INFO)  # 设置日志级别
    logger.handlers.clear()  # 清空旧处理器避免重复输出
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")  # 定义日志格式
    file_path = Path(log_file)  # 构造日志文件路径
    file_path.parent.mkdir(parents=True, exist_ok=True)  # 确保日志目录存在
    file_handler = logging.FileHandler(file_path, encoding="utf-8")  # 创建文件处理器
    file_handler.setFormatter(formatter)  # 设置文件日志格式
    logger.addHandler(file_handler)  # 绑定文件处理器
    stream_handler = logging.StreamHandler()  # 创建控制台处理器
    stream_handler.setFormatter(formatter)  # 设置控制台日志格式
    logger.addHandler(stream_handler)  # 绑定控制台处理器
    return logger  # 返回日志器

