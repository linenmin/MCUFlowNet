"""EdgeFlowNAS Multi-Agent 自主架构搜索 — CLI 入口。

用法:
    python wrappers/run_agentic_search.py --config configs/search_v1.yaml
    python wrappers/run_agentic_search.py --config configs/search_v1.yaml --resume
    python wrappers/run_agentic_search.py --config configs/search_v1.yaml --experiment_name my_exp
"""

import argparse
import logging
import os
import sys

import yaml

# 将项目根目录加入 sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from efnas.search.coordinator import SearchCoordinator
from efnas.search.file_io import find_latest_experiment_dir, init_experiment_dir


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="EdgeFlowNAS Multi-Agent 自主架构搜索引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/search_v1.yaml",
        help="搜索配置文件路径 (默认: configs/search_v1.yaml)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="nas_search",
        help="实验名称标签 (默认: nas_search)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点恢复模式：查找最新实验目录并继续搜索",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )
    return parser


def _setup_logging(level_str: str) -> None:
    """配置全局日志格式。"""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 降低第三方库日志等级
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


def main() -> int:
    """主入口函数。"""
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.log_level)
    logger = logging.getLogger("run_agentic_search")

    # 加载配置
    config_path = os.path.join(_PROJECT_ROOT, args.config)
    if not os.path.exists(config_path):
        logger.error("配置文件不存在: %s", config_path)
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("配置已加载: %s", config_path)

    # 确定实验目录
    output_root = os.path.join(_PROJECT_ROOT, cfg["paths"]["output_root"])

    if args.resume:
        exp_dir = find_latest_experiment_dir(output_root)
        if exp_dir is None:
            logger.error("断点恢复失败: 未找到已有实验目录 (%s)", output_root)
            return 1
        logger.info("断点恢复: 使用已有实验目录 %s", exp_dir)
    else:
        exp_dir = init_experiment_dir(output_root, args.experiment_name)
        logger.info("新实验目录已创建: %s", exp_dir)

    # 启动协调引擎
    coordinator = SearchCoordinator(
        cfg=cfg,
        exp_dir=exp_dir,
        project_root=_PROJECT_ROOT,
    )

    try:
        coordinator.run()
    except KeyboardInterrupt:
        logger.info("用户中断，搜索已安全停止。")
        return 0
    except Exception:
        logger.exception("搜索引擎遇到致命错误")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
