"""CLI entry for the EdgeFlowNAS NSGA-II baseline search."""

import argparse
import logging
import os
import sys

import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from efnas.baselines.nsga2_search import NSGA2SearchRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EdgeFlowNAS NSGA-II baseline search")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nsga2_v2.yaml",
        help="baseline config path",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="nsga2_search",
        help="experiment name label",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest experiment under output_root",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="global log level",
    )
    return parser


def _setup_logging(level_text: str) -> None:
    level = getattr(logging, level_text.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging(args.log_level)
    logger = logging.getLogger("run_nsga2_search")

    config_path = os.path.join(_PROJECT_ROOT, args.config)
    if not os.path.exists(config_path):
        logger.error("config not found: %s", config_path)
        return 1

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    output_root = os.path.join(_PROJECT_ROOT, cfg["paths"]["output_root"])
    from efnas.search.file_io import find_latest_experiment_dir, init_experiment_dir

    if args.resume:
        exp_dir = find_latest_experiment_dir(output_root)
        if exp_dir is None:
            logger.error("resume failed: no experiment found under %s", output_root)
            return 1
        logger.info("resume from %s", exp_dir)
    else:
        exp_dir = init_experiment_dir(output_root, args.experiment_name)
        logger.info("new experiment dir: %s", exp_dir)

    runner = NSGA2SearchRunner(cfg=cfg, exp_dir=exp_dir, project_root=_PROJECT_ROOT)
    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("user interrupted baseline search")
        return 0
    except Exception:
        logger.exception("NSGA-II baseline failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
