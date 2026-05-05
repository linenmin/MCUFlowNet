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
    parser.add_argument("--supernet_experiment_dir", type=str, default=None, help="trained supernet experiment folder for evaluator")
    parser.add_argument("--gpu_devices", type=str, default=None, help="comma-separated GPU ids assigned to eval workers")
    parser.add_argument("--max_workers", type=int, default=None, help="override concurrent evaluation workers")
    parser.add_argument("--num_workers", type=int, default=None, help="override per-eval FC2 loader workers")
    parser.add_argument("--prefetch_batches", type=int, default=None, help="override per-eval prefetch depth")
    parser.add_argument("--max_fc2_val_samples", type=int, default=None, help="optional FC2 val cap for pilots")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="override cfg.paths.output_root (e.g., outputs/ablation_phase5)",
    )
    # Phase 2 (search_hybrid_v1): warm-start agent flag
    parser.add_argument(
        "--enable_warmstart",
        action="store_true",
        help="invoke warmstart_agent (LLM) to seed NSGA-II generation 0 instead of random init",
    )
    parser.add_argument(
        "--warmstart_role",
        type=str,
        default="warmstart_agent",
        help="LLM client role name for warmstart agent (must be configured in cfg.llm.models)",
    )
    # Phase 3 (search_hybrid_v1): scientist agent flags
    parser.add_argument(
        "--enable_scientist",
        action="store_true",
        help="invoke scientist_agent every K generations (3-stage reflection on history)",
    )
    parser.add_argument(
        "--scientist_interval",
        type=int,
        default=3,
        help="K (default 3): scientist fires after generation indices where (gen+1) %% K == 0",
    )
    parser.add_argument(
        "--scientist_sandbox_timeout",
        type=int,
        default=30,
        help="max seconds per verification code execution in scientist sandbox",
    )
    # Phase 4 (search_hybrid_v1): supervisor agent flag
    parser.add_argument(
        "--enable_supervisor",
        action="store_true",
        help="invoke supervisor_agent every scientist_interval generations to "
             "tune NSGA-II 5 levers (mutation_prob, crossover_prob, "
             "per_dim_mutation_multiplier, tournament_size, reseed_bottom_pct)",
    )
    parser.add_argument(
        "--supervisor_role",
        type=str,
        default="supervisor_agent",
        help="LLM client role name for supervisor agent",
    )
    # Phase 5 (search_hybrid_v1): convenience flag for ablation experiments.
    # Auto-expands to the corresponding combination of enable_warmstart /
    # enable_scientist / enable_supervisor flags.
    parser.add_argument(
        "--ablation_group",
        type=str,
        default=None,
        choices=["a", "b", "c", "d"],
        help="Ablation group convenience: a=NSGA-II only, b=+warmstart, "
             "c=+scientist, d=+supervisor (full system). Overrides any "
             "manually-set --enable_* flags below.",
    )
    return parser


def _apply_ablation_group(args: argparse.Namespace) -> None:
    """Phase 5: 把 --ablation_group 展开成对应的 enable flags."""
    group = args.ablation_group
    if group is None:
        return
    args.enable_warmstart = group in ("b", "c", "d")
    args.enable_scientist = group in ("c", "d")
    args.enable_supervisor = group == "d"


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply runtime CLI overrides to a loaded NSGA-II config."""
    if args.supernet_experiment_dir is not None:
        cfg.setdefault("evaluation", {})["supernet_experiment_dir"] = args.supernet_experiment_dir
    if args.gpu_devices is not None:
        cfg.setdefault("concurrency", {})["gpu_devices"] = args.gpu_devices
    if args.max_workers is not None:
        cfg.setdefault("concurrency", {})["max_workers"] = int(args.max_workers)
    if args.num_workers is not None:
        cfg.setdefault("evaluation", {})["num_workers"] = int(args.num_workers)
    if args.prefetch_batches is not None:
        cfg.setdefault("evaluation", {})["prefetch_batches"] = int(args.prefetch_batches)
    if args.max_fc2_val_samples is not None:
        cfg.setdefault("evaluation", {})["max_fc2_val_samples"] = int(args.max_fc2_val_samples)
    return cfg


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
    _apply_ablation_group(args)
    _setup_logging(args.log_level)
    logger = logging.getLogger("run_nsga2_search")
    if args.ablation_group is not None:
        logger.info(
            "ablation_group=%s expanded to: warmstart=%s scientist=%s supervisor=%s",
            args.ablation_group,
            args.enable_warmstart,
            args.enable_scientist,
            args.enable_supervisor,
        )

    config_path = os.path.join(_PROJECT_ROOT, args.config)
    if not os.path.exists(config_path):
        logger.error("config not found: %s", config_path)
        return 1

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = _apply_cli_overrides(cfg, args)

    if args.output_root is not None:
        cfg.setdefault("paths", {})["output_root"] = args.output_root
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

    # Phase 2: warm-start agent (optional, only on fresh experiments —— resume
    # 不重新调用 warmstart, 避免重复污染 generation 0).
    external_initial_population = None
    if args.enable_warmstart and not args.resume:
        from efnas.search.llm_client import LLMClient
        from efnas.search.warmstart_agent import warmstart_pipeline
        from efnas.baselines.nsga2_search import load_search_space

        pop_size = int(cfg["search"]["population_size"])
        search_space_module = cfg["search"].get("search_space_module", "efnas.nas.search_space_v2")
        search_space = load_search_space(search_space_module)
        llm_client = LLMClient(cfg)
        llm_model = cfg.get("llm", {}).get("models", {}).get(args.warmstart_role, "")
        logger.info("warmstart enabled: role=%s model=%s pop_size=%d",
                    args.warmstart_role, llm_model, pop_size)
        external_initial_population = warmstart_pipeline(
            llm_client,
            exp_dir,
            search_space=search_space,
            population_size=pop_size,
            llm_model=str(llm_model),
            role=args.warmstart_role,
        )
        logger.info("warmstart returned %d valid arch_codes (will partial-fill if < %d)",
                    len(external_initial_population), pop_size)
    elif args.enable_warmstart and args.resume:
        logger.info("--enable_warmstart ignored on --resume (warmstart only seeds gen 0)")

    # Phase 3: scientist agent (复用 warmstart 创建的 LLMClient 如果可能, 否则
    # 单独 build).
    scientist_llm = None
    if args.enable_scientist:
        from efnas.search.llm_client import LLMClient
        # 如果 warmstart 也启用了, 上面已经 build 过 llm_client; 复用它
        try:
            scientist_llm = llm_client  # type: ignore[name-defined]
        except NameError:
            scientist_llm = LLMClient(cfg)
        logger.info(
            "scientist enabled: interval=%d sandbox_timeout=%ds",
            args.scientist_interval, args.scientist_sandbox_timeout,
        )

    # Phase 4: supervisor agent (复用 LLMClient if any of warmstart/scientist
    # already built one)
    supervisor_llm = None
    if args.enable_supervisor:
        from efnas.search.llm_client import LLMClient
        try:
            supervisor_llm = scientist_llm if scientist_llm is not None else llm_client  # type: ignore[name-defined]
        except NameError:
            supervisor_llm = LLMClient(cfg)
        logger.info(
            "supervisor enabled: triggers with scientist (every %d gens)",
            args.scientist_interval,
        )

    runner = NSGA2SearchRunner(
        cfg=cfg,
        exp_dir=exp_dir,
        project_root=_PROJECT_ROOT,
        external_initial_population=external_initial_population,
        scientist_llm=scientist_llm,
        scientist_interval=args.scientist_interval,
        scientist_sandbox_timeout=args.scientist_sandbox_timeout,
        supervisor_llm=supervisor_llm,
    )
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
