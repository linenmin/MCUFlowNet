"""协调引擎：Multi-Agent NAS 搜索的主循环状态机。

本模块是整个系统的唯一物理驱动中枢。它按照以下时序调度各 Agent：
1. 断点恢复 (Rescue)
2. 科学家大反思 (Agent D-1 / D-2 / D-3) — 每 N 个 Epoch 触发
3. 猜想验证与升格 (Engine)
4. 战略规划 (Agent A)
5. 编码生成 (Agent B)
6. 去重过滤 (Engine)
7. 多线程评估 + 硬件蒸馏 (Worker + Agent C) — Map 阶段
8. 结果归并 (Engine) — Reduce 阶段
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from efnas.search import agents, file_io
from efnas.search.eval_worker import evaluate_single_arch
from efnas.search.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SearchCoordinator:
    """Multi-Agent NAS 搜索协调器。"""

    def __init__(
        self,
        cfg: Dict[str, Any],
        exp_dir: str,
        project_root: str,
    ) -> None:
        """初始化协调器。

        Args:
            cfg: 从 search_v1.yaml 加载的完整配置字典。
            exp_dir: 本次实验的根目录（含 metadata/, dashboard/, scripts/）。
            project_root: EdgeFlowNAS 项目根目录。
        """
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.project_root = project_root

        search_cfg = cfg["search"]
        self.total_epochs: int = search_cfg["total_epochs"]
        self.batch_size: int = search_cfg["batch_size"]
        self.scientist_interval: int = search_cfg["scientist_trigger_interval"]
        self.confidence_threshold: float = search_cfg["assumption_confidence_threshold"]
        self.max_workers: int = cfg["concurrency"]["max_workers"]

        self.llm = LLMClient(cfg)

    # ===============================================================
    # 主循环入口
    # ===============================================================

    def run(self, start_epoch: int = 0) -> None:
        """启动自主搜索主循环。

        Args:
            start_epoch: 起始轮次（断点恢复时可从非零开始）。
        """
        logger.info("=" * 60)
        logger.info("EdgeFlowNAS Agentic Search 启动")
        logger.info("实验目录: %s", self.exp_dir)
        logger.info("总轮次: %d, 批量: %d, 工作线程: %d",
                     self.total_epochs, self.batch_size, self.max_workers)
        logger.info("=" * 60)

        # Step 0: 断点恢复 — 抢救 tmp_workers/ 中的遗留结果
        rescued = file_io.rescue_orphaned_results(self.exp_dir)
        if rescued > 0:
            logger.info("断点恢复完成: 抢救 %d 条结果", rescued)

        # 推断 start_epoch（从已有数据中找最大 epoch）
        if start_epoch == 0:
            start_epoch = self._infer_start_epoch()

        for epoch in range(start_epoch, self.total_epochs):
            logger.info("=" * 50)
            logger.info("=== Epoch %d / %d 开始 ===", epoch, self.total_epochs)
            logger.info("=" * 50)

            try:
                self._run_single_epoch(epoch)
            except KeyboardInterrupt:
                logger.warning("用户中断 (Ctrl+C)，正在执行 Reduce 抢救...")
                file_io.collect_and_commit_worker_results(self.exp_dir)
                logger.info("Reduce 抢救完成，安全退出。")
                return
            except Exception:
                logger.exception("Epoch %d 发生未捕获异常", epoch)
                # 尝试抢救已有结果
                file_io.collect_and_commit_worker_results(self.exp_dir)
                raise

            logger.info("=== Epoch %d 完成 ===", epoch)

        logger.info("=" * 60)
        logger.info("搜索完毕！共 %d 个 Epoch。", self.total_epochs)
        logger.info("=" * 60)

    # ===============================================================
    # 单轮执行
    # ===============================================================

    def _run_single_epoch(self, epoch: int) -> None:
        """执行单个 Epoch 的完整流程。"""

        # -------------------------------------------------------
        # Phase 1: 触发科学家大反思 (每 N 个 Epoch)
        # -------------------------------------------------------
        if epoch > 0 and epoch % self.scientist_interval == 0:
            logger.info("[Phase 1] 触发科学家大反思 (interval=%d)", self.scientist_interval)
            self._execute_scientist_macro_reflection()

        # -------------------------------------------------------
        # Phase 2: 验证已有猜想并升格
        # -------------------------------------------------------
        logger.info("[Phase 2] 验证已有猜想")
        self._evaluate_pending_assumptions()

        # -------------------------------------------------------
        # Phase 3: 战略规划 -> Agent A
        # -------------------------------------------------------
        logger.info("[Phase 3] 调用 Agent A (战略规划)")
        agent_a_result = agents.invoke_agent_a(
            self.llm, self.exp_dir, epoch, self.batch_size,
        )
        allocation = agent_a_result.get("allocation", {})

        # -------------------------------------------------------
        # Phase 4: 编码生成 -> Agent B
        # -------------------------------------------------------
        logger.info("[Phase 4] 调用 Agent B (编码生成)")
        candidates = agents.invoke_agent_b(
            self.llm, self.exp_dir, allocation, self.batch_size,
        )

        if not candidates:
            logger.warning("Agent B 未返回有效候选，跳过本轮评估")
            return

        # -------------------------------------------------------
        # Phase 5: 去重过滤
        # -------------------------------------------------------
        evaluated = file_io.get_evaluated_arch_codes(self.exp_dir)
        new_archs = [c for c in candidates if c not in evaluated]
        skipped = len(candidates) - len(new_archs)

        logger.info("[Phase 5] 去重: 候选 %d, 新增 %d, 跳过 %d",
                     len(candidates), len(new_archs), skipped)

        if not new_archs:
            logger.info("本轮无新架构需要评估，跳过 Map-Reduce")
            return

        # -------------------------------------------------------
        # Phase 6: Map 阶段 — 多线程评估 + Agent C 蒸馏
        # -------------------------------------------------------
        logger.info("[Phase 6] Map 阶段: %d 个架构, %d 个工作线程",
                     len(new_archs), self.max_workers)
        self._map_evaluate(new_archs, epoch)

        # -------------------------------------------------------
        # Phase 7: Reduce 阶段 — 归并到全局 CSV
        # -------------------------------------------------------
        logger.info("[Phase 7] Reduce 阶段")
        committed = file_io.collect_and_commit_worker_results(self.exp_dir)
        logger.info("本轮提交 %d 条评估结果", committed)

    # ===============================================================
    # 子流程：科学家大反思 (D-1 -> D-2)
    # ===============================================================

    def _execute_scientist_macro_reflection(self) -> None:
        """执行科学家反思循环：D-1 提出猜想 -> D-2 写验证代码。"""
        logger.info("[Scientist] Session D-1: 提出猜想")
        new_assumptions = agents.invoke_agent_d1(self.llm, self.exp_dir)

        if not new_assumptions:
            logger.info("[Scientist] 未产生新猜想")
            return

        # 为每个新猜想生成验证脚本
        for assumption in new_assumptions:
            logger.info("[Scientist] Session D-2: 为猜想 %s 生成验证脚本", assumption.get("id"))
            agents.invoke_agent_d2(self.llm, self.exp_dir, assumption)

    # ===============================================================
    # 子流程：猜想验证与升格
    # ===============================================================

    def _evaluate_pending_assumptions(self) -> None:
        """遍历所有待验证猜想，执行验证脚本，满足置信度则升格为 Finding。"""
        assumptions = file_io.read_assumptions(self.exp_dir)
        if not assumptions:
            return

        data_csv = os.path.join(self.exp_dir, "metadata", "history_archive.csv")

        for assumption in list(assumptions):  # 使用 list() 复制，避免迭代中修改
            aid = assumption.get("id", "unknown")
            script_name = f"eval_assumption_{aid}.py"
            script_path = os.path.join(self.exp_dir, "scripts", script_name)

            if not os.path.exists(script_path):
                logger.debug("猜想 %s 尚无验证脚本，跳过", aid)
                continue

            logger.info("执行验证脚本: %s", script_path)
            result = agents.execute_verification_script(script_path, data_csv)

            if result is None:
                logger.warning("验证脚本执行失败: %s", script_path)
                continue

            confidence = result.get("confidence", 0.0)
            total = result.get("total_triggered", 0)
            logger.info("猜想 %s 验证结果: confidence=%.4f, total_triggered=%d",
                         aid, confidence, total)

            # 置信度门槛检查 + 最少触发样本数检查（避免小样本误判）
            if confidence >= self.confidence_threshold and total >= 5:
                logger.info("猜想 %s 达到升格条件！触发 D-3 写入 Findings", aid)
                agents.invoke_agent_d3(self.llm, self.exp_dir, assumption, confidence)
            else:
                logger.info("猜想 %s 尚未达标 (threshold=%.2f, min_samples=5)",
                             aid, self.confidence_threshold)

    # ===============================================================
    # 子流程：Map 阶段 (多线程评估)
    # ===============================================================

    def _map_evaluate(self, arch_codes: List[str], epoch: int) -> None:
        """使用 ThreadPoolExecutor 并行评估多个架构。

        每个 Worker 独立写入 tmp_workers/ 下的 JSON 文件。
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_single_arch,
                    arch_code_str=arch,
                    epoch=epoch,
                    exp_dir=self.exp_dir,
                    project_root=self.project_root,
                    cfg=self.cfg,
                    llm_client=self.llm,
                ): arch
                for arch in arch_codes
            }

            for future in as_completed(futures):
                arch = futures[future]
                try:
                    result = future.result()
                    if result is None:
                        logger.warning("架构评估失败 (返回 None): %s", arch)
                except Exception:
                    logger.exception("架构评估异常: %s", arch)

    # ===============================================================
    # 辅助方法
    # ===============================================================

    def _infer_start_epoch(self) -> int:
        """从已有历史数据推断应该从哪个 epoch 开始。"""
        df = file_io.read_history(self.exp_dir)
        if df.empty or "epoch" not in df.columns:
            return 0
        try:
            max_epoch = int(df["epoch"].dropna().astype(int).max())
            return max_epoch + 1
        except (ValueError, TypeError):
            return 0
