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
        self.search_space_size: int = int(search_cfg.get("search_space_size", 3 ** 9))
        self.max_workers: int = cfg["concurrency"]["max_workers"]
        self.prune_tflite_after_reduce: bool = bool(
            cfg.get("evaluation", {}).get("vela_prune_tflite_after_reduce", False)
        )

        # P1c: 上轮有效产出率反馈 (传给 Agent A)
        self._last_yield_info: str = file_io.read_run_state(exp_dir).get("last_yield_info", "")

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
        self._maybe_prune_vela_tflite(stage="startup_rescue")

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
                self._maybe_prune_vela_tflite(stage="interrupt_rescue")
                logger.info("Reduce 抢救完成，安全退出。")
                return
            except Exception:
                logger.exception("Epoch %d 发生未捕获异常", epoch)
                # 尝试抢救已有结果
                file_io.collect_and_commit_worker_results(self.exp_dir)
                self._maybe_prune_vela_tflite(stage="exception_rescue")
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
        state = file_io.read_run_state(self.exp_dir)
        if state.get("current_epoch") != epoch:
            state = file_io.begin_epoch_state(
                self.exp_dir,
                epoch,
                last_yield_info=self._last_yield_info,
            )

        # -------------------------------------------------------
        # Phase 1: 触发科学家大反思 (每 N 个 Epoch)
        # -------------------------------------------------------
        if (
            epoch > 0
            and epoch % self.scientist_interval == 0
            and not state.get("scientist_done", False)
        ):
            logger.info("[Phase 1] 触发科学家大反思 (interval=%d)", self.scientist_interval)
            self._execute_scientist_macro_reflection()
            state = file_io.update_run_state(
                self.exp_dir,
                scientist_done=True,
                phase="scientist_done",
            )

        # -------------------------------------------------------
        # Phase 2: 验证已有猜想并升格 + Finding 再验证降级
        # -------------------------------------------------------
        if not state.get("assumptions_evaluated", False):
            logger.info("[Phase 2] 验证已有猜想")
            self._evaluate_pending_assumptions()
            state = file_io.update_run_state(
                self.exp_dir,
                assumptions_evaluated=True,
                phase="assumptions_evaluated",
            )

        if not state.get("findings_revalidated", False):
            logger.info("[Phase 2b] 再验证已有 Findings")
            self._revalidate_findings()
            state = file_io.update_run_state(
                self.exp_dir,
                findings_revalidated=True,
                phase="findings_revalidated",
            )

        # -------------------------------------------------------
        # Phase 3: 战略规划 -> Agent A
        # -------------------------------------------------------
        agent_a_result = state.get("agent_a_result")
        if not agent_a_result:
            logger.info("[Phase 3] 调用 Agent A (战略规划)")
            agent_a_result = agents.invoke_agent_a(
                self.llm, self.exp_dir, epoch, self.batch_size,
                last_yield_info=self._last_yield_info,
            )
            state = file_io.update_run_state(
                self.exp_dir,
                agent_a_result=agent_a_result,
                phase="agent_a_done",
            )
        allocation = agent_a_result.get("allocation", {})

        # -------------------------------------------------------
        # Phase 4: 编码生成 -> Agent B
        # -------------------------------------------------------
        candidates = list(state.get("candidates") or [])
        if not candidates:
            logger.info("[Phase 4] 调用 Agent B (编码生成)")
            candidates = agents.invoke_agent_b(
                self.llm, self.exp_dir, allocation, self.batch_size,
            )
            state = file_io.update_run_state(
                self.exp_dir,
                candidates=candidates,
                phase="agent_b_done",
            )

        if not candidates:
            logger.warning("Agent B 未返回有效候选，跳过本轮评估")
            return

        # -------------------------------------------------------
        # Phase 5: 去重过滤
        # -------------------------------------------------------
        new_archs = list(state.get("new_archs") or [])
        skipped = int(state.get("duplicates", 0) or 0)
        rule_rejected = int(state.get("rule_rejected", 0) or 0)
        if not new_archs and state.get("phase") not in {"dedup_done", "map_done", "reduce_done"}:
            evaluated = file_io.get_evaluated_arch_codes(self.exp_dir)
            deduped_archs = [c for c in candidates if c not in evaluated]
            skipped = len(candidates) - len(deduped_archs)
            filtered_archs, rule_rejected = agents.filter_candidates_by_findings(
                self.exp_dir,
                deduped_archs,
                context={"allocation": allocation, "epoch": epoch},
            )
            new_archs = filtered_archs

            logger.info(
                "[Phase 5] 去重/规则过滤: 候选 %d, 新增 %d, 历史重复 %d, 规则拒绝 %d",
                len(candidates),
                len(new_archs),
                skipped,
                rule_rejected,
            )

            yield_pct = len(new_archs) / len(candidates) * 100 if candidates else 0
            self._last_yield_info = (
                f"请求 {self.batch_size} 个候选, Agent B 生成 {len(candidates)} 个, "
                f"历史去重后 {len(candidates) - skipped} 个, 规则拒绝 {rule_rejected} 个, "
                f"实际新评估 {len(new_archs)} 个 (有效率 {yield_pct:.0f}%)"
            )
            state = file_io.update_run_state(
                self.exp_dir,
                new_archs=new_archs,
                duplicates=skipped,
                rule_rejected=rule_rejected,
                last_yield_info=self._last_yield_info,
                phase="dedup_done",
            )
        else:
            evaluated = file_io.get_evaluated_arch_codes(self.exp_dir)
            new_archs = [c for c in new_archs if c not in evaluated]

        if not new_archs:
            logger.info("本轮无新架构需要评估，跳过 Map-Reduce")
            self._record_epoch_metrics(epoch, 0, skipped, rule_rejected=rule_rejected)
            file_io.clear_epoch_state(self.exp_dir, last_yield_info=self._last_yield_info)
            return

        # -------------------------------------------------------
        # Phase 6: Map 阶段 — 多线程评估 + Agent C 蒸馏
        # -------------------------------------------------------
        if not state.get("map_done", False):
            logger.info("[Phase 6] Map 阶段: %d 个架构, %d 个工作线程",
                         len(new_archs), self.max_workers)
            self._map_evaluate(new_archs, epoch)
            state = file_io.update_run_state(
                self.exp_dir,
                map_done=True,
                phase="map_done",
            )

        # -------------------------------------------------------
        # Phase 7: Reduce 阶段 — 归并到全局 CSV
        # -------------------------------------------------------
        committed = 0
        if not state.get("reduce_done", False):
            logger.info("[Phase 7] Reduce 阶段")
            committed = file_io.collect_and_commit_worker_results(self.exp_dir)
            self._maybe_prune_vela_tflite(stage=f"epoch_{epoch}_reduce")
            logger.info("本轮提交 %d 条评估结果", committed)
            state = file_io.update_run_state(
                self.exp_dir,
                reduce_done=True,
                phase="reduce_done",
            )
        else:
            logger.info("[Phase 7] Reduce 已完成，跳过重复提交")

        # -------------------------------------------------------
        # Phase 8: 记录 Epoch 级可观测指标
        # -------------------------------------------------------
        self._record_epoch_metrics(epoch, committed, skipped, rule_rejected=rule_rejected)
        file_io.clear_epoch_state(self.exp_dir, last_yield_info=self._last_yield_info)

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
            script_path = self._resolve_rule_script_path(aid)

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
            if confidence >= self.confidence_threshold and total >= 30:
                logger.info("猜想 %s 达到升格条件！触发 D-3 写入 Findings", aid)
                agents.invoke_agent_d3(self.llm, self.exp_dir, assumption, confidence)
            else:
                logger.info("猜想 %s 尚未达标 (threshold=%.2f, min_samples=30)",
                             aid, self.confidence_threshold)

    # ===============================================================
    # 子流程：Finding 再验证与降级 (P3)
    # ===============================================================

    def _revalidate_findings(self) -> None:
        """重新验证所有 Findings，置信度低于阈值的降级回猜想队列。"""
        findings = [
            {"id": f.get("id", "")}
            for f in file_io.read_findings_registry(self.exp_dir)
            if f.get("active", True)
        ]
        if not findings:
            return

        data_csv = os.path.join(self.exp_dir, "metadata", "history_archive.csv")
        demoted_ids: list[str] = []

        for finding in findings:
            fid = finding["id"]
            script_path = self._resolve_rule_script_path(fid)

            if not os.path.exists(script_path):
                logger.debug("Finding %s 无验证脚本，跳过再验证", fid)
                continue

            result = agents.execute_verification_script(script_path, data_csv)
            if result is None:
                logger.warning("Finding %s 再验证脚本执行失败，保留", fid)
                continue

            confidence = result.get("confidence", 0.0)
            total = result.get("total_triggered", 0)
            logger.info("Finding %s 再验证: confidence=%.4f, total=%d",
                         fid, confidence, total)

            if confidence < self.confidence_threshold:
                logger.info("Finding %s 降级！confidence=%.4f < threshold=%.2f",
                             fid, confidence, self.confidence_threshold)
                demoted_ids.append(fid)

        # 批量执行降级：将 finding 置为 inactive，并加回 assumptions.json
        for fid in demoted_ids:
            file_io.remove_finding_by_id(self.exp_dir, fid)
            desc = f"(降级自 Finding) 原 Finding {fid} 再验证未达标，回退到猜想队列重新验证。"
            file_io.append_assumptions(self.exp_dir, [{"id": fid, "description": desc}])
            logger.info("Finding %s 已降级为猜想", fid)

        if demoted_ids:
            logger.info("[P3] 本轮共降级 %d 条 Finding: %s",
                         len(demoted_ids), ", ".join(demoted_ids))

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
        state = file_io.read_run_state(self.exp_dir)
        current_epoch = state.get("current_epoch")
        phase = str(state.get("phase", "idle"))
        if current_epoch is not None and phase not in {"idle"} and not state.get("reduce_done", False):
            try:
                return int(current_epoch)
            except (ValueError, TypeError):
                pass

        df = file_io.read_history(self.exp_dir)
        if df.empty or "epoch" not in df.columns:
            return 0
        try:
            max_epoch = int(df["epoch"].dropna().astype(int).max())
            return max_epoch + 1
        except (ValueError, TypeError):
            return 0

    def _maybe_prune_vela_tflite(self, stage: str) -> None:
        """Optionally prune Vela tflite artifacts in main thread after reduce/rescue."""
        if not self.prune_tflite_after_reduce:
            return
        removed = file_io.prune_vela_tflite_artifacts(self.exp_dir)
        if removed > 0:
            logger.info("[Prune] %s: 删除 %d 个 Vela tflite 文件", stage, removed)

    def _resolve_rule_script_path(self, rule_id: str) -> str:
        """Resolve new rule_Axx.py name first, then fall back to legacy eval_assumption_Axx.py."""
        candidates = [
            os.path.join(self.exp_dir, "scripts", f"rule_{rule_id}.py"),
            os.path.join(self.exp_dir, "scripts", f"eval_assumption_{rule_id}.py"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _record_epoch_metrics(self, epoch: int, new_evaluated: int, duplicates: int, *, rule_rejected: int = 0) -> None:
        """记录本轮搜索的可观测性指标。"""
        df = file_io.read_history(self.exp_dir)
        total = len(df)
        total_space = max(1, int(self.search_space_size))

        best_epe = float("inf")
        best_fps = float("-inf")
        pareto_count = 0
        if not df.empty and "epe" in df.columns:
            try:
                epe_vals = df["epe"].astype(float)
                best_epe = float(epe_vals.min())
                if "fps" in df.columns:
                    fps_vals = df["fps"].astype(float)
                    best_fps = float(fps_vals.max())
                    pareto_count = agents._count_pareto_2d(epe_vals, fps_vals)
            except (ValueError, TypeError):
                pass

        metrics = {
            "epoch": epoch,
            "total_evaluated": total,
            "new_evaluated": new_evaluated,
            "duplicates": duplicates,
            "rule_rejected": rule_rejected,
            "best_epe": round(best_epe, 6) if best_epe != float("inf") else "",
            "best_fps": round(best_fps, 6) if best_fps != float("-inf") else "",
            "pareto_count": pareto_count,
            "findings_count": file_io.count_findings(self.exp_dir),
            "assumptions_count": len(file_io.read_assumptions(self.exp_dir)),
            "coverage_pct": round(total / total_space * 100, 2),
        }
        file_io.append_epoch_metrics(self.exp_dir, metrics)
        logger.info("[Phase 8] Epoch指标: evaluated=%d, best_epe=%s, best_fps=%s, pareto=%d, coverage=%.1f%%",
                     total, metrics["best_epe"], metrics["best_fps"], pareto_count, metrics["coverage_pct"])
