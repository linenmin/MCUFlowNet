"""LiteLLM 统一网关封装层 (search_hybrid_v2).

为每个 Agent 角色提供独立的模型路由 + 自动按 provider 过滤参数 + 重试 +
强制 JSON 输出.

支持的 provider (通过 model_id 前缀自动识别):
  - "gemini/..."     -> Google Gemini (3.x 系列 temperature 强制 1.0;
                       可选 thinking_budget)
  - "anthropic/..."  -> Anthropic Claude (支持 temperature + thinking_budget)
  - "openai/..."     -> OpenAI GPT (支持 temperature + reasoning_effort)
  - 其他前缀         -> 透传 (LiteLLM 处理)

Per-role config 示例 (configs/nsga2_*.yaml):
  llm:
    models:
      warmstart_agent: "anthropic/claude-opus-4-5-20250929"
      supervisor_agent: "openai/gpt-5"
    temperature:
      warmstart_agent: 0.7
      supervisor_agent: 0.4
    thinking_budget:           # optional, for Anthropic / Gemini 3+
      warmstart_agent: 8192
    reasoning_effort:          # optional, for OpenAI reasoning models
      supervisor_agent: "high"
    cost_log_path: "outputs/.../metadata/llm_cost_log.jsonl"  # optional
    max_tokens: 8192
    request_timeout: 180
    max_retries: 3
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import litellm

logger = logging.getLogger(__name__)


# ===================================================================
# Provider detection
# ===================================================================

def _detect_provider(model_id: str) -> str:
    """从 model_id 前缀识别 provider.

    Returns: "gemini" | "anthropic" | "openai" | "other"
    """
    if not isinstance(model_id, str):
        return "other"
    prefix = model_id.split("/", 1)[0].lower()
    if prefix in ("gemini", "google", "vertex_ai"):
        return "gemini"
    if prefix in ("anthropic", "claude"):
        return "anthropic"
    if prefix in ("openai", "azure"):
        return "openai"
    return "other"


def _is_gemini_3_or_newer(model_id: str) -> bool:
    """Gemini 3+ 不接受 temperature ≠ 1.0 (Google reasoning model 服务端限制).

    检测规则: model_id 含 'gemini-3' 或 'gemini-4'. 'gemini-2.*' 不命中.
    """
    if not isinstance(model_id, str):
        return False
    lc = model_id.lower()
    return "gemini-3" in lc or "gemini-4" in lc


# ===================================================================
# LLMClient
# ===================================================================

class LLMClient:
    """多 provider 路由的 LiteLLM 统一调用客户端 (v2).

    Phase 2-4 (search_hybrid_v1+) 5 个 role:
      - warmstart_agent (Phase 2)
      - scientist_stage_a / scientist_stage_b1 / scientist_stage_b2 (Phase 3)
      - supervisor_agent (Phase 4)

    v2 改动:
    - _detect_provider 按 model_id 前缀路由
    - per-provider 参数过滤:
        * Gemini 3+: temperature 强制 1.0, 支持 thinking_budget
        * Anthropic: 支持 temperature + thinking_budget
        * OpenAI: 支持 temperature + reasoning_effort
    - 可选 cost tracking (累计 input/output tokens + 美元成本)
    """

    ROLE_TO_CONFIG_KEY = {
        "warmstart_agent": "warmstart_agent",
        "scientist_stage_a": "scientist_stage_a",
        "scientist_stage_b1": "scientist_stage_b1",
        "scientist_stage_b2": "scientist_stage_b2",
        "supervisor_agent": "supervisor_agent",
    }

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """初始化客户端.

        Args:
            cfg: 从 nsga2_*.yaml 加载的完整配置字典 (含 llm: 段).
        """
        llm_cfg = cfg["llm"]
        self._models: Dict[str, str] = llm_cfg["models"]
        self._max_tokens: int = llm_cfg.get("max_tokens", 4096)
        self._timeout: int = llm_cfg.get("request_timeout", 120)
        self._max_retries: int = llm_cfg.get("max_retries", 3)

        # 支持 per-role 温度 dict 或全局标量
        raw_temp = llm_cfg.get("temperature", 0.4)
        if isinstance(raw_temp, dict):
            self._temperature_map: Dict[str, float] = raw_temp
            self._default_temperature: float = 0.4
        else:
            self._temperature_map = {}
            self._default_temperature = float(raw_temp)

        # v2: optional per-provider extras
        self._reasoning_effort_map: Dict[str, str] = (
            llm_cfg.get("reasoning_effort", {}) or {}
        )
        self._thinking_budget_map: Dict[str, int] = (
            llm_cfg.get("thinking_budget", {}) or {}
        )

        # v2: cost tracking (best-effort, falls back to no-op on any error)
        self._cost_log_path: Optional[str] = llm_cfg.get("cost_log_path")
        self._total_cost_usd: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

        # 关闭 LiteLLM 自身日志刷屏
        litellm.suppress_debug_info = True

    # -----------------------------------------------------------------
    # 路由 / 取参
    # -----------------------------------------------------------------

    def _resolve_model(self, role: str) -> str:
        """根据 Agent 角色返回对应的模型标识符."""
        config_key = self.ROLE_TO_CONFIG_KEY.get(role)
        if config_key is None:
            raise ValueError(
                f"未知的 Agent 角色: {role}, "
                f"合法值: {list(self.ROLE_TO_CONFIG_KEY.keys())}"
            )
        model_id = self._models.get(config_key)
        if model_id is None:
            raise ValueError(
                f"配置文件中缺少模型映射: llm.models.{config_key}"
            )
        return model_id

    def _resolve_temperature(self, role: str) -> float:
        return self._temperature_map.get(role, self._default_temperature)

    def _build_kwargs(
        self,
        role: str,
        model_id: str,
        messages: List[Dict[str, str]],
        *,
        force_json: bool,
        temperature_override: Optional[float],
        max_tokens_override: Optional[int],
    ) -> Dict[str, Any]:
        """按 provider 过滤参数, 返回 litellm.completion 的 kwargs.

        提取此方法是为了让"哪些参数被透传/丢弃"显式化, 便于在 v2 调试模型切换.
        """
        provider = _detect_provider(model_id)
        temp = (
            temperature_override
            if temperature_override is not None
            else self._resolve_temperature(role)
        )
        max_tok = (
            max_tokens_override
            if max_tokens_override is not None
            else self._max_tokens
        )

        kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tok,
            "timeout": self._timeout,
            "num_retries": self._max_retries,
        }

        if provider == "gemini" and _is_gemini_3_or_newer(model_id):
            # Gemini 3+: temperature 强制 1.0 (服务端限制)
            if abs(float(temp) - 1.0) > 1e-6:
                logger.warning(
                    "[LLMClient] %s on Gemini 3+ model %s: "
                    "temperature=%.3f -> 强制 1.0 (provider limit)",
                    role, model_id, temp,
                )
            kwargs["temperature"] = 1.0
            tb = self._thinking_budget_map.get(role)
            if tb is not None:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": int(tb),
                }
        elif provider == "anthropic":
            kwargs["temperature"] = float(temp)
            tb = self._thinking_budget_map.get(role)
            if tb is not None:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": int(tb),
                }
        elif provider == "openai":
            kwargs["temperature"] = float(temp)
            eff = self._reasoning_effort_map.get(role)
            if eff is not None:
                kwargs["reasoning_effort"] = str(eff)
        else:
            kwargs["temperature"] = float(temp)

        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        return kwargs

    # -----------------------------------------------------------------
    # Cost tracking (v2)
    # -----------------------------------------------------------------

    def _track_cost(self, role: str, model_id: str, response: Any) -> None:
        """记录单次调用的 token 消耗 + 美元成本到日志文件 (best-effort, non-fatal).

        日志格式: JSONL, 每行一个 entry. 默认关闭 (需 cfg.llm.cost_log_path).
        """
        if not self._cost_log_path:
            return
        try:
            usage = getattr(response, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0)) if usage else 0
            out_tok = int(getattr(usage, "completion_tokens", 0)) if usage else 0
            try:
                cost = float(litellm.completion_cost(completion_response=response))
            except Exception:
                cost = 0.0
            self._total_input_tokens += in_tok
            self._total_output_tokens += out_tok
            self._total_cost_usd += cost
            entry = {
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "model": model_id,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": round(cost, 6),
                "cumulative_input_tokens": self._total_input_tokens,
                "cumulative_output_tokens": self._total_output_tokens,
                "cumulative_cost_usd": round(self._total_cost_usd, 6),
            }
            os.makedirs(os.path.dirname(self._cost_log_path), exist_ok=True)
            with open(self._cost_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug(
                "[LLMClient] cost tracking 失败 (non-fatal)", exc_info=True,
            )

    @property
    def cumulative_cost_usd(self) -> float:
        return self._total_cost_usd

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def chat(
        self,
        role: str,
        system_prompt: str,
        user_message: str,
        *,
        force_json: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """发起一次 LLM 对话调用.

        Args:
            role: Agent 角色标识. 见 ROLE_TO_CONFIG_KEY 的合法 key.
            system_prompt: 系统提示词.
            user_message: 用户消息 (含注入的上下文数据).
            force_json: 是否强制要求模型输出 JSON 格式.
            temperature: 覆盖默认温度 (可选; Gemini 3+ 仍被强制 1.0).
            max_tokens: 覆盖默认最大 token 数 (可选).

        Returns:
            模型返回的文本内容 (已去除首尾空白).
        """
        model_id = self._resolve_model(role)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        kwargs = self._build_kwargs(
            role, model_id, messages,
            force_json=force_json,
            temperature_override=temperature,
            max_tokens_override=max_tokens,
        )

        logger.info(
            "LLM 调用 -> 角色=%s, 模型=%s, provider=%s, JSON强制=%s",
            role, model_id, _detect_provider(model_id), force_json,
        )

        try:
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content.strip()
            self._track_cost(role, model_id, response)
            logger.debug("LLM 返回 [%s]: %s...", role, content[:200])
            return content
        except Exception:
            logger.exception("LLM 调用失败: 角色=%s, 模型=%s", role, model_id)
            raise

    def chat_json(
        self,
        role: str,
        system_prompt: str,
        user_message: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """调用 LLM 并自动解析返回的 JSON.

        Returns:
            解析后的 Python 字典.

        Raises:
            json.JSONDecodeError: 模型返回内容两次都无法解析为 JSON.
        """
        raw = self.chat(role, system_prompt, user_message, force_json=True, **kwargs)
        try:
            return json.loads(self._strip_json_fence(raw))
        except json.JSONDecodeError:
            logger.error("JSON 解析失败, 原始返回:\n%s", raw)

        repair_msg = (
            f"{user_message}\n\n"
            "你上一次返回的 JSON 不合法或被截断. "
            "请只重新输出一个完整、可被 json.loads 解析的 JSON 对象. "
            "不要解释, 不要补充文本."
        )
        repaired = self.chat(role, system_prompt, repair_msg, force_json=True, **kwargs)
        return json.loads(self._strip_json_fence(repaired))

    @staticmethod
    def _strip_json_fence(raw: str) -> str:
        """去掉 ```json fenced block 并返回干净 JSON 文本."""
        stripped = raw.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            stripped = "\n".join(lines).strip()
        return stripped
