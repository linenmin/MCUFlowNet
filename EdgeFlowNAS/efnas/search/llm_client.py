"""LiteLLM 统一网关封装层。

为每个 Agent 角色提供独立的模型路由，自动处理重试与 JSON 强制输出。
通过 YAML 配置文件动态选择 Gemini Pro / Flash 等不同模型。
"""

import json
import logging
from typing import Any, Dict, Optional

import litellm

logger = logging.getLogger(__name__)


class LLMClient:
    """多模型路由的 LiteLLM 统一调用客户端。"""

    # Agent 角色 -> YAML 配置键 的映射
    ROLE_TO_CONFIG_KEY = {
        "agent_a": "agent_a_strategist",
        "agent_b": "agent_b_generator",
        "agent_c": "agent_c_distiller",
        "agent_d1": "agent_d_scientist",
        "agent_d2": "agent_d_coder",
        "agent_d3": "agent_d_rule_manager",
    }

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """初始化客户端。

        Args:
            cfg: 从 search_v1.yaml 加载的完整配置字典。
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

        # 关闭 LiteLLM 自身日志刷屏
        litellm.suppress_debug_info = True

    def _resolve_model(self, role: str) -> str:
        """根据 Agent 角色返回对应的模型标识符。"""
        config_key = self.ROLE_TO_CONFIG_KEY.get(role)
        if config_key is None:
            raise ValueError(f"未知的 Agent 角色: {role}，合法值: {list(self.ROLE_TO_CONFIG_KEY.keys())}")
        model_id = self._models.get(config_key)
        if model_id is None:
            raise ValueError(f"配置文件中缺少模型映射: llm.models.{config_key}")
        return model_id

    def _resolve_temperature(self, role: str) -> float:
        """根据 Agent 角色返回对应的温度值。"""
        return self._temperature_map.get(role, self._default_temperature)

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
        """发起一次 LLM 对话调用。

        Args:
            role: Agent 角色标识 ("agent_a", "agent_b", "agent_c", "agent_d1", "agent_d2", "agent_d3")。
            system_prompt: 系统提示词。
            user_message: 用户消息（包含注入的上下文数据）。
            force_json: 是否强制要求模型输出 JSON 格式。
            temperature: 覆盖默认温度（可选）。
            max_tokens: 覆盖默认最大 token 数（可选）。

        Returns:
            模型返回的文本内容（已去除首尾空白）。
        """
        model_id = self._resolve_model(role)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._resolve_temperature(role),
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "timeout": self._timeout,
            "num_retries": self._max_retries,
        }

        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        logger.info("LLM 调用 -> 角色=%s, 模型=%s, JSON强制=%s", role, model_id, force_json)

        try:
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content.strip()
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
        """调用 LLM 并自动解析返回的 JSON。

        Returns:
            解析后的 Python 字典。

        Raises:
            json.JSONDecodeError: 模型返回内容无法解析为 JSON。
        """
        raw = self.chat(role, system_prompt, user_message, force_json=True, **kwargs)
        try:
            return json.loads(self._strip_json_fence(raw))
        except json.JSONDecodeError:
            logger.error("JSON 解析失败，原始返回:\n%s", raw)

        repair_msg = (
            f"{user_message}\n\n"
            "你上一次返回的 JSON 不合法或被截断。"
            "请只重新输出一个完整、可被 json.loads 解析的 JSON 对象。"
            "不要解释，不要补充文本。"
        )
        repaired = self.chat(role, system_prompt, repair_msg, force_json=True, **kwargs)
        return json.loads(self._strip_json_fence(repaired))

    @staticmethod
    def _strip_json_fence(raw: str) -> str:
        """去掉 ```json fenced block 并返回干净 JSON 文本。"""
        stripped = raw.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            stripped = "\n".join(lines).strip()
        return stripped
