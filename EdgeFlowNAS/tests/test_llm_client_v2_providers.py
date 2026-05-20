"""测试 LLMClient v2 的 multi-provider 参数路由 (search_hybrid_v2 Phase 3)."""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

if "litellm" not in sys.modules:
    sys.modules["litellm"] = types.SimpleNamespace(
        suppress_debug_info=True,
        completion=lambda **kw: None,
        completion_cost=lambda **kw: 0.0,
    )

from efnas.search.llm_client import (
    LLMClient,
    _detect_provider,
    _is_gemini_3_or_newer,
    _is_overload_or_transient,
)


# ===================================================================
# Provider detection
# ===================================================================

class TestDetectProvider(unittest.TestCase):
    def test_gemini_prefix(self):
        self.assertEqual(_detect_provider("gemini/gemini-3.1-pro-preview"), "gemini")
        self.assertEqual(_detect_provider("gemini/gemini-2.5-pro"), "gemini")

    def test_anthropic_prefix(self):
        self.assertEqual(
            _detect_provider("anthropic/claude-opus-4-5-20250929"), "anthropic",
        )
        self.assertEqual(_detect_provider("claude/claude-sonnet-4-5"), "anthropic")

    def test_openai_prefix(self):
        self.assertEqual(_detect_provider("openai/gpt-5"), "openai")
        self.assertEqual(_detect_provider("azure/gpt-4o"), "openai")

    def test_vertex_ai_treated_as_gemini(self):
        self.assertEqual(_detect_provider("vertex_ai/gemini-3-pro"), "gemini")

    def test_unknown_returns_other(self):
        self.assertEqual(_detect_provider("local/llama-3"), "other")
        self.assertEqual(_detect_provider("plainmodelname"), "other")

    def test_non_string_returns_other(self):
        self.assertEqual(_detect_provider(None), "other")
        self.assertEqual(_detect_provider(123), "other")


class TestIsGemini3OrNewer(unittest.TestCase):
    def test_gemini_3_variants(self):
        self.assertTrue(_is_gemini_3_or_newer("gemini/gemini-3.1-pro-preview"))
        self.assertTrue(_is_gemini_3_or_newer("gemini/gemini-3-flash"))
        self.assertTrue(_is_gemini_3_or_newer("gemini/gemini-3-ultra"))

    def test_gemini_4_variant(self):
        self.assertTrue(_is_gemini_3_or_newer("gemini/gemini-4-pro"))

    def test_gemini_2_not_locked(self):
        self.assertFalse(_is_gemini_3_or_newer("gemini/gemini-2.5-pro"))
        self.assertFalse(_is_gemini_3_or_newer("gemini/gemini-1.5-pro"))

    def test_non_gemini_returns_false(self):
        self.assertFalse(_is_gemini_3_or_newer("anthropic/claude-opus-4-5"))
        self.assertFalse(_is_gemini_3_or_newer("openai/gpt-5"))


# ===================================================================
# _build_kwargs per-provider filtering
# ===================================================================

def _make_client(model_id, role="warmstart_agent", **extra):
    """Helper: 构造 LLMClient 走指定 model_id."""
    cfg = {
        "llm": {
            "models": {
                "warmstart_agent": model_id,
                "scientist_stage_a": model_id,
                "scientist_stage_b1": model_id,
                "scientist_stage_b2": model_id,
                "supervisor_agent": model_id,
            },
            "temperature": extra.get("temperature", 0.5),
            "max_tokens": 4096,
            "request_timeout": 60,
            "max_retries": 1,
            **{
                k: v for k, v in extra.items()
                if k in ("reasoning_effort", "thinking_budget", "cost_log_path")
            },
        }
    }
    return LLMClient(cfg)


class TestBuildKwargsGemini3(unittest.TestCase):
    def test_gemini_3_forces_temperature_1(self):
        client = _make_client("gemini/gemini-3.1-pro-preview", temperature=0.5)
        kwargs = client._build_kwargs(
            "warmstart_agent",
            "gemini/gemini-3.1-pro-preview",
            messages=[{"role": "user", "content": "x"}],
            force_json=True,
            temperature_override=None,
            max_tokens_override=None,
        )
        self.assertEqual(kwargs["temperature"], 1.0)

    def test_gemini_3_thinking_budget_injected(self):
        client = _make_client(
            "gemini/gemini-3.1-pro-preview",
            temperature=1.0,
            thinking_budget={"warmstart_agent": 8192},
        )
        kwargs = client._build_kwargs(
            "warmstart_agent",
            "gemini/gemini-3.1-pro-preview",
            messages=[],
            force_json=False,
            temperature_override=None,
            max_tokens_override=None,
        )
        self.assertEqual(kwargs["thinking"]["budget_tokens"], 8192)

    def test_gemini_3_no_reasoning_effort(self):
        client = _make_client(
            "gemini/gemini-3.1-pro-preview",
            reasoning_effort={"warmstart_agent": "high"},
        )
        kwargs = client._build_kwargs(
            "warmstart_agent",
            "gemini/gemini-3.1-pro-preview",
            messages=[],
            force_json=False,
            temperature_override=None,
            max_tokens_override=None,
        )
        self.assertNotIn("reasoning_effort", kwargs)


class TestBuildKwargsAnthropic(unittest.TestCase):
    def test_anthropic_preserves_temperature(self):
        client = _make_client(
            "anthropic/claude-opus-4-5-20250929", temperature=0.7,
        )
        kwargs = client._build_kwargs(
            "warmstart_agent",
            "anthropic/claude-opus-4-5-20250929",
            messages=[],
            force_json=True,
            temperature_override=None,
            max_tokens_override=None,
        )
        self.assertEqual(kwargs["temperature"], 0.7)

    def test_anthropic_thinking_budget_injected(self):
        client = _make_client(
            "anthropic/claude-opus-4-5-20250929",
            thinking_budget={"warmstart_agent": 16384},
        )
        kwargs = client._build_kwargs(
            "warmstart_agent",
            "anthropic/claude-opus-4-5-20250929",
            messages=[],
            force_json=False,
            temperature_override=None,
            max_tokens_override=None,
        )
        self.assertEqual(kwargs["thinking"]["budget_tokens"], 16384)

    def test_anthropic_no_reasoning_effort(self):
        client = _make_client(
            "anthropic/claude-opus-4-5-20250929",
            reasoning_effort={"warmstart_agent": "high"},
        )
        kwargs = client._build_kwargs(
            "warmstart_agent",
            "anthropic/claude-opus-4-5-20250929",
            messages=[],
            force_json=False,
            temperature_override=None,
            max_tokens_override=None,
        )
        self.assertNotIn("reasoning_effort", kwargs)


class TestBuildKwargsOpenAI(unittest.TestCase):
    def test_openai_preserves_temperature(self):
        client = _make_client("openai/gpt-5", temperature=0.4)
        kwargs = client._build_kwargs(
            "supervisor_agent",
            "openai/gpt-5",
            messages=[],
            force_json=True,
            temperature_override=None,
            max_tokens_override=None,
        )
        self.assertEqual(kwargs["temperature"], 0.4)

    def test_openai_reasoning_effort_injected(self):
        client = _make_client(
            "openai/gpt-5",
            reasoning_effort={"supervisor_agent": "high"},
        )
        kwargs = client._build_kwargs(
            "supervisor_agent",
            "openai/gpt-5",
            messages=[],
            force_json=False,
            temperature_override=None,
            max_tokens_override=None,
        )
        self.assertEqual(kwargs["reasoning_effort"], "high")

    def test_openai_no_thinking_budget(self):
        client = _make_client(
            "openai/gpt-5",
            thinking_budget={"supervisor_agent": 8192},
        )
        kwargs = client._build_kwargs(
            "supervisor_agent",
            "openai/gpt-5",
            messages=[],
            force_json=False,
            temperature_override=None,
            max_tokens_override=None,
        )
        self.assertNotIn("thinking", kwargs)


class TestBuildKwargsCommon(unittest.TestCase):
    def test_force_json_adds_response_format(self):
        client = _make_client("anthropic/claude-opus-4-5-20250929")
        kwargs = client._build_kwargs(
            "warmstart_agent",
            "anthropic/claude-opus-4-5-20250929",
            messages=[],
            force_json=True,
            temperature_override=None,
            max_tokens_override=None,
        )
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_force_json_false_omits_response_format(self):
        client = _make_client("anthropic/claude-opus-4-5-20250929")
        kwargs = client._build_kwargs(
            "warmstart_agent",
            "anthropic/claude-opus-4-5-20250929",
            messages=[],
            force_json=False,
            temperature_override=None,
            max_tokens_override=None,
        )
        self.assertNotIn("response_format", kwargs)

    def test_temperature_override_wins(self):
        client = _make_client("anthropic/claude-opus-4-5-20250929", temperature=0.5)
        kwargs = client._build_kwargs(
            "warmstart_agent",
            "anthropic/claude-opus-4-5-20250929",
            messages=[],
            force_json=False,
            temperature_override=0.2,
            max_tokens_override=None,
        )
        self.assertAlmostEqual(kwargs["temperature"], 0.2)

    def test_max_tokens_override_wins(self):
        client = _make_client("anthropic/claude-opus-4-5-20250929")
        kwargs = client._build_kwargs(
            "warmstart_agent",
            "anthropic/claude-opus-4-5-20250929",
            messages=[],
            force_json=False,
            temperature_override=None,
            max_tokens_override=2048,
        )
        self.assertEqual(kwargs["max_tokens"], 2048)


# ===================================================================
# Backward compatibility
# ===================================================================

class TestBackwardCompat(unittest.TestCase):
    def test_v1_config_without_extras_still_works(self):
        """v1 配置 (无 thinking_budget / reasoning_effort / cost_log_path) 应正常初始化."""
        cfg = {
            "llm": {
                "models": {
                    "warmstart_agent": "gemini/gemini-3.1-pro-preview",
                    "scientist_stage_a": "gemini/gemini-3.1-pro-preview",
                    "scientist_stage_b1": "gemini/gemini-3.1-pro-preview",
                    "scientist_stage_b2": "gemini/gemini-3.1-pro-preview",
                    "supervisor_agent": "gemini/gemini-3.1-pro-preview",
                },
                "temperature": {
                    "warmstart_agent": 1.0,
                    "scientist_stage_a": 1.0,
                    "scientist_stage_b1": 1.0,
                    "scientist_stage_b2": 1.0,
                    "supervisor_agent": 1.0,
                },
                "max_tokens": 8192,
                "request_timeout": 180,
                "max_retries": 3,
            }
        }
        client = LLMClient(cfg)
        self.assertEqual(client._max_tokens, 8192)
        self.assertEqual(client._resolve_temperature("warmstart_agent"), 1.0)
        # v2 fields all empty/None
        self.assertEqual(client._reasoning_effort_map, {})
        self.assertEqual(client._thinking_budget_map, {})
        self.assertIsNone(client._cost_log_path)

    def test_chat_api_unchanged(self):
        """chat() / chat_json() 签名应保持兼容."""
        cfg = {
            "llm": {
                "models": {
                    "warmstart_agent": "gemini/test",
                    "scientist_stage_a": "gemini/test",
                    "scientist_stage_b1": "gemini/test",
                    "scientist_stage_b2": "gemini/test",
                    "supervisor_agent": "gemini/test",
                },
                "temperature": 1.0,
                "max_retries": 0,
            }
        }
        client = LLMClient(cfg)
        # 不调用 — 只看签名存在
        self.assertTrue(callable(client.chat))
        self.assertTrue(callable(client.chat_json))


# ===================================================================
# Transient error retry (Anthropic 529 / OpenAI 429)
# ===================================================================

class TestIsOverloadOrTransient(unittest.TestCase):
    def test_anthropic_529_overloaded(self):
        exc = RuntimeError(
            'AnthropicError - {"type":"error","error":{"type":"overloaded_error",'
            '"message":"Overloaded"},"request_id":"req_xxx"}'
        )
        self.assertTrue(_is_overload_or_transient(exc))

    def test_litellm_internal_server_error(self):
        exc = RuntimeError("litellm.InternalServerError: Overloaded")
        self.assertTrue(_is_overload_or_transient(exc))

    def test_rate_limit(self):
        exc = RuntimeError("RateLimitError: Too many requests")
        self.assertTrue(_is_overload_or_transient(exc))

    def test_timeout(self):
        exc = RuntimeError("Request timed out after 60s")
        self.assertTrue(_is_overload_or_transient(exc))

    def test_503(self):
        exc = RuntimeError("Server error '503 Service Unavailable'")
        self.assertTrue(_is_overload_or_transient(exc))

    def test_auth_error_not_transient(self):
        exc = RuntimeError("AuthenticationError: Invalid API key")
        self.assertFalse(_is_overload_or_transient(exc))

    def test_bad_request_not_transient(self):
        exc = RuntimeError("BadRequestError: model not found")
        self.assertFalse(_is_overload_or_transient(exc))


class TestChatRetryBehavior(unittest.TestCase):
    """验证 chat() 在 transient 错误下指数退避重试 / non-transient 立刻 fail."""

    def _build_client(self):
        cfg = {
            "llm": {
                "models": {
                    "warmstart_agent": "anthropic/claude-opus-4-7",
                    "scientist_stage_a": "anthropic/claude-opus-4-7",
                    "scientist_stage_b1": "anthropic/claude-opus-4-7",
                    "scientist_stage_b2": "anthropic/claude-opus-4-7",
                    "supervisor_agent": "anthropic/claude-opus-4-7",
                },
                "temperature": 1.0,
                "max_retries": 0,
            }
        }
        return LLMClient(cfg)

    def _make_response(self, content):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=content)
            )],
            usage=None,
        )

    def test_transient_error_retries_then_succeeds(self):
        """529 第一次失败, 第二次成功 -> chat() 返回内容."""
        client = self._build_client()
        # 把 sleep 替换成 no-op 以加快测试
        with patch("efnas.search.llm_client.time.sleep") as mock_sleep, \
             patch("efnas.search.llm_client.litellm.completion") as mock_completion:
            mock_completion.side_effect = [
                RuntimeError("AnthropicError - overloaded_error 529"),
                self._make_response('{"ok": true}'),
            ]
            out = client.chat("warmstart_agent", "sys", "user", force_json=False)
        self.assertEqual(out, '{"ok": true}')
        # 一次失败 + 一次成功 = 总 2 次 completion 调用
        self.assertEqual(mock_completion.call_count, 2)
        # 触发了 1 次 sleep
        self.assertEqual(mock_sleep.call_count, 1)
        # 首次 sleep 用第一档延迟 30s
        mock_sleep.assert_called_with(30)

    def test_non_transient_error_no_retry(self):
        """auth error 不应重试, 立刻抛."""
        client = self._build_client()
        with patch("efnas.search.llm_client.time.sleep") as mock_sleep, \
             patch("efnas.search.llm_client.litellm.completion") as mock_completion:
            mock_completion.side_effect = RuntimeError(
                "AuthenticationError: Invalid API key"
            )
            with self.assertRaises(RuntimeError):
                client.chat("warmstart_agent", "sys", "user", force_json=False)
        # 只调用 1 次 (不重试)
        self.assertEqual(mock_completion.call_count, 1)
        # 完全没 sleep
        mock_sleep.assert_not_called()

    def test_all_transient_retries_exhausted_raises(self):
        """如果所有 transient 重试都失败, 最终抛最后的异常."""
        client = self._build_client()
        with patch("efnas.search.llm_client.time.sleep"), \
             patch("efnas.search.llm_client.litellm.completion") as mock_completion:
            mock_completion.side_effect = RuntimeError(
                "AnthropicError - overloaded_error 529"
            )
            with self.assertRaises(RuntimeError):
                client.chat("warmstart_agent", "sys", "user", force_json=False)
        # 初次 + 5 档退避重试 = 6 次调用
        self.assertEqual(mock_completion.call_count, 6)


if __name__ == "__main__":
    unittest.main()
