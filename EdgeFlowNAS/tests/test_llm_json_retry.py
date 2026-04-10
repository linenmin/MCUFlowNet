import sys
import types
import unittest
from unittest.mock import patch

if "litellm" not in sys.modules:
    sys.modules["litellm"] = types.SimpleNamespace(suppress_debug_info=True)

from efnas.search.llm_client import LLMClient


class TestLLMJsonRetry(unittest.TestCase):
    def test_chat_json_retries_once_after_truncated_json(self) -> None:
        cfg = {
            "llm": {
                "models": {
                    "agent_a_strategist": "gemini/test",
                    "agent_b_generator": "gemini/test",
                    "agent_c_distiller": "gemini/test",
                    "agent_d_scientist": "gemini/test",
                    "agent_d_coder": "gemini/test",
                    "agent_d_rule_manager": "gemini/test",
                },
                "temperature": 1.0,
                "max_retries": 0,
            }
        }
        client = LLMClient(cfg)

        with patch.object(
            client,
            "chat",
            side_effect=[
                '{"generated_candidates": ["0,0,0',
                '{"generated_candidates": ["0,0,0,0,0,0,0,0,0,0,0"]}',
            ],
        ) as mock_chat:
            result = client.chat_json("agent_b", "sys", "user")

        self.assertEqual(result["generated_candidates"], ["0,0,0,0,0,0,0,0,0,0,0"])
        self.assertEqual(mock_chat.call_count, 2)


if __name__ == "__main__":
    unittest.main()
