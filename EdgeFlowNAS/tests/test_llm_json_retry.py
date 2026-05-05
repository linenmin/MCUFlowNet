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

        with patch.object(
            client,
            "chat",
            side_effect=[
                '{"arch_codes": ["0,0,0',
                '{"arch_codes": ["0,0,0,0,0,0,0,0,0,0,0"]}',
            ],
        ) as mock_chat:
            result = client.chat_json("warmstart_agent", "sys", "user")

        self.assertEqual(result["arch_codes"], ["0,0,0,0,0,0,0,0,0,0,0"])
        self.assertEqual(mock_chat.call_count, 2)


if __name__ == "__main__":
    unittest.main()
