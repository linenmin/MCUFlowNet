import unittest

from efnas.search import prompts


class TestPromptTemplateFormatting(unittest.TestCase):
    def test_agent_d2_prompt_formats_with_csv_columns(self) -> None:
        rendered = prompts.AGENT_D2_SYSTEM.format(csv_columns="arch_code,epe,fps")
        self.assertIn("arch_code,epe,fps", rendered)
        self.assertIn('"enforcement":"hard_filter"', rendered)


if __name__ == "__main__":
    unittest.main()
