"""Phase 3.1 (search_hybrid_v1): sandbox 测试."""

import json
import unittest

from efnas.search import sandbox


class TestValidateImports(unittest.TestCase):
    def test_whitelisted_import_ok(self) -> None:
        ok, err = sandbox.validate_imports("import pandas\nimport numpy as np\n")
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_whitelisted_import_from_ok(self) -> None:
        ok, err = sandbox.validate_imports(
            "from scipy.stats import ttest_ind\nimport math\n"
        )
        self.assertTrue(ok)

    def test_sys_argv_allowed(self) -> None:
        ok, _ = sandbox.validate_imports("import sys\nprint(sys.argv)\n")
        self.assertTrue(ok)

    def test_disallowed_import_os(self) -> None:
        ok, err = sandbox.validate_imports("import os\nos.remove('x')\n")
        self.assertFalse(ok)
        self.assertIn("os", err)

    def test_disallowed_import_subprocess(self) -> None:
        ok, err = sandbox.validate_imports(
            "from subprocess import call\ncall(['ls'])\n"
        )
        self.assertFalse(ok)

    def test_disallowed_pathlib(self) -> None:
        ok, err = sandbox.validate_imports("import pathlib\n")
        self.assertFalse(ok)

    def test_disallowed_relative_import(self) -> None:
        ok, err = sandbox.validate_imports(
            "from . import something\n"
        )
        self.assertFalse(ok)
        self.assertIn("relative", err)

    def test_syntax_error_detected(self) -> None:
        ok, err = sandbox.validate_imports("def f(:\n    pass\n")
        self.assertFalse(ok)
        self.assertIn("SyntaxError", err)


class TestExecutePython(unittest.TestCase):
    def test_simple_ok(self) -> None:
        result = sandbox.execute_python("print('hello')")
        self.assertEqual(result["status"], "ok")
        self.assertIn("hello", result["stdout"])
        self.assertEqual(result["returncode"], 0)

    def test_validation_error_blocks_execution(self) -> None:
        result = sandbox.execute_python("import os\nprint(os.getcwd())")
        self.assertEqual(result["status"], "validation_error")
        self.assertEqual(result["returncode"], None)
        self.assertIn("os", result["error"])
        self.assertEqual(result["stdout"], "")

    def test_syntax_error_blocks_execution(self) -> None:
        result = sandbox.execute_python("def f(:\n    pass\n")
        self.assertEqual(result["status"], "syntax_error")
        self.assertEqual(result["returncode"], None)

    def test_nonzero_exit(self) -> None:
        result = sandbox.execute_python(
            "import sys\nsys.exit(2)"
        )
        self.assertEqual(result["status"], "nonzero_exit")
        self.assertEqual(result["returncode"], 2)

    def test_timeout(self) -> None:
        # 写一个会卡住的循环, timeout=1s 强行 kill
        code = "import math\nwhile True:\n    math.sqrt(2.0)\n"
        result = sandbox.execute_python(code, timeout=1)
        self.assertEqual(result["status"], "timeout")
        self.assertIn("1s", result["error"])

    def test_argv_passed(self) -> None:
        code = "import sys\nprint('|'.join(sys.argv[1:]))"
        result = sandbox.execute_python(code, args=["foo", "bar"])
        self.assertEqual(result["status"], "ok")
        self.assertIn("foo|bar", result["stdout"])


class TestExecuteVerification(unittest.TestCase):
    def test_parses_json_from_last_line(self) -> None:
        code = (
            "import json\n"
            "print('some preamble')\n"
            "print(json.dumps({'count': 42}))\n"
        )
        result = sandbox.execute_verification(code)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["parsed_json"], {"count": 42})

    def test_no_json_status_changed(self) -> None:
        result = sandbox.execute_verification("print('plain text only')")
        self.assertEqual(result["status"], "ok_no_json")
        self.assertIsNone(result["parsed_json"])

    def test_validation_failure_flows_through(self) -> None:
        result = sandbox.execute_verification("import os\n")
        self.assertEqual(result["status"], "validation_error")
        self.assertIsNone(result["parsed_json"])

    def test_finds_json_among_multiple_lines(self) -> None:
        code = (
            "import json\n"
            "print('debug')\n"
            "print(json.dumps([1, 2, 3]))\n"
            "print('trailing log')\n"
        )
        result = sandbox.execute_verification(code)
        # last JSON-shaped line is the array
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["parsed_json"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
