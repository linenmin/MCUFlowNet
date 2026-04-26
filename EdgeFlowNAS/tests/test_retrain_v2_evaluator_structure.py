"""Structure tests for retrain_v2 evaluator graph handling."""

import ast
import unittest
from pathlib import Path


class TestRetrainV2EvaluatorStructure(unittest.TestCase):
    def test_setup_eval_uses_independent_graph_without_reset_default_graph(self) -> None:
        source_path = Path(__file__).resolve().parents[1] / "efnas" / "engine" / "retrain_v2_evaluator.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        has_reset_default_graph = False
        has_tf_graph_ctor = False
        has_graph_as_default = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "reset_default_graph":
                    has_reset_default_graph = True
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "Graph"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "tf"
                ):
                    has_tf_graph_ctor = True
                if isinstance(func, ast.Attribute) and func.attr == "as_default":
                    has_graph_as_default = True

        self.assertFalse(has_reset_default_graph, "nested eval graph must not call reset_default_graph()")
        self.assertTrue(has_tf_graph_ctor, "setup_retrain_v2_eval_model should create an explicit tf.Graph()")
        self.assertTrue(has_graph_as_default, "setup_retrain_v2_eval_model should build tensors under graph.as_default()")


if __name__ == "__main__":
    unittest.main()
