import inspect
import unittest

from efnas.nas import supernet_subnet_distribution_v2


class TestSupernetV2VelaExport(unittest.TestCase):
    def test_tflite_export_uses_fixed_subnet_builder(self) -> None:
        source = inspect.getsource(supernet_subnet_distribution_v2._build_tflite_for_arch_v2)

        self.assertIn("_FixedSubnetForExportV2", source)
        self.assertNotIn("MultiScaleResNetSupernetV2", source)


if __name__ == "__main__":
    unittest.main()
