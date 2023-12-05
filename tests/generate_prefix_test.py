import unittest
import torch
from core import generate_prefix


class TestGeneratePrefixMethod(unittest.TestCase):
    def test_generate_prefix(self):
        # Test case 1
        result = generate_prefix(
            torch.tensor([[464, 1492, 318, 2266, 13]]),
            torch.tensor([[2061, 3124, 318, 262, 1492, 30]]),
        )
        self.assertEqual(result.tolist(), [[445]])

        # Add more test cases to cover different scenarios


if __name__ == "__main__":
    unittest.main()
