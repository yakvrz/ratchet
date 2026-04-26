from __future__ import annotations

import unittest

from ratchet.pricing import estimate_cost_usd


class PricingTests(unittest.TestCase):
    def test_estimate_cost_uses_model_specific_rates(self) -> None:
        self.assertAlmostEqual(
            estimate_cost_usd("gpt-4o-2024-08-06", input_tokens=1000, output_tokens=100),
            0.0035,
        )


if __name__ == "__main__":
    unittest.main()
