from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

from judo_aligned_ce import GatedResidualCrossAttention  # noqa: E402


class AlignedCETest(unittest.TestCase):
    def test_zero_gate_is_exact_identity_and_receives_gradient(self) -> None:
        torch.manual_seed(7)
        module = GatedResidualCrossAttention(
            hidden_size=32,
            bottleneck_size=16,
            num_heads=4,
        )
        base = torch.randn(11, 32)
        comparison = torch.randn(5, 32)
        output = module(base, comparison)
        self.assertTrue(torch.equal(output, base))
        output.square().mean().backward()
        self.assertIsNotNone(module.global_scale.grad)
        self.assertTrue(torch.isfinite(module.global_scale.grad))


if __name__ == "__main__":
    unittest.main()
