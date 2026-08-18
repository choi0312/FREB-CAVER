from __future__ import annotations

import unittest

import torch

from judo_semantic_refdiff import SemanticRefDiffMemory


class SemanticRefDiffMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.module = SemanticRefDiffMemory(
            hidden_size=32,
            bottleneck_size=16,
            num_heads=4,
            injection_layers=(1, 3, 5),
            max_relative_rms=0.005,
            direction_floor=0.10,
        )
        self.query = torch.randn(2, 5, 32)
        self.reference = torch.randn(2, 5, 32)
        self.hidden = torch.randn(2, 11, 32)

    def test_zero_scale_is_exact_identity(self) -> None:
        self.module.build_memory(self.query, self.reference)
        fused = self.module.inject(self.hidden, 0)
        self.assertTrue(torch.equal(fused, self.hidden))
        self.assertEqual(float(self.module.global_scale.detach()), 0.0)
        self.assertGreater(int(torch.count_nonzero(self.module.out_proj.weight)), 0)

    def test_memory_contains_signed_and_energy_halves(self) -> None:
        memory = self.module.build_memory(self.query, self.reference)
        self.assertEqual(tuple(memory.shape), (2, 10, 32))
        signed, energy = memory.chunk(2, dim=1)
        self.assertTrue(torch.equal(energy, signed.abs()))

    def test_identical_streams_produce_zero_effect_at_active_scale(self) -> None:
        self.module.global_scale.data.fill_(20.0)
        self.module.build_memory(self.query, self.query.clone())
        fused = self.module.inject(self.hidden, 1)
        self.assertTrue(torch.equal(fused, self.hidden))

    def test_each_site_respects_rms_bound(self) -> None:
        self.module.global_scale.data.fill_(20.0)
        self.module.site_logits.data.fill_(20.0)
        self.module.build_memory(self.query, self.reference)
        fused = self.module.inject(self.hidden, 2)
        base_rms = self.hidden.square().mean(dim=-1).sqrt()
        residual_rms = (fused - self.hidden).square().mean(dim=-1).sqrt()
        self.assertTrue(torch.all(residual_rms <= 0.005001 * base_rms))

    def test_question_states_can_select_different_memory(self) -> None:
        self.module.global_scale.data.fill_(1.0)
        self.module.build_memory(self.query, self.reference)
        fused = self.module.inject(self.hidden, 0)
        residual = fused - self.hidden
        question_delta = (residual[:, 0] - residual[:, 1]).detach().abs().sum()
        self.assertGreater(float(question_delta), 0.0)

    def test_decision_mask_ignores_format_example_inside_initial_prompt(self) -> None:
        self.module.configure_decision_prefix((27, 9217, 29))
        ids = torch.tensor([[1, 27, 9217, 29, 2, 3]])
        mask = self.module.prepare_decision_mask(
            ids, attention_mask=torch.ones_like(ids), initial_call=True
        )
        self.assertEqual(float(mask.sum()), 0.0)

    def test_decision_mask_tracks_prefix_across_cached_generation(self) -> None:
        self.module.configure_decision_prefix((27, 9217, 29))
        prompt = torch.tensor([[1, 2, 3]])
        self.module.prepare_decision_mask(
            prompt, attention_mask=torch.ones_like(prompt), initial_call=True
        )
        for token in (27, 9217):
            mask = self.module.prepare_decision_mask(
                torch.tensor([[token]]), attention_mask=None, initial_call=False
            )
            self.assertEqual(float(mask.sum()), 0.0)
        mask = self.module.prepare_decision_mask(
            torch.tensor([[29]]), attention_mask=None, initial_call=False
        )
        self.assertEqual(mask.tolist(), [[1.0]])

    def test_decision_mask_preserves_all_nondecision_tokens_exactly(self) -> None:
        self.module.global_scale.data.fill_(20.0)
        self.module.site_logits.data.fill_(20.0)
        self.module.build_memory(self.query, self.reference)
        self.module.configure_decision_prefix((27, 9217, 29))
        ids = torch.tensor([[1] * 10 + [29], [1] * 10 + [29]])
        self.module.runtime_decision_mask = torch.zeros_like(ids, dtype=torch.float32)
        self.module.runtime_decision_mask[:, -1] = 1.0
        fused = self.module.inject(self.hidden, 0)
        self.assertTrue(torch.equal(fused[:, :-1], self.hidden[:, :-1]))
        decision_delta = (fused[:, -1] - self.hidden[:, -1]).detach().abs().sum()
        self.assertGreater(float(decision_delta), 0.0)


if __name__ == "__main__":
    unittest.main()
