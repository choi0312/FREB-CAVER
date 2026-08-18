from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


ABLATION = Path(__file__).resolve().parents[1]
if str(ABLATION) not in sys.path:
    sys.path.insert(0, str(ABLATION))

from judo_graft import GroundedReferenceAnchoredReplayAdapter


def adapter() -> GroundedReferenceAnchoredReplayAdapter:
    value = GroundedReferenceAnchoredReplayAdapter(
        16,
        replay_max_relative_rms=0.02,
        replay_scale_fraction=0.8,
        unmatched_hidden=8,
        unmatched_prior=0.05,
        transport_rank=8,
        transport_temperature=0.07,
        hyper_rank=8,
        bottleneck_size=8,
        num_heads=2,
        injection_layers=(0,),
        max_relative_rms=0.08,
        fixed_scale_fraction=0.8,
    )
    value.configure_decision_prefix((20, 21))
    value.configure_phase_tokens(assistant_prefix_ids=(90, 91), seg_start_ids=(10, 11))
    return value


class GraftPhaseMaskTests(unittest.TestCase):
    def test_system_prompt_tag_is_not_replayed(self) -> None:
        value = adapter()
        ids = torch.tensor([[10, 11, 7, 90, 91, 20, 21]])
        value.prepare_decision_mask(ids, attention_mask=None, initial_call=True)
        replay = value.prepare_replay_mask(ids, attention_mask=None, initial_call=True)
        self.assertEqual(int(replay.sum()), 0)
        self.assertEqual(int(value.runtime_decision_mask.sum()), 1)
        self.assertEqual(int(value.runtime_decision_mask[0, -1]), 1)

    def test_teacher_forced_cot_replays_only_after_final_assistant(self) -> None:
        value = adapter()
        ids = torch.tensor([[10, 11, 7, 90, 91, 10, 11, 5, 6, 20, 21]])
        value.prepare_decision_mask(ids, attention_mask=None, initial_call=True)
        replay = value.prepare_replay_mask(ids, attention_mask=None, initial_call=True)
        active = replay[0].nonzero().flatten().tolist()
        self.assertEqual(active, [6, 7, 8, 9])
        self.assertEqual(int(value.runtime_decision_mask[0, 10]), 1)
        self.assertEqual(int((replay * value.runtime_decision_mask).sum()), 0)

    def test_cached_generation_phase_automaton(self) -> None:
        value = adapter()
        initial = torch.tensor([[90, 91]])
        value.prepare_decision_mask(initial, attention_mask=None, initial_call=True)
        value.prepare_replay_mask(initial, attention_mask=None, initial_call=True)
        observed = []
        decisions = []
        for token in (10, 11, 3, 20, 21):
            ids = torch.tensor([[token]])
            value.prepare_decision_mask(ids, attention_mask=None, initial_call=False)
            observed.append(int(value.prepare_replay_mask(ids, attention_mask=None, initial_call=False).item()))
            decisions.append(int(value.runtime_decision_mask.item()))
        self.assertEqual(observed, [0, 1, 1, 1, 0])
        self.assertEqual(decisions, [0, 0, 0, 0, 1])


class GraftArchitectureTests(unittest.TestCase):
    def test_zero_initialized_adapter_is_exact_identity(self) -> None:
        value = adapter()
        query = torch.randn(2, 4, 16)
        reference = torch.randn(2, 4, 16)
        value.build_native_memory(query, reference)
        value.runtime_decision_mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        value.runtime_replay_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        hidden = torch.randn(2, 2, 16)
        actual = value.inject(hidden, 0)
        self.assertTrue(torch.equal(actual, hidden))

    def test_dual_evidence_and_auxiliary_losses_exist(self) -> None:
        value = adapter()
        query = torch.randn(2, 4, 16)
        reference = torch.randn(2, 4, 16)
        memory = value.build_native_memory(query, reference)
        self.assertEqual(tuple(memory.shape), (2, 12, 16))
        self.assertEqual(tuple(value.runtime_defect_memory.shape), (2, 8, 16))
        self.assertEqual(tuple(value.runtime_normal_memory.shape), (2, 4, 16))
        self.assertEqual(tuple(value.last_visual_verdict_logits.shape), (2, 2))
        self.assertEqual(value.last_subspace_orthogonality.ndim, 0)
        self.assertTrue(torch.isfinite(value.last_subspace_orthogonality))

    def test_replay_trust_region_is_stricter_than_answer_region(self) -> None:
        value = adapter()
        self.assertLess(value.replay_max_relative_rms, value.max_relative_rms)
        stats = value.statistics()
        self.assertTrue(stats["predecision_cot_states_modified"])
        self.assertEqual(stats["dual_subspaces"], ["matched_normal", "unmatched_defect"])


if __name__ == "__main__":
    unittest.main()
