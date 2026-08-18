from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


ABLATION = Path(__file__).resolve().parents[1]
if str(ABLATION) not in sys.path:
    sys.path.insert(0, str(ABLATION))

from judo_caver import ARCHITECTURE, CausalVisualBeliefReplayAdapter, normal_null


def adapter() -> CausalVisualBeliefReplayAdapter:
    value = CausalVisualBeliefReplayAdapter(
        16,
        belief_rank=4,
        belief_temperature=1.0,
        belief_scale_fraction=0.5,
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


class CaverArchitectureTests(unittest.TestCase):
    def test_zero_initialized_binding_is_exact_identity(self) -> None:
        value = adapter()
        query, reference = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
        value.build_native_memory(query, reference)
        value.runtime_decision_mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        value.runtime_replay_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        hidden = torch.randn(2, 2, 16)
        self.assertTrue(torch.equal(value.inject(hidden, 0), hidden))

    def test_binding_path_is_causal_after_training(self) -> None:
        value = adapter()
        with torch.no_grad():
            value.belief_up[0].weight.fill_(0.1)
        query, reference = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
        value.build_native_memory(query, reference)
        value.runtime_decision_mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        value.runtime_replay_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        hidden = torch.randn(2, 2, 16)
        actual = value.inject(hidden, 0)
        self.assertFalse(torch.equal(actual, hidden))
        self.assertTrue(torch.isfinite(actual).all())

    def test_only_causal_modules_are_exposed_for_continuation(self) -> None:
        value = adapter()
        names = {name for name, parameter in value.named_parameters() if any(
            parameter is candidate for candidate in value.causal_parameters()
        )}
        self.assertTrue(any(name.startswith("visual_verdict.") for name in names))
        self.assertTrue(any(name.startswith("belief_up.") for name in names))
        self.assertFalse(any(name.startswith("q_proj.") for name in names))
        self.assertEqual(value.statistics()["architecture"], ARCHITECTURE)


class CaverCounterfactualTests(unittest.TestCase):
    def test_normal_null_uses_reference_as_both_images_and_no_target(self) -> None:
        row = {
            "sample_id": "x",
            "image": "query.png",
            "template_image": "normal.png",
            "label": "anomaly",
            "correct_answer": "A",
            "options": {"A": "Yes", "B": "No", "C": "Maybe", "D": "Unknown"},
        }
        value = normal_null(row)
        self.assertEqual(value["image"], "normal.png")
        self.assertEqual(value["template_image"], "normal.png")
        self.assertEqual(value["correct_answer"], "B")
        self.assertEqual(value["label"], "normal")
        self.assertEqual(value["teacher_segmentation"], "None")

    def test_normal_null_resolves_shuffled_semantic_no_option(self) -> None:
        row = {
            "sample_id": "x",
            "image": "query.png",
            "template_image": "normal.png",
            "label": "anomaly",
            "correct_answer": "B",
            "options": {"A": "Unknown.", "B": "Yes.", "C": "No.", "D": "Maybe."},
        }
        value = normal_null(row)
        self.assertEqual(value["correct_answer"], "C")


if __name__ == "__main__":
    unittest.main()
