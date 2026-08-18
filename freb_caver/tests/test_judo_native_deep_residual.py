from __future__ import annotations

import unittest

import torch

from judo_native_deep_residual import (
    CounterfactualCausalHyperAdapter,
    DefectPreservingPartialTransportCausalAdapter,
    NativeDeepCounterfactualResidual,
    TransportEquivariantCausalAdapter,
)


class NativeDeepCounterfactualResidualTests(unittest.TestCase):
    def make_adapter(self) -> NativeDeepCounterfactualResidual:
        adapter = NativeDeepCounterfactualResidual(
            16,
            bottleneck_size=8,
            num_heads=2,
            injection_layers=(0,),
            max_relative_rms=0.08,
            fixed_scale_fraction=0.90,
        )
        adapter.configure_decision_prefix((11, 13))
        return adapter

    def prepare(self, adapter: NativeDeepCounterfactualResidual) -> torch.Tensor:
        adapter.build_native_memory(torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        ids = torch.tensor([[1, 11, 13], [2, 11, 13]])
        adapter.prepare_decision_mask(ids, attention_mask=torch.ones_like(ids), initial_call=True)
        return torch.randn(2, 3, 16)

    def test_zero_initialized_residual_is_exact_identity(self) -> None:
        adapter = self.make_adapter()
        hidden = self.prepare(adapter)
        output = adapter.inject(hidden.clone(), 0)
        self.assertTrue(torch.equal(output, hidden))

    def test_only_answer_decision_state_changes(self) -> None:
        adapter = self.make_adapter()
        torch.nn.init.normal_(adapter.out_proj.weight, std=0.1)
        hidden = self.prepare(adapter)
        output = adapter.inject(hidden.clone(), 0)
        self.assertTrue(torch.equal(output[:, :-1], hidden[:, :-1]))
        self.assertGreater(int(torch.count_nonzero(output[:, -1] - hidden[:, -1])), 0)
        observed = adapter.last_residual_ratios[-1][:, -1]
        self.assertTrue(torch.all(observed <= adapter.max_relative_rms + 1e-5))

    def test_first_backward_reaches_zero_initialized_output(self) -> None:
        adapter = self.make_adapter()
        hidden = self.prepare(adapter)
        output = adapter.inject(hidden, 0)
        output[:, -1].sum().backward()
        gradient = adapter.out_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_anomaly_auxiliary_path_receives_gradients(self) -> None:
        adapter = self.make_adapter()
        adapter.build_native_memory(torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        loss = adapter.last_anomaly_logits.square().mean()
        loss.backward()
        gradient = adapter.anomaly_mlp[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_hyperadapter_is_exact_then_learns_on_first_backward(self) -> None:
        adapter = CounterfactualCausalHyperAdapter(
            16,
            hyper_rank=8,
            bottleneck_size=8,
            num_heads=2,
            injection_layers=(0,),
            max_relative_rms=0.10,
            fixed_scale_fraction=0.90,
        )
        adapter.configure_decision_prefix((11, 13))
        hidden = self.prepare(adapter)
        output = adapter.inject(hidden, 0)
        self.assertTrue(torch.equal(output, hidden))
        output[:, -1].sum().backward()
        self.assertGreater(float(adapter.hyper_up[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(adapter.out_proj.weight.grad.abs().sum()), 0.0)

    def test_transport_adapter_has_exact_identity_and_trainable_cycle(self) -> None:
        adapter = TransportEquivariantCausalAdapter(
            16,
            transport_rank=8,
            transport_temperature=0.1,
            hyper_rank=8,
            bottleneck_size=8,
            num_heads=2,
            injection_layers=(0,),
            max_relative_rms=0.10,
            fixed_scale_fraction=0.90,
        )
        adapter.configure_decision_prefix((11, 13))
        hidden = self.prepare(adapter)
        output = adapter.inject(hidden, 0)
        self.assertTrue(torch.equal(output, hidden))
        self.assertIsNotNone(adapter.last_transport_cycle_loss)
        loss = output[:, -1].sum() + adapter.last_transport_cycle_loss
        loss.backward()
        self.assertGreater(float(adapter.transport_q.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(adapter.hyper_up[0].weight.grad.abs().sum()), 0.0)

    def test_transport_is_permutation_equivariant_without_position_assumption(self) -> None:
        adapter = TransportEquivariantCausalAdapter(
            16,
            transport_rank=8,
            hyper_rank=8,
            bottleneck_size=8,
            num_heads=2,
            injection_layers=(0,),
        )
        query, reference = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
        first = adapter.build_native_memory(query, reference).detach()
        permutation = torch.tensor([2, 0, 3, 1])
        second = adapter.build_native_memory(query, reference[:, permutation]).detach()
        self.assertTrue(torch.allclose(first, second, atol=2e-5, rtol=2e-5))

    def test_partial_transport_has_calibrated_trainable_unmatched_mass(self) -> None:
        adapter = DefectPreservingPartialTransportCausalAdapter(
            16,
            unmatched_hidden=8,
            unmatched_prior=0.05,
            transport_rank=8,
            hyper_rank=8,
            bottleneck_size=8,
            num_heads=2,
            injection_layers=(0,),
        )
        adapter.configure_decision_prefix((11, 13))
        hidden = self.prepare(adapter)
        output = adapter.inject(hidden, 0)
        self.assertTrue(torch.equal(output, hidden))
        self.assertIsNotNone(adapter.last_unmatched_mass)
        expected = torch.full_like(adapter.last_unmatched_mass, 0.05)
        self.assertTrue(torch.allclose(adapter.last_unmatched_mass, expected, atol=1e-6, rtol=0.0))
        loss = output[:, -1].sum() + adapter.last_unmatched_mass.square().mean()
        loss.backward()
        self.assertGreater(float(adapter.unmatched_score[-1].weight.grad.abs().sum()), 0.0)

    def test_partial_transport_preserves_reference_permutation_equivariance(self) -> None:
        adapter = DefectPreservingPartialTransportCausalAdapter(
            16,
            unmatched_hidden=8,
            transport_rank=8,
            hyper_rank=8,
            bottleneck_size=8,
            num_heads=2,
            injection_layers=(0,),
        )
        query, reference = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
        first = adapter.build_native_memory(query, reference).detach()
        first_mass = adapter.last_unmatched_mass.detach().clone()
        permutation = torch.tensor([2, 0, 3, 1])
        second = adapter.build_native_memory(query, reference[:, permutation]).detach()
        self.assertTrue(torch.allclose(first, second, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(first_mass, adapter.last_unmatched_mass, atol=2e-5, rtol=2e-5))


if __name__ == "__main__":
    unittest.main()
