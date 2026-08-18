#!/usr/bin/env python3
"""Decision-causal native reference residuals inside JUDO decoder layers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import types
from typing import Any, Sequence

import torch
from torch import nn

from judo_semantic_refdiff import SemanticRefDiffMemory


SCHEMA = "judo-native-deep-residual-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeDeepCounterfactualResidual(SemanticRefDiffMemory):
    """Inject native signed/energy/agreement evidence only at answer causality."""

    def __init__(
        self,
        hidden_size: int,
        *,
        bottleneck_size: int = 256,
        num_heads: int = 8,
        injection_layers: Sequence[int] = (14, 21, 25),
        max_relative_rms: float = 0.08,
        fixed_scale_fraction: float = 0.90,
        direction_floor: float = 0.10,
    ) -> None:
        if not 0.0 < fixed_scale_fraction < 1.0:
            raise ValueError("fixed scale fraction must be in (0,1)")
        super().__init__(
            hidden_size,
            bottleneck_size=bottleneck_size,
            num_heads=num_heads,
            injection_layers=injection_layers,
            max_relative_rms=max_relative_rms,
            direction_floor=direction_floor,
        )
        self.fixed_scale_fraction = float(fixed_scale_fraction)
        self.native_norm = nn.LayerNorm(hidden_size)
        self.memory_type = nn.Parameter(torch.zeros(3, hidden_size, dtype=torch.float32))
        self.anomaly_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, 256),
            nn.SiLU(),
            nn.Linear(256, 2),
        )
        # Exact zero function at initialization without the zero-times-zero
        # gradient degeneracy of a learned global gate.
        nn.init.zeros_(self.out_proj.weight)
        self.global_scale.data.fill_(math.atanh(self.fixed_scale_fraction))
        self.global_scale.requires_grad_(False)
        self.last_anomaly_logits: torch.Tensor | None = None
        self._native_rms_sum = 0.0
        self._native_batches = 0

    def reset_runtime(self) -> None:
        super().reset_runtime()
        self.last_anomaly_logits = None

    def build_native_memory(self, query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if query.shape != reference.shape or query.ndim != 3:
            raise ValueError("aligned query/reference native features are required")
        q = self.native_norm(query.float())
        r = self.native_norm(reference.float())
        signed = q - r
        energy = signed.abs()
        agreement = q * r
        self.runtime_memory = torch.cat(
            [value + self.memory_type[index].view(1, 1, -1) for index, value in enumerate((signed, energy, agreement))],
            dim=1,
        )
        self.last_anomaly_logits = self.anomaly_mlp(
            torch.cat((energy.mean(dim=1), energy.amax(dim=1)), dim=-1)
        )
        self._native_rms_sum += float(signed.detach().square().mean().sqrt().cpu())
        self._native_batches += 1
        return self.runtime_memory

    def statistics(self) -> dict[str, Any]:
        value = super().statistics()
        value.update(
            {
                "architecture": "decision-causal-native-deep-residual-v1",
                "bottleneck_size": self.bottleneck_size,
                "num_heads": self.num_heads,
                "fixed_scale_fraction": self.fixed_scale_fraction,
                "mean_native_difference_rms": self._native_rms_sum / max(1, self._native_batches),
                "native_memory_batches": self._native_batches,
            }
        )
        return value


class CounterfactualCausalHyperAdapter(NativeDeepCounterfactualResidual):
    """Comparison-conditioned low-rank adaptation at the exact answer state.

    The adapter combines token-level cross-attention with a native-difference
    conditioned hyper-adapter.  It is causal-token gated, so every hidden
    state that produced the original CoT remains bitwise untouched.
    """

    def __init__(self, hidden_size: int, *, hyper_rank: int = 256, **kwargs: Any) -> None:
        if hyper_rank < 8:
            raise ValueError("hyper_rank must be at least 8")
        super().__init__(hidden_size, **kwargs)
        self.hyper_rank = int(hyper_rank)
        self.condition_norm = nn.LayerNorm(hidden_size * 4)
        self.condition_proj = nn.Sequential(
            nn.Linear(hidden_size * 4, hyper_rank * 2),
            nn.SiLU(),
            nn.Linear(hyper_rank * 2, hyper_rank),
        )
        self.hyper_down = nn.ModuleList(
            nn.Linear(hidden_size, hyper_rank, bias=False) for _ in self.injection_layers
        )
        self.hyper_up = nn.ModuleList(
            nn.Linear(hyper_rank, hidden_size, bias=False) for _ in self.injection_layers
        )
        for layer in self.hyper_up:
            nn.init.zeros_(layer.weight)
        self._active_hyper_site = 0

    def _attention(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended = super()._attention(hidden_states, memory)
        if memory.shape[1] % 3:
            raise RuntimeError("hyper-adapter requires signed/energy/agreement memory")
        signed, energy, agreement = memory.chunk(3, dim=1)
        condition = torch.cat(
            (
                signed.mean(dim=1),
                energy.mean(dim=1),
                energy.amax(dim=1),
                agreement.mean(dim=1),
            ),
            dim=-1,
        )
        condition = self.condition_proj(self.condition_norm(condition)).unsqueeze(1)
        site = self._active_hyper_site
        low_rank = torch.nn.functional.silu(self.hyper_down[site](hidden_states) + condition)
        return attended + self.hyper_up[site](low_rank)

    def inject(self, hidden_states: torch.Tensor, site_index: int) -> torch.Tensor:
        self._active_hyper_site = int(site_index)
        return super().inject(hidden_states, site_index)

    def statistics(self) -> dict[str, Any]:
        value = super().statistics()
        value.update(
            {
                "architecture": "counterfactual-causal-hyperadapter-v1",
                "hyper_rank": self.hyper_rank,
                "cot_preservation": "exact causal-token gating",
            }
        )
        return value


class TransportEquivariantCausalAdapter(CounterfactualCausalHyperAdapter):
    """Align reference patches before constructing causal comparison evidence.

    Industrial query and normal-reference images are rarely registered.  A
    same-index subtraction therefore confounds pose/background displacement
    with defects.  This module learns a shared low-dimensional metric and a
    bidirectional soft transport plan before forming signed, energy, and
    agreement evidence.  The downstream causal hyper-adapter is unchanged, so
    the exact zero-function initialization and answer-state-only contract are
    preserved.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        transport_rank: int = 64,
        transport_temperature: float = 0.07,
        **kwargs: Any,
    ) -> None:
        if transport_rank < 8:
            raise ValueError("transport_rank must be at least 8")
        if not 0.01 <= transport_temperature <= 1.0:
            raise ValueError("transport_temperature must be in [0.01, 1.0]")
        super().__init__(hidden_size, **kwargs)
        self.transport_rank = int(transport_rank)
        self.transport_q = nn.Linear(hidden_size, transport_rank, bias=False)
        self.transport_k = nn.Linear(hidden_size, transport_rank, bias=False)
        nn.init.normal_(self.transport_q.weight, std=hidden_size ** -0.5)
        self.transport_k.weight.data.copy_(self.transport_q.weight.data)
        self.transport_log_temperature = nn.Parameter(
            torch.tensor(math.log(float(transport_temperature)), dtype=torch.float32)
        )
        self.last_transport_cycle_loss: torch.Tensor | None = None
        self._transport_entropy_sum = 0.0
        self._transport_batches = 0

    def reset_runtime(self) -> None:
        super().reset_runtime()
        self.last_transport_cycle_loss = None

    def build_native_memory(self, query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if query.shape != reference.shape or query.ndim != 3:
            raise ValueError("aligned token counts are required for soft transport")
        q = self.native_norm(query.float())
        r = self.native_norm(reference.float())
        q_key = torch.nn.functional.normalize(self.transport_q(q), dim=-1)
        r_key = torch.nn.functional.normalize(self.transport_k(r), dim=-1)
        temperature = self.transport_log_temperature.exp().clamp(0.01, 1.0)
        scores = torch.bmm(q_key, r_key.transpose(1, 2)) / temperature
        q_to_r = scores.softmax(dim=-1)
        r_to_q = scores.transpose(1, 2).softmax(dim=-1)
        aligned_reference = torch.bmm(q_to_r, r)
        aligned_query = torch.bmm(r_to_q, q)
        cycled_query = torch.bmm(q_to_r, aligned_query)

        signed = q - aligned_reference
        energy = signed.abs()
        agreement = q * aligned_reference
        self.runtime_memory = torch.cat(
            [value + self.memory_type[index].view(1, 1, -1)
             for index, value in enumerate((signed, energy, agreement))],
            dim=1,
        )
        self.last_anomaly_logits = self.anomaly_mlp(
            torch.cat((energy.mean(dim=1), energy.amax(dim=1)), dim=-1)
        )
        # Stop the native target, but let correspondence projections learn a
        # cycle-consistent transport plan through cycled_query.
        self.last_transport_cycle_loss = torch.nn.functional.smooth_l1_loss(
            cycled_query, q.detach()
        )
        entropy = -(q_to_r.clamp_min(1e-8) * q_to_r.clamp_min(1e-8).log()).sum(dim=-1).mean()
        self._transport_entropy_sum += float(entropy.detach().cpu())
        self._transport_batches += 1
        self._native_rms_sum += float(signed.detach().square().mean().sqrt().cpu())
        self._native_batches += 1
        return self.runtime_memory

    def statistics(self) -> dict[str, Any]:
        value = super().statistics()
        value.update(
            {
                "architecture": "transport-equivariant-causal-adapter-v1",
                "transport_rank": self.transport_rank,
                "transport_temperature": float(
                    self.transport_log_temperature.detach().exp().clamp(0.01, 1.0).cpu()
                ),
                "mean_transport_entropy": self._transport_entropy_sum / max(1, self._transport_batches),
                "transport_batches": self._transport_batches,
            }
        )
        return value


class DefectPreservingPartialTransportCausalAdapter(TransportEquivariantCausalAdapter):
    """Allow defect patches to remain unmatched to the normal reference.

    Balanced softmax transport must assign every query patch to some normal
    patch.  That can erase a defect whenever an unrelated normal patch happens
    to look similar.  This variant learns a feature-dependent dustbin mass for
    both transport directions.  Real-patch correspondence is renormalized, and
    the unmatched query mass amplifies difference evidence while attenuating
    false agreement.  The downstream intervention remains exactly zero at
    initialization and answer-state causal.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        unmatched_hidden: int = 64,
        unmatched_prior: float = 0.05,
        **kwargs: Any,
    ) -> None:
        if unmatched_hidden < 8:
            raise ValueError("unmatched_hidden must be at least 8")
        if not 0.001 <= unmatched_prior <= 0.25:
            raise ValueError("unmatched_prior must be in [0.001, 0.25]")
        super().__init__(hidden_size, **kwargs)
        self.unmatched_hidden = int(unmatched_hidden)
        self.unmatched_prior = float(unmatched_prior)
        self.unmatched_score = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, unmatched_hidden),
            nn.SiLU(),
            nn.Linear(unmatched_hidden, 1),
        )
        nn.init.zeros_(self.unmatched_score[-1].weight)
        nn.init.zeros_(self.unmatched_score[-1].bias)
        self.register_buffer(
            "unmatched_prior_logit",
            torch.tensor(math.log(unmatched_prior / (1.0 - unmatched_prior))),
            persistent=True,
        )
        self.last_unmatched_mass: torch.Tensor | None = None
        self._unmatched_mass_sum = 0.0
        self._unmatched_batches = 0

    def reset_runtime(self) -> None:
        super().reset_runtime()
        self.last_unmatched_mass = None

    def _partial_plan(
        self,
        scores: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Anchoring the dustbin logit to logsumexp(real scores) makes the
        # initial unmatched probability equal to `unmatched_prior` regardless
        # of token count or correspondence-temperature scale.
        offset = self.unmatched_score(features).squeeze(-1)
        dustbin = (
            torch.logsumexp(scores, dim=-1).detach()
            + self.unmatched_prior_logit.to(scores.dtype)
            + offset
        )
        plan = torch.cat((scores, dustbin.unsqueeze(-1)), dim=-1).softmax(dim=-1)
        return plan[..., :-1], plan[..., -1]

    def build_native_memory(self, query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if query.shape != reference.shape or query.ndim != 3:
            raise ValueError("aligned token counts are required for partial soft transport")
        q = self.native_norm(query.float())
        r = self.native_norm(reference.float())
        q_key = torch.nn.functional.normalize(self.transport_q(q), dim=-1)
        r_key = torch.nn.functional.normalize(self.transport_k(r), dim=-1)
        temperature = self.transport_log_temperature.exp().clamp(0.01, 1.0)
        scores = torch.bmm(q_key, r_key.transpose(1, 2)) / temperature
        q_real, q_unmatched = self._partial_plan(scores, q)
        r_real, _r_unmatched = self._partial_plan(scores.transpose(1, 2), r)
        q_mass = q_real.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        r_mass = r_real.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned_reference = torch.bmm(q_real, r) / q_mass
        aligned_query = torch.bmm(r_real, q) / r_mass
        cycled_query = torch.bmm(q_real, aligned_query) / q_mass

        signed = q - aligned_reference
        energy = signed.abs() * (1.0 + q_unmatched.unsqueeze(-1))
        agreement = q * aligned_reference * (1.0 - q_unmatched.unsqueeze(-1))
        self.runtime_memory = torch.cat(
            [value + self.memory_type[index].view(1, 1, -1)
             for index, value in enumerate((signed, energy, agreement))],
            dim=1,
        )
        self.last_anomaly_logits = self.anomaly_mlp(
            torch.cat((energy.mean(dim=1), energy.amax(dim=1)), dim=-1)
        )
        cycle_per_patch = torch.nn.functional.smooth_l1_loss(
            cycled_query, q.detach(), reduction="none"
        ).mean(dim=-1)
        matched = (1.0 - q_unmatched).detach()
        self.last_transport_cycle_loss = (
            cycle_per_patch * matched
        ).sum() / matched.sum().clamp_min(1.0)
        self.last_unmatched_mass = q_unmatched
        normalized_real = q_real / q_mass
        entropy = -(
            normalized_real.clamp_min(1e-8)
            * normalized_real.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        self._transport_entropy_sum += float(entropy.detach().cpu())
        self._transport_batches += 1
        self._unmatched_mass_sum += float(q_unmatched.detach().mean().cpu())
        self._unmatched_batches += 1
        self._native_rms_sum += float(signed.detach().square().mean().sqrt().cpu())
        self._native_batches += 1
        return self.runtime_memory

    def statistics(self) -> dict[str, Any]:
        value = super().statistics()
        value.update(
            {
                "architecture": "defect-preserving-partial-transport-causal-adapter-v1",
                "unmatched_hidden": self.unmatched_hidden,
                "unmatched_prior": self.unmatched_prior,
                "mean_unmatched_mass": self._unmatched_mass_sum / max(1, self._unmatched_batches),
                "unmatched_batches": self._unmatched_batches,
            }
        )
        return value


def install_native_deep_residual(
    model: Any,
    *,
    decision_prefix_ids: Sequence[int],
    bottleneck_size: int = 256,
    num_heads: int = 8,
    injection_layers: Sequence[int] = (14, 21, 25),
    max_relative_rms: float = 0.08,
    fixed_scale_fraction: float = 0.90,
    direction_floor: float = 0.10,
) -> NativeDeepCounterfactualResidual:
    model.requires_grad_(False)
    adapter = NativeDeepCounterfactualResidual(
        int(model.config.vision_config.out_hidden_size),
        bottleneck_size=bottleneck_size,
        num_heads=num_heads,
        injection_layers=injection_layers,
        max_relative_rms=max_relative_rms,
        fixed_scale_fraction=fixed_scale_fraction,
        direction_floor=direction_floor,
    )
    adapter.to(device=next(model.parameters()).device, dtype=torch.float32)
    adapter.configure_decision_prefix(decision_prefix_ids)
    model.model.native_deep_counterfactual_residual = adapter

    layers = getattr(getattr(model.model, "language_model", None), "layers", None)
    if layers is None:
        layers = getattr(model.model, "layers", None)
    if layers is None:
        raise AttributeError("unable to locate decoder layers")
    if any(index < 0 or index >= len(layers) for index in adapter.injection_layers):
        raise ValueError("native residual injection layer is invalid")

    def get_image_features(
        self: Any,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required")
        self.native_deep_counterfactual_residual.reset_runtime()
        pixel_values = pixel_values.type(self.visual.dtype)
        with torch.no_grad():
            result = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = result[0] if isinstance(result, tuple) else result
        sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        splits = list(torch.split(image_embeds, sizes))
        if len(splits) % 2:
            raise RuntimeError("complete query/reference pairs are required")
        query_splits = [splits[index].detach() for index in range(0, len(splits), 2)]
        reference_splits = [splits[index].detach() for index in range(1, len(splits), 2)]
        if len({tuple(value.shape) for value in query_splits + reference_splits}) != 1:
            raise RuntimeError("query/reference token grids are not aligned")
        self.native_deep_counterfactual_residual.build_native_memory(
            torch.stack(query_splits), torch.stack(reference_splits)
        )
        return [value.detach() for value in splits]

    model.model.get_image_features = types.MethodType(get_image_features, model.model)
    handles = []
    for site, layer_index in enumerate(adapter.injection_layers):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, _site: int = site) -> Any:
            if isinstance(output, tuple):
                return (adapter.inject(output[0], _site), *output[1:])
            return adapter.inject(output, _site)

        handles.append(layers[layer_index].register_forward_hook(hook))

    def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            embeds = kwargs.get("inputs_embeds")
            if embeds is None:
                raise RuntimeError("native deep residual requires input ids")
            adapter.runtime_decision_mask = torch.zeros(embeds.shape[:2], device=embeds.device)
            return None
        cache_position, past = kwargs.get("cache_position"), kwargs.get("past_key_values")
        if cache_position is not None and cache_position.numel():
            initial = int(cache_position.reshape(-1)[0].item()) == 0
        elif past is None:
            initial = True
        else:
            get_length = getattr(past, "get_seq_length", None)
            initial = bool(get_length is not None and int(get_length()) == 0)
        adapter.prepare_decision_mask(
            input_ids, attention_mask=kwargs.get("attention_mask"), initial_call=initial
        )
        return None

    handles.append(model.register_forward_pre_hook(pre_hook, with_kwargs=True))
    model._native_deep_residual_handles = handles
    return adapter


def install_counterfactual_causal_hyperadapter(
    model: Any,
    *,
    decision_prefix_ids: Sequence[int],
    hyper_rank: int = 256,
    bottleneck_size: int = 256,
    num_heads: int = 8,
    injection_layers: Sequence[int] = (18, 20, 22, 24, 26),
    max_relative_rms: float = 0.10,
    fixed_scale_fraction: float = 0.90,
    direction_floor: float = 0.10,
) -> CounterfactualCausalHyperAdapter:
    """Install the stronger conditional adapter while reusing pair extraction."""
    base = install_native_deep_residual(
        model,
        decision_prefix_ids=decision_prefix_ids,
        bottleneck_size=bottleneck_size,
        num_heads=num_heads,
        injection_layers=injection_layers,
        max_relative_rms=max_relative_rms,
        fixed_scale_fraction=fixed_scale_fraction,
        direction_floor=direction_floor,
    )
    for handle in model._native_deep_residual_handles:
        handle.remove()
    adapter = CounterfactualCausalHyperAdapter(
        int(model.config.vision_config.out_hidden_size),
        hyper_rank=hyper_rank,
        bottleneck_size=bottleneck_size,
        num_heads=num_heads,
        injection_layers=injection_layers,
        max_relative_rms=max_relative_rms,
        fixed_scale_fraction=fixed_scale_fraction,
        direction_floor=direction_floor,
    )
    adapter.to(device=next(model.parameters()).device, dtype=torch.float32)
    adapter.configure_decision_prefix(decision_prefix_ids)
    model.model.native_deep_counterfactual_residual = adapter
    del base

    layers = getattr(getattr(model.model, "language_model", None), "layers", None)
    if layers is None:
        layers = getattr(model.model, "layers", None)
    if layers is None:
        raise AttributeError("unable to locate decoder layers")
    handles = []
    for site, layer_index in enumerate(adapter.injection_layers):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, _site: int = site) -> Any:
            if isinstance(output, tuple):
                return (adapter.inject(output[0], _site), *output[1:])
            return adapter.inject(output, _site)

        handles.append(layers[layer_index].register_forward_hook(hook))

    def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            embeds = kwargs.get("inputs_embeds")
            if embeds is None:
                raise RuntimeError("causal hyper-adapter requires input ids")
            adapter.runtime_decision_mask = torch.zeros(embeds.shape[:2], device=embeds.device)
            return None
        cache_position, past = kwargs.get("cache_position"), kwargs.get("past_key_values")
        if cache_position is not None and cache_position.numel():
            initial = int(cache_position.reshape(-1)[0].item()) == 0
        elif past is None:
            initial = True
        else:
            get_length = getattr(past, "get_seq_length", None)
            initial = bool(get_length is not None and int(get_length()) == 0)
        adapter.prepare_decision_mask(
            input_ids, attention_mask=kwargs.get("attention_mask"), initial_call=initial
        )
        return None

    handles.append(model.register_forward_pre_hook(pre_hook, with_kwargs=True))
    model._native_deep_residual_handles = handles
    return adapter


def install_transport_equivariant_causal_adapter(
    model: Any,
    *,
    decision_prefix_ids: Sequence[int],
    transport_rank: int = 64,
    transport_temperature: float = 0.07,
    hyper_rank: int = 256,
    bottleneck_size: int = 256,
    num_heads: int = 8,
    injection_layers: Sequence[int] = (18, 20, 22, 24, 26),
    max_relative_rms: float = 0.10,
    fixed_scale_fraction: float = 0.90,
    direction_floor: float = 0.10,
) -> TransportEquivariantCausalAdapter:
    """Install soft patch transport followed by the causal hyper-adapter."""
    previous = install_counterfactual_causal_hyperadapter(
        model,
        decision_prefix_ids=decision_prefix_ids,
        hyper_rank=hyper_rank,
        bottleneck_size=bottleneck_size,
        num_heads=num_heads,
        injection_layers=injection_layers,
        max_relative_rms=max_relative_rms,
        fixed_scale_fraction=fixed_scale_fraction,
        direction_floor=direction_floor,
    )
    for handle in model._native_deep_residual_handles:
        handle.remove()
    adapter = TransportEquivariantCausalAdapter(
        int(model.config.vision_config.out_hidden_size),
        transport_rank=transport_rank,
        transport_temperature=transport_temperature,
        hyper_rank=hyper_rank,
        bottleneck_size=bottleneck_size,
        num_heads=num_heads,
        injection_layers=injection_layers,
        max_relative_rms=max_relative_rms,
        fixed_scale_fraction=fixed_scale_fraction,
        direction_floor=direction_floor,
    )
    adapter.to(device=next(model.parameters()).device, dtype=torch.float32)
    adapter.configure_decision_prefix(decision_prefix_ids)
    model.model.native_deep_counterfactual_residual = adapter
    del previous

    layers = getattr(getattr(model.model, "language_model", None), "layers", None)
    if layers is None:
        layers = getattr(model.model, "layers", None)
    if layers is None:
        raise AttributeError("unable to locate decoder layers")
    handles = []
    for site, layer_index in enumerate(adapter.injection_layers):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, _site: int = site) -> Any:
            if isinstance(output, tuple):
                return (adapter.inject(output[0], _site), *output[1:])
            return adapter.inject(output, _site)

        handles.append(layers[layer_index].register_forward_hook(hook))

    def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            embeds = kwargs.get("inputs_embeds")
            if embeds is None:
                raise RuntimeError("transport causal adapter requires input ids")
            adapter.runtime_decision_mask = torch.zeros(embeds.shape[:2], device=embeds.device)
            return None
        cache_position, past = kwargs.get("cache_position"), kwargs.get("past_key_values")
        if cache_position is not None and cache_position.numel():
            initial = int(cache_position.reshape(-1)[0].item()) == 0
        elif past is None:
            initial = True
        else:
            get_length = getattr(past, "get_seq_length", None)
            initial = bool(get_length is not None and int(get_length()) == 0)
        adapter.prepare_decision_mask(
            input_ids, attention_mask=kwargs.get("attention_mask"), initial_call=initial
        )
        return None

    handles.append(model.register_forward_pre_hook(pre_hook, with_kwargs=True))
    model._native_deep_residual_handles = handles
    return adapter


def install_defect_preserving_partial_transport_causal_adapter(
    model: Any,
    *,
    decision_prefix_ids: Sequence[int],
    unmatched_hidden: int = 64,
    unmatched_prior: float = 0.05,
    transport_rank: int = 64,
    transport_temperature: float = 0.07,
    hyper_rank: int = 256,
    bottleneck_size: int = 256,
    num_heads: int = 8,
    injection_layers: Sequence[int] = (18, 20, 22, 24, 26),
    max_relative_rms: float = 0.08,
    fixed_scale_fraction: float = 0.80,
    direction_floor: float = 0.10,
) -> DefectPreservingPartialTransportCausalAdapter:
    """Install defect-preserving partial transport at the answer boundary."""
    previous = install_transport_equivariant_causal_adapter(
        model,
        decision_prefix_ids=decision_prefix_ids,
        transport_rank=transport_rank,
        transport_temperature=transport_temperature,
        hyper_rank=hyper_rank,
        bottleneck_size=bottleneck_size,
        num_heads=num_heads,
        injection_layers=injection_layers,
        max_relative_rms=max_relative_rms,
        fixed_scale_fraction=fixed_scale_fraction,
        direction_floor=direction_floor,
    )
    for handle in model._native_deep_residual_handles:
        handle.remove()
    adapter = DefectPreservingPartialTransportCausalAdapter(
        int(model.config.vision_config.out_hidden_size),
        unmatched_hidden=unmatched_hidden,
        unmatched_prior=unmatched_prior,
        transport_rank=transport_rank,
        transport_temperature=transport_temperature,
        hyper_rank=hyper_rank,
        bottleneck_size=bottleneck_size,
        num_heads=num_heads,
        injection_layers=injection_layers,
        max_relative_rms=max_relative_rms,
        fixed_scale_fraction=fixed_scale_fraction,
        direction_floor=direction_floor,
    )
    adapter.to(device=next(model.parameters()).device, dtype=torch.float32)
    adapter.configure_decision_prefix(decision_prefix_ids)
    model.model.native_deep_counterfactual_residual = adapter
    del previous

    layers = getattr(getattr(model.model, "language_model", None), "layers", None)
    if layers is None:
        layers = getattr(model.model, "layers", None)
    if layers is None:
        raise AttributeError("unable to locate decoder layers")
    handles = []
    for site, layer_index in enumerate(adapter.injection_layers):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, _site: int = site) -> Any:
            if isinstance(output, tuple):
                return (adapter.inject(output[0], _site), *output[1:])
            return adapter.inject(output, _site)

        handles.append(layers[layer_index].register_forward_hook(hook))

    def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            embeds = kwargs.get("inputs_embeds")
            if embeds is None:
                raise RuntimeError("partial transport causal adapter requires input ids")
            adapter.runtime_decision_mask = torch.zeros(embeds.shape[:2], device=embeds.device)
            return None
        cache_position, past = kwargs.get("cache_position"), kwargs.get("past_key_values")
        if cache_position is not None and cache_position.numel():
            initial = int(cache_position.reshape(-1)[0].item()) == 0
        elif past is None:
            initial = True
        else:
            get_length = getattr(past, "get_seq_length", None)
            initial = bool(get_length is not None and int(get_length()) == 0)
        adapter.prepare_decision_mask(
            input_ids, attention_mask=kwargs.get("attention_mask"), initial_call=initial
        )
        return None

    handles.append(model.register_forward_pre_hook(pre_hook, with_kwargs=True))
    model._native_deep_residual_handles = handles
    return adapter


def trainable_contract(model: Any, adapter: nn.Module, trainable: bool) -> dict[str, int]:
    model.requires_grad_(False)
    adapter.requires_grad_(trainable)
    if trainable and hasattr(adapter, "global_scale"):
        adapter.global_scale.requires_grad_(False)
    actual = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    expected = sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad)
    if actual != expected:
        raise RuntimeError(f"native deep residual trainable contract failed: {actual} != {expected}")
    return {
        "adapter_parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "trainable_parameter_count": actual,
    }


def save_adapter(adapter: NativeDeepCounterfactualResidual, output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    weights = output_dir / "native_deep_residual.safetensors"
    temporary = weights.with_suffix(".tmp")
    state = {name: value.detach().cpu().contiguous() for name, value in adapter.state_dict().items()}
    save_file(state, temporary, metadata={"format": "pt", "schema": SCHEMA})
    temporary.replace(weights)
    identity = {
        "schema_version": SCHEMA,
        "weights_sha256": sha256_file(weights),
        "parameter_count": sum(value.numel() for value in state.values()),
        "statistics": adapter.statistics(),
        **metadata,
    }
    (output_dir / "native_deep_residual_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identity


def load_adapter(adapter: NativeDeepCounterfactualResidual, directory: Path) -> dict[str, Any]:
    from safetensors import safe_open

    identity = json.loads((directory / "native_deep_residual_identity.json").read_text())
    weights = directory / "native_deep_residual.safetensors"
    if identity.get("schema_version") != SCHEMA or identity.get("weights_sha256") != sha256_file(weights):
        raise ValueError("native deep residual identity mismatch")
    tensors = {}
    with safe_open(weights, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensors[name] = handle.get_tensor(name)
    adapter.load_state_dict(tensors, strict=True)
    return identity


__all__ = [
    "CounterfactualCausalHyperAdapter",
    "NativeDeepCounterfactualResidual",
    "TransportEquivariantCausalAdapter",
    "install_counterfactual_causal_hyperadapter",
    "install_native_deep_residual",
    "install_transport_equivariant_causal_adapter",
    "load_adapter",
    "save_adapter",
    "trainable_contract",
]
