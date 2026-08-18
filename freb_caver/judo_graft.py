#!/usr/bin/env python3
"""Grounded Reference-Anchored Faithful Tuning (GRAFT) for JUDO.

GRAFT extends the defect-preserving partial-transport adapter in two ways:

* matched normal evidence and unmatched defect evidence are kept in separate
  attention branches; and
* the resulting evidence is replayed through the generated ``<seg>`` and
  ``<think>`` trajectory, instead of being introduced only at the final
  answer token.

The base JUDO checkpoint stays frozen.  Both branches have zero-initialized
output projections, so installing an untrained adapter is an exact functional
identity even though the phase gates are already active.
"""

from __future__ import annotations

import math
import types
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from judo_native_deep_residual import (
    DefectPreservingPartialTransportCausalAdapter,
    install_defect_preserving_partial_transport_causal_adapter,
)


ARCHITECTURE = "grounded-reference-anchored-faithful-tuning-v1"


def _last_subsequence(tokens: list[int], pattern: tuple[int, ...], start: int = 0) -> int:
    """Return the last start index of ``pattern`` at or after ``start``."""
    if not pattern or len(tokens) < len(pattern):
        return -1
    for index in range(len(tokens) - len(pattern), start - 1, -1):
        if tuple(tokens[index : index + len(pattern)]) == pattern:
            return index
    return -1


class GroundedReferenceAnchoredReplayAdapter(
    DefectPreservingPartialTransportCausalAdapter
):
    """Dual-subspace partial transport with causal visual-evidence replay."""

    def __init__(
        self,
        hidden_size: int,
        *,
        replay_max_relative_rms: float = 0.02,
        replay_scale_fraction: float = 0.80,
        **kwargs: Any,
    ) -> None:
        if not 0.0 < replay_max_relative_rms <= 0.05:
            raise ValueError("replay_max_relative_rms must be in (0, 0.05]")
        if not 0.0 < replay_scale_fraction < 1.0:
            raise ValueError("replay_scale_fraction must be in (0, 1)")
        super().__init__(hidden_size, **kwargs)
        self.replay_max_relative_rms = float(replay_max_relative_rms)
        self.replay_scale_fraction = float(replay_scale_fraction)
        self.register_buffer(
            "replay_scale_raw",
            torch.tensor(math.atanh(replay_scale_fraction), dtype=torch.float32),
            persistent=True,
        )

        # The inherited q/k/v/out path is the defect branch.  This second path
        # consumes matched agreement evidence and is intentionally independent
        # of the defect branch after the shared hidden-state query projection.
        self.normal_k_proj = nn.Linear(hidden_size, self.bottleneck_size, bias=False)
        self.normal_v_proj = nn.Linear(hidden_size, self.bottleneck_size, bias=False)
        self.normal_out_proj = nn.Linear(self.bottleneck_size, hidden_size, bias=False)
        nn.init.zeros_(self.normal_out_proj.weight)
        self.branch_gate_norm = nn.LayerNorm(hidden_size)
        self.branch_gate = nn.Linear(hidden_size, 2)
        nn.init.zeros_(self.branch_gate.weight)
        nn.init.zeros_(self.branch_gate.bias)

        # A balanced visual verdict is an auxiliary training signal.  It is
        # never thresholded at inference and therefore is not an external
        # router or a post-hoc calibration head.
        self.visual_verdict = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, 256),
            nn.SiLU(),
            nn.Linear(256, 2),
        )

        self.runtime_defect_memory: torch.Tensor | None = None
        self.runtime_normal_memory: torch.Tensor | None = None
        self.last_defect_summary: torch.Tensor | None = None
        self.last_normal_summary: torch.Tensor | None = None
        self.last_subspace_orthogonality: torch.Tensor | None = None
        self.last_visual_verdict_logits: torch.Tensor | None = None

        self.runtime_replay_mask: torch.Tensor | None = None
        self.assistant_prefix_ids: tuple[int, ...] | None = None
        self.seg_start_ids: tuple[int, ...] | None = None
        self._replay_suffixes: list[list[int]] = []
        self._replay_active: list[bool] = []
        self._replay_hits = 0
        self._replay_calls = 0
        self._decision_calls = 0
        self._replay_ratio_sum = 0.0
        self._decision_ratio_sum = 0.0
        self._active_batch_indices: torch.Tensor | None = None

    def reset_runtime(self) -> None:
        super().reset_runtime()
        self.runtime_defect_memory = None
        self.runtime_normal_memory = None
        self.last_defect_summary = None
        self.last_normal_summary = None
        self.last_subspace_orthogonality = None
        self.last_visual_verdict_logits = None
        self.runtime_replay_mask = None

    def configure_phase_tokens(
        self,
        *,
        assistant_prefix_ids: Sequence[int],
        seg_start_ids: Sequence[int],
    ) -> None:
        assistant = tuple(int(value) for value in assistant_prefix_ids)
        seg = tuple(int(value) for value in seg_start_ids)
        if not assistant or not seg:
            raise ValueError("assistant and <seg> token patterns must be non-empty")
        self.assistant_prefix_ids = assistant
        self.seg_start_ids = seg
        self._replay_suffixes = []
        self._replay_active = []
        self.runtime_replay_mask = None

    def prepare_replay_mask(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        initial_call: bool,
    ) -> torch.Tensor:
        """Activate replay only in the actual assistant SEG/CoT trajectory.

        The system prompt itself contains literal format examples.  On an
        initial prefill we therefore locate the final assistant-generation
        prefix first and ignore all earlier tags.  Cached generation calls are
        handled by a small per-row suffix automaton.
        """
        if input_ids.ndim != 2:
            raise ValueError("replay masking requires rank-2 input_ids")
        assistant, seg, answer = (
            self.assistant_prefix_ids,
            self.seg_start_ids,
            self.decision_prefix_ids,
        )
        if assistant is None or seg is None or answer is None:
            raise RuntimeError("phase tokens and answer decision prefix must be configured")
        batch, length = input_ids.shape
        mask = torch.zeros((batch, length), device=input_ids.device, dtype=torch.float32)
        rows = input_ids.detach().cpu().tolist()
        if attention_mask is None:
            valid_rows = [[True] * length for _ in range(batch)]
        else:
            valid_rows = attention_mask[:, -length:].detach().cpu().bool().tolist()

        max_keep = max(len(seg), len(answer)) - 1
        if initial_call or len(self._replay_suffixes) != batch:
            self._replay_suffixes = [[] for _ in range(batch)]
            self._replay_active = [False for _ in range(batch)]
            for row_index, (row, valid) in enumerate(zip(rows, valid_rows)):
                positions = [index for index, keep in enumerate(valid) if keep]
                tokens = [int(row[index]) for index in positions]
                assistant_at = _last_subsequence(tokens, assistant)
                content_at = assistant_at + len(assistant) if assistant_at >= 0 else len(tokens)
                seg_at = _last_subsequence(tokens, seg, content_at)
                answer_at = _last_subsequence(tokens, answer, content_at)
                active = seg_at >= 0 and (answer_at < 0 or seg_at < answer_at)
                if seg_at >= 0:
                    replay_start = seg_at + len(seg) - 1
                    replay_stop = answer_at + len(answer) - 1 if answer_at >= 0 else len(tokens)
                    for token_position in range(replay_start, replay_stop):
                        mask[row_index, positions[token_position]] = 1.0
                self._replay_active[row_index] = bool(active and answer_at < 0)
                self._replay_suffixes[row_index] = tokens[-max_keep:] if max_keep else []
        else:
            for row_index, (row, valid) in enumerate(zip(rows, valid_rows)):
                suffix = list(self._replay_suffixes[row_index])
                active = bool(self._replay_active[row_index])
                for position, (token, keep_token) in enumerate(zip(row, valid)):
                    if not keep_token:
                        continue
                    suffix.append(int(token))
                    if len(suffix) >= len(seg) and tuple(suffix[-len(seg) :]) == seg:
                        active = True
                    if len(suffix) >= len(answer) and tuple(suffix[-len(answer) :]) == answer:
                        active = False
                    if active:
                        mask[row_index, position] = 1.0
                    suffix = suffix[-max_keep:] if max_keep else []
                self._replay_active[row_index] = active
                self._replay_suffixes[row_index] = suffix

        # The final answer-decision state is governed by the larger answer
        # trust region, never by both scales simultaneously.
        if self.runtime_decision_mask is not None:
            mask = mask * (1.0 - self.runtime_decision_mask.to(mask.dtype))
        self._replay_hits += int(mask.sum().item())
        self.runtime_replay_mask = mask
        return mask

    def build_native_memory(
        self,
        query: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        memory = super().build_native_memory(query, reference)
        if memory.shape[1] % 3:
            raise RuntimeError("GRAFT requires signed/energy/agreement memory")
        signed, energy, agreement = memory.chunk(3, dim=1)
        signed = signed - self.memory_type[0].view(1, 1, -1)
        energy = energy - self.memory_type[1].view(1, 1, -1)
        agreement = agreement - self.memory_type[2].view(1, 1, -1)
        self.runtime_defect_memory = torch.cat((signed, energy), dim=1)
        self.runtime_normal_memory = agreement
        self.last_defect_summary = energy.mean(dim=1)
        self.last_normal_summary = agreement.mean(dim=1)
        defect_unit = F.normalize(self.last_defect_summary, dim=-1)
        normal_unit = F.normalize(self.last_normal_summary, dim=-1)
        self.last_subspace_orthogonality = (defect_unit * normal_unit).sum(dim=-1).square().mean()
        self.last_visual_verdict_logits = self.visual_verdict(
            torch.cat(
                (
                    energy.mean(dim=1),
                    energy.amax(dim=1),
                    agreement.mean(dim=1),
                ),
                dim=-1,
            )
        )
        # Keep the inherited name populated for generic training/evaluation
        # diagnostics, but train against the stronger balanced verdict.
        self.last_anomaly_logits = self.last_visual_verdict_logits
        return memory

    def _projected_attention(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        *,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        memory_length = memory.shape[1]
        q = self.q_proj(self.hidden_norm(hidden_states))
        k, v = k_proj(memory), v_proj(memory)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, memory_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, memory_length, self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, length, self.bottleneck_size)
        return out_proj(attended)

    def _attention(self, hidden_states: torch.Tensor, _memory: torch.Tensor) -> torch.Tensor:
        defect_memory, normal_memory = self.runtime_defect_memory, self.runtime_normal_memory
        defect_summary, normal_summary = self.last_defect_summary, self.last_normal_summary
        if any(value is None for value in (defect_memory, normal_memory, defect_summary, normal_summary)):
            raise RuntimeError("GRAFT dual evidence was not built")
        indices = self._active_batch_indices
        if indices is not None:
            defect_memory = defect_memory.index_select(0, indices)
            normal_memory = normal_memory.index_select(0, indices)
            defect_summary = defect_summary.index_select(0, indices)
            normal_summary = normal_summary.index_select(0, indices)
        defect = self._projected_attention(
            hidden_states,
            defect_memory,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            out_proj=self.out_proj,
        )
        normal = self._projected_attention(
            hidden_states,
            normal_memory,
            k_proj=self.normal_k_proj,
            v_proj=self.normal_v_proj,
            out_proj=self.normal_out_proj,
        )
        weights = self.branch_gate(self.branch_gate_norm(hidden_states)).softmax(dim=-1)
        attended = weights[..., :1] * defect + weights[..., 1:] * normal

        # Retain the comparison-conditioned hyper-adapter from DPTCA, but feed
        # it an explicitly factorized condition.
        condition = torch.cat(
            (defect_summary, defect_summary, defect_summary, normal_summary), dim=-1
        )
        condition = self.condition_proj(self.condition_norm(condition)).unsqueeze(1)
        site = self._active_hyper_site
        low_rank = F.silu(self.hyper_down[site](hidden_states) + condition)
        return attended + self.hyper_up[site](low_rank)

    def inject(self, hidden_states: torch.Tensor, site_index: int) -> torch.Tensor:
        memory = self.runtime_memory
        if memory is None:
            raise RuntimeError("GRAFT evidence was not built before decoder injection")
        if hidden_states.ndim != 3 or hidden_states.shape[0] != memory.shape[0]:
            raise RuntimeError("GRAFT hidden/evidence batch mismatch")
        decision = self.runtime_decision_mask
        replay = self.runtime_replay_mask
        if decision is None and replay is None:
            return hidden_states
        decision = torch.zeros(hidden_states.shape[:2], device=hidden_states.device) if decision is None else decision
        replay = torch.zeros(hidden_states.shape[:2], device=hidden_states.device) if replay is None else replay
        if decision.shape != hidden_states.shape[:2] or replay.shape != hidden_states.shape[:2]:
            raise RuntimeError("GRAFT phase-mask mismatch")
        active = (decision.bool() | replay.bool()).nonzero(as_tuple=False)
        if active.numel() == 0:
            return hidden_states

        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            hidden = hidden_states.float()
            batch_indices, token_indices = active[:, 0], active[:, 1]
            selected = hidden[batch_indices, token_indices].unsqueeze(1)
            self._active_hyper_site = int(site_index)
            self._active_batch_indices = batch_indices
            raw_delta = self._attention(selected, memory).squeeze(1)
            self._active_batch_indices = None
            base_rms = selected.squeeze(1).square().mean(dim=-1, keepdim=True).add(1e-12).sqrt().detach()
            delta_rms = raw_delta.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
            site_fraction = torch.sigmoid(self.site_logits[site_index])
            answer_scale = self.max_relative_rms * torch.tanh(self.global_scale) * site_fraction
            replay_scale = (
                self.replay_max_relative_rms
                * torch.tanh(self.replay_scale_raw)
                * site_fraction
            )
            selected_is_decision = decision[batch_indices, token_indices].bool().unsqueeze(-1)
            scales = torch.where(selected_is_decision, answer_scale, replay_scale)
            selected_residual = scales * base_rms * raw_delta / (self.direction_floor + delta_rms)
            residual = torch.zeros_like(hidden).index_put(
                (batch_indices, token_indices), selected_residual, accumulate=False
            )
            fused = hidden + residual
            if not torch.isfinite(fused).all():
                raise FloatingPointError("non-finite GRAFT residual")
            ratios = selected_residual.square().mean(dim=-1).add(1e-12).sqrt() / base_rms.squeeze(-1)
            decision_ratios = ratios[selected_is_decision.squeeze(-1)]
            replay_ratios = ratios[~selected_is_decision.squeeze(-1)]
            if decision_ratios.numel():
                self._decision_calls += 1
                self._decision_ratio_sum += float(decision_ratios.detach().mean().cpu())
            if replay_ratios.numel():
                self._replay_calls += 1
                self._replay_ratio_sum += float(replay_ratios.detach().mean().cpu())
            self.last_residual_ratios.append(
                residual.square().mean(dim=-1).add(1e-12).sqrt()
                / hidden.square().mean(dim=-1).add(1e-12).sqrt().detach()
            )
            self._calls += 1
            self._residual_ratio_sum += float(ratios.detach().mean().cpu())
        return fused.to(hidden_states.dtype)

    def statistics(self) -> dict[str, Any]:
        value = super().statistics()
        value.update(
            {
                "architecture": ARCHITECTURE,
                "predecision_cot_states_modified": True,
                "visual_evidence_replay": "assistant <seg>/<think> trajectory",
                "dual_subspaces": ["matched_normal", "unmatched_defect"],
                "replay_max_relative_rms_per_site": self.replay_max_relative_rms,
                "replay_scale_fraction": self.replay_scale_fraction,
                "replay_hits": self._replay_hits,
                "replay_calls": self._replay_calls,
                "decision_calls": self._decision_calls,
                "mean_replay_residual_ratio_per_site": self._replay_ratio_sum / max(1, self._replay_calls),
                "mean_decision_residual_ratio_per_site": self._decision_ratio_sum / max(1, self._decision_calls),
            }
        )
        return value


def install_graft_adapter(
    model: Any,
    *,
    decision_prefix_ids: Sequence[int],
    assistant_prefix_ids: Sequence[int],
    seg_start_ids: Sequence[int],
    replay_max_relative_rms: float = 0.02,
    replay_scale_fraction: float = 0.80,
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
) -> GroundedReferenceAnchoredReplayAdapter:
    """Install GRAFT while reusing DPTCA's pair-isolated vision extraction."""
    previous = install_defect_preserving_partial_transport_causal_adapter(
        model,
        decision_prefix_ids=decision_prefix_ids,
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
    for handle in model._native_deep_residual_handles:
        handle.remove()
    adapter = GroundedReferenceAnchoredReplayAdapter(
        int(model.config.vision_config.out_hidden_size),
        replay_max_relative_rms=replay_max_relative_rms,
        replay_scale_fraction=replay_scale_fraction,
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
    adapter.configure_phase_tokens(
        assistant_prefix_ids=assistant_prefix_ids,
        seg_start_ids=seg_start_ids,
    )
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
                raise RuntimeError("GRAFT requires input ids")
            shape = embeds.shape[:2]
            adapter.runtime_decision_mask = torch.zeros(shape, device=embeds.device)
            adapter.runtime_replay_mask = torch.zeros(shape, device=embeds.device)
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
            input_ids,
            attention_mask=kwargs.get("attention_mask"),
            initial_call=initial,
        )
        adapter.prepare_replay_mask(
            input_ids,
            attention_mask=kwargs.get("attention_mask"),
            initial_call=initial,
        )
        return None

    handles.append(model.register_forward_pre_hook(pre_hook, with_kwargs=True))
    model._native_deep_residual_handles = handles
    return adapter


__all__ = [
    "ARCHITECTURE",
    "GroundedReferenceAnchoredReplayAdapter",
    "install_graft_adapter",
]
