#!/usr/bin/env python3
"""CAVER: causally bind comparison evidence to the JUDO reasoning state.

CAVER is a small continuation module for a trained GRAFT adapter.  GRAFT's
visual verdict is useful supervision, but an auxiliary head can be ignored by
the language decoder.  CAVER closes that causal gap: the *same* frozen visual
belief is written into every grounded CoT state and the final answer state by
a zero-initialized, prompt-conditioned low-rank path.

The module remains an exact functional identity before continuation training.
It uses no external router and no inference-time decision threshold.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

from judo_graft import GroundedReferenceAnchoredReplayAdapter, install_graft_adapter


ARCHITECTURE = "causal-anomaly-visual-evidence-recurrent-binding-v1"
LETTERS = "ABCD"


def normal_null(row: dict[str, Any]) -> dict[str, Any]:
    """Create the do(query := normal-reference) counterfactual.

    The question and option ordering are retained, while both visual inputs
    are set to the same normal anchor.  The semantic target is therefore the
    option whose text is exactly ``No``.  Resolving the option by meaning,
    rather than assuming a fixed letter, prevents an option-order shortcut.
    """
    options = row.get("options")
    if not isinstance(options, dict):
        raise ValueError("normal-null intervention requires an option mapping")
    def normalize(value: Any) -> str:
        return str(value).strip().rstrip(".").strip().casefold()

    matches = [
        index
        for index, letter in enumerate(LETTERS)
        if normalize(options.get(letter, "")) == "no"
    ]
    if len(matches) != 1:
        raise ValueError("normal-null intervention requires exactly one semantic No option")
    if not row.get("template_image"):
        raise ValueError("normal-null intervention requires a normal reference")
    no = matches[0]
    value = dict(row)
    value.update(
        {
            "sample_id": f"{row['sample_id']}::normal-null",
            "image": str(row["template_image"]),
            "label": "normal",
            "correct_answer": LETTERS[no],
            "teacher_answer": LETTERS[no],
            "teacher_correct": True,
            "teacher_segmentation": "None",
            "teacher_thinking": (
                "The query is identical to the normal reference. "
                "There is no localized difference or defect evidence."
            ),
        }
    )
    return value


class CausalVisualBeliefReplayAdapter(GroundedReferenceAnchoredReplayAdapter):
    """GRAFT plus a recurrent causal visual-belief binding path."""

    def __init__(
        self,
        hidden_size: int,
        *,
        belief_rank: int = 128,
        belief_temperature: float = 1.0,
        belief_scale_fraction: float = 0.50,
        **kwargs: Any,
    ) -> None:
        if belief_rank < 1:
            raise ValueError("belief_rank must be positive")
        if belief_temperature <= 0:
            raise ValueError("belief_temperature must be positive")
        if not 0.0 < belief_scale_fraction < 1.0:
            raise ValueError("belief_scale_fraction must be in (0, 1)")
        super().__init__(hidden_size, **kwargs)
        self.belief_rank = int(belief_rank)
        self.belief_temperature = float(belief_temperature)
        self.belief_scale_fraction = float(belief_scale_fraction)
        self.register_buffer(
            "belief_scale_raw",
            torch.tensor(math.atanh(belief_scale_fraction), dtype=torch.float32),
            persistent=True,
        )
        self.belief_norm = nn.LayerNorm(hidden_size)
        self.belief_down = nn.ModuleList(
            nn.Linear(hidden_size, belief_rank, bias=False)
            for _ in self.injection_layers
        )
        self.belief_up = nn.ModuleList(
            nn.Linear(belief_rank, hidden_size, bias=False)
            for _ in self.injection_layers
        )
        self.belief_gate = nn.ModuleList(
            nn.Linear(hidden_size, 1) for _ in self.injection_layers
        )
        for layer in self.belief_up:
            nn.init.zeros_(layer.weight)
        for layer in self.belief_gate:
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, -1.0)
        self.last_visual_belief: torch.Tensor | None = None
        self.last_belief_gate_mean: torch.Tensor | None = None
        self.belief_enabled = True
        self._belief_calls = 0
        self._belief_abs_sum = 0.0
        self._belief_gate_sum = 0.0

    def reset_runtime(self) -> None:
        super().reset_runtime()
        self.last_visual_belief = None
        self.last_belief_gate_mean = None

    def build_native_memory(
        self,
        query: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        memory = super().build_native_memory(query, reference)
        if self.last_visual_verdict_logits is None:
            raise RuntimeError("CAVER requires the GRAFT visual verdict")
        self.last_visual_belief = (
            self.last_visual_verdict_logits[:, 1]
            - self.last_visual_verdict_logits[:, 0]
        )
        return memory

    def _attention(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        base = super()._attention(hidden_states, memory)
        if not self.belief_enabled:
            return base
        belief = self.last_visual_belief
        if belief is None:
            raise RuntimeError("CAVER visual belief was not built")
        indices = self._active_batch_indices
        if indices is not None:
            belief = belief.index_select(0, indices)
        site = int(self._active_hyper_site)
        normalized = self.belief_norm(hidden_states)
        latent = F.silu(self.belief_down[site](normalized))
        signed = torch.tanh(belief / self.belief_temperature).view(-1, 1, 1)
        gate = torch.sigmoid(self.belief_gate[site](normalized))
        bound = self.belief_up[site](latent * signed) * gate
        bound = bound * torch.tanh(self.belief_scale_raw)
        self.last_belief_gate_mean = gate.mean()
        self._belief_calls += 1
        self._belief_abs_sum += float(signed.detach().abs().mean().cpu())
        self._belief_gate_sum += float(gate.detach().mean().cpu())
        return base + bound

    def causal_parameters(self) -> list[nn.Parameter]:
        modules: list[nn.Module] = [
            self.visual_verdict,
            self.belief_norm,
            self.belief_down,
            self.belief_up,
            self.belief_gate,
        ]
        return [parameter for module in modules for parameter in module.parameters()]

    def statistics(self) -> dict[str, Any]:
        value = super().statistics()
        value.update(
            {
                "architecture": ARCHITECTURE,
                "causal_visual_belief_binding": True,
                "normal_null_intervention_training": True,
                "trajectory_preference_training": True,
                "belief_rank": self.belief_rank,
                "belief_temperature": self.belief_temperature,
                "belief_scale_fraction": self.belief_scale_fraction,
                "belief_calls": self._belief_calls,
                "mean_abs_visual_belief": self._belief_abs_sum / max(1, self._belief_calls),
                "mean_belief_gate": self._belief_gate_sum / max(1, self._belief_calls),
            }
        )
        return value


def load_graft_seed(
    adapter: CausalVisualBeliefReplayAdapter,
    directory: Path,
) -> dict[str, Any]:
    """Load a frozen GRAFT seed while requiring only CAVER keys to be new."""
    identity_path = directory / "native_deep_residual_identity.json"
    weights_path = directory / "native_deep_residual.safetensors"
    if not identity_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"incomplete GRAFT seed: {directory}")
    import json

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("statistics", {}).get("architecture") != (
        "grounded-reference-anchored-faithful-tuning-v1"
    ):
        raise ValueError("CAVER seed is not a GRAFT adapter")
    state = load_file(str(weights_path), device="cpu")
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    allowed = (
        "belief_scale_raw",
        "belief_norm.",
        "belief_down.",
        "belief_up.",
        "belief_gate.",
    )
    invalid_missing = [name for name in missing if not name.startswith(allowed)]
    if unexpected or invalid_missing:
        raise RuntimeError(
            f"GRAFT->CAVER state mismatch missing={invalid_missing} unexpected={unexpected}"
        )
    return identity


def install_caver_adapter(
    model: Any,
    *,
    decision_prefix_ids: Sequence[int],
    assistant_prefix_ids: Sequence[int],
    seg_start_ids: Sequence[int],
    belief_rank: int = 128,
    belief_temperature: float = 1.0,
    belief_scale_fraction: float = 0.50,
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
) -> CausalVisualBeliefReplayAdapter:
    """Install CAVER while retaining GRAFT's pair-isolated vision path."""
    previous = install_graft_adapter(
        model,
        decision_prefix_ids=decision_prefix_ids,
        assistant_prefix_ids=assistant_prefix_ids,
        seg_start_ids=seg_start_ids,
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
    for handle in model._native_deep_residual_handles:
        handle.remove()
    adapter = CausalVisualBeliefReplayAdapter(
        int(model.config.vision_config.out_hidden_size),
        belief_rank=belief_rank,
        belief_temperature=belief_temperature,
        belief_scale_fraction=belief_scale_fraction,
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
                raise RuntimeError("CAVER requires input ids")
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
    "CausalVisualBeliefReplayAdapter",
    "install_caver_adapter",
    "load_graft_seed",
    "normal_null",
]
