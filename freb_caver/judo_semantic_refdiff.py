#!/usr/bin/env python3
"""Question-conditioned semantic RefDiff memory for frozen JUDO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import types
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from judo_aligned_ce import (
    EXPECTED_CE_PARAMETERS,
    configure_processor_for_residual,
    install_pair_isolation,
)


ADAPTER_SCHEMA = "judo-semantic-refdiff-adapter-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SemanticRefDiffMemory(nn.Module):
    """Inject query-conditioned signed/energy comparison memory at LM layers.

    Unlike image-boundary fusion, the attention queries are frozen JUDO
    language hidden states.  The same comparison memory can therefore affect
    anomaly, localization, description, or object questions differently
    without an external router.  A zero-initialized global scale gives exact
    functional parity, and every injection is constrained by a tokenwise RMS
    trust region.
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 128,
        num_heads: int = 8,
        injection_layers: Sequence[int] = (7, 14, 21),
        max_relative_rms: float = 0.005,
        direction_floor: float = 0.10,
    ) -> None:
        super().__init__()
        if bottleneck_size % num_heads:
            raise ValueError("bottleneck size must be divisible by num_heads")
        if not injection_layers or len(set(injection_layers)) != len(injection_layers):
            raise ValueError("injection_layers must be non-empty and unique")
        if not 0.0 < max_relative_rms <= 0.10:
            raise ValueError("max_relative_rms must be in (0, 0.10]")
        if direction_floor <= 0.0:
            raise ValueError("direction_floor must be positive")
        self.hidden_size = int(hidden_size)
        self.bottleneck_size = int(bottleneck_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.bottleneck_size // self.num_heads
        self.injection_layers = tuple(int(value) for value in injection_layers)
        self.max_relative_rms = float(max_relative_rms)
        self.direction_floor = float(direction_floor)

        self.hidden_norm = nn.LayerNorm(self.hidden_size)
        self.compare_norm = nn.LayerNorm(self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.bottleneck_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.bottleneck_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.bottleneck_size, bias=False)
        self.out_proj = nn.Linear(self.bottleneck_size, self.hidden_size, bias=False)
        self.global_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.site_logits = nn.Parameter(torch.zeros(len(self.injection_layers), dtype=torch.float32))

        self.runtime_memory: torch.Tensor | None = None
        self.runtime_decision_mask: torch.Tensor | None = None
        self.decision_prefix_ids: tuple[int, ...] | None = None
        self._decision_suffixes: list[list[int]] = []
        self._decision_hits = 0
        self.last_residual_ratios: list[torch.Tensor] = []
        self._calls = 0
        self._residual_ratio_sum = 0.0
        self._memory_difference_rms_sum = 0.0
        self._memory_batches = 0

    def reset_runtime(self) -> None:
        self.runtime_memory = None
        self.last_residual_ratios = []

    def configure_decision_prefix(self, token_ids: Sequence[int] | None) -> None:
        """Restrict evidence injection to the causal answer-decision state.

        The prefix is the token sequence immediately preceding an answer
        choice (for JUDO, the common prefix of ``<answer>A`` ...
        ``<answer>D``).  Prefix state is tracked across cached generation
        calls, while an initial prompt is only considered at its final
        non-padding token.  This prevents the literal format example in the
        system prompt from activating the adapter.
        """
        if token_ids is None:
            self.decision_prefix_ids = None
            self._decision_suffixes = []
            self.runtime_decision_mask = None
            return
        prefix = tuple(int(value) for value in token_ids)
        if not prefix:
            raise ValueError("decision prefix must contain at least one token")
        self.decision_prefix_ids = prefix
        self._decision_suffixes = []
        self.runtime_decision_mask = None

    def prepare_decision_mask(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        initial_call: bool,
    ) -> torch.Tensor:
        """Build the token mask used by decoder-layer injection hooks."""
        if input_ids.ndim != 2:
            raise ValueError("decision masking requires rank-2 input_ids")
        prefix = self.decision_prefix_ids
        batch, length = input_ids.shape
        mask = torch.zeros((batch, length), device=input_ids.device, dtype=torch.float32)
        if prefix is None:
            mask.fill_(1.0)
            self.runtime_decision_mask = mask
            return mask

        rows = input_ids.detach().cpu().tolist()
        if attention_mask is None:
            valid_rows = [[True] * length for _ in range(batch)]
        else:
            local_attention = attention_mask[:, -length:].detach().cpu().bool().tolist()
            valid_rows = local_attention

        if initial_call or len(self._decision_suffixes) != batch:
            self._decision_suffixes = [[] for _ in range(batch)]
            for row_index, (row, valid) in enumerate(zip(rows, valid_rows)):
                positions = [index for index, keep in enumerate(valid) if keep]
                tokens = [row[index] for index in positions]
                if len(tokens) >= len(prefix) and tuple(tokens[-len(prefix) :]) == prefix:
                    mask[row_index, positions[-1]] = 1.0
                keep = max(0, len(prefix) - 1)
                self._decision_suffixes[row_index] = tokens[-keep:] if keep else []
        else:
            for row_index, (row, valid) in enumerate(zip(rows, valid_rows)):
                suffix = list(self._decision_suffixes[row_index])
                for position, (token, keep_token) in enumerate(zip(row, valid)):
                    if not keep_token:
                        continue
                    suffix.append(int(token))
                    if len(suffix) >= len(prefix) and tuple(suffix[-len(prefix) :]) == prefix:
                        mask[row_index, position] = 1.0
                    keep = max(0, len(prefix) - 1)
                    suffix = suffix[-keep:] if keep else []
                self._decision_suffixes[row_index] = suffix

        self._decision_hits += int(mask.sum().item())
        self.runtime_decision_mask = mask
        return mask

    def build_memory(self, query_compare: torch.Tensor, reference_compare: torch.Tensor) -> torch.Tensor:
        if query_compare.shape != reference_compare.shape or query_compare.ndim != 3:
            raise ValueError("semantic RefDiff expects aligned [batch, tokens, hidden] streams")
        if query_compare.shape[-1] != self.hidden_size:
            raise ValueError("comparison hidden size mismatch")
        query = self.compare_norm(query_compare.float())
        reference = self.compare_norm(reference_compare.float())
        signed = query - reference
        # Signed evidence captures direction; energy captures mismatch strength
        # without discarding direction from the first half of the memory.
        energy = signed.abs()
        memory = torch.cat((signed, energy), dim=1)
        self.runtime_memory = memory
        self.last_residual_ratios = []
        self._memory_difference_rms_sum += float(signed.detach().square().mean().sqrt().cpu())
        self._memory_batches += 1
        return memory

    def _attention(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        memory_length = memory.shape[1]
        q = self.q_proj(self.hidden_norm(hidden_states))
        k = self.k_proj(memory)
        v = self.v_proj(memory)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, memory_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, memory_length, self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, length, self.bottleneck_size)
        return self.out_proj(attended)

    def inject(self, hidden_states: torch.Tensor, site_index: int) -> torch.Tensor:
        memory = self.runtime_memory
        if memory is None:
            raise RuntimeError("semantic RefDiff memory was not built before decoder injection")
        if hidden_states.ndim != 3 or hidden_states.shape[0] != memory.shape[0]:
            raise RuntimeError(
                f"semantic RefDiff batch mismatch: hidden={tuple(hidden_states.shape)}, memory={tuple(memory.shape)}"
            )
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            hidden = hidden_states.float()
            site_fraction = torch.sigmoid(self.site_logits[site_index])
            scale = self.max_relative_rms * torch.tanh(self.global_scale) * site_fraction
            decision_mask = self.runtime_decision_mask
            if decision_mask is not None:
                if decision_mask.shape != hidden_states.shape[:2]:
                    raise RuntimeError(
                        "semantic RefDiff decision-mask mismatch: "
                        f"mask={tuple(decision_mask.shape)}, hidden={tuple(hidden_states.shape)}"
                    )
                active = decision_mask.to(device=hidden.device).bool().nonzero(as_tuple=False)
                if active.numel() == 0:
                    return hidden_states
                batch_indices, token_indices = active[:, 0], active[:, 1]
                selected_hidden = hidden[batch_indices, token_indices].unsqueeze(1)
                selected_memory = memory.index_select(0, batch_indices)
                raw_delta = self._attention(selected_hidden, selected_memory).squeeze(1)
                base_rms_selected = selected_hidden.squeeze(1).square().mean(dim=-1, keepdim=True).add(1e-12).sqrt().detach()
                delta_rms = raw_delta.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
                selected_residual = scale * base_rms_selected * raw_delta / (self.direction_floor + delta_rms)
                residual = torch.zeros_like(hidden).index_put(
                    (batch_indices, token_indices), selected_residual, accumulate=False
                )
                base_rms = hidden.square().mean(dim=-1).add(1e-12).sqrt().detach()
                ratios = residual.square().mean(dim=-1).add(1e-12).sqrt() / base_rms
            else:
                raw_delta = self._attention(hidden, memory)
                base_rms = hidden.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt().detach()
                delta_rms = raw_delta.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
                direction = raw_delta / (self.direction_floor + delta_rms)
                residual = scale * base_rms * direction
                ratios = residual.square().mean(dim=-1).add(1e-12).sqrt() / base_rms.squeeze(-1)
            fused = hidden + residual
            if not torch.isfinite(fused).all():
                raise FloatingPointError("non-finite semantic RefDiff output")
            self.last_residual_ratios.append(ratios)
            self._calls += 1
            self._residual_ratio_sum += float(ratios.detach().mean().cpu())
        return fused.to(hidden_states.dtype)

    def statistics(self) -> dict[str, Any]:
        return {
            "injection_layers": list(self.injection_layers),
            "injection_calls": self._calls,
            "max_relative_rms_per_site": self.max_relative_rms,
            "direction_floor": self.direction_floor,
            "global_scale_raw": float(self.global_scale.detach().cpu()),
            "global_scale_fraction": float(torch.tanh(self.global_scale.detach()).cpu()),
            "site_fractions": torch.sigmoid(self.site_logits.detach()).cpu().tolist(),
            "mean_observed_residual_ratio_per_site": self._residual_ratio_sum / max(1, self._calls),
            "mean_memory_signed_difference_rms": self._memory_difference_rms_sum / max(1, self._memory_batches),
            "memory_batches": self._memory_batches,
            "decision_prefix_ids": list(self.decision_prefix_ids) if self.decision_prefix_ids is not None else None,
            "decision_hits": self._decision_hits,
        }


def install_semantic_refdiff(
    model: Any,
    *,
    bottleneck_size: int = 128,
    num_heads: int = 8,
    injection_layers: Sequence[int] = (7, 14, 21),
    max_relative_rms: float = 0.005,
    direction_floor: float = 0.10,
    decision_prefix_ids: Sequence[int] | None = None,
    counters: dict[str, Any] | None = None,
) -> SemanticRefDiffMemory:
    """Register comparison memory and decoder-layer injection hooks."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    model.requires_grad_(False)
    install_pair_isolation(model, counters)
    hidden_size = int(model.config.vision_config.out_hidden_size)
    adapter = SemanticRefDiffMemory(
        hidden_size,
        bottleneck_size=bottleneck_size,
        num_heads=num_heads,
        injection_layers=injection_layers,
        max_relative_rms=max_relative_rms,
        direction_floor=direction_floor,
    )
    adapter.to(device=next(model.parameters()).device, dtype=torch.float32)
    adapter.configure_decision_prefix(decision_prefix_ids)
    model.model.semantic_refdiff_adapter = adapter

    decoder_layers = getattr(getattr(model.model, "language_model", None), "layers", None)
    if decoder_layers is None:
        decoder_layers = getattr(model.model, "layers", None)
    if decoder_layers is None:
        raise AttributeError("unable to locate Qwen decoder layers in the loaded model hierarchy")
    layer_count = len(decoder_layers)
    if any(index < 0 or index >= layer_count for index in adapter.injection_layers):
        raise ValueError(f"injection layers {adapter.injection_layers} invalid for {layer_count} decoder layers")

    def semantic_get_image_features(
        self: Any,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
    ) -> list[torch.Tensor]:
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required")
        self.semantic_refdiff_adapter.reset_runtime()
        pixel_values = pixel_values.type(self.visual.dtype)
        with torch.no_grad():
            image_embeds, compare_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        base_splits = list(torch.split(image_embeds, split_sizes))
        compare_splits = list(compare_embeds)
        if len(base_splits) != len(compare_splits) or len(base_splits) % 2:
            raise RuntimeError("semantic RefDiff requires complete [query, normal] pairs")
        query = torch.stack([compare_splits[index].detach() for index in range(0, len(compare_splits), 2)])
        reference = torch.stack([compare_splits[index].detach() for index in range(1, len(compare_splits), 2)])
        self.semantic_refdiff_adapter.build_memory(query, reference)
        return [value.detach() for value in base_splits]

    def standard_rope(self: Any, *args: Any, **kwargs: Any) -> Any:
        return Qwen2_5_VLModel.get_rope_index(self, *args, **kwargs)

    model.model.get_image_features = types.MethodType(semantic_get_image_features, model.model)
    model.model.get_rope_index = types.MethodType(standard_rope, model.model)

    handles = []
    for site_index, layer_index in enumerate(adapter.injection_layers):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, _site: int = site_index) -> Any:
            if isinstance(output, tuple):
                return (adapter.inject(output[0], _site), *output[1:])
            return adapter.inject(output, _site)

        handles.append(decoder_layers[layer_index].register_forward_hook(hook))
    model.model._semantic_refdiff_hook_handles = handles

    if decision_prefix_ids is not None:
        def decision_synchronous_pre_hook(
            _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is None:
                inputs_embeds = kwargs.get("inputs_embeds")
                if inputs_embeds is None:
                    raise RuntimeError("decision-synchronous injection requires input_ids or inputs_embeds")
                adapter.runtime_decision_mask = torch.zeros(
                    inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.float32
                )
            else:
                cache_position = kwargs.get("cache_position")
                past_key_values = kwargs.get("past_key_values")
                if cache_position is not None and cache_position.numel():
                    initial_call = int(cache_position.reshape(-1)[0].item()) == 0
                elif past_key_values is None:
                    initial_call = True
                else:
                    get_seq_length = getattr(past_key_values, "get_seq_length", None)
                    initial_call = bool(get_seq_length is not None and int(get_seq_length()) == 0)
                adapter.prepare_decision_mask(
                    input_ids,
                    attention_mask=kwargs.get("attention_mask"),
                    initial_call=initial_call,
                )
            return None

        # A pre-hook leaves the public ``forward`` signature untouched.  The
        # generation stack introspects that signature to decide whether it
        # must construct Qwen's multimodal ``position_ids``; wrapping forward
        # with ``*args, **kwargs`` silently disables that path.
        model._semantic_refdiff_decision_pre_hook = model.register_forward_pre_hook(
            decision_synchronous_pre_hook, with_kwargs=True
        )
    return adapter


def adapter_parameter_count(adapter: nn.Module) -> int:
    return sum(parameter.numel() for parameter in adapter.parameters())


def trainable_contract(model: Any, adapter: nn.Module, trainable: bool) -> dict[str, int]:
    model.requires_grad_(False)
    adapter.requires_grad_(trainable)
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    expected = adapter_parameter_count(adapter) if trainable else 0
    if trainable_count != expected:
        raise RuntimeError(f"semantic RefDiff trainable contract failed: {trainable_count} != {expected}")
    ce_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".compare_visual_encoder." in f".{name}."
    )
    if ce_parameters != EXPECTED_CE_PARAMETERS:
        raise RuntimeError(f"comparison encoder parameter count changed: {ce_parameters}")
    return {
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "adapter_parameter_count": adapter_parameter_count(adapter),
        "trainable_parameter_count": trainable_count,
        "ce_parameter_count": ce_parameters,
    }


def save_adapter(adapter: SemanticRefDiffMemory, output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    weights = output_dir / "semantic_refdiff_adapter.safetensors"
    temporary = weights.with_suffix(weights.suffix + ".tmp")
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in adapter.state_dict().items()}
    save_file(state, temporary, metadata={"format": "pt", "schema": ADAPTER_SCHEMA})
    temporary.replace(weights)
    identity = {
        "schema_version": ADAPTER_SCHEMA,
        "architecture": "question-conditioned-signed-energy-memory-v1",
        "weights_sha256": sha256_file(weights),
        "tensor_count": len(state),
        "parameter_count": adapter_parameter_count(adapter),
        "hidden_size": adapter.hidden_size,
        "bottleneck_size": adapter.bottleneck_size,
        "num_heads": adapter.num_heads,
        "injection_layers": list(adapter.injection_layers),
        "max_relative_rms": adapter.max_relative_rms,
        "direction_floor": adapter.direction_floor,
        "statistics": adapter.statistics(),
        **metadata,
    }
    (output_dir / "semantic_refdiff_adapter_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identity


def load_adapter(adapter: SemanticRefDiffMemory, directory: Path) -> dict[str, Any]:
    from safetensors import safe_open

    identity_path = directory / "semantic_refdiff_adapter_identity.json"
    weights = directory / "semantic_refdiff_adapter.safetensors"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("schema_version") != ADAPTER_SCHEMA or identity.get("weights_sha256") != sha256_file(weights):
        raise ValueError("semantic RefDiff identity/hash mismatch")
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(weights, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensors[name] = handle.get_tensor(name)
    if set(tensors) != set(adapter.state_dict()):
        raise ValueError("semantic RefDiff state keys do not match architecture")
    adapter.load_state_dict(tensors, strict=True)
    return identity


__all__ = [
    "ADAPTER_SCHEMA",
    "SemanticRefDiffMemory",
    "adapter_parameter_count",
    "configure_processor_for_residual",
    "install_semantic_refdiff",
    "load_adapter",
    "save_adapter",
    "sha256_file",
    "trainable_contract",
]
