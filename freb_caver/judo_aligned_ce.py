#!/usr/bin/env python3
"""Runtime components for a zero-initialized residual CE alignment adapter."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading
import types
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


EXPECTED_CE_TENSORS = 36
EXPECTED_CE_PARAMETERS = 63_801_344
EXPECTED_COMPARE_TOKENS = 100
ADAPTER_SCHEMA = "judo-aligned-ce-adapter-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GatedResidualCrossAttention(nn.Module):
    """Low-rank CE-to-JUDO alignment with an exact zero residual at init."""

    def __init__(self, hidden_size: int, bottleneck_size: int = 256, num_heads: int = 8):
        super().__init__()
        if bottleneck_size % num_heads:
            raise ValueError("bottleneck size must be divisible by the number of heads")
        self.hidden_size = int(hidden_size)
        self.bottleneck_size = int(bottleneck_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.bottleneck_size // self.num_heads
        self.base_norm = nn.LayerNorm(self.hidden_size)
        self.ce_norm = nn.LayerNorm(self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.bottleneck_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.bottleneck_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.bottleneck_size, bias=False)
        self.out_proj = nn.Linear(self.bottleneck_size, self.hidden_size, bias=False)
        gate_hidden = max(32, self.bottleneck_size // 4)
        self.local_gate = nn.Sequential(
            nn.Linear(self.hidden_size, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )
        # tanh(0) is exactly zero, so installing the adapter initially leaves
        # every original JUDO image embedding unchanged.
        self.global_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.last_alphas: list[torch.Tensor] = []
        self._gate_sum = 0.0
        self._gate_count = 0

    def reset_last_alphas(self) -> None:
        self.last_alphas = []

    def gate_statistics(self) -> dict[str, float | int]:
        return {
            "global_scale_raw": float(self.global_scale.detach().cpu()),
            "global_scale_tanh": float(torch.tanh(self.global_scale.detach()).cpu()),
            "observed_image_gates": self._gate_count,
            "mean_effective_alpha": self._gate_sum / self._gate_count if self._gate_count else 0.0,
        }

    def forward(self, base_tokens: torch.Tensor, ce_tokens: torch.Tensor) -> torch.Tensor:
        if base_tokens.ndim != 2 or ce_tokens.ndim != 2:
            raise ValueError("adapter expects two rank-2 token tensors")
        if base_tokens.shape[-1] != self.hidden_size or ce_tokens.shape[-1] != self.hidden_size:
            raise ValueError("adapter hidden dimension mismatch")
        # Keep adapter math in fp32 even when the frozen 8B model runs in bf16.
        with torch.autocast(device_type=base_tokens.device.type, enabled=False):
            base = base_tokens.float()
            ce = ce_tokens.float()
            q = self.q_proj(self.base_norm(base))
            k = self.k_proj(self.ce_norm(ce))
            v = self.v_proj(self.ce_norm(ce))
            q = q.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            attended = attended.transpose(1, 2).reshape(-1, self.bottleneck_size)
            delta = self.out_proj(attended)
            local = torch.sigmoid(self.local_gate(ce.mean(dim=0))).reshape(())
            alpha = torch.tanh(self.global_scale) * local
            if not torch.isfinite(delta).all() or not torch.isfinite(alpha):
                raise FloatingPointError("non-finite residual adapter output")
            self.last_alphas.append(alpha)
            self._gate_sum += float(alpha.detach().cpu())
            self._gate_count += 1
            fused = base + alpha * delta
        return fused.to(dtype=base_tokens.dtype)


def install_pair_isolation(model: Any, counters: dict[str, Any] | None = None) -> None:
    encoder = model.model.visual.compare_visual_encoder
    if getattr(encoder, "_judo_pair_isolated", False):
        return
    original_forward = encoder.forward
    lock = threading.Lock()

    def paired_forward(_encoder: Any, image_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        if not image_hidden_states or len(image_hidden_states) % 2:
            raise RuntimeError("comparison encoder requires complete [query, normal] pairs")
        outputs = []
        for index in range(0, len(image_hidden_states), 2):
            query = image_hidden_states[index]
            normal = image_hidden_states[index + 1]
            native = original_forward([normal, query])
            if tuple(native.shape[:2]) != (2, EXPECTED_COMPARE_TOKENS):
                raise RuntimeError(f"unexpected comparison output shape: {tuple(native.shape)}")
            outputs.append(native[[1, 0]])
        if counters is not None:
            with lock:
                counters["ce_forward_calls"] = int(counters.get("ce_forward_calls", 0)) + 1
                counters["question_pairs"] = int(counters.get("question_pairs", 0)) + len(image_hidden_states) // 2
        return torch.cat(outputs, dim=0)

    encoder.forward = types.MethodType(paired_forward, encoder)
    encoder._judo_pair_isolated = True


def install_aligned_image_path(
    model: Any,
    *,
    bottleneck_size: int = 256,
    num_heads: int = 8,
    counters: dict[str, Any] | None = None,
) -> GatedResidualCrossAttention:
    """Freeze JUDO+CE and replace appended CE tokens with a gated residual."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    model.requires_grad_(False)
    install_pair_isolation(model, counters)
    hidden_size = int(model.config.vision_config.out_hidden_size)
    adapter = GatedResidualCrossAttention(hidden_size, bottleneck_size, num_heads)
    adapter.to(device=next(model.parameters()).device, dtype=torch.float32)
    model.model.alignment_adapter = adapter

    def aligned_get_image_features(
        self: Any,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
    ) -> list[torch.Tensor]:
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required")
        self.alignment_adapter.reset_last_alphas()
        pixel_values = pixel_values.type(self.visual.dtype)
        # Both vision encoders are frozen.  Detaching here avoids retaining
        # their activation graphs while preserving gradients through the new
        # adapter and the frozen language model back to the adapter output.
        with torch.no_grad():
            image_embeds, compare_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_splits = torch.split(image_embeds, split_sizes)
        if len(image_splits) != len(compare_embeds):
            raise RuntimeError("base and comparison image counts differ")
        return [
            self.alignment_adapter(base.detach(), compare.detach())
            for base, compare in zip(image_splits, compare_embeds)
        ]

    def standard_rope(self: Any, *args: Any, **kwargs: Any) -> Any:
        return Qwen2_5_VLModel.get_rope_index(self, *args, **kwargs)

    model.model.get_image_features = types.MethodType(aligned_get_image_features, model.model)
    model.model.get_rope_index = types.MethodType(standard_rope, model.model)
    return adapter


def configure_processor_for_residual(processor: Any) -> None:
    # The AD-Copilot processor normally reserves 100 extra image-token slots.
    # Residual fusion keeps JUDO's original sequence length, so reserve none.
    processor.compare_token_size = 0
    processor.tokenizer.padding_side = "left"


def adapter_parameter_count(adapter: nn.Module) -> int:
    return sum(parameter.numel() for parameter in adapter.parameters())


def save_adapter(
    adapter: GatedResidualCrossAttention,
    output_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    weights = output_dir / "alignment_adapter.safetensors"
    temporary = weights.with_suffix(".safetensors.tmp")
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in adapter.state_dict().items()}
    save_file(state, temporary, metadata={"format": "pt", "schema": ADAPTER_SCHEMA})
    temporary.replace(weights)
    identity = {
        "schema_version": ADAPTER_SCHEMA,
        "weights_sha256": sha256_file(weights),
        "tensor_count": len(state),
        "parameter_count": adapter_parameter_count(adapter),
        "hidden_size": adapter.hidden_size,
        "bottleneck_size": adapter.bottleneck_size,
        "num_heads": adapter.num_heads,
        "gate_statistics": adapter.gate_statistics(),
        **metadata,
    }
    identity_path = output_dir / "adapter_identity.json"
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return identity


def load_adapter(adapter: GatedResidualCrossAttention, directory: Path) -> dict[str, Any]:
    from safetensors import safe_open

    identity = json.loads((directory / "adapter_identity.json").read_text(encoding="utf-8"))
    weights_path = directory / "alignment_adapter.safetensors"
    if identity.get("schema_version") != ADAPTER_SCHEMA or identity.get("weights_sha256") != sha256_file(weights_path):
        raise ValueError("adapter identity or weight hash mismatch")
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensors[name] = handle.get_tensor(name)
    expected = set(adapter.state_dict())
    if set(tensors) != expected:
        raise ValueError("adapter state keys differ from the declared architecture")
    adapter.load_state_dict(tensors, strict=True)
    return identity


def trainable_contract(model: Any, adapter: GatedResidualCrossAttention, trainable: bool) -> dict[str, int]:
    model.requires_grad_(False)
    adapter.requires_grad_(trainable)
    total_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    expected = adapter_parameter_count(adapter) if trainable else 0
    if total_trainable != expected:
        raise RuntimeError(f"trainable-parameter contract failed: {total_trainable} != {expected}")
    ce_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".compare_visual_encoder." in f".{name}."
    )
    if ce_parameters != EXPECTED_CE_PARAMETERS:
        raise RuntimeError(f"comparison encoder parameter contract failed: {ce_parameters}")
    return {
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "adapter_parameter_count": adapter_parameter_count(adapter),
        "trainable_parameter_count": total_trainable,
        "ce_parameter_count": ce_parameters,
    }
