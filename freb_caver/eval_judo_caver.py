#!/usr/bin/env python3
"""Autoregressive MMAD evaluation for frozen JUDO + CAVER Stage 4b."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

import eval_manifest as base_eval
from judo_caver import ARCHITECTURE, install_caver_adapter
from judo_native_deep_residual import load_adapter, trainable_contract
from train_decision_aligned_refdiff import answer_protocol


LOADED: list[Any] = []
ATTESTATION: dict[str, Any] = {
    "schema_version": "judo-caver-runtime-v1",
    "architecture": ARCHITECTURE,
    "external_router": False,
    "threshold_tuning": False,
    "predecision_cot_states_modified": True,
    "answer_decision_state_modified": True,
    "normal_null_intervention_training": True,
    "causal_visual_belief_binding": True,
}


def output_dir() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("--output-dir required") from error


class Engine(base_eval.official_eval.OptimizedMultiGPUEngine):
    def _load_models(self) -> None:
        from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        adapter_dir = Path(os.environ["JUDO_CAVER_ADAPTER_DIR"])
        identity = json.loads(
            (adapter_dir / "native_deep_residual_identity.json").read_text(encoding="utf-8")
        )
        stats = identity["statistics"]
        if stats.get("architecture") != ARCHITECTURE:
            raise RuntimeError(f"not a CAVER adapter: {stats.get('architecture')}")
        protocol = identity["decision_protocol"]
        phase = identity["phase_protocol"]
        for gpu_id in range(self.num_gpus):
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": f"cuda:{gpu_id}"},
                local_files_only=True,
                attn_implementation=self.attn_implementation,
            )
            processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            processor.tokenizer.padding_side = "left"
            prefix, prefix_text, _candidates, actual = answer_protocol(tokenizer)
            if actual != protocol:
                raise RuntimeError("CAVER answer protocol mismatch")
            actual_assistant = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            actual_seg = tokenizer.encode("<seg>", add_special_tokens=False)
            if actual_assistant != phase["assistant_prefix_ids"] or actual_seg != phase["seg_start_ids"]:
                raise RuntimeError("CAVER phase-token protocol mismatch")
            adapter = install_caver_adapter(
                model,
                decision_prefix_ids=prefix,
                assistant_prefix_ids=actual_assistant,
                seg_start_ids=actual_seg,
                belief_rank=int(stats["belief_rank"]),
                belief_temperature=float(stats["belief_temperature"]),
                belief_scale_fraction=float(stats["belief_scale_fraction"]),
                replay_max_relative_rms=float(stats["replay_max_relative_rms_per_site"]),
                replay_scale_fraction=float(stats["replay_scale_fraction"]),
                unmatched_hidden=int(stats["unmatched_hidden"]),
                unmatched_prior=float(stats["unmatched_prior"]),
                transport_rank=int(stats["transport_rank"]),
                transport_temperature=float(stats["transport_temperature"]),
                hyper_rank=int(stats["hyper_rank"]),
                bottleneck_size=int(stats["bottleneck_size"]),
                num_heads=int(stats["num_heads"]),
                injection_layers=tuple(int(value) for value in stats["injection_layers"]),
                max_relative_rms=float(stats["max_relative_rms_per_site"]),
                fixed_scale_fraction=float(stats["fixed_scale_fraction"]),
                direction_floor=float(stats["direction_floor"]),
            )
            loaded = load_adapter(adapter, adapter_dir)
            contract = trainable_contract(model, adapter, False)
            model.eval(); adapter.eval()
            self.models[gpu_id] = model
            self.processors[gpu_id] = processor
            self.tokenizers[gpu_id] = tokenizer
            LOADED.append(adapter)
            ATTESTATION.update(
                {
                    "adapter_weights_sha256": loaded["weights_sha256"],
                    "adapter_parameter_count": contract["adapter_parameter_count"],
                    "serialized_tensor_element_count": loaded["parameter_count"],
                    "selected_epoch": loaded["selected_epoch"],
                    "graft_seed_weights_sha256": identity["graft_seed_weights_sha256"],
                    "decision_protocol": protocol,
                    "decision_prefix_text": prefix_text,
                    "phase_protocol": phase,
                    "injection_layers": stats["injection_layers"],
                    "max_relative_rms_per_site": stats["max_relative_rms_per_site"],
                    "replay_max_relative_rms_per_site": stats["replay_max_relative_rms_per_site"],
                    "belief_rank": stats["belief_rank"],
                    "trainable_parameter_count": contract["trainable_parameter_count"],
                }
            )
            print(json.dumps({"event": "caver_ready", **ATTESTATION}, sort_keys=True), flush=True)


def main() -> None:
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    base_eval.official_eval.OptimizedMultiGPUEngine = Engine
    try:
        base_eval.main()
    finally:
        path = destination / "caver_runtime_attestation.json"
        preserve_existing = not LOADED and path.is_file()
        if LOADED:
            ATTESTATION["adapter_statistics_current_process"] = LOADED[0].statistics()
        if not preserve_existing:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(ATTESTATION, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)


if __name__ == "__main__":
    main()
