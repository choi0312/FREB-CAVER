#!/usr/bin/env python3
"""Teacher-forced public-validation screen for one frozen CAVER adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import eval_manifest as eval_entry
from judo_caver import ARCHITECTURE, install_caver_adapter
from judo_native_deep_residual import load_adapter, trainable_contract
from train_decision_aligned_refdiff import answer_protocol
from train_judo_aligned_ce import atomic_json, read_jsonl
from train_judo_caver_stage4b import evaluate
from train_judo_graft_stage4 import direct_text
from train_judo_semantic_refdiff import paired_delta, prediction_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--baseline-logits", type=Path, required=True)
    parser.add_argument("--graft-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = read_jsonl(args.validation_manifest)
    if len(rows) != 560:
        raise ValueError("public validation must contain exactly 560 rows")
    for row in rows:
        for field in ("image", "template_image"):
            path = args.data_root / str(row[field])
            if not path.is_file():
                raise FileNotFoundError(path)

    identity = json.loads(
        (args.adapter / "native_deep_residual_identity.json").read_text(encoding="utf-8")
    )
    stats = identity["statistics"]
    if stats.get("architecture") != ARCHITECTURE:
        raise ValueError("adapter is not CAVER")
    homotopy = identity.get("homotopy")
    if not homotopy or homotopy.get("schema_version") != "graft-caver-residual-homotopy-v1":
        raise ValueError("adapter is not the preregistered homotopy diagnostic")

    from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    device = torch.device("cuda:0")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    prefix_ids, prefix_text, candidates, protocol = answer_protocol(tokenizer)
    assistant_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    seg_ids = tokenizer.encode("<seg>", add_special_tokens=False)
    if protocol != identity["decision_protocol"]:
        raise ValueError("decision protocol mismatch")
    if assistant_ids != identity["phase_protocol"]["assistant_prefix_ids"]:
        raise ValueError("assistant phase protocol mismatch")
    if seg_ids != identity["phase_protocol"]["seg_start_ids"]:
        raise ValueError("segmentation phase protocol mismatch")

    adapter = install_caver_adapter(
        model,
        decision_prefix_ids=prefix_ids,
        assistant_prefix_ids=assistant_ids,
        seg_start_ids=seg_ids,
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
    loaded = load_adapter(adapter, args.adapter)
    contract = trainable_contract(model, adapter, False)
    model.eval()
    adapter.eval()

    system_prompt = eval_entry.official_eval.make_system_prompt(16, 16)
    metrics, logits, persistence = evaluate(
        model,
        adapter,
        processor,
        rows,
        args.data_root,
        system_prompt,
        prefix_text,
        candidates,
        args.batch_size,
    )
    baseline_all = {
        str(row["sample_id"]): list(row["logits"])
        for row in read_jsonl(args.baseline_logits)
    }
    baseline_logits = {
        str(row["sample_id"]): baseline_all[str(row["sample_id"])] for row in rows
    }
    baseline_metrics = prediction_metrics(rows, baseline_logits)
    graft_payload = json.loads(args.graft_validation.read_text(encoding="utf-8"))
    graft_logits = {
        str(key): list(value) for key, value in graft_payload["logits"].items()
    }
    graft_metrics = graft_payload["metrics"]
    metrics["paired_vs_public_baseline"] = paired_delta(rows, baseline_logits, logits)
    metrics["paired_vs_graft_seed"] = paired_delta(rows, graft_logits, logits)
    per_task_safe = all(
        metrics["per_task_accuracy"][task]
        >= baseline_metrics["per_task_accuracy"][task] - 0.0125
        for task in baseline_metrics["per_task_accuracy"]
    )
    screen_pass = bool(
        per_task_safe
        and metrics["accuracy"] > graft_metrics["accuracy"]
        and metrics["ad_balanced_accuracy"] > graft_metrics["ad_balanced_accuracy"]
    )
    output = {
        "schema_version": "judo-caver-homotopy-screen-v1",
        "status": "complete",
        "screen_pass": screen_pass,
        "holdout_evaluated": False,
        "samples": len(rows),
        "alpha": homotopy["alpha"],
        "public_baseline_validation": baseline_metrics,
        "graft_seed_validation": graft_metrics,
        "candidate_validation": metrics,
        "candidate_visual_persistence": persistence,
        "adapter_weights_sha256": loaded["weights_sha256"],
        "adapter_parameter_count": contract["adapter_parameter_count"],
        "serialized_tensor_element_count": loaded["parameter_count"],
        "runtime_statistics": adapter.statistics(),
        "external_router": False,
        "inference_threshold": False,
        "teacher_forced_public_validation_only": True,
    }
    atomic_json(args.output, output)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
