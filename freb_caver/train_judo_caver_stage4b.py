#!/usr/bin/env python3
"""Stage-4b CAVER interventional trajectory-preference continuation.

The frozen public JUDO model and a completed GRAFT adapter are the starting
point.  Only the visual-verdict head and CAVER's zero-initialized causal
binding path are optimized.  Normal-null interventions (REFERENCE, REFERENCE)
are paired with hard-normal and anomalous queries that share one anchor.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

import eval_manifest as eval_entry
from judo_caver import ARCHITECTURE, install_caver_adapter, load_graft_seed, normal_null
from judo_native_deep_residual import save_adapter
from train_decision_aligned_refdiff import answer_protocol, decision_text, labels
from train_judo_aligned_ce import atomic_json, read_jsonl, update_progress
from train_judo_graft_stage4 import (
    answer_margin,
    build_same_anchor_pairs,
    direct_text,
    evaluate,
    prepare_text_batch,
    semantic_anomaly_margin,
)
from train_judo_semantic_refdiff import paired_delta, prediction_metrics
from train_transport_equivariant_adapter import answer_logits, batches


STATE_SCHEMA = "judo-caver-stage4b-resume-v1"
def partial_cot_text(
    processor: Any,
    row: dict[str, Any],
    system_prompt: str,
    prefix_text: str,
) -> str:
    prompt = direct_text(processor, row, system_prompt, "")
    segmentation = str(row.get("teacher_segmentation", "None")).strip() or "None"
    thinking = str(row.get("teacher_thinking", "")).strip()
    first = thinking.split(".", 1)[0].strip()
    if first:
        first += "."
    return prompt + f"<seg>{segmentation}</seg>\n<think>{first}</think>\n{prefix_text}"


def stage_forward(
    model: Any,
    adapter: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    data_root: Path,
    texts: list[str],
    candidates: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = answer_logits(
        model,
        prepare_text_batch(processor, rows, data_root, texts, device),
        candidates,
    )
    belief = adapter.last_visual_belief
    if belief is None or belief.shape[0] != len(rows):
        raise RuntimeError("CAVER visual belief is missing or misaligned")
    return values, belief.float()


def save_state(
    path: Path,
    adapter: Any,
    optimizer: Any,
    *,
    next_step: int,
    loss_sums: dict[str, float],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": STATE_SCHEMA,
            "adapter": {name: value.detach().cpu() for name, value in adapter.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "next_step": int(next_step),
            "loss_sums": loss_sums,
        },
        temporary,
    )
    temporary.replace(path)


def load_or_measure_seed_validation(
    path: Path,
    *,
    model: Any,
    adapter: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    data_root: Path,
    system_prompt: str,
    prefix_text: str,
    candidates: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, list[float]], dict[str, float]]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        logits = {str(key): list(value) for key, value in payload["logits"].items()}
        return payload["metrics"], logits, payload["visual_persistence"]
    adapter.belief_enabled = False
    metrics, logits, persistence = evaluate(
        model,
        adapter,
        processor,
        rows,
        data_root,
        system_prompt,
        prefix_text,
        candidates,
        batch_size,
    )
    adapter.belief_enabled = True
    atomic_json(
        path,
        {"schema_version": "judo-caver-seed-validation-v1", "metrics": metrics,
         "logits": logits, "visual_persistence": persistence},
    )
    return metrics, logits, persistence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--graft-seed", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--baseline-logits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--null-weight", type=float, default=1.5)
    parser.add_argument("--progressive-weight", type=float, default=0.5)
    parser.add_argument("--trajectory-weight", type=float, default=1.0)
    parser.add_argument("--pair-rank-weight", type=float, default=1.0)
    parser.add_argument("--null-rank-weight", type=float, default=0.5)
    parser.add_argument("--belief-weight", type=float, default=0.5)
    parser.add_argument("--binding-weight", type=float, default=0.5)
    parser.add_argument("--safety-kl-weight", type=float, default=1.0)
    parser.add_argument("--safety-frequency", type=int, default=4)
    parser.add_argument("--pair-margin", type=float, default=1.0)
    parser.add_argument("--null-margin", type=float, default=0.5)
    parser.add_argument("--state-save-steps", type=int, default=25)
    parser.add_argument("--belief-rank", type=int, default=128)
    parser.add_argument("--belief-temperature", type=float, default=1.0)
    parser.add_argument("--belief-scale-fraction", type=float, default=0.5)
    parser.add_argument("--expected-train-rows", type=int, default=5600)
    parser.add_argument("--expected-validation-rows", type=int, default=560)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if min(args.epochs, args.eval_batch_size, args.safety_frequency, args.state_save_steps) < 1:
        raise ValueError("invalid CAVER training sizes")
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows = read_jsonl(args.train_manifest), read_jsonl(args.validation_manifest)
    if len(train_rows) != args.expected_train_rows or len(validation_rows) != args.expected_validation_rows:
        raise ValueError("locked CAVER split mismatch")
    train_assets = {str(row[field]) for row in train_rows for field in ("image", "template_image")}
    validation_assets = {str(row[field]) for row in validation_rows for field in ("image", "template_image")}
    if train_assets & validation_assets:
        raise ValueError("CAVER train/validation asset leakage")
    missing = [
        args.data_root / str(row[field])
        for row in train_rows + validation_rows
        for field in ("image", "template_image")
        if not (args.data_root / str(row[field])).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{len(missing)} CAVER assets missing; first={missing[0]}")

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
    seed_identity = json.loads(
        (args.graft_seed / "native_deep_residual_identity.json").read_text(encoding="utf-8")
    )
    seed_stats = seed_identity["statistics"]
    adapter = install_caver_adapter(
        model,
        decision_prefix_ids=prefix_ids,
        assistant_prefix_ids=assistant_ids,
        seg_start_ids=seg_ids,
        belief_rank=args.belief_rank,
        belief_temperature=args.belief_temperature,
        belief_scale_fraction=args.belief_scale_fraction,
        replay_max_relative_rms=float(seed_stats["replay_max_relative_rms_per_site"]),
        replay_scale_fraction=float(seed_stats["replay_scale_fraction"]),
        unmatched_hidden=int(seed_stats["unmatched_hidden"]),
        unmatched_prior=float(seed_stats["unmatched_prior"]),
        transport_rank=int(seed_stats["transport_rank"]),
        transport_temperature=float(seed_stats["transport_temperature"]),
        hyper_rank=int(seed_stats["hyper_rank"]),
        bottleneck_size=int(seed_stats["bottleneck_size"]),
        num_heads=int(seed_stats["num_heads"]),
        injection_layers=tuple(int(value) for value in seed_stats["injection_layers"]),
        max_relative_rms=float(seed_stats["max_relative_rms_per_site"]),
        fixed_scale_fraction=float(seed_stats["fixed_scale_fraction"]),
        direction_floor=float(seed_stats["direction_floor"]),
    )
    loaded_seed = load_graft_seed(adapter, args.graft_seed)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    causal_parameters = adapter.causal_parameters()
    for parameter in causal_parameters:
        parameter.requires_grad_(True)
    if len({id(value) for value in causal_parameters}) != len(causal_parameters):
        raise RuntimeError("duplicate CAVER causal parameters")
    causal_ids = {id(value) for value in causal_parameters}
    actual_trainable_ids = {id(value) for value in model.parameters() if value.requires_grad}
    if actual_trainable_ids != causal_ids:
        raise RuntimeError(
            "CAVER trainable contract failed: only causal parameters may be optimized"
        )
    causal_named = {
        name: parameter
        for name, parameter in adapter.named_parameters()
        if id(parameter) in causal_ids
    }
    trainable = sum(parameter.numel() for parameter in causal_parameters)
    if trainable < 1:
        raise RuntimeError("CAVER has no trainable causal parameters")

    system_prompt = eval_entry.official_eval.make_system_prompt(16, 16)
    baseline_rows = {
        str(row["sample_id"]): list(row["logits"])
        for row in read_jsonl(args.baseline_logits)
    }
    baseline_validation_logits = {
        str(row["sample_id"]): baseline_rows[str(row["sample_id"])]
        for row in validation_rows
    }
    baseline_metrics = prediction_metrics(validation_rows, baseline_validation_logits)
    seed_metrics, seed_validation_logits, seed_persistence = load_or_measure_seed_validation(
        args.output_dir / "graft_seed_validation.json",
        model=model,
        adapter=adapter,
        processor=processor,
        rows=validation_rows,
        data_root=args.data_root,
        system_prompt=system_prompt,
        prefix_text=prefix_text,
        candidates=candidates,
        batch_size=args.eval_batch_size,
    )
    pairs = build_same_anchor_pairs(train_rows, args.seed)
    non_ad = [row for row in train_rows if row.get("question_type") != "Anomaly Detection"]
    random.Random(args.seed).shuffle(non_ad)
    total_steps = len(pairs) * args.epochs
    optimizer = torch.optim.AdamW(
        causal_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    names = (
        "total", "full_ce", "progressive_ce", "trajectory", "pair_rank",
        "null_rank", "belief", "binding", "safety_kl",
    )
    sums = {name: 0.0 for name in names}
    state_path = args.output_dir / "training_state.pt"
    start = 0
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != STATE_SCHEMA:
            raise ValueError("CAVER resume schema mismatch")
        adapter.load_state_dict(state["adapter"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["next_step"])
        sums = {name: float(state["loss_sums"][name]) for name in names}
        print(json.dumps({"event": "resume-caver-stage4b", "next_step": start}, sort_keys=True), flush=True)

    model.eval(); adapter.train()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(start, total_steps):
        anomaly, normal = pairs[step % len(pairs)]
        null = normal_null(anomaly)
        main_rows = [anomaly, normal, null]
        main_target = labels(main_rows, device)
        direct_texts = [direct_text(processor, row, system_prompt, prefix_text) for row in main_rows]
        partial_texts = [partial_cot_text(processor, row, system_prompt, prefix_text) for row in main_rows]
        full_texts = [decision_text(processor, row, system_prompt, prefix_text) for row in main_rows]

        safety_rows: list[dict[str, Any]] = []
        seed_safety: torch.Tensor | None = None
        if step % args.safety_frequency == 0:
            safety_rows = [non_ad[(step // args.safety_frequency) % len(non_ad)]]
            safety_texts = [decision_text(processor, safety_rows[0], system_prompt, prefix_text)]
            adapter.belief_enabled = False
            with torch.no_grad():
                seed_safety = answer_logits(
                    model,
                    prepare_text_batch(processor, safety_rows, args.data_root, safety_texts, device),
                    candidates,
                )
            adapter.belief_enabled = True
            full_rows = [*main_rows, *safety_rows]
            full_texts = [*full_texts, *safety_texts]
        else:
            full_rows = main_rows

        full_logits_all, full_belief_all = stage_forward(
            model, adapter, processor, full_rows, args.data_root, full_texts, candidates, device
        )
        full_logits, full_belief = full_logits_all[:3], full_belief_all[:3]
        partial_logits, partial_belief = stage_forward(
            model, adapter, processor, main_rows, args.data_root, partial_texts, candidates, device
        )
        direct_logits, direct_belief = stage_forward(
            model, adapter, processor, main_rows, args.data_root, direct_texts, candidates, device
        )

        row_weights = torch.tensor([1.0, 1.0, args.null_weight], device=device)
        full_ce = (F.cross_entropy(full_logits, main_target, reduction="none") * row_weights).mean()
        progressive_ce = 0.5 * (
            (F.cross_entropy(partial_logits, main_target, reduction="none") * row_weights).mean()
            + (F.cross_entropy(direct_logits, main_target, reduction="none") * row_weights).mean()
        )
        direct_margin = answer_margin(direct_logits, main_target).detach()
        partial_margin = answer_margin(partial_logits, main_target)
        full_margin = answer_margin(full_logits, main_target)
        trajectory = (
            F.relu(direct_margin - partial_margin).mean()
            + F.relu(torch.maximum(direct_margin, partial_margin.detach()) - full_margin).mean()
        )

        stage_logits = (direct_logits, partial_logits, full_logits)
        semantic = [semantic_anomaly_margin(value, main_rows) for value in stage_logits]
        pair_rank = torch.stack(
            [F.softplus(args.pair_margin - value[0] + value[1]) for value in semantic]
        ).mean()
        null_rank = torch.stack(
            [F.softplus(args.null_margin - value[1] + value[2]) for value in semantic]
        ).mean()
        intervention = torch.stack(
            [F.softplus(args.pair_margin + args.null_margin - value[0] + value[2]) for value in semantic]
        ).mean()
        pair_rank = 0.5 * (pair_rank + intervention)

        visual_target = torch.tensor([1, 0, 0], device=device)
        beliefs = (direct_belief, partial_belief, full_belief)
        belief_loss = torch.stack(
            [F.binary_cross_entropy_with_logits(value, visual_target.float()) for value in beliefs]
        ).mean()
        signs = visual_target.float().mul(2).sub(1)
        binding = torch.stack(
            [
                F.smooth_l1_loss(
                    torch.tanh(sem_value / 2.0),
                    torch.tanh(belief_value.detach() / 2.0),
                )
                + 0.25 * F.softplus(-signs * sem_value).mean()
                for sem_value, belief_value in zip(semantic, beliefs)
            ]
        ).mean()

        safety_kl = full_ce.new_zeros(())
        if safety_rows:
            if seed_safety is None:
                raise RuntimeError("CAVER safety seed logits missing")
            candidate_safety = full_logits_all[3:]
            safety_kl = F.kl_div(
                F.log_softmax(candidate_safety, dim=-1),
                F.softmax(seed_safety.detach(), dim=-1),
                reduction="batchmean",
            )

        loss = (
            full_ce
            + args.progressive_weight * progressive_ce
            + args.trajectory_weight * trajectory
            + args.pair_rank_weight * pair_rank
            + args.null_rank_weight * null_rank
            + args.belief_weight * belief_loss
            + args.binding_weight * binding
            + args.safety_kl_weight * safety_kl
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(causal_parameters, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite CAVER gradient")
        first_step_gradients: dict[str, float] = {}
        if step == 0:
            first_step_gradients = {
                name: float(parameter.grad.detach().float().norm().cpu())
                if parameter.grad is not None else 0.0
                for name, parameter in causal_named.items()
            }
            if not any(
                value > 0.0 for name, value in first_step_gradients.items()
                if name.startswith("belief_up.")
            ):
                raise RuntimeError("CAVER first step did not reach the causal belief writer")
            if not any(
                value > 0.0 for name, value in first_step_gradients.items()
                if name.startswith("visual_verdict.")
            ):
                raise RuntimeError("CAVER first step did not train the visual verdict")
            frozen_with_grad = [
                name for name, parameter in adapter.named_parameters()
                if id(parameter) not in causal_ids and parameter.grad is not None
            ]
            if frozen_with_grad:
                raise RuntimeError(f"frozen CAVER seed received gradients: {frozen_with_grad[:3]}")
        optimizer.step()
        if step == 0:
            belief_writer_max = max(
                float(layer.weight.detach().abs().max().cpu()) for layer in adapter.belief_up
            )
            if belief_writer_max <= 0.0:
                raise RuntimeError("CAVER optimizer did not update the causal belief writer")
            atomic_json(
                args.output_dir / "G4_SMOKE_SUCCESS.json",
                {
                    "schema_version": "judo-caver-g4-smoke-v1",
                    "status": "passed",
                    "actual_model": str(args.model),
                    "actual_data_rows": [row["sample_id"] for row in main_rows],
                    "trajectory_forwards": ["direct", "partial-cot", "full-cot"],
                    "finite_loss": bool(torch.isfinite(loss.detach()).item()),
                    "clipped_gradient_norm": float(norm.detach().cpu()),
                    "gradient_norms": first_step_gradients,
                    "belief_writer_max_abs_after_update": belief_writer_max,
                    "trainable_parameter_count": trainable,
                    "frozen_parameter_gradient_count": 0,
                    "max_cuda_memory_allocated_gib": (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    ),
                },
            )
        values = {
            "total": loss,
            "full_ce": full_ce,
            "progressive_ce": progressive_ce,
            "trajectory": trajectory,
            "pair_rank": pair_rank,
            "null_rank": null_rank,
            "belief": belief_loss,
            "binding": binding,
            "safety_kl": safety_kl,
        }
        for name, value in values.items():
            sums[name] += float(value.detach().cpu())
        if step == 0 or (step + 1) % args.state_save_steps == 0 or step + 1 == total_steps:
            save_state(
                state_path,
                adapter,
                optimizer,
                next_step=step + 1,
                loss_sums=sums,
            )
            update_progress(
                args.progress_json,
                "train-caver-stage4b",
                step + 1,
                total_steps,
                "normal-null-trajectory-preference",
            )

    metrics, candidate_logits, persistence = evaluate(
        model,
        adapter,
        processor,
        validation_rows,
        args.data_root,
        system_prompt,
        prefix_text,
        candidates,
        args.eval_batch_size,
    )
    metrics["paired_vs_public_baseline"] = paired_delta(
        validation_rows, baseline_validation_logits, candidate_logits
    )
    metrics["paired_vs_graft_seed"] = paired_delta(
        validation_rows, seed_validation_logits, candidate_logits
    )
    per_task_safe = all(
        metrics["per_task_accuracy"][task]
        >= baseline_metrics["per_task_accuracy"][task] - 0.0125
        for task in baseline_metrics["per_task_accuracy"]
    )
    screen_pass = (
        per_task_safe
        and metrics["accuracy"] >= seed_metrics["accuracy"]
        and metrics["accuracy"] > baseline_metrics["accuracy"]
        and metrics["ad_balanced_accuracy"] > baseline_metrics["ad_balanced_accuracy"]
    )
    identity = save_adapter(
        adapter,
        args.output_dir / "final",
        {
            "selected_epoch": args.epochs,
            "decision_protocol": protocol,
            "phase_protocol": {
                "assistant_prefix_ids": assistant_ids,
                "seg_start_ids": seg_ids,
            },
            "validation_metrics": metrics,
            "visual_persistence": persistence,
            "graft_seed_weights_sha256": loaded_seed["weights_sha256"],
            "screen_pass": screen_pass,
        },
    )
    atomic_json(args.output_dir / "candidate_validation_logits.json", candidate_logits)
    summary = {
        "schema_version": "judo-caver-stage4b-training-summary-v1",
        "status": "complete",
        "screen_pass": screen_pass,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "same_anchor_pairs": len(pairs),
        "total_steps": total_steps,
        "trainable_parameter_count": trainable,
        "graft_seed_weights_sha256": loaded_seed["weights_sha256"],
        "public_baseline_validation": baseline_metrics,
        "graft_seed_validation": seed_metrics,
        "graft_seed_visual_persistence": seed_persistence,
        "candidate_validation": metrics,
        "candidate_visual_persistence": persistence,
        "mean_losses": {name: value / max(1, total_steps) for name, value in sums.items()},
        "identity": identity,
        "hyperparameters": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
    }
    atomic_json(args.output_dir / "caver_stage4b_training_summary.json", summary)
    save_state(
        state_path,
        adapter,
        optimizer,
        next_step=total_steps,
        loss_sums=sums,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "screen_pass": screen_pass,
                "public_accuracy": baseline_metrics["accuracy"],
                "graft_accuracy": seed_metrics["accuracy"],
                "candidate_accuracy": metrics["accuracy"],
                "public_ad_balanced": baseline_metrics["ad_balanced_accuracy"],
                "candidate_ad_balanced": metrics["ad_balanced_accuracy"],
                "architecture": ARCHITECTURE,
                "adapter_sha256": identity["weights_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
