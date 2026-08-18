#!/usr/bin/env python3
"""Train question-conditioned semantic RefDiff memory on all MMAD tasks."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import random
from typing import Any

import torch
from torch.nn import functional as F

import eval_manifest as eval_entry
from judo_aligned_ce import configure_processor_for_residual
from judo_semantic_refdiff import install_semantic_refdiff, save_adapter, trainable_contract
from train_judo_aligned_ce import atomic_json, prepare_batch, read_jsonl, update_progress


SCHEMA_VERSION = "judo-semantic-refdiff-training-state-v1"
LETTERS = ("A", "B", "C", "D")


def deterministic_batches(rows: list[dict[str, Any]], batch_size: int, seed: int) -> list[list[dict[str, Any]]]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [[rows[index] for index in order[start : start + batch_size]] for start in range(0, len(order), batch_size)]


def build_counterfactual_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Create outcome-preserving reference interventions using train assets only.

    Normal queries use themselves as their null comparison.  Anomalous
    queries receive a deterministic different normal reference from the same
    source/category whenever one exists.  The question and answer never
    change, so this is an invariance intervention rather than relabeling.
    """
    pools: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (str(row["source"]), str(row["category"]))
        pools.setdefault(key, []).append(str(row["template_image"]))
    pools = {key: sorted(set(values)) for key, values in pools.items()}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        alternate = dict(row)
        if str(row["label"]) == "normal":
            alternate["template_image"] = str(row["image"])
        else:
            values = pools[(str(row["source"]), str(row["category"]))]
            current = str(row["template_image"])
            if len(values) > 1:
                position = values.index(current) if current in values else -1
                alternate["template_image"] = values[(position + 1) % len(values)]
        result[str(row["sample_id"])] = alternate
    return result


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers or len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("injection layers must be unique comma-separated integers")
    return layers


def sample_residual_ratios(adapter: Any, expected_batch: int) -> torch.Tensor:
    if not adapter.last_residual_ratios:
        raise RuntimeError("semantic RefDiff did not record decoder residuals")
    values = torch.stack([tensor.mean(dim=1) for tensor in adapter.last_residual_ratios]).mean(dim=0)
    if values.numel() != expected_batch:
        raise RuntimeError(f"semantic residual batch mismatch: {values.numel()} != {expected_batch}")
    return values


def candidate_ids(tokenizer: Any) -> torch.Tensor:
    values = []
    for answer in LETTERS:
        ids = tokenizer(answer, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"answer {answer!r} is not one tokenizer token: {ids}")
        values.append(ids[0])
    if len(set(values)) != len(values):
        raise ValueError("candidate letters do not map to unique tokenizer tokens")
    return torch.tensor(values, dtype=torch.long)


def answer_logits(model: Any, inputs: dict[str, torch.Tensor], candidates: torch.Tensor) -> torch.Tensor:
    outputs = model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)
    return outputs.logits[:, -1, :].index_select(-1, candidates.to(outputs.logits.device)).float()


def labels_for(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.tensor([LETTERS.index(str(row["correct_answer"])) for row in rows], device=device)


def append_logits(path: Path, rows: list[dict[str, Any]], logits: torch.Tensor) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row, values in zip(rows, logits.detach().cpu().tolist()):
            handle.write(
                json.dumps(
                    {"sample_id": row["sample_id"], "logits": values},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def read_logits(path: Path) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    result: dict[str, list[float]] = {}
    valid: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            path.write_text("\n".join(valid) + ("\n" if valid else ""), encoding="utf-8")
            break
        valid.append(line)
        values = [float(value) for value in row["logits"]]
        if len(values) != 4:
            raise ValueError("teacher cache does not contain four candidate logits")
        result[str(row["sample_id"])] = values
    return result


def prediction_metrics(rows: list[dict[str, Any]], logits_by_id: dict[str, list[float]]) -> dict[str, Any]:
    cell_total: Counter[tuple[str, str]] = Counter()
    cell_correct: Counter[tuple[str, str]] = Counter()
    task_total: Counter[str] = Counter()
    task_correct: Counter[str] = Counter()
    ad_total: Counter[str] = Counter()
    ad_correct: Counter[str] = Counter()
    correct = 0
    for row in rows:
        target = LETTERS.index(str(row["correct_answer"]))
        pred = max(range(4), key=lambda index: logits_by_id[str(row["sample_id"])][index])
        hit = int(pred == target)
        source, task = str(row["source"]), str(row["question_type"])
        cell_total[(source, task)] += 1
        cell_correct[(source, task)] += hit
        task_total[task] += 1
        task_correct[task] += hit
        correct += hit
        if task == "Anomaly Detection":
            label = str(row["label"])
            ad_total[label] += 1
            ad_correct[label] += hit
    per_cell = {
        f"{source}|{task}": cell_correct[(source, task)] / cell_total[(source, task)]
        for source, task in sorted(cell_total)
    }
    per_task = {task: task_correct[task] / task_total[task] for task in sorted(task_total)}
    anomaly = ad_correct["anomaly"] / ad_total["anomaly"]
    normal = ad_correct["normal"] / ad_total["normal"]
    return {
        "samples": len(rows),
        "accuracy": correct / len(rows),
        "source_task_macro_accuracy": sum(per_cell.values()) / len(per_cell),
        "per_cell_accuracy": per_cell,
        "per_task_accuracy": per_task,
        "ad_anomaly_recall": anomaly,
        "ad_normal_specificity": normal,
        "ad_balanced_accuracy": (anomaly + normal) / 2,
    }


def paired_delta(
    rows: list[dict[str, Any]],
    baseline: dict[str, list[float]],
    candidate: dict[str, list[float]],
) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        key = str(row["sample_id"])
        target = LETTERS.index(str(row["correct_answer"]))
        b = max(range(4), key=lambda index: baseline[key][index]) == target
        c = max(range(4), key=lambda index: candidate[key][index]) == target
        counts["both_correct" if b and c else "regressions" if b else "rescues" if c else "both_wrong"] += 1
    counts["net_rescues"] = counts["rescues"] - counts["regressions"]
    return dict(counts)


def evaluate(
    model: Any,
    adapter: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    data_root: Path,
    system_prompt: str,
    candidates: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    output: dict[str, list[float]] = {}
    model.eval()
    adapter.eval()
    with torch.no_grad():
        for batch in deterministic_batches(rows, batch_size, 0):
            inputs = prepare_batch(processor, batch, data_root, system_prompt, next(model.parameters()).device)
            logits = answer_logits(model, inputs, candidates)
            for row, values in zip(batch, logits.cpu().tolist()):
                output[str(row["sample_id"])] = values
    return prediction_metrics(rows, output), output


def save_state(
    path: Path,
    adapter: Any,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    next_batch: int,
    history: list[dict[str, Any]],
    best: dict[str, Any],
) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "adapter": {name: tensor.detach().cpu() for name, tensor in adapter.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "next_batch": next_batch,
            "history": history,
            "best": best,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judo-model", type=Path, required=True)
    parser.add_argument("--hybrid-model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--scale-learning-rate", type=float, default=5e-2)
    parser.add_argument("--ce-correct-weight", type=float, default=0.5)
    parser.add_argument("--ce-wrong-weight", type=float, default=2.0)
    parser.add_argument("--anchor-correct-weight", type=float, default=1.0)
    parser.add_argument("--anchor-wrong-weight", type=float, default=0.05)
    parser.add_argument("--counterfactual-weight", type=float, default=0.05)
    parser.add_argument("--reference-consistency-weight", type=float, default=0.20)
    parser.add_argument("--repair-margin-weight", type=float, default=0.50)
    parser.add_argument("--repair-margin-gain", type=float, default=1.0)
    parser.add_argument("--bottleneck-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-relative-rms", type=float, default=0.02)
    parser.add_argument("--direction-floor", type=float, default=0.10)
    parser.add_argument("--injection-layers", type=parse_layers, default=(7, 14, 21))
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("one BF16 CUDA GPU is required")
    if min(args.gradient_accumulation, args.batch_size, args.teacher_batch_size, args.eval_batch_size) < 1:
        raise ValueError("all batch sizes and gradient accumulation must be positive")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_manifest)
    validation_rows = read_jsonl(args.validation_manifest)
    if not train_rows or not validation_rows:
        raise ValueError("training and validation manifests must be non-empty")
    counterfactual_rows = build_counterfactual_rows(train_rows)
    required_teacher_fields = ("teacher_segmentation", "teacher_thinking")
    for row in train_rows + validation_rows:
        if any(field not in row for field in required_teacher_fields):
            raise ValueError("teacher fields are missing from the training manifest")
    train_assets = {str(row[field]) for row in train_rows for field in ("image", "template_image")}
    validation_assets = {str(row[field]) for row in validation_rows for field in ("image", "template_image")}
    if train_assets & validation_assets:
        raise ValueError("training and validation assets overlap")
    required_paths = {
        args.data_root / str(row[field])
        for row in train_rows + validation_rows
        for field in ("image", "template_image")
    }
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} training assets are missing; first={missing[0]}")

    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from transformers import Qwen2_5_VLForConditionalGeneration

    device = torch.device("cuda:0")
    model = AutoModelForImageTextToText.from_pretrained(
        args.hybrid_model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.hybrid_model, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hybrid_model, trust_remote_code=True, local_files_only=True)
    configure_processor_for_residual(processor)
    adapter = install_semantic_refdiff(
        model,
        bottleneck_size=args.bottleneck_size,
        num_heads=args.num_heads,
        injection_layers=args.injection_layers,
        max_relative_rms=args.max_relative_rms,
        direction_floor=args.direction_floor,
    )
    contract = trainable_contract(model, adapter, trainable=True)
    candidates = candidate_ids(tokenizer)
    system_prompt = eval_entry.official_eval.make_system_prompt(16, 16)

    parity_rows = validation_rows[:8]
    aligned_inputs = prepare_batch(processor, parity_rows, args.data_root, system_prompt, device)
    with torch.no_grad():
        aligned_logits = answer_logits(model, aligned_inputs, candidates).cpu()
    baseline_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.judo_model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    baseline_processor = AutoProcessor.from_pretrained(args.judo_model, local_files_only=True)
    baseline_processor.tokenizer.padding_side = "left"
    baseline_inputs = prepare_batch(baseline_processor, parity_rows, args.data_root, system_prompt, device)
    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        if not torch.equal(aligned_inputs[key], baseline_inputs[key]):
            raise RuntimeError(f"zero-init processor parity failed for {key}")
    with torch.no_grad():
        baseline_logits = answer_logits(baseline_model, baseline_inputs, candidates).cpu()
    max_abs = float((aligned_logits - baseline_logits).abs().max())
    argmax_match = torch.equal(aligned_logits.argmax(dim=-1), baseline_logits.argmax(dim=-1))
    logit_tolerance = 1e-4
    zero_global_scale = float(adapter.global_scale.detach().cpu()) == 0.0
    nonzero_output_projection = int(torch.count_nonzero(adapter.out_proj.weight.detach()).cpu()) > 0
    tolerance_match = bool(torch.allclose(aligned_logits, baseline_logits, rtol=0.0, atol=logit_tolerance))
    accepted = argmax_match and tolerance_match and zero_global_scale and nonzero_output_projection
    parity = {
        "samples": len(parity_rows),
        "exact_abcd_logit_match": torch.equal(aligned_logits, baseline_logits),
        "abcd_argmax_match": argmax_match,
        "zero_initialized_global_scale": zero_global_scale,
        "injection_layers": list(adapter.injection_layers),
        "nonzero_initialized_output_projection": nonzero_output_projection,
        "absolute_logit_tolerance": logit_tolerance,
        "within_absolute_logit_tolerance": tolerance_match,
        "accepted_functional_parity": accepted,
        "max_abs_abcd_logit_difference": max_abs,
        "aligned_logits": aligned_logits.tolist(),
        "baseline_logits": baseline_logits.tolist(),
    }
    atomic_json(args.output_dir / "zero_init_parity.json", parity)
    if not accepted:
        raise RuntimeError(
            "zero-init RefDiff functional parity failed: "
            f"max_abs={max_abs}, argmax_match={argmax_match}, "
            f"zero_global_scale={zero_global_scale}, "
            f"nonzero_output_projection={nonzero_output_projection}"
        )
    del baseline_model, baseline_inputs, aligned_inputs
    gc.collect()
    torch.cuda.empty_cache()

    cache_path = args.output_dir / "baseline_teacher_logits.jsonl"
    cached = read_logits(cache_path)
    all_rows = train_rows + validation_rows
    remaining = [row for row in all_rows if str(row["sample_id"]) not in cached]
    model.eval()
    adapter.eval()
    with torch.no_grad():
        pending_rows: list[dict[str, Any]] = []
        pending_logits: list[torch.Tensor] = []
        teacher_batches = deterministic_batches(remaining, args.teacher_batch_size, 1)
        flush_every = max(1, 64 // args.teacher_batch_size)
        for index, batch in enumerate(teacher_batches):
            inputs = prepare_batch(processor, batch, args.data_root, system_prompt, device)
            logits = answer_logits(model, inputs, candidates)
            pending_rows.extend(batch)
            pending_logits.append(logits.cpu())
            if (index + 1) % flush_every == 0 or index + 1 == len(teacher_batches):
                append_logits(cache_path, pending_rows, torch.cat(pending_logits))
                pending_rows.clear()
                pending_logits.clear()
    cached = read_logits(cache_path)
    if set(cached) != {str(row["sample_id"]) for row in all_rows}:
        raise RuntimeError("baseline teacher cache is incomplete")
    baseline_validation = prediction_metrics(validation_rows, cached)

    direction_parameters = [
        parameter for name, parameter in adapter.named_parameters() if name != "global_scale"
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": direction_parameters, "lr": args.learning_rate, "weight_decay": 0.01},
            {"params": [adapter.global_scale], "lr": args.scale_learning_rate, "weight_decay": 0.0},
        ]
    )
    state_path = args.output_dir / "training_state.pt"
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"epoch": 0, "metrics": baseline_validation, "safe": True}
    best_dir = args.output_dir / "best_adapter"
    if not (best_dir / "semantic_refdiff_adapter.safetensors").is_file():
        save_adapter(
            adapter,
            best_dir,
            {
                "selected_epoch": 0,
                "selection_reason": "exact public-JUDO zero-scale semantic-memory identity candidate",
                "validation_metrics": baseline_validation,
            },
        )
    start_epoch, start_batch = 1, 0
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("training state schema mismatch")
        adapter.load_state_dict(state["adapter"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        history = list(state["history"])
        best = dict(state["best"])
        start_epoch, start_batch = int(state["epoch"]), int(state["next_batch"])

    total_work = len(all_rows) + args.epochs * (len(train_rows) + len(validation_rows))
    for epoch in range(start_epoch, args.epochs + 1):
        batches = deterministic_batches(train_rows, args.batch_size, args.seed + epoch)
        model.eval()
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sums = Counter()
        for batch_index in range(start_batch if epoch == start_epoch else 0, len(batches)):
            batch = batches[batch_index]
            inputs = prepare_batch(processor, batch, args.data_root, system_prompt, device)
            logits = answer_logits(model, inputs, candidates)
            primary_residual_ratios = sample_residual_ratios(adapter, len(batch))
            labels = labels_for(batch, device)
            teacher = torch.tensor([cached[str(row["sample_id"])] for row in batch], device=device)
            teacher_pred = teacher.argmax(dim=-1)
            baseline_correct = teacher_pred.eq(labels)

            ce_values = F.cross_entropy(logits, labels, reduction="none")
            ce_weights = torch.where(
                baseline_correct,
                torch.full_like(ce_values, args.ce_correct_weight),
                torch.full_like(ce_values, args.ce_wrong_weight),
            )
            ce_loss = (ce_values * ce_weights).mean()
            teacher_probs = F.softmax(teacher, dim=-1)
            kl_values = F.kl_div(F.log_softmax(logits, dim=-1), teacher_probs, reduction="none").sum(-1)
            anchor_weights = torch.where(
                baseline_correct,
                torch.full_like(kl_values, args.anchor_correct_weight),
                torch.full_like(kl_values, args.anchor_wrong_weight),
            )
            anchor_loss = (kl_values * anchor_weights).mean()

            competitor_mask = F.one_hot(labels, num_classes=len(LETTERS)).bool()
            candidate_margin = logits.gather(1, labels[:, None]).squeeze(1) - logits.masked_fill(
                competitor_mask, float("-inf")
            ).max(dim=-1).values
            teacher_margin = teacher.gather(1, labels[:, None]).squeeze(1) - teacher.masked_fill(
                competitor_mask, float("-inf")
            ).max(dim=-1).values
            wrong_indices = ~baseline_correct
            if wrong_indices.any():
                margin_improvement = candidate_margin[wrong_indices] - teacher_margin[wrong_indices]
                repair_loss = F.softplus(args.repair_margin_gain - margin_improvement).mean()
            else:
                repair_loss = logits.new_zeros(())

            normal_ad_indices = [
                index
                for index, row in enumerate(batch)
                if row["question_type"] == "Anomaly Detection" and row["label"] == "normal"
            ]
            if normal_ad_indices:
                counterfactual_loss = torch.stack(
                    [
                        (primary_residual_ratios[index] / adapter.max_relative_rms).square()
                        for index in normal_ad_indices
                    ]
                ).mean()
            else:
                counterfactual_loss = logits.new_zeros(())
            primary_loss = (
                ce_loss
                + anchor_loss
                + args.repair_margin_weight * repair_loss
                + args.counterfactual_weight * counterfactual_loss
            )
            primary_probs = F.softmax(logits.detach(), dim=-1)
            (primary_loss / args.gradient_accumulation).backward()

            # Backpropagate the intervention in a second graph.  This keeps
            # peak memory close to one frozen-LLM forward instead of retaining
            # two 8B-model activation graphs simultaneously.
            alternate_batch = [counterfactual_rows[str(row["sample_id"])] for row in batch]
            alternate_inputs = prepare_batch(processor, alternate_batch, args.data_root, system_prompt, device)
            alternate_logits = answer_logits(model, alternate_inputs, candidates)
            alternate_log_probs = F.log_softmax(alternate_logits, dim=-1)
            reference_consistency_loss = F.kl_div(
                alternate_log_probs,
                primary_probs,
                reduction="batchmean",
            )
            consistency_term = args.reference_consistency_weight * reference_consistency_loss
            (consistency_term / args.gradient_accumulation).backward()
            loss = primary_loss.detach() + consistency_term.detach()
            loss_sums["total"] += float(loss.detach().cpu())
            loss_sums["ce"] += float(ce_loss.detach().cpu())
            loss_sums["anchor"] += float(anchor_loss.detach().cpu())
            loss_sums["repair_margin"] += float(repair_loss.detach().cpu())
            loss_sums["reference_consistency"] += float(reference_consistency_loss.detach().cpu())
            loss_sums["counterfactual"] += float(counterfactual_loss.detach().cpu())
            if (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(batches):
                grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite refdiff gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (batch_index + 1) % 50 == 0:
                save_state(state_path, adapter, optimizer, epoch, batch_index + 1, history, best)
            if (batch_index + 1) % 25 == 0 or batch_index + 1 == len(batches):
                completed = len(all_rows) + (epoch - 1) * (len(train_rows) + len(validation_rows)) + min(
                    (batch_index + 1) * args.batch_size, len(train_rows)
                )
                update_progress(args.progress_json, "train-refdiff", completed, total_work, f"epoch-{epoch}-train")

        validation_metrics, validation_logits = evaluate(
            model, adapter, processor, validation_rows, args.data_root, system_prompt, candidates, args.eval_batch_size
        )
        validation_metrics["paired_vs_baseline"] = paired_delta(validation_rows, cached, validation_logits)
        per_task_safe = all(
            validation_metrics["per_task_accuracy"][task]
            >= baseline_validation["per_task_accuracy"][task] - 0.02
            for task in baseline_validation["per_task_accuracy"]
        )
        recall_safe = validation_metrics["ad_anomaly_recall"] >= baseline_validation["ad_anomaly_recall"] - 0.02
        safe = per_task_safe and recall_safe
        record = {
            "epoch": epoch,
            "mean_losses": {name: value / len(batches) for name, value in loss_sums.items()},
            "validation_metrics": validation_metrics,
            "safe_noninferiority": safe,
            "per_task_safe": per_task_safe,
            "ad_recall_safe": recall_safe,
            "adapter_statistics": adapter.statistics(),
        }
        history.append(record)
        atomic_json(
            args.output_dir / "training_history.json",
            {"baseline_validation": baseline_validation, "epochs": history},
        )
        epoch_dir = args.output_dir / f"epoch-{epoch}"
        save_adapter(
            adapter,
            epoch_dir,
            {
                "selected_epoch": epoch,
                "selection_reason": "epoch snapshot for independent autoregressive selection",
                "validation_metrics": validation_metrics,
                "safe_noninferiority": safe,
            },
        )
        if safe and validation_metrics["source_task_macro_accuracy"] > best["metrics"]["source_task_macro_accuracy"]:
            best = {"epoch": epoch, "metrics": validation_metrics, "safe": True}
            save_adapter(
                adapter,
                best_dir,
                {
                    "selected_epoch": epoch,
                    "selection_reason": "best safe source-task macro teacher-forced validation accuracy",
                    "validation_metrics": validation_metrics,
                    "safe_noninferiority": True,
                },
            )
        next_epoch = epoch + 1
        save_state(state_path, adapter, optimizer, next_epoch, 0, history, best)
        start_batch = 0

    summary = {
        "schema_version": "judo-semantic-refdiff-training-summary-v1",
        "status": "complete",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "contract": contract,
        "hyperparameters": vars(args) | {
            "judo_model": str(args.judo_model),
            "hybrid_model": str(args.hybrid_model),
            "data_root": str(args.data_root),
            "train_manifest": str(args.train_manifest),
            "validation_manifest": str(args.validation_manifest),
            "output_dir": str(args.output_dir),
            "progress_json": str(args.progress_json) if args.progress_json else None,
        },
        "baseline_validation": baseline_validation,
        "best": best,
        "adapter_statistics": adapter.statistics(),
    }
    atomic_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
