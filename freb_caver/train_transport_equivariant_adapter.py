#!/usr/bin/env python3
"""Train transport-equivariant causal evidence inside frozen JUDO."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

import eval_manifest as eval_entry
from judo_native_deep_residual import (
    install_defect_preserving_partial_transport_causal_adapter,
    install_transport_equivariant_causal_adapter,
    save_adapter,
    trainable_contract,
)
from train_decision_aligned_refdiff import answer_protocol, labels, prepare_batch
from train_judo_aligned_ce import atomic_json, read_jsonl, update_progress
from train_judo_semantic_refdiff import paired_delta, prediction_metrics


def batches(rows: list[dict[str, Any]], size: int, seed: int) -> list[list[dict[str, Any]]]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [[rows[index] for index in order[start : start + size]] for start in range(0, len(order), size)]


def alternate_references(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    pools: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        pools.setdefault((str(row["source"]), str(row["category"])), []).append(
            str(row["template_image"])
        )
    pools = {key: sorted(set(values)) for key, values in pools.items()}
    result = {}
    for row in rows:
        alternate = dict(row)
        if row["label"] == "normal":
            alternate["template_image"] = str(row["image"])
        else:
            values = pools[(str(row["source"]), str(row["category"]))]
            current = str(row["template_image"])
            if len(values) > 1:
                position = values.index(current) if current in values else -1
                alternate["template_image"] = values[(position + 1) % len(values)]
        result[str(row["sample_id"])] = alternate
    return result


def answer_logits(model: Any, inputs: dict[str, torch.Tensor], candidates: torch.Tensor) -> torch.Tensor:
    output = model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)
    return output.logits[:, -1].index_select(-1, candidates.to(output.logits.device)).float()


def evaluate(model: Any, adapter: Any, processor: Any, rows: list[dict[str, Any]], data_root: Path, system_prompt: str, prefix_text: str, candidates: torch.Tensor, batch_size: int) -> tuple[dict[str, Any], dict[str, list[float]]]:
    output = {}
    model.eval(); adapter.eval()
    with torch.no_grad():
        for batch in batches(rows, batch_size, 0):
            inputs = prepare_batch(processor, batch, data_root, system_prompt, prefix_text, next(model.parameters()).device)
            values = answer_logits(model, inputs, candidates)
            for row, value in zip(batch, values.cpu().tolist()):
                output[str(row["sample_id"])] = value
    return prediction_metrics(rows, output), output


def save_state(
    path: Path,
    adapter: Any,
    optimizer: Any,
    *,
    next_batch: int,
    best: dict[str, Any],
    loss_sums: dict[str, float],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": "judo-native-deep-residual-resume-v1",
            "adapter": {name: value.detach().cpu() for name, value in adapter.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "next_batch": int(next_batch),
            "best": best,
            "loss_sums": loss_sums,
        },
        temporary,
    )
    temporary.replace(path)


def load_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    """Read a resumable JSONL cache, ignoring only a torn final record."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    raw = path.read_bytes().splitlines()
    for index, line in enumerate(raw):
        if not line.strip():
            continue
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index != len(raw) - 1:
                raise
            break
        if not isinstance(value, dict) or not value.get("sample_id"):
            raise ValueError(f"invalid baseline-logit row in {path}")
        rows.append(value)
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs in baseline-logit cache")
    temporary = path.with_suffix(path.suffix + ".repair")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)
    return rows


def cache_baseline_logits(
    model: Any,
    adapter: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    data_root: Path,
    system_prompt: str,
    prefix_text: str,
    candidates: torch.Tensor,
    batch_size: int,
    path: Path,
    progress_json: Path | None,
) -> dict[str, list[float]]:
    """Materialize frozen-JUDO decision logits once with batch-level durability."""
    cached_rows = load_jsonl_tolerant(path)
    expected = {str(row["sample_id"]) for row in rows}
    cached = {str(row["sample_id"]): row["logits"] for row in cached_rows}
    if not set(cached).issubset(expected):
        raise ValueError("baseline-logit cache contains foreign sample IDs")
    if any(not isinstance(value, list) or len(value) != 4 for value in cached.values()):
        raise ValueError("baseline logits must contain four answer branches")
    missing = [row for row in rows if str(row["sample_id"]) not in cached]
    model.eval(); adapter.eval()
    with torch.no_grad():
        for batch in batches(missing, batch_size, 0):
            inputs = prepare_batch(
                processor, batch, data_root, system_prompt, prefix_text,
                next(model.parameters()).device,
            )
            values = answer_logits(model, inputs, candidates).cpu().tolist()
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                for row, logits in zip(batch, values):
                    record = {"sample_id": str(row["sample_id"]), "logits": logits}
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    cached[record["sample_id"]] = logits
                handle.flush()
                os.fsync(handle.fileno())
            update_progress(
                progress_json, "cache-baseline-logits", len(cached), len(rows),
                "frozen-public-JUDO",
            )
    if set(cached) != expected:
        raise RuntimeError("baseline-logit cache is incomplete")
    return cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--baseline-logits", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--correct-weight", type=float, default=0.25)
    parser.add_argument("--wrong-weight", type=float, default=2.0)
    parser.add_argument("--alternate-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--anomaly-weight", type=float, default=0.05)
    parser.add_argument("--anchor-kl-weight", type=float, default=1.0)
    parser.add_argument("--margin-retention-weight", type=float, default=2.0)
    parser.add_argument("--cycle-weight", type=float, default=0.01)
    parser.add_argument("--preserve-weight", type=float, default=1.0)
    parser.add_argument("--bottleneck-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--hyper-rank", type=int, default=256)
    parser.add_argument("--transport-rank", type=int, default=64)
    parser.add_argument("--transport-temperature", type=float, default=0.07)
    parser.add_argument("--partial-transport", action="store_true")
    parser.add_argument("--unmatched-hidden", type=int, default=64)
    parser.add_argument("--unmatched-prior", type=float, default=0.05)
    parser.add_argument("--unmatched-weight", type=float, default=0.02)
    parser.add_argument("--injection-layers", default="18,20,22,24,26")
    parser.add_argument("--max-relative-rms", type=float, default=0.10)
    parser.add_argument("--fixed-scale-fraction", type=float, default=0.90)
    parser.add_argument("--state-save-steps", type=int, default=10)
    parser.add_argument("--expected-train-rows", type=int, default=1120)
    parser.add_argument("--expected-validation-rows", type=int, default=560)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if min(args.batch_size, args.eval_batch_size, args.state_save_steps) < 1:
        raise ValueError("invalid training sizes")
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows = read_jsonl(args.train_manifest), read_jsonl(args.validation_manifest)
    if len(train_rows) != args.expected_train_rows or len(validation_rows) != args.expected_validation_rows:
        raise ValueError("locked native deep split mismatch")
    required_teacher_fields = ("teacher_segmentation", "teacher_thinking")
    if any(field not in row for row in train_rows + validation_rows for field in required_teacher_fields):
        raise ValueError("training and validation rows require frozen public-JUDO prefixes")
    train_assets = {str(row[field]) for row in train_rows for field in ("image", "template_image")}
    validation_assets = {str(row[field]) for row in validation_rows for field in ("image", "template_image")}
    if train_assets & validation_assets:
        raise ValueError("train/validation asset leakage")
    missing = [args.data_root / str(row[field]) for row in train_rows + validation_rows for field in ("image", "template_image") if not (args.data_root / str(row[field])).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} assets missing; first={missing[0]}")
    alternates = alternate_references(train_rows)

    from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    device = torch.device("cuda:0")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, local_files_only=True, attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    prefix_ids, prefix_text, candidates, protocol = answer_protocol(tokenizer)
    layers = tuple(int(value) for value in args.injection_layers.split(","))
    install_kwargs = {
        "decision_prefix_ids": prefix_ids,
        "bottleneck_size": args.bottleneck_size,
        "num_heads": args.num_heads,
        "injection_layers": layers,
        "max_relative_rms": args.max_relative_rms,
        "fixed_scale_fraction": args.fixed_scale_fraction,
        "hyper_rank": args.hyper_rank,
        "transport_rank": args.transport_rank,
        "transport_temperature": args.transport_temperature,
    }
    if args.partial_transport:
        adapter = install_defect_preserving_partial_transport_causal_adapter(
            model,
            unmatched_hidden=args.unmatched_hidden,
            unmatched_prior=args.unmatched_prior,
            **install_kwargs,
        )
    else:
        adapter = install_transport_equivariant_causal_adapter(model, **install_kwargs)
    contract = trainable_contract(model, adapter, True)
    system_prompt = eval_entry.official_eval.make_system_prompt(16, 16)
    baseline_path = args.baseline_logits or (args.output_dir / "baseline_teacher_logits.jsonl")
    baseline_by_id = cache_baseline_logits(
        model, adapter, processor, train_rows + validation_rows, args.data_root,
        system_prompt, prefix_text, candidates, args.eval_batch_size,
        baseline_path, args.progress_json,
    )
    baseline_validation_logits = {
        str(row["sample_id"]): baseline_by_id[str(row["sample_id"])]
        for row in validation_rows
    }
    baseline_metrics = prediction_metrics(validation_rows, baseline_validation_logits)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapter.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    best = {
        "epoch": 0,
        "metrics": baseline_metrics,
        "state": {name: value.detach().cpu() for name, value in adapter.state_dict().items()},
    }
    sums = {name: 0.0 for name in (
        "total", "primary", "alternate", "consistency", "anomaly",
        "anchor_kl", "margin_retention", "cycle", "preserve", "unmatched"
    )}
    state_path = args.output_dir / "training_state.pt"
    start = 0
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != "judo-native-deep-residual-resume-v1":
            raise ValueError("native deep resume mismatch")
        adapter.load_state_dict(state["adapter"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["next_batch"])
        best = state["best"]
        restored_sums = state.get("loss_sums")
        if restored_sums is not None:
            if set(restored_sums) != set(sums):
                raise ValueError("resume loss accumulator mismatch")
            sums = {name: float(value) for name, value in restored_sums.items()}
        print(json.dumps({"event": "resume-native-deep-residual", "next_batch": start}, sort_keys=True), flush=True)

    train_batches = batches(train_rows, args.batch_size, args.seed)
    model.eval(); adapter.train()
    for batch_index in range(start, len(train_batches)):
        batch = train_batches[batch_index]
        alternate = [alternates[str(row["sample_id"])] for row in batch]
        combined = [*batch, *alternate]
        inputs = prepare_batch(processor, combined, args.data_root, system_prompt, prefix_text, device)
        values = answer_logits(model, inputs, candidates)
        size = len(batch); primary, alternate_values = values[:size], values[size:]
        target = labels(batch, device)
        base = torch.tensor(
            [baseline_by_id[str(row["sample_id"])] for row in batch],
            device=device,
            dtype=primary.dtype,
        )
        correct = base.argmax(dim=-1).eq(target)
        weights = [args.correct_weight if hit else args.wrong_weight for hit in correct.tolist()]
        primary_loss = (F.cross_entropy(primary, target, reduction="none") * torch.tensor(weights, device=device)).mean()
        alternate_loss = F.cross_entropy(alternate_values, target)
        p_log, a_log = F.log_softmax(primary, -1), F.log_softmax(alternate_values, -1)
        consistency = 0.5 * (F.kl_div(p_log, a_log.exp().detach(), reduction="batchmean") + F.kl_div(a_log, p_log.exp().detach(), reduction="batchmean"))
        if adapter.last_anomaly_logits is None:
            raise RuntimeError("native deep anomaly evidence missing")
        anomaly_target = torch.tensor([1 if row["label"] == "anomaly" else 0 for row in combined], device=device)
        anomaly_loss = F.cross_entropy(adapter.last_anomaly_logits, anomaly_target)
        unmatched_loss = values.new_zeros(())
        if args.partial_transport:
            unmatched = getattr(adapter, "last_unmatched_mass", None)
            if unmatched is None:
                raise RuntimeError("partial transport unmatched evidence missing")
            top_count = max(1, unmatched.shape[1] // 20)
            top_mass = unmatched.topk(top_count, dim=1).values.mean(dim=1)
            mean_mass = unmatched.mean(dim=1)
            normal = anomaly_target.eq(0)
            anomalous = anomaly_target.eq(1)
            normal_term = mean_mass[normal].square().mean() if normal.any() else values.new_zeros(())
            anomaly_term = (
                F.relu(0.10 - top_mass[anomalous]).square().mean()
                if anomalous.any() else values.new_zeros(())
            )
            unmatched_loss = normal_term + anomaly_term
        per_row_kl = F.kl_div(
            F.log_softmax(primary, dim=-1), F.softmax(base, dim=-1), reduction="none"
        ).sum(dim=-1)
        anchor_kl = per_row_kl[correct].mean() if correct.any() else values.new_zeros(())
        target_mask = F.one_hot(target, num_classes=4).bool()
        base_correct = base.gather(1, target[:, None]).squeeze(1)
        adapted_correct = primary.gather(1, target[:, None]).squeeze(1)
        base_other = base.masked_fill(target_mask, -torch.inf).amax(dim=-1)
        adapted_other = primary.masked_fill(target_mask, -torch.inf).amax(dim=-1)
        base_margin = (base_correct - base_other).detach()
        adapted_margin = adapted_correct - adapted_other
        margin_rows = F.relu(base_margin - adapted_margin)
        margin_retention = margin_rows[correct].mean() if correct.any() else values.new_zeros(())
        if adapter.last_transport_cycle_loss is None:
            raise RuntimeError("transport cycle evidence missing")
        cycle = adapter.last_transport_cycle_loss
        if not adapter.last_residual_ratios:
            raise RuntimeError("native deep residual was not applied")
        ratios = torch.stack([value.amax(dim=1) for value in adapter.last_residual_ratios]).mean(dim=0)
        preserve = ratios[:size][correct].square().mean() if correct.any() else values.new_zeros(())
        loss = (
            primary_loss
            + args.alternate_weight * alternate_loss
            + args.consistency_weight * consistency
            + args.anomaly_weight * anomaly_loss
            + args.anchor_kl_weight * anchor_kl
            + args.margin_retention_weight * margin_retention
            + args.cycle_weight * cycle
            + args.preserve_weight * preserve
            + args.unmatched_weight * unmatched_loss
        )
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        if not torch.isfinite(norm): raise FloatingPointError("non-finite native deep gradient")
        optimizer.step()
        for name, value in (
            ("total", loss), ("primary", primary_loss), ("alternate", alternate_loss),
            ("consistency", consistency), ("anomaly", anomaly_loss),
            ("anchor_kl", anchor_kl), ("margin_retention", margin_retention),
            ("cycle", cycle), ("preserve", preserve),
            ("unmatched", unmatched_loss),
        ):
            sums[name] += float(value.detach().cpu())
        if (batch_index + 1) % args.state_save_steps == 0 or batch_index + 1 == len(train_batches):
            save_state(
                state_path, adapter, optimizer, next_batch=batch_index + 1,
                best=best, loss_sums=sums,
            )
            update_progress(args.progress_json, "train-native-deep-residual", batch_index + 1, len(train_batches), "epoch-1")

    metrics, candidate_logits = evaluate(
        model, adapter, processor, validation_rows, args.data_root, system_prompt, prefix_text, candidates, args.eval_batch_size
    )
    metrics["paired_vs_baseline"] = paired_delta(validation_rows, baseline_validation_logits, candidate_logits)
    safe = all(metrics["per_task_accuracy"][task] >= baseline_metrics["per_task_accuracy"][task] - 0.0125 for task in baseline_metrics["per_task_accuracy"]) and metrics["ad_balanced_accuracy"] >= baseline_metrics["ad_balanced_accuracy"] - 0.0125
    if safe and (metrics["accuracy"], metrics["source_task_macro_accuracy"]) > (best["metrics"]["accuracy"], best["metrics"]["source_task_macro_accuracy"]):
        best = {"epoch": 1, "metrics": metrics, "state": {name: value.detach().cpu() for name, value in adapter.state_dict().items()}}
    trained_identity = save_adapter(adapter, args.output_dir / "epoch-1", {"selected_epoch": 1, "decision_protocol": protocol, "validation_metrics": metrics, "safe_noninferiority": safe})
    adapter.load_state_dict(best["state"], strict=True)
    best_identity = save_adapter(adapter, args.output_dir / "best_adapter", {"selected_epoch": best["epoch"], "decision_protocol": protocol, "validation_metrics": best["metrics"], "selection_reason": "locked validation accuracy with per-task and AD noninferiority"})
    save_state(
        state_path, adapter, optimizer, next_batch=len(train_batches),
        best=best, loss_sums=sums,
    )
    summary = {
        "schema_version": "judo-native-deep-residual-training-summary-v1",
        "status": "complete",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "contract": contract,
        "baseline_validation": baseline_metrics,
        "epoch1": {"mean_losses": {name: value / max(1, len(train_batches)) for name, value in sums.items()}, "metrics": metrics, "safe_noninferiority": safe, "identity": trained_identity},
        "best": {"epoch": best["epoch"], "metrics": best["metrics"], "identity": best_identity},
        "hyperparameters": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    atomic_json(args.output_dir / "native_deep_training_summary.json", summary)
    print(json.dumps({"status": "complete", "baseline_accuracy": baseline_metrics["accuracy"], "epoch1_accuracy": metrics["accuracy"], "best_epoch": best["epoch"], "adapter_sha256": best_identity["weights_sha256"]}, sort_keys=True))
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
