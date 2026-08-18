#!/usr/bin/env python3
"""Train only a zero-initialized residual adapter between frozen JUDO and CE."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable

from PIL import Image
import torch
import torch.nn.functional as F

import eval_manifest as eval_entry
from judo_aligned_ce import (
    configure_processor_for_residual,
    install_aligned_image_path,
    load_adapter,
    save_adapter,
    trainable_contract,
)


SCHEMA_VERSION = "judo-aligned-ce-training-v1"
SOURCES = ("GoodsAD", "MVTec-AD", "MVTec-LOCO", "VisA")
LABELS = ("anomaly", "normal")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def update_progress(path: Path | None, stage: str, completed: int, total: int, detail: str) -> None:
    if path is None:
        return
    atomic_json(
        path,
        {
            "schema_version": "judo-aligned-ce-progress-v1",
            "stage": stage,
            "detail": detail,
            "completed": completed,
            "total": total,
            "percent": 100.0 * completed / total if total else 0.0,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)


def teacher_text(processor: Any, row: dict[str, Any], system_prompt: str) -> str:
    system = {"role": "system", "content": system_prompt}
    user = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "\nFirst image = QUERY; Second image = NORMAL template.\n"
                    f"Question: {row['question']}\n{row['options_text']}"
                ),
            },
        ],
    }
    prompt = processor.apply_chat_template([system, user], tokenize=False, add_generation_prompt=True)
    segmentation = str(row.get("teacher_segmentation", "None")) or "None"
    thinking = str(row.get("teacher_thinking", ""))
    return prompt + f"<seg>{segmentation}</seg>\n<think>{thinking}</think>\n<answer>"


def prepare_batch(
    processor: Any,
    rows: list[dict[str, Any]],
    data_root: Path,
    system_prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    texts = [teacher_text(processor, row, system_prompt) for row in rows]
    images: list[Image.Image] = []
    for row in rows:
        images.append(load_image(data_root / str(row["image"])))
        images.append(load_image(data_root / str(row["template_image"])))
    encoded = processor(text=texts, images=images, padding=True, return_tensors="pt")
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in encoded.items()}


def candidate_ids(tokenizer: Any) -> torch.Tensor:
    values = []
    for answer in ("A", "B"):
        ids = tokenizer(answer, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"answer {answer!r} is not a single tokenizer token: {ids}")
        values.append(ids[0])
    if len(set(values)) != 2:
        raise ValueError("A and B resolve to the same token")
    return torch.tensor(values, dtype=torch.long)


def answer_logits(model: Any, inputs: dict[str, torch.Tensor], candidates: torch.Tensor) -> torch.Tensor:
    outputs = model(
        **inputs,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    logits = outputs.logits[:, -1, :]
    return logits.index_select(-1, candidates.to(logits.device)).float()


def labels_for(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.tensor([0 if row["correct_answer"] == "A" else 1 for row in rows], device=device)


def prediction_metrics(rows: list[dict[str, Any]], logits_by_id: dict[str, list[float]]) -> dict[str, Any]:
    cell = Counter()
    correct = Counter()
    for row in rows:
        key = (str(row["source"]), str(row["label"]))
        target = 0 if row["correct_answer"] == "A" else 1
        pred = max(range(2), key=lambda index: logits_by_id[str(row["sample_id"])][index])
        cell[key] += 1
        correct[key] += int(pred == target)
    per_cell = {
        f"{source}|{label}": correct[(source, label)] / cell[(source, label)]
        for source in SOURCES
        for label in LABELS
    }
    anomaly = sum(correct[(source, "anomaly")] for source in SOURCES) / sum(
        cell[(source, "anomaly")] for source in SOURCES
    )
    normal = sum(correct[(source, "normal")] for source in SOURCES) / sum(
        cell[(source, "normal")] for source in SOURCES
    )
    return {
        "balanced_accuracy": (anomaly + normal) / 2,
        "anomaly_recall": anomaly,
        "normal_specificity": normal,
        "per_cell_accuracy": per_cell,
    }


def append_logits(path: Path, rows: list[dict[str, Any]], logits: torch.Tensor) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row, values in zip(rows, logits.detach().cpu().tolist()):
            handle.write(
                json.dumps(
                    {"sample_id": row["sample_id"], "a_logit": values[0], "b_logit": values[1]},
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
    result = {}
    valid_lines: list[str] = []
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            # A VM loss can interrupt only the final append.  Remove that
            # incomplete tail before resuming so the next JSON row cannot be
            # concatenated onto corrupt bytes.
            path.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")
            break
        valid_lines.append(line)
        result[str(row["sample_id"])] = [float(row["a_logit"]), float(row["b_logit"])]
    return result


def deterministic_batches(rows: list[dict[str, Any]], batch_size: int, seed: int) -> list[list[dict[str, Any]]]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [[rows[index] for index in order[start : start + batch_size]] for start in range(0, len(order), batch_size)]


def evaluate_teacher_forced(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    data_root: Path,
    system_prompt: str,
    candidates: torch.Tensor,
    batch_size: int,
    progress_path: Path | None,
    progress_offset: int,
    progress_total: int,
    detail: str,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    output: dict[str, list[float]] = {}
    model.eval()
    model.model.alignment_adapter.eval()
    batches = deterministic_batches(rows, batch_size, 0)
    with torch.no_grad():
        for index, batch in enumerate(batches):
            inputs = prepare_batch(processor, batch, data_root, system_prompt, next(model.parameters()).device)
            logits = answer_logits(model, inputs, candidates)
            for row, values in zip(batch, logits.cpu().tolist()):
                output[str(row["sample_id"])] = values
            if (index + 1) % 25 == 0 or index + 1 == len(batches):
                update_progress(
                    progress_path,
                    "train-aligned-adapter",
                    progress_offset + min((index + 1) * batch_size, len(rows)),
                    progress_total,
                    detail,
                )
    return prediction_metrics(rows, output), output


def save_training_state(
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--gate-weight", type=float, default=1e-3)
    parser.add_argument("--bottleneck-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("one BF16 CUDA GPU is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_manifest)
    validation_rows = read_jsonl(args.validation_manifest)
    if len(train_rows) != 1600 or len(validation_rows) != 400:
        raise ValueError("expected train=1600 and validation=400")
    train_ids = {str(row["sample_id"]) for row in train_rows}
    validation_ids = {str(row["sample_id"]) for row in validation_rows}
    if train_ids & validation_ids:
        raise ValueError("train and validation IDs overlap")
    train_refs = {str(row["template_image"]) for row in train_rows}
    validation_refs = {str(row["template_image"]) for row in validation_rows}
    if train_refs & validation_refs:
        raise ValueError("train and validation normal references overlap")
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
    adapter = install_aligned_image_path(
        model,
        bottleneck_size=args.bottleneck_size,
        num_heads=args.num_heads,
    )
    contract = trainable_contract(model, adapter, trainable=True)
    candidates = candidate_ids(tokenizer)
    system_prompt = eval_entry.official_eval.make_system_prompt(16, 16)

    # Functional preservation test: at scale zero, the custom residual path
    # must reproduce public JUDO's A/B logits exactly on eight held-out rows.
    parity_rows = validation_rows[:8]
    aligned_inputs = prepare_batch(processor, parity_rows, args.data_root, system_prompt, device)
    with torch.no_grad():
        aligned_logits = answer_logits(model, aligned_inputs, candidates).cpu()
    baseline = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
        baseline_logits = answer_logits(baseline, baseline_inputs, candidates).cpu()
    max_abs = float((aligned_logits - baseline_logits).abs().max())
    exact = torch.equal(aligned_logits, baseline_logits)
    parity = {
        "samples": len(parity_rows),
        "exact_ab_logit_match": exact,
        "max_abs_ab_logit_difference": max_abs,
        "aligned_logits": aligned_logits.tolist(),
        "baseline_logits": baseline_logits.tolist(),
    }
    atomic_json(args.output_dir / "zero_init_parity.json", parity)
    if not exact:
        raise RuntimeError(f"zero-init residual is not functionally identical to JUDO: max_abs={max_abs}")
    del baseline, baseline_inputs, aligned_inputs
    gc.collect()
    torch.cuda.empty_cache()

    cache_path = args.output_dir / "baseline_teacher_logits.jsonl"
    cached = read_logits(cache_path)
    all_rows = train_rows + validation_rows
    remaining = [row for row in all_rows if str(row["sample_id"]) not in cached]
    cache_batches = deterministic_batches(remaining, args.batch_size, 1)
    model.eval()
    adapter.eval()
    pending_rows: list[dict[str, Any]] = []
    pending_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for index, batch in enumerate(cache_batches):
            inputs = prepare_batch(processor, batch, args.data_root, system_prompt, device)
            logits = answer_logits(model, inputs, candidates)
            pending_rows.extend(batch)
            pending_logits.append(logits.detach().cpu())
            if (index + 1) % 25 == 0 or index + 1 == len(cache_batches):
                append_logits(cache_path, pending_rows, torch.cat(pending_logits, dim=0))
                pending_rows.clear()
                pending_logits.clear()
                update_progress(
                    args.progress_json,
                    "train-aligned-adapter",
                    len(cached) + min((index + 1) * args.batch_size, len(remaining)),
                    len(all_rows) + args.epochs * len(train_rows) + args.epochs * len(validation_rows),
                    "cache-zero-init-teacher-logits",
                )
    cached = read_logits(cache_path)
    if set(cached) != {str(row["sample_id"]) for row in all_rows}:
        raise RuntimeError("teacher-logit cache is incomplete")
    baseline_validation = prediction_metrics(validation_rows, cached)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=0.01)
    state_path = args.output_dir / "training_state.pt"
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {
        "epoch": 0,
        "metrics": baseline_validation,
        "feasible": True,
        "checkpoint": "best_adapter",
    }
    best_dir = args.output_dir / "best_adapter"
    if not (best_dir / "alignment_adapter.safetensors").is_file():
        save_adapter(
            adapter,
            best_dir,
            {
                "selected_epoch": 0,
                "selection_reason": "zero-init JUDO baseline candidate",
                "validation_metrics": baseline_validation,
            },
        )
    start_epoch, start_batch = 1, 0
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("training-state schema mismatch")
        adapter.load_state_dict(state["adapter"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        history = list(state["history"])
        best = dict(state["best"])
        start_epoch = int(state["epoch"])
        start_batch = int(state["next_batch"])

    total_work = len(all_rows) + args.epochs * len(train_rows) + args.epochs * len(validation_rows)
    for epoch in range(start_epoch, args.epochs + 1):
        batches = deterministic_batches(train_rows, args.batch_size, args.seed + epoch)
        model.eval()
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for batch_index in range(start_batch if epoch == start_epoch else 0, len(batches)):
            batch = batches[batch_index]
            inputs = prepare_batch(processor, batch, args.data_root, system_prompt, device)
            logits = answer_logits(model, inputs, candidates)
            labels = labels_for(batch, device)
            teacher = torch.tensor([cached[str(row["sample_id"])] for row in batch], device=device)
            ce_loss = F.cross_entropy(logits, labels)
            kl_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(teacher, dim=-1),
                reduction="batchmean",
            )
            gate_loss = torch.tanh(adapter.global_scale).abs()
            loss = ce_loss + args.kl_weight * kl_loss + args.gate_weight * gate_loss
            (loss / args.gradient_accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(batches):
                grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite adapter gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (batch_index + 1) % 100 == 0:
                save_training_state(state_path, adapter, optimizer, epoch, batch_index + 1, history, best)
            if (batch_index + 1) % 25 == 0 or batch_index + 1 == len(batches):
                completed = len(all_rows) + (epoch - 1) * (len(train_rows) + len(validation_rows)) + min(
                    (batch_index + 1) * args.batch_size, len(train_rows)
                )
                update_progress(args.progress_json, "train-aligned-adapter", completed, total_work, f"epoch-{epoch}-train")

        validation_offset = len(all_rows) + (epoch - 1) * (len(train_rows) + len(validation_rows)) + len(train_rows)
        metrics, validation_logits = evaluate_teacher_forced(
            model,
            processor,
            validation_rows,
            args.data_root,
            system_prompt,
            candidates,
            args.batch_size,
            args.progress_json,
            validation_offset,
            total_work,
            f"epoch-{epoch}-validation",
        )
        feasible = metrics["anomaly_recall"] >= baseline_validation["anomaly_recall"] - 0.02
        record = {
            "epoch": epoch,
            "mean_training_loss": sum(losses) / len(losses),
            "validation_metrics": metrics,
            "feasible_anomaly_recall_constraint": feasible,
            "gate_statistics": adapter.gate_statistics(),
        }
        history.append(record)
        atomic_json(args.output_dir / "training_history.json", {"baseline_validation": baseline_validation, "epochs": history})
        better = feasible and (
            not best.get("feasible", False)
            or metrics["balanced_accuracy"] > best["metrics"]["balanced_accuracy"]
            or (
                math.isclose(metrics["balanced_accuracy"], best["metrics"]["balanced_accuracy"])
                and metrics["normal_specificity"] > best["metrics"]["normal_specificity"]
            )
        )
        if better:
            best = {"epoch": epoch, "metrics": metrics, "feasible": feasible, "checkpoint": "best_adapter"}
            save_adapter(
                adapter,
                best_dir,
                {
                    "selected_epoch": epoch,
                    "selection_reason": "maximum validation balanced accuracy subject to anomaly recall >= baseline - 0.02",
                    "validation_metrics": metrics,
                    "baseline_validation_metrics": baseline_validation,
                    "train_manifest_sha256": sha256_file(args.train_manifest),
                    "validation_manifest_sha256": sha256_file(args.validation_manifest),
                },
            )
        save_training_state(state_path, adapter, optimizer, epoch + 1, 0, history, best)
        start_batch = 0

    selected_identity_path = best_dir / "adapter_identity.json"
    selected_identity = json.loads(selected_identity_path.read_text(encoding="utf-8"))
    selected_identity["training_pipeline_completed"] = True
    selected_identity["epochs_evaluated"] = len(history)
    selected_identity["selection_fell_back_to_zero_init"] = int(selected_identity["selected_epoch"]) == 0
    atomic_json(selected_identity_path, selected_identity)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "seed": args.seed,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_unique_references": len(train_refs),
        "validation_unique_references": len(validation_refs),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "kl_weight": args.kl_weight,
            "gate_weight": args.gate_weight,
            "bottleneck_size": args.bottleneck_size,
            "num_heads": args.num_heads,
        },
        "parameter_contract": contract,
        "zero_init_parity": parity,
        "baseline_validation": baseline_validation,
        "history": history,
        "selected": best,
        "selected_adapter_identity": selected_identity,
    }
    atomic_json(args.output_dir / "training_summary.json", summary)
    update_progress(args.progress_json, "train-aligned-adapter", total_work, total_work, "complete")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
