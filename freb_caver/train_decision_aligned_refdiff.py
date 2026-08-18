#!/usr/bin/env python3
"""Train RefDiff at the exact autoregressive token boundary used by JUDO."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

import eval_manifest as eval_entry
from judo_aligned_ce import configure_processor_for_residual
from judo_semantic_refdiff import install_semantic_refdiff, load_adapter, save_adapter, trainable_contract
from train_judo_aligned_ce import atomic_json, read_jsonl, update_progress
from train_judo_semantic_refdiff import prediction_metrics, paired_delta


LETTERS = ("A", "B", "C", "D")
STATE_SCHEMA = "judo-decision-aligned-refdiff-state-v1"


def answer_protocol(tokenizer: Any) -> tuple[list[int], str, torch.Tensor, dict[str, Any]]:
    variants = [tokenizer.encode(f"<answer>{letter}", add_special_tokens=False) for letter in LETTERS]
    common = 0
    for values in zip(*variants):
        if len(set(values)) != 1:
            break
        common += 1
    prefix = variants[0][:common]
    if not prefix or any(len(value) <= common for value in variants):
        raise RuntimeError(f"invalid autoregressive answer protocol: {variants}")
    candidates = [int(value[common]) for value in variants]
    if len(set(candidates)) != len(LETTERS):
        raise RuntimeError(f"answer branches are not unique at the decision boundary: {variants}")
    prefix_text = tokenizer.decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if tokenizer.encode(prefix_text, add_special_tokens=False) != prefix:
        raise RuntimeError("answer decision prefix does not round-trip through the tokenizer")
    protocol = {
        "variants": variants,
        "common_prefix_ids": prefix,
        "common_prefix_text": prefix_text,
        "first_branch_token_ids": candidates,
        "first_branch_token_text": [tokenizer.decode([value]) for value in candidates],
    }
    return [int(value) for value in prefix], prefix_text, torch.tensor(candidates), protocol


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)


def decision_text(processor: Any, row: dict[str, Any], system_prompt: str, prefix_text: str) -> str:
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
    segmentation = str(row.get("teacher_segmentation", "None")).strip() or "None"
    thinking = str(row.get("teacher_thinking", "")).strip()
    return prompt + f"<seg>{segmentation}</seg>\n<think>{thinking}</think>\n{prefix_text}"


def prepare_batch(
    processor: Any,
    rows: list[dict[str, Any]],
    data_root: Path,
    system_prompt: str,
    prefix_text: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    texts = [decision_text(processor, row, system_prompt, prefix_text) for row in rows]
    images: list[Image.Image] = []
    for row in rows:
        images.extend((load_image(data_root / str(row["image"])), load_image(data_root / str(row["template_image"]))))
    encoded = processor(text=texts, images=images, padding=True, return_tensors="pt")
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in encoded.items()}


def batches(rows: list[dict[str, Any]], size: int, seed: int) -> list[list[dict[str, Any]]]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [[rows[index] for index in order[start : start + size]] for start in range(0, len(order), size)]


def labels(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.tensor([LETTERS.index(str(row["correct_answer"])) for row in rows], device=device)


def answer_logits(model: Any, inputs: dict[str, torch.Tensor], candidates: torch.Tensor) -> torch.Tensor:
    output = model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)
    return output.logits[:, -1].index_select(-1, candidates.to(output.logits.device)).float()


def evaluate(
    model: Any,
    adapter: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    data_root: Path,
    system_prompt: str,
    prefix_text: str,
    candidates: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    result: dict[str, list[float]] = {}
    model.eval(); adapter.eval()
    with torch.no_grad():
        for batch in batches(rows, batch_size, 0):
            inputs = prepare_batch(processor, batch, data_root, system_prompt, prefix_text, next(model.parameters()).device)
            values = answer_logits(model, inputs, candidates)
            for row, logits in zip(batch, values.cpu().tolist()):
                result[str(row["sample_id"])] = logits
    return prediction_metrics(rows, result), result


def save_state(path: Path, adapter: Any, optimizer: Any, next_batch: int, history: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": STATE_SCHEMA,
            "adapter": {name: value.detach().cpu() for name, value in adapter.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "next_batch": next_batch,
            "history": history,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-model", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--scale-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-relative-rms", type=float, default=0.04)
    parser.add_argument("--wrong-weight", type=float, default=8.0)
    parser.add_argument("--wrong-normal-ad-weight", type=float, default=12.0)
    parser.add_argument("--correct-weight", type=float, default=0.75)
    parser.add_argument("--anchor-correct-weight", type=float, default=1.5)
    parser.add_argument("--anchor-wrong-weight", type=float, default=0.05)
    parser.add_argument("--repair-margin-weight", type=float, default=0.5)
    parser.add_argument("--repair-margin-gain", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if min(args.batch_size, args.gradient_accumulation, args.eval_batch_size) < 1:
        raise ValueError("invalid batch configuration")
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows = read_jsonl(args.train_manifest), read_jsonl(args.validation_manifest)
    if len(train_rows) != 1120 or len(validation_rows) != 560:
        raise ValueError("locked split size mismatch")

    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    device = torch.device("cuda:0")
    model = AutoModelForImageTextToText.from_pretrained(
        args.hybrid_model, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        trust_remote_code=True, local_files_only=True, attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(args.hybrid_model, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hybrid_model, trust_remote_code=True, local_files_only=True)
    configure_processor_for_residual(processor)
    prefix_ids, prefix_text, candidate_tensor, protocol = answer_protocol(tokenizer)
    identity = json.loads((args.initial_adapter / "semantic_refdiff_adapter_identity.json").read_text())
    adapter = install_semantic_refdiff(
        model,
        bottleneck_size=int(identity["bottleneck_size"]), num_heads=int(identity["num_heads"]),
        injection_layers=tuple(int(value) for value in identity["injection_layers"]),
        max_relative_rms=args.max_relative_rms, direction_floor=float(identity["direction_floor"]),
        decision_prefix_ids=prefix_ids,
    )
    loaded = load_adapter(adapter, args.initial_adapter)
    contract = trainable_contract(model, adapter, trainable=True)
    system_prompt = eval_entry.official_eval.make_system_prompt(16, 16)

    all_rows = train_rows + validation_rows
    original_scale = adapter.global_scale.detach().clone()
    adapter.global_scale.data.zero_()
    baseline_metrics, baseline_logits = evaluate(
        model, adapter, processor, all_rows, args.data_root, system_prompt, prefix_text,
        candidate_tensor, args.eval_batch_size
    )
    adapter.global_scale.data.copy_(original_scale)
    baseline_validation = prediction_metrics(
        validation_rows, {str(row["sample_id"]): baseline_logits[str(row["sample_id"])] for row in validation_rows}
    )

    direction = [parameter for name, parameter in adapter.named_parameters() if name != "global_scale"]
    optimizer = torch.optim.AdamW(
        [
            {"params": direction, "lr": args.learning_rate, "weight_decay": 0.01},
            {"params": [adapter.global_scale], "lr": args.scale_learning_rate, "weight_decay": 0.0},
        ]
    )
    state_path = args.output_dir / "decision_training_state.pt"
    history: list[dict[str, Any]] = []
    start = 0
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != STATE_SCHEMA:
            raise ValueError("decision training state schema mismatch")
        adapter.load_state_dict(state["adapter"]); optimizer.load_state_dict(state["optimizer"])
        start = int(state["next_batch"]); history = list(state["history"])

    train_batches = batches(train_rows, args.batch_size, args.seed)
    optimizer.zero_grad(set_to_none=True)
    sums: Counter[str] = Counter()
    for batch_index in range(start, len(train_batches)):
        batch = train_batches[batch_index]
        inputs = prepare_batch(processor, batch, args.data_root, system_prompt, prefix_text, device)
        logits = answer_logits(model, inputs, candidate_tensor)
        target = labels(batch, device)
        teacher = torch.tensor([baseline_logits[str(row["sample_id"])] for row in batch], device=device)
        baseline_correct = teacher.argmax(-1).eq(target)
        ce_values = F.cross_entropy(logits, target, reduction="none")
        ce_weights = []
        for row, correct in zip(batch, baseline_correct.tolist()):
            if correct:
                ce_weights.append(args.correct_weight)
            elif row["question_type"] == "Anomaly Detection" and row["label"] == "normal":
                ce_weights.append(args.wrong_normal_ad_weight)
            else:
                ce_weights.append(args.wrong_weight)
        ce_loss = (ce_values * torch.tensor(ce_weights, device=device)).mean()
        kl_values = F.kl_div(F.log_softmax(logits, -1), F.softmax(teacher, -1), reduction="none").sum(-1)
        anchor_weights = torch.where(
            baseline_correct, torch.full_like(kl_values, args.anchor_correct_weight),
            torch.full_like(kl_values, args.anchor_wrong_weight)
        )
        anchor_loss = (kl_values * anchor_weights).mean()
        one_hot = F.one_hot(target, num_classes=4).bool()
        margin = logits.gather(1, target[:, None]).squeeze(1) - logits.masked_fill(one_hot, -torch.inf).max(-1).values
        teacher_margin = teacher.gather(1, target[:, None]).squeeze(1) - teacher.masked_fill(one_hot, -torch.inf).max(-1).values
        wrong = ~baseline_correct
        repair_loss = F.softplus(args.repair_margin_gain - (margin[wrong] - teacher_margin[wrong])).mean() if wrong.any() else logits.new_zeros(())
        loss = ce_loss + anchor_loss + args.repair_margin_weight * repair_loss
        (loss / args.gradient_accumulation).backward()
        sums.update(total=float(loss.detach()), ce=float(ce_loss.detach()), anchor=float(anchor_loss.detach()), repair=float(repair_loss.detach()))
        if (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(train_batches):
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite decision-aligned gradient")
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if (batch_index + 1) % 50 == 0:
            save_state(state_path, adapter, optimizer, batch_index + 1, history)
        if (batch_index + 1) % 25 == 0 or batch_index + 1 == len(train_batches):
            update_progress(args.progress_json, "decision-aligned-training", batch_index + 1, len(train_batches), None)

    validation_metrics, validation_logits = evaluate(
        model, adapter, processor, validation_rows, args.data_root, system_prompt,
        prefix_text, candidate_tensor, args.eval_batch_size
    )
    validation_metrics["paired_vs_baseline"] = paired_delta(validation_rows, baseline_logits, validation_logits)
    record = {
        "mean_losses": {name: value / max(1, len(train_batches) - start) for name, value in sums.items()},
        "baseline_validation": baseline_validation,
        "validation_metrics": validation_metrics,
    }
    history.append(record)
    metadata = {
        "selected_epoch": 1,
        "selection_reason": "generation-tokenization-aligned answer-decision training epoch 1",
        "initial_adapter_weights_sha256": loaded["weights_sha256"],
        "decision_protocol": protocol,
        "training_metrics": record,
    }
    saved = save_adapter(adapter, args.output_dir / "epoch-1", metadata)
    save_state(state_path, adapter, optimizer, len(train_batches), history)
    summary = {
        "schema_version": "judo-decision-aligned-refdiff-summary-v1", "status": "complete",
        "train_rows": len(train_rows), "validation_rows": len(validation_rows), "contract": contract,
        "decision_protocol": protocol, "initial_adapter_weights_sha256": loaded["weights_sha256"],
        "baseline_all_rows": baseline_metrics, "result": record, "adapter_identity": saved,
        "hyperparameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    atomic_json(args.output_dir / "decision_training_summary.json", summary)
    print(json.dumps({"status": "complete", "validation_accuracy": validation_metrics["accuracy"], "adapter_sha256": saved["weights_sha256"]}, sort_keys=True))
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
