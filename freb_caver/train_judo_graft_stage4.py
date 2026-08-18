#!/usr/bin/env python3
"""Stage-4 GRAFT training on a frozen public JUDO checkpoint.

The objective explicitly couples two failure modes observed in JUDO:

1. the correct visual margin must not collapse after the teacher-forced CoT;
2. anomaly and hard-normal queries sharing one normal anchor must be ordered
   correctly, with additional null/reference-invariance constraints.

This is a supervised structural screen.  Grounded policy optimization is only
run after this lower-cost screen establishes that evidence replay improves both
overall accuracy and balanced anomaly detection.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

import eval_manifest as eval_entry
from judo_graft import install_graft_adapter
from judo_native_deep_residual import save_adapter, trainable_contract
from train_decision_aligned_refdiff import (
    answer_protocol,
    decision_text,
    labels,
    load_image,
)
from train_judo_aligned_ce import atomic_json, read_jsonl, update_progress
from train_judo_semantic_refdiff import paired_delta, prediction_metrics
from train_transport_equivariant_adapter import (
    alternate_references,
    answer_logits,
    batches,
    cache_baseline_logits,
)


STATE_SCHEMA = "judo-graft-stage4-resume-v1"


def normalize_option(value: Any) -> str:
    return str(value).strip().rstrip(".").strip().casefold()


def yes_no_indices(row: dict[str, Any]) -> tuple[int, int]:
    options = row.get("options") or {}
    yes = [index for index, letter in enumerate("ABCD") if normalize_option(options.get(letter)) == "yes"]
    no = [index for index, letter in enumerate("ABCD") if normalize_option(options.get(letter)) == "no"]
    if len(yes) != 1 or len(no) != 1:
        raise ValueError(f"AD row lacks unique Yes/No options: {row.get('sample_id')}")
    return yes[0], no[0]


def semantic_anomaly_margin(logits: torch.Tensor, rows: list[dict[str, Any]]) -> torch.Tensor:
    values = []
    for index, row in enumerate(rows):
        yes, no = yes_no_indices(row)
        values.append(logits[index, yes] - logits[index, no])
    return torch.stack(values)


def direct_text(processor: Any, row: dict[str, Any], system_prompt: str, prefix_text: str) -> str:
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
    prompt = processor.apply_chat_template(
        [system, user], tokenize=False, add_generation_prompt=True
    )
    return prompt + prefix_text


def prepare_text_batch(
    processor: Any,
    rows: list[dict[str, Any]],
    data_root: Path,
    texts: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if len(rows) != len(texts):
        raise ValueError("row/text batch mismatch")
    images: list[Image.Image] = []
    for row in rows:
        images.extend(
            (
                load_image(data_root / str(row["image"])),
                load_image(data_root / str(row["template_image"])),
            )
        )
    encoded = processor(text=texts, images=images, padding=True, return_tensors="pt")
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in encoded.items()
    }


def prepare_cot_batch(
    processor: Any,
    rows: list[dict[str, Any]],
    data_root: Path,
    system_prompt: str,
    prefix_text: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return prepare_text_batch(
        processor,
        rows,
        data_root,
        [decision_text(processor, row, system_prompt, prefix_text) for row in rows],
        device,
    )


def prepare_direct_batch(
    processor: Any,
    rows: list[dict[str, Any]],
    data_root: Path,
    system_prompt: str,
    prefix_text: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return prepare_text_batch(
        processor,
        rows,
        data_root,
        [direct_text(processor, row, system_prompt, prefix_text) for row in rows],
        device,
    )


def answer_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = F.one_hot(target, num_classes=logits.shape[-1]).bool()
    correct = logits.gather(1, target[:, None]).squeeze(1)
    other = logits.masked_fill(mask, -torch.inf).amax(dim=-1)
    return correct - other


def build_same_anchor_pairs(rows: list[dict[str, Any]], seed: int) -> list[list[dict[str, Any]]]:
    pools: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("question_type") != "Anomaly Detection":
            continue
        key = (str(row["source"]), str(row["category"]), str(row["label"]))
        pools.setdefault(key, []).append(row)
    result: list[list[dict[str, Any]]] = []
    keys = sorted({(source, category) for source, category, _label in pools})
    for source, category in keys:
        anomalies = sorted(
            pools.get((source, category, "anomaly"), []),
            key=lambda row: str(row["sample_id"]),
        )
        normals = sorted(
            pools.get((source, category, "normal"), []),
            key=lambda row: str(row["sample_id"]),
        )
        if not anomalies or not normals:
            continue
        count = max(len(anomalies), len(normals))
        for index in range(count):
            anomaly = dict(anomalies[index % len(anomalies)])
            normal = dict(normals[index % len(normals)])
            # Holding the normal anchor fixed makes this a query-state
            # comparison rather than a reference-style shortcut.
            anchor = str(normal["template_image"])
            anomaly["template_image"] = anchor
            normal["template_image"] = anchor
            result.append([anomaly, normal])
    random.Random(seed).shuffle(result)
    if not result:
        raise ValueError("no same-anchor AD pairs could be built")
    return result


def sample_weights(
    rows: list[dict[str, Any]],
    base_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    correct_weight: float,
    wrong_weight: float,
    ad_normal_correct_weight: float,
    ad_normal_wrong_weight: float,
    ad_anomaly_correct_weight: float,
    ad_anomaly_wrong_weight: float,
) -> torch.Tensor:
    correct = base_logits.argmax(dim=-1).eq(target).tolist()
    values = []
    for row, hit in zip(rows, correct):
        if row.get("question_type") == "Anomaly Detection" and row.get("label") == "normal":
            values.append(ad_normal_correct_weight if hit else ad_normal_wrong_weight)
        elif row.get("question_type") == "Anomaly Detection":
            values.append(ad_anomaly_correct_weight if hit else ad_anomaly_wrong_weight)
        else:
            values.append(correct_weight if hit else wrong_weight)
    return torch.tensor(values, device=target.device, dtype=base_logits.dtype)


def save_state(
    path: Path,
    adapter: Any,
    optimizer: Any,
    *,
    next_batch: int,
    loss_sums: dict[str, float],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": STATE_SCHEMA,
            "adapter": {
                name: value.detach().cpu() for name, value in adapter.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "next_batch": int(next_batch),
            "loss_sums": loss_sums,
        },
        temporary,
    )
    temporary.replace(path)


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
) -> tuple[dict[str, Any], dict[str, list[float]], dict[str, float]]:
    result: dict[str, list[float]] = {}
    persistence = {"eligible": 0, "preserved": 0, "direct_margin": 0.0, "cot_margin": 0.0}
    model.eval(); adapter.eval()
    with torch.no_grad():
        for batch in batches(rows, batch_size, 0):
            cot = answer_logits(
                model,
                prepare_cot_batch(processor, batch, data_root, system_prompt, prefix_text, next(model.parameters()).device),
                candidates,
            )
            direct = answer_logits(
                model,
                prepare_direct_batch(processor, batch, data_root, system_prompt, prefix_text, next(model.parameters()).device),
                candidates,
            )
            target = labels(batch, cot.device)
            direct_margin, cot_margin = answer_margin(direct, target), answer_margin(cot, target)
            eligible = direct.argmax(dim=-1).eq(target)
            persistence["eligible"] += int(eligible.sum().item())
            persistence["preserved"] += int((cot_margin[eligible] >= direct_margin[eligible]).sum().item())
            persistence["direct_margin"] += float(direct_margin[eligible].sum().cpu())
            persistence["cot_margin"] += float(cot_margin[eligible].sum().cpu())
            for row, value in zip(batch, cot.cpu().tolist()):
                result[str(row["sample_id"])] = value
    denom = max(1, int(persistence["eligible"]))
    diagnostic = {
        "eligible_direct_correct": int(persistence["eligible"]),
        "cot_margin_not_below_direct_rate": float(persistence["preserved"]) / denom,
        "mean_direct_correct_margin": float(persistence["direct_margin"]) / denom,
        "mean_cot_margin_on_direct_correct": float(persistence["cot_margin"]) / denom,
    }
    return prediction_metrics(rows, result), result, diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--baseline-logits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--correct-weight", type=float, default=0.5)
    parser.add_argument("--wrong-weight", type=float, default=1.5)
    parser.add_argument("--ad-normal-correct-weight", type=float, default=1.5)
    parser.add_argument("--ad-normal-wrong-weight", type=float, default=8.0)
    parser.add_argument("--ad-anomaly-correct-weight", type=float, default=0.75)
    parser.add_argument("--ad-anomaly-wrong-weight", type=float, default=4.0)
    parser.add_argument("--alternate-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.15)
    parser.add_argument("--visual-verdict-weight", type=float, default=0.20)
    parser.add_argument("--persistence-weight", type=float, default=1.0)
    parser.add_argument("--pair-rank-weight", type=float, default=0.50)
    parser.add_argument("--pair-evidence-weight", type=float, default=0.10)
    parser.add_argument("--anchor-kl-weight", type=float, default=1.0)
    parser.add_argument("--margin-retention-weight", type=float, default=1.5)
    parser.add_argument("--cycle-weight", type=float, default=0.01)
    parser.add_argument("--orthogonality-weight", type=float, default=0.01)
    parser.add_argument("--preserve-weight", type=float, default=0.50)
    parser.add_argument("--unmatched-weight", type=float, default=0.05)
    parser.add_argument("--pair-frequency", type=int, default=4)
    parser.add_argument("--pair-margin", type=float, default=1.0)
    parser.add_argument("--evidence-margin", type=float, default=0.02)
    parser.add_argument("--bottleneck-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--hyper-rank", type=int, default=256)
    parser.add_argument("--transport-rank", type=int, default=64)
    parser.add_argument("--transport-temperature", type=float, default=0.07)
    parser.add_argument("--unmatched-hidden", type=int, default=64)
    parser.add_argument("--unmatched-prior", type=float, default=0.05)
    parser.add_argument("--injection-layers", default="18,20,22,24,26")
    parser.add_argument("--max-relative-rms", type=float, default=0.08)
    parser.add_argument("--replay-max-relative-rms", type=float, default=0.02)
    parser.add_argument("--fixed-scale-fraction", type=float, default=0.80)
    parser.add_argument("--replay-scale-fraction", type=float, default=0.80)
    parser.add_argument("--state-save-steps", type=int, default=50)
    parser.add_argument("--expected-train-rows", type=int, default=5600)
    parser.add_argument("--expected-validation-rows", type=int, default=560)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    if min(args.batch_size, args.eval_batch_size, args.state_save_steps, args.pair_frequency) < 1:
        raise ValueError("invalid Stage-4 training sizes")
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows = read_jsonl(args.train_manifest), read_jsonl(args.validation_manifest)
    if len(train_rows) != args.expected_train_rows or len(validation_rows) != args.expected_validation_rows:
        raise ValueError("locked GRAFT split mismatch")
    required = ("teacher_segmentation", "teacher_thinking")
    if any(field not in row for row in train_rows + validation_rows for field in required):
        raise ValueError("GRAFT requires frozen public-JUDO CoT prefixes")
    train_assets = {str(row[field]) for row in train_rows for field in ("image", "template_image")}
    validation_assets = {str(row[field]) for row in validation_rows for field in ("image", "template_image")}
    if train_assets & validation_assets:
        raise ValueError("GRAFT train/validation asset leakage")
    missing = [
        args.data_root / str(row[field])
        for row in train_rows + validation_rows
        for field in ("image", "template_image")
        if not (args.data_root / str(row[field])).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{len(missing)} GRAFT assets missing; first={missing[0]}")

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
    if not assistant_ids or not seg_ids:
        raise RuntimeError("GRAFT phase-token preflight failed")
    adapter = install_graft_adapter(
        model,
        decision_prefix_ids=prefix_ids,
        assistant_prefix_ids=assistant_ids,
        seg_start_ids=seg_ids,
        replay_max_relative_rms=args.replay_max_relative_rms,
        replay_scale_fraction=args.replay_scale_fraction,
        unmatched_hidden=args.unmatched_hidden,
        unmatched_prior=args.unmatched_prior,
        transport_rank=args.transport_rank,
        transport_temperature=args.transport_temperature,
        hyper_rank=args.hyper_rank,
        bottleneck_size=args.bottleneck_size,
        num_heads=args.num_heads,
        injection_layers=tuple(int(value) for value in args.injection_layers.split(",")),
        max_relative_rms=args.max_relative_rms,
        fixed_scale_fraction=args.fixed_scale_fraction,
    )
    contract = trainable_contract(model, adapter, True)
    system_prompt = eval_entry.official_eval.make_system_prompt(16, 16)
    baseline_by_id = cache_baseline_logits(
        model,
        adapter,
        processor,
        train_rows + validation_rows,
        args.data_root,
        system_prompt,
        prefix_text,
        candidates,
        args.eval_batch_size,
        args.baseline_logits,
        args.progress_json,
    )
    baseline_validation_logits = {
        str(row["sample_id"]): baseline_by_id[str(row["sample_id"])]
        for row in validation_rows
    }
    baseline_metrics = prediction_metrics(validation_rows, baseline_validation_logits)
    alternates = alternate_references(train_rows)
    same_anchor_pairs = build_same_anchor_pairs(train_rows, args.seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapter.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    names = (
        "total", "primary", "alternate", "consistency", "visual_verdict",
        "persistence", "pair_rank", "pair_evidence", "anchor_kl",
        "margin_retention", "cycle", "orthogonality", "preserve", "unmatched",
    )
    sums = {name: 0.0 for name in names}
    state_path = args.output_dir / "training_state.pt"
    start = 0
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != STATE_SCHEMA:
            raise ValueError("GRAFT resume schema mismatch")
        adapter.load_state_dict(state["adapter"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["next_batch"])
        sums = {name: float(state["loss_sums"][name]) for name in names}
        print(json.dumps({"event": "resume-graft-stage4", "next_batch": start}, sort_keys=True), flush=True)

    train_batches = batches(train_rows, args.batch_size, args.seed)
    model.eval(); adapter.train()
    for batch_index in range(start, len(train_batches)):
        batch = train_batches[batch_index]
        alternate = [alternates[str(row["sample_id"])] for row in batch]
        combined = [*batch, *alternate]
        cot_logits = answer_logits(
            model,
            prepare_cot_batch(processor, combined, args.data_root, system_prompt, prefix_text, device),
            candidates,
        )
        size = len(batch)
        primary, alternate_logits = cot_logits[:size], cot_logits[size:]
        target = labels(batch, device)
        base = torch.tensor(
            [baseline_by_id[str(row["sample_id"])] for row in batch],
            device=device,
            dtype=primary.dtype,
        )
        correct = base.argmax(dim=-1).eq(target)
        weights = sample_weights(
            batch,
            base,
            target,
            correct_weight=args.correct_weight,
            wrong_weight=args.wrong_weight,
            ad_normal_correct_weight=args.ad_normal_correct_weight,
            ad_normal_wrong_weight=args.ad_normal_wrong_weight,
            ad_anomaly_correct_weight=args.ad_anomaly_correct_weight,
            ad_anomaly_wrong_weight=args.ad_anomaly_wrong_weight,
        )
        primary_loss = (F.cross_entropy(primary, target, reduction="none") * weights).mean()
        alternate_loss = F.cross_entropy(alternate_logits, target)
        p_log, a_log = F.log_softmax(primary, -1), F.log_softmax(alternate_logits, -1)
        consistency = 0.5 * (
            F.kl_div(p_log, a_log.exp().detach(), reduction="batchmean")
            + F.kl_div(a_log, p_log.exp().detach(), reduction="batchmean")
        )
        combined_labels = torch.tensor(
            [1 if row["label"] == "anomaly" else 0 for row in combined], device=device
        )
        if adapter.last_visual_verdict_logits is None:
            raise RuntimeError("GRAFT visual verdict missing")
        visual_verdict = F.cross_entropy(adapter.last_visual_verdict_logits, combined_labels)
        if adapter.last_subspace_orthogonality is None or adapter.last_transport_cycle_loss is None:
            raise RuntimeError("GRAFT structural evidence missing")
        orthogonality = adapter.last_subspace_orthogonality
        cycle = adapter.last_transport_cycle_loss
        unmatched = adapter.last_unmatched_mass
        if unmatched is None:
            raise RuntimeError("GRAFT unmatched evidence missing")
        top_count = max(1, unmatched.shape[1] // 20)
        top_mass = unmatched.topk(top_count, dim=1).values.mean(dim=1)
        normal_mask, anomaly_mask = combined_labels.eq(0), combined_labels.eq(1)
        normal_unmatched = unmatched.mean(dim=1)[normal_mask].square().mean() if normal_mask.any() else primary.new_zeros(())
        anomaly_unmatched = F.relu(0.10 - top_mass[anomaly_mask]).square().mean() if anomaly_mask.any() else primary.new_zeros(())
        unmatched_loss = normal_unmatched + anomaly_unmatched
        if not adapter.last_residual_ratios:
            raise RuntimeError("GRAFT replay residual was not applied")
        residual_ratios = torch.stack([value.amax(dim=1) for value in adapter.last_residual_ratios]).mean(dim=0)
        preserve = residual_ratios[:size][correct].square().mean() if correct.any() else primary.new_zeros(())

        direct_logits = answer_logits(
            model,
            prepare_direct_batch(processor, batch, args.data_root, system_prompt, prefix_text, device),
            candidates,
        )
        direct_margin = answer_margin(direct_logits, target).detach()
        cot_margin = answer_margin(primary, target)
        direct_correct = direct_logits.argmax(dim=-1).eq(target)
        persistence = F.relu(direct_margin[direct_correct] - cot_margin[direct_correct]).mean() if direct_correct.any() else primary.new_zeros(())

        per_row_kl = F.kl_div(
            F.log_softmax(primary, dim=-1), F.softmax(base, dim=-1), reduction="none"
        ).sum(dim=-1)
        anchor_kl = per_row_kl[correct].mean() if correct.any() else primary.new_zeros(())
        base_margin = answer_margin(base, target).detach()
        retention_rows = F.relu(base_margin - cot_margin)
        margin_retention = retention_rows[correct].mean() if correct.any() else primary.new_zeros(())

        pair_rank = primary.new_zeros(())
        pair_evidence = primary.new_zeros(())
        if batch_index % args.pair_frequency == 0:
            pair_rows = same_anchor_pairs[(batch_index // args.pair_frequency) % len(same_anchor_pairs)]
            pair_logits = answer_logits(
                model,
                prepare_cot_batch(processor, pair_rows, args.data_root, system_prompt, prefix_text, device),
                candidates,
            )
            semantic = semantic_anomaly_margin(pair_logits, pair_rows)
            pair_rank = F.softplus(args.pair_margin - semantic[0] + semantic[1])
            pair_unmatched = adapter.last_unmatched_mass
            if pair_unmatched is None:
                raise RuntimeError("same-anchor unmatched evidence missing")
            pair_top = pair_unmatched.topk(max(1, pair_unmatched.shape[1] // 20), dim=1).values.mean(dim=1)
            pair_evidence = F.softplus(args.evidence_margin - pair_top[0] + pair_top[1])

        loss = (
            primary_loss
            + args.alternate_weight * alternate_loss
            + args.consistency_weight * consistency
            + args.visual_verdict_weight * visual_verdict
            + args.persistence_weight * persistence
            + args.pair_rank_weight * pair_rank
            + args.pair_evidence_weight * pair_evidence
            + args.anchor_kl_weight * anchor_kl
            + args.margin_retention_weight * margin_retention
            + args.cycle_weight * cycle
            + args.orthogonality_weight * orthogonality
            + args.preserve_weight * preserve
            + args.unmatched_weight * unmatched_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite GRAFT gradient")
        optimizer.step()
        values = {
            "total": loss,
            "primary": primary_loss,
            "alternate": alternate_loss,
            "consistency": consistency,
            "visual_verdict": visual_verdict,
            "persistence": persistence,
            "pair_rank": pair_rank,
            "pair_evidence": pair_evidence,
            "anchor_kl": anchor_kl,
            "margin_retention": margin_retention,
            "cycle": cycle,
            "orthogonality": orthogonality,
            "preserve": preserve,
            "unmatched": unmatched_loss,
        }
        for name, value in values.items():
            sums[name] += float(value.detach().cpu())
        if (batch_index + 1) % args.state_save_steps == 0 or batch_index + 1 == len(train_batches):
            save_state(
                state_path,
                adapter,
                optimizer,
                next_batch=batch_index + 1,
                loss_sums=sums,
            )
            update_progress(
                args.progress_json,
                "train-graft-stage4",
                batch_index + 1,
                len(train_batches),
                "supervised-structural-screen",
            )

    metrics, candidate_logits, persistence_diagnostic = evaluate(
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
    metrics["paired_vs_baseline"] = paired_delta(
        validation_rows, baseline_validation_logits, candidate_logits
    )
    per_task_safe = all(
        metrics["per_task_accuracy"][task]
        >= baseline_metrics["per_task_accuracy"][task] - 0.0125
        for task in baseline_metrics["per_task_accuracy"]
    )
    screen_pass = (
        per_task_safe
        and metrics["accuracy"] > baseline_metrics["accuracy"]
        and metrics["ad_balanced_accuracy"] > baseline_metrics["ad_balanced_accuracy"]
    )
    identity = save_adapter(
        adapter,
        args.output_dir / "epoch-1",
        {
            "selected_epoch": 1,
            "decision_protocol": protocol,
            "phase_protocol": {
                "assistant_prefix_ids": assistant_ids,
                "seg_start_ids": seg_ids,
            },
            "validation_metrics": metrics,
            "visual_persistence": persistence_diagnostic,
            "screen_pass": screen_pass,
        },
    )
    summary = {
        "schema_version": "judo-graft-stage4-training-summary-v1",
        "status": "complete",
        "screen_pass": screen_pass,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "same_anchor_pairs": len(same_anchor_pairs),
        "contract": contract,
        "baseline_validation": baseline_metrics,
        "epoch1": {
            "metrics": metrics,
            "visual_persistence": persistence_diagnostic,
            "mean_losses": {
                name: value / max(1, len(train_batches)) for name, value in sums.items()
            },
            "identity": identity,
        },
        "hyperparameters": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
    }
    atomic_json(args.output_dir / "graft_stage4_training_summary.json", summary)
    save_state(
        state_path,
        adapter,
        optimizer,
        next_batch=len(train_batches),
        loss_sums=sums,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "screen_pass": screen_pass,
                "baseline_accuracy": baseline_metrics["accuracy"],
                "candidate_accuracy": metrics["accuracy"],
                "baseline_ad_balanced": baseline_metrics["ad_balanced_accuracy"],
                "candidate_ad_balanced": metrics["ad_balanced_accuracy"],
                "adapter_sha256": identity["weights_sha256"],
            },
            sort_keys=True,
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
