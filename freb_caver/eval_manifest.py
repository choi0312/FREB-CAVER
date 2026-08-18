#!/usr/bin/env python3
"""Run the official JUDO model path against a frozen MMAD manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
JUDO_REPO = Path(
    os.environ.get("JUDO_REPO", REPO_ROOT / "third_party" / "JUDO")
).expanduser().resolve()
EVAL_DIR = JUDO_REPO / "eval"
if not (EVAL_DIR / "eval_seg_mult.py").is_file():
    raise RuntimeError(
        "JUDO's official evaluator was not found. Clone woodavid31/JUDO and set "
        "JUDO_REPO to that checkout before running training or evaluation."
    )
sys.path.insert(0, str(EVAL_DIR))

import eval_seg_mult as official_eval  # noqa: E402

from provenance import checkpoint_fingerprint  # noqa: E402

from mmad_tools import normalize_task, read_jsonl, sha256_file, write_json  # noqa: E402


SCHEMA_VERSION = "judo-ablation-eval-v1"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    valid = []
    raw_lines = path.read_bytes().splitlines()
    for line_number, raw in enumerate(raw_lines, 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if line_number != len(raw_lines):
                raise ValueError(f"corrupt prediction record at {path}:{line_number}")
            break
        if not isinstance(row, dict) or not row.get("sample_id"):
            raise ValueError(f"invalid prediction record at {path}:{line_number}")
        valid.append(row)

    # Rewriting valid rows makes append-resume safe after a torn final write.
    temporary = path.with_suffix(path.suffix + ".repair")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in valid:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
    ids = [row["sample_id"] for row in valid]
    if len(ids) != len(set(ids)):
        raise ValueError("prediction checkpoint has duplicate sample IDs")
    return valid


def runtime_gate_payload(
    predictions: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    required_samples: int,
    max_sec_per_sample: float,
) -> dict[str, Any] | None:
    """Build a deterministic first-N inference gate from committed batches."""
    if len(predictions) < required_samples:
        return None
    prefix = predictions[:required_samples]
    batch_segments = [row for row in segments if row.get("kind") == "batch"]
    covered = 0
    selected_segments = []
    for segment in batch_segments:
        count = int(segment.get("new_samples", 0))
        if count < 1:
            raise ValueError("runtime gate observed a batch segment without samples")
        if covered + count > required_samples:
            raise ValueError("runtime gate sample boundary does not align with committed batches")
        selected_segments.append(segment)
        covered += count
        if covered == required_samples:
            break
    if covered != required_samples:
        raise ValueError("runtime gate predictions are not backed by committed batch timing")

    inference_seconds = sum(float(row["inference_seconds"]) for row in selected_segments)
    wall_seconds = sum(float(row["wall_seconds"]) for row in selected_segments)
    invalid_answers = sum(row.get("gpt_answer") not in {"A", "B", "C", "D"} for row in prefix)
    missing_margins = sum(
        row.get("answer_margin_captured") is not True
        or row.get("answer_semantic_margin") is None
        for row in prefix
    )
    inference_sec_per_sample = inference_seconds / required_samples
    sec_per_sample = wall_seconds / required_samples
    passed = (
        invalid_answers == 0
        and missing_margins == 0
        and sec_per_sample < max_sec_per_sample
    )
    return {
        "schema_version": "judo-ablation-teacher-runtime-gate-v1",
        "samples": required_samples,
        "invalid_answers": invalid_answers,
        "missing_answer_margins": missing_margins,
        "inference_seconds": inference_seconds,
        "wall_seconds": wall_seconds,
        "inference_sec_per_sample": inference_sec_per_sample,
        "sec_per_sample": sec_per_sample,
        "max_sec_per_sample_exclusive": max_sec_per_sample,
        "peak_vram_allocated_mib": max(
            float(row.get("peak_vram_allocated_mib", 0.0)) for row in selected_segments
        ),
        "peak_vram_reserved_mib": max(
            float(row.get("peak_vram_reserved_mib", 0.0)) for row in selected_segments
        ),
        "passed": passed,
    }


def enforce_runtime_gate(
    output_dir: Path,
    predictions: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    required_samples: int,
    max_sec_per_sample: float,
) -> dict[str, Any] | None:
    payload = runtime_gate_payload(
        predictions,
        segments,
        required_samples,
        max_sec_per_sample,
    )
    if payload is None:
        return None
    gate_path = output_dir / "runtime_gate.json"
    if gate_path.is_file():
        previous = json.loads(gate_path.read_text(encoding="utf-8"))
        if previous != payload:
            raise ValueError("runtime gate artifact changed across resume")
    else:
        write_json(gate_path, payload)
    if payload["passed"] is not True:
        raise RuntimeError(f"first-batch teacher runtime gate failed: {payload}")
    return payload


def accuracy_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_stats: dict[str, Counter[str]] = defaultdict(Counter)
    label_detection: dict[str, Counter[str]] = defaultdict(Counter)
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)
    official_cells: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    official_detection: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    valid_predictions = 0
    correct = 0
    for row in rows:
        prediction = str(row.get("gpt_answer", "X"))
        target = str(row.get("correct_answer", ""))
        is_correct = prediction == target
        valid_predictions += int(prediction in {"A", "B", "C", "D"})
        correct += int(is_correct)
        task = normalize_task(
            str(row.get("question_type_normalized") or row.get("question_type", "unknown"))
        )
        task_stats[task]["total"] += 1
        task_stats[task]["correct"] += int(is_correct)
        source = str(row.get("source") or row.get("source_raw") or "unknown")
        source_stats[source]["total"] += 1
        source_stats[source]["correct"] += int(is_correct)
        source_raw = str(row.get("source_raw") or "unknown")
        official_cells[(source_raw, task)]["total"] += 1
        official_cells[(source_raw, task)]["correct"] += int(is_correct)
        if task == "Anomaly Detection":
            label = str(row.get("label", "unknown"))
            label_detection[label]["total"] += 1
            label_detection[label]["correct"] += int(is_correct)
            official_detection[(source_raw, label)]["total"] += 1
            official_detection[(source_raw, label)]["correct"] += int(is_correct)

    task_accuracy = {}
    official_task_accuracy = {}
    for task, stats in sorted(task_stats.items()):
        raw_accuracy = stats["correct"] / stats["total"] if stats["total"] else 0.0
        task_accuracy[task] = raw_accuracy
        if task == "Anomaly Detection":
            label_accuracies = [
                label_detection[label]["correct"] / label_detection[label]["total"]
                for label in ("normal", "anomaly")
                if label_detection[label]["total"]
            ]
            official_task_accuracy[task] = (
                sum(label_accuracies) / len(label_accuracies) if label_accuracies else 0.0
            )
        else:
            official_task_accuracy[task] = raw_accuracy

    official_matrix: dict[str, dict[str, float]] = {}
    sources_raw = sorted({source for source, _ in official_cells})
    tasks = sorted(task_stats)
    for source_raw in sources_raw:
        official_matrix[source_raw] = {}
        for task in tasks:
            stats = official_cells[(source_raw, task)]
            if task == "Anomaly Detection":
                label_accuracies = []
                for label in ("normal", "anomaly"):
                    label_stats = official_detection[(source_raw, label)]
                    label_accuracies.append(
                        label_stats["correct"] / label_stats["total"]
                        if label_stats["total"]
                        else 0.0
                    )
                value = sum(label_accuracies) / 2
            else:
                value = stats["correct"] / stats["total"] if stats["total"] else 0.0
            official_matrix[source_raw][task] = value
    official_values = [
        value for source_values in official_matrix.values() for value in source_values.values()
    ]
    official_source_average = {
        source: sum(values.values()) / len(values) if values else 0.0
        for source, values in official_matrix.items()
    }
    official_task_average = {
        task: (
            sum(official_matrix[source][task] for source in sources_raw) / len(sources_raw)
            if sources_raw
            else 0.0
        )
        for task in tasks
    }

    return {
        "samples": len(rows),
        "correct": correct,
        "valid_predictions": valid_predictions,
        "invalid_predictions": len(rows) - valid_predictions,
        "overall_accuracy": correct / len(rows) if rows else 0.0,
        "micro_accuracy": correct / len(rows) if rows else 0.0,
        "global_task_macro_accuracy": (
            sum(official_task_accuracy.values()) / len(official_task_accuracy)
            if official_task_accuracy
            else 0.0
        ),
        "mmad_official_overall_accuracy": (
            sum(official_values) / len(official_values) if official_values else 0.0
        ),
        "task_accuracy": task_accuracy,
        "global_official_task_accuracy": official_task_accuracy,
        "mmad_official_source_task_accuracy": official_matrix,
        "mmad_official_source_average": official_source_average,
        "mmad_official_task_average": official_task_average,
        "task_counts": {task: dict(stats) for task, stats in sorted(task_stats.items())},
        "anomaly_detection_by_label": {
            label: {
                **dict(stats),
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
            }
            for label, stats in sorted(label_detection.items())
        },
        "source_accuracy": {
            source: {
                **dict(stats),
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
            }
            for source, stats in sorted(source_stats.items())
        },
    }


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)


def prepare_items(
    manifest_rows: list[dict[str, Any]], data_root: Path, workers: int
) -> list[tuple[Any, Any, dict[str, Any], str, str, str]]:
    paths = sorted(
        {
            str(row[field])
            for row in manifest_rows
            for field in ("image", "template_image")
        }
    )
    missing = [relative for relative in paths if not (data_root / relative).is_file()]
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"{len(missing)} unique evaluation images are missing under {data_root}:\n{preview}"
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        loaded = dict(
            zip(paths, executor.map(lambda value: load_image(data_root / value), paths))
        )

    items = []
    for row in manifest_rows:
        qdata = {
            "sample_id": row["sample_id"],
            "text": row["question"],
            "opts_text": row["options_text"],
            "question_type_normalized": row["question_type"],
            "source": row["source"],
            "source_raw": row["source_raw"],
            "category": row["category"],
            "label": row["label"],
            "options": row.get("options", {}),
            "template_image": row["template_image"],
        }
        items.append(
            (
                loaded[row["image"]],
                loaded[row["template_image"]],
                qdata,
                row["image"],
                row["correct_answer"],
                row["question_type_raw"],
            )
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sub-batch-size", type=int, default=8)
    parser.add_argument("--image-workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--enable-kv-cache", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--capture-answer-margin",
        action="store_true",
        help=(
            "For A/B Yes/No manifests, record the semantic Yes-minus-No logit "
            "when generation reaches <answer>. Disabled by default."
        ),
    )
    parser.add_argument(
        "--allow-missing-answer-margin-for-format-invalid",
        action="store_true",
        help=(
            "Allow a missing answer-token margin only when the strict parser "
            "attests that no valid answer field was produced. A strict parsed "
            "answer without a margin remains a hard failure."
        ),
    )
    parser.add_argument("--runtime-gate-samples", type=int)
    parser.add_argument("--runtime-gate-max-sec-per-sample", type=float)
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="sdpa",
        help="G4/Blackwell uses sdpa; flash_attention_2 preserves the official H200 path.",
    )
    args = parser.parse_args()

    if args.batch_size < 1 or args.sub_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if (args.runtime_gate_samples is None) != (
        args.runtime_gate_max_sec_per_sample is None
    ):
        raise ValueError("runtime gate sample and latency arguments must be supplied together")
    if args.runtime_gate_samples is not None:
        if args.runtime_gate_samples != args.batch_size:
            raise ValueError("runtime gate must align exactly with the first evaluation batch")
        if args.runtime_gate_max_sec_per_sample <= 0:
            raise ValueError("runtime gate latency must be positive")
        if not args.capture_answer_margin:
            raise ValueError("runtime gate requires strict answer-margin capture")
    if args.allow_missing_answer_margin_for_format_invalid and not args.capture_answer_margin:
        raise ValueError("the missing-margin exception requires answer-margin capture")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"this entry point is intentionally single-GPU; found {torch.cuda.device_count()} GPUs"
        )

    manifest_path = Path(args.manifest).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha256_file(manifest_path)
    rows = read_jsonl(manifest_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    manifest_ids = [row["sample_id"] for row in rows]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("manifest contains duplicate sample IDs")

    model_root = Path(args.model_path).resolve()
    model_info = checkpoint_fingerprint(model_root)
    model_fingerprint = model_info["fingerprint"]
    config = {
        "schema_version": SCHEMA_VERSION,
        "run_name": args.run_name,
        "model_path": str(model_root),
        "model_fingerprint": model_fingerprint,
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "seed": args.seed,
        "grid_size": args.grid_size,
        "batch_size": args.batch_size,
        "sub_batch_size": args.sub_batch_size,
        "max_samples": args.max_samples,
        "enable_kv_cache": args.enable_kv_cache,
        "compile_model": args.compile_model,
        "attn_implementation": args.attn_implementation,
        "runtime_gate_samples": args.runtime_gate_samples,
        "runtime_gate_max_sec_per_sample": args.runtime_gate_max_sec_per_sample,
    }
    # Omitting the false default keeps existing run_config files resume-compatible.
    if args.capture_answer_margin:
        config["capture_answer_margin"] = True
    if args.allow_missing_answer_margin_for_format_invalid:
        config["allow_missing_answer_margin_for_format_invalid"] = True
    config_path = output_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("refusing to resume with a different evaluation configuration")
    write_json(config_path, config)

    predictions_path = output_dir / "predictions.jsonl"
    existing = load_predictions(predictions_path)
    for row in existing:
        if row.get("manifest_sha256") != manifest_sha:
            raise ValueError("prediction checkpoint has a different manifest fingerprint")
        if row.get("model_fingerprint") != model_fingerprint:
            raise ValueError("prediction checkpoint has a different model fingerprint")
        if row.get("run_name") != args.run_name:
            raise ValueError("prediction checkpoint has a different run name")
    existing_by_id = {row["sample_id"]: row for row in existing}
    unknown = set(existing_by_id) - set(manifest_ids)
    if unknown:
        raise ValueError(f"prediction checkpoint contains {len(unknown)} IDs outside this manifest")
    remaining_rows = [row for row in rows if row["sample_id"] not in existing_by_id]

    segments_path = output_dir / "eval_segments.jsonl"
    previous_segments = read_jsonl(segments_path) if segments_path.exists() else []
    new_segments: list[dict[str, Any]] = []
    committed_rows = list(existing)
    if (
        args.runtime_gate_samples is not None
        and 0 < len(existing) < args.runtime_gate_samples
    ):
        raise RuntimeError(
            "cannot safely resume a teacher run interrupted before its first-batch runtime gate"
        )
    if args.runtime_gate_samples is not None and len(existing) >= args.runtime_gate_samples:
        enforce_runtime_gate(
            output_dir,
            existing,
            previous_segments,
            args.runtime_gate_samples,
            args.runtime_gate_max_sec_per_sample,
        )
    if remaining_rows:
        torch.cuda.reset_peak_memory_stats()
        official_eval.SYSTEM_PROMPT = official_eval.make_system_prompt(
            args.grid_size, args.grid_size
        )
        load_started = time.perf_counter()
        engine = official_eval.OptimizedMultiGPUEngine(
            model_path=str(model_root),
            num_gpus=1,
            enable_kv_cache=args.enable_kv_cache,
            compile_model=args.compile_model,
            attn_implementation=args.attn_implementation,
            capture_answer_margin=args.capture_answer_margin,
        )
        load_segment = {
            "kind": "model_load",
            "new_samples": 0,
            "wall_seconds": time.perf_counter() - load_started,
            "inference_seconds": 0.0,
            "peak_vram_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 2),
            "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024**2, 2),
        }
        append_jsonl(segments_path, load_segment)
        new_segments.append(load_segment)
        with predictions_path.open("a", encoding="utf-8", newline="\n") as handle:
            for start in range(0, len(remaining_rows), args.batch_size):
                batch_started = time.perf_counter()
                batch_rows = remaining_rows[start : start + args.batch_size]
                batch = prepare_items(batch_rows, data_root, args.image_workers)
                inference_started = time.perf_counter()
                results = engine.infer_batch(batch, args.sub_batch_size)
                batch_inference_seconds = time.perf_counter() - inference_started
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"inference returned {len(results)} rows for a batch of {len(batch)}"
                    )
                for result in results:
                    result["manifest_sha256"] = manifest_sha
                    result["model_fingerprint"] = model_fingerprint
                    result["run_name"] = args.run_name
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                committed_rows.extend(results)
                batch_segment = {
                    "kind": "batch",
                    "new_samples": len(results),
                    "wall_seconds": time.perf_counter() - batch_started,
                    "inference_seconds": batch_inference_seconds,
                    "peak_vram_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 2),
                    "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024**2, 2),
                }
                append_jsonl(segments_path, batch_segment)
                new_segments.append(batch_segment)
                if (
                    args.runtime_gate_samples is not None
                    and len(existing) + sum(
                        int(row.get("new_samples", 0))
                        for row in new_segments
                        if row.get("kind") == "batch"
                    )
                    >= args.runtime_gate_samples
                ):
                    enforce_runtime_gate(
                        output_dir,
                        committed_rows,
                        previous_segments + new_segments,
                        args.runtime_gate_samples,
                        args.runtime_gate_max_sec_per_sample,
                    )

    final_rows = load_predictions(predictions_path)
    if args.runtime_gate_samples is not None:
        gate = enforce_runtime_gate(
            output_dir,
            final_rows,
            previous_segments + new_segments,
            args.runtime_gate_samples,
            args.runtime_gate_max_sec_per_sample,
        )
        if gate is None:
            raise RuntimeError("evaluation ended before the teacher runtime gate")
    if args.capture_answer_margin:
        missing_rows = [
            row
            for row in final_rows
            if not row.get("answer_margin_captured")
            or row.get("answer_semantic_margin") is None
        ]
        forbidden_missing = [
            row["sample_id"]
            for row in missing_rows
            if not args.allow_missing_answer_margin_for_format_invalid
            or row.get("strict_format") is not False
        ]
        if forbidden_missing:
            raise RuntimeError(
                "answer-token margin was not captured for "
                f"{len(forbidden_missing)} required samples; first IDs: "
                f"{forbidden_missing[:10]}"
            )
    all_segments = previous_segments + new_segments
    cumulative_wall_seconds = sum(float(row.get("wall_seconds", 0.0)) for row in all_segments)
    cumulative_inference_seconds = sum(
        float(row.get("inference_seconds", 0.0)) for row in all_segments
    )
    segment_wall_seconds = sum(float(row.get("wall_seconds", 0.0)) for row in new_segments)
    segment_inference_seconds = sum(
        float(row.get("inference_seconds", 0.0)) for row in new_segments
    )
    metrics = accuracy_payload(final_rows)
    metrics.update(
        {
            "schema_version": SCHEMA_VERSION,
            "run_name": args.run_name,
            "manifest_sha256": manifest_sha,
            "model_fingerprint": model_fingerprint,
            "expected_samples": len(rows),
            "completed": len(final_rows),
            "segment_new_samples": len(remaining_rows),
            "segment_wall_seconds": segment_wall_seconds,
            "segment_inference_seconds": segment_inference_seconds,
            "segment_sec_per_sample": (
                segment_wall_seconds / len(remaining_rows) if remaining_rows else 0.0
            ),
            "segment_inference_sec_per_sample": (
                segment_inference_seconds / len(remaining_rows) if remaining_rows else 0.0
            ),
            "runtime_seconds": cumulative_wall_seconds,
            "inference_seconds": cumulative_inference_seconds,
            "sec_per_sample": cumulative_wall_seconds / len(final_rows) if final_rows else 0.0,
            "mean_recorded_inference_seconds": (
                sum(float(row.get("inference_time", 0.0)) for row in final_rows) / len(final_rows)
                if final_rows
                else 0.0
            ),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_mib": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2, 2
            ),
            "peak_vram_allocated_mib": max(
                (float(row.get("peak_vram_allocated_mib", 0.0)) for row in all_segments),
                default=0.0,
            ),
            "peak_vram_reserved_mib": max(
                (float(row.get("peak_vram_reserved_mib", 0.0)) for row in all_segments),
                default=0.0,
            ),
            "torch_version": torch.__version__,
        }
    )
    write_json(output_dir / "eval_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    if len(final_rows) != len(rows):
        raise RuntimeError(f"evaluation incomplete: {len(final_rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
