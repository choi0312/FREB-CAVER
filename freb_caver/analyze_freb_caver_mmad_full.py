#!/usr/bin/env python3
"""Audit and score a complete 39,670-row MMAD FREB-CAVER inference run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCES = ("GoodsAD", "MVTec-AD", "MVTec-LOCO", "VisA")
TASKS = (
    "Anomaly Detection",
    "Defect Analysis",
    "Defect Classification",
    "Defect Description",
    "Defect Localization",
    "Object Analysis",
    "Object Classification",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accuracy(rows: list[dict[str, Any]], answer_field: str) -> float:
    if not rows:
        raise ValueError("accuracy requires at least one row")
    return sum(str(row.get(answer_field, "X")) == str(row["correct_answer"]) for row in rows) / len(rows)


def ad_balanced(rows: list[dict[str, Any]], answer_field: str) -> dict[str, Any]:
    by_label: dict[str, Any] = {}
    for label in ("anomaly", "normal"):
        subset = [row for row in rows if row["label"] == label]
        if not subset:
            raise ValueError(f"AD subset is missing label {label}")
        by_label[label] = {
            "accuracy": accuracy(subset, answer_field),
            "correct": sum(str(row.get(answer_field, "X")) == str(row["correct_answer"]) for row in subset),
            "total": len(subset),
        }
    return {
        "balanced_accuracy": (by_label["anomaly"]["accuracy"] + by_label["normal"]["accuracy"]) / 2,
        "anomaly_recall": by_label["anomaly"]["accuracy"],
        "normal_specificity": by_label["normal"]["accuracy"],
        "normal_false_positive_rate": 1.0 - by_label["normal"]["accuracy"],
        "by_label": by_label,
    }


def source_task_cells(rows: list[dict[str, Any]], answer_field: str) -> dict[tuple[str, str], float]:
    cells: dict[tuple[str, str], float] = {}
    for source in SOURCES:
        for task in TASKS:
            subset = [row for row in rows if row["source"] == source and row["question_type"] == task]
            if not subset:
                raise ValueError(f"missing normalized source/task cell: {source}|{task}")
            cells[(source, task)] = (
                ad_balanced(subset, answer_field)["balanced_accuracy"]
                if task == "Anomaly Detection"
                else accuracy(subset, answer_field)
            )
    return cells


def score(rows: list[dict[str, Any]], answer_field: str) -> dict[str, Any]:
    cells = source_task_cells(rows, answer_field)
    ad_rows = [row for row in rows if row["question_type"] == "Anomaly Detection"]
    return {
        "answer_field": answer_field,
        "micro_accuracy": accuracy(rows, answer_field),
        "normalized_28_cell_macro_accuracy": sum(cells.values()) / len(cells),
        "per_task_source_macro_accuracy": {
            task: sum(cells[(source, task)] for source in SOURCES) / len(SOURCES)
            for task in TASKS
        },
        "per_source_task_macro_accuracy": {
            source: sum(cells[(source, task)] for task in TASKS) / len(TASKS)
            for source in SOURCES
        },
        "per_cell_accuracy": {f"{source}|{task}": cells[(source, task)] for source in SOURCES for task in TASKS},
        "anomaly_detection": ad_balanced(ad_rows, answer_field),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--eval-metrics", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    predictions = read_jsonl(args.predictions)
    if len(manifest) != 39670 or len(predictions) != 39670:
        raise ValueError(f"full MMAD cardinality mismatch: manifest={len(manifest)} predictions={len(predictions)}")
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    predictions_by_id = {str(row["sample_id"]): row for row in predictions}
    if len(manifest_by_id) != len(manifest) or len(predictions_by_id) != len(predictions):
        raise ValueError("blank or duplicate sample IDs")
    if set(manifest_by_id) != set(predictions_by_id):
        raise ValueError("predictions do not cover the complete manifest")

    joined: list[dict[str, Any]] = []
    for sample_id, source in manifest_by_id.items():
        prediction = predictions_by_id[sample_id]
        if (
            prediction.get("correct_answer") != source["correct_answer"]
            or prediction.get("source") != source["source"]
            or prediction.get("question_type_normalized") != source["question_type"]
            or prediction.get("image") != source["image"]
            or prediction.get("template_image") != source["template_image"]
        ):
            raise ValueError(f"prediction metadata mismatch: {sample_id}")
        joined.append(
            {
                **source,
                "gpt_answer": prediction.get("gpt_answer", "X"),
                "strict_gpt_answer": prediction.get("strict_gpt_answer", "X"),
                "strict_format": bool(prediction.get("strict_format", False)),
            }
        )

    expected_cells = {(source, task) for source in SOURCES for task in TASKS}
    observed_cells = {(str(row["source"]), str(row["question_type"])) for row in joined}
    if observed_cells != expected_cells:
        raise ValueError("full MMAD rows do not cover the expected 28 normalized cells")

    metrics = json.loads(args.eval_metrics.read_text(encoding="utf-8"))
    attestation = json.loads(args.attestation.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(args.manifest)
    if metrics.get("manifest_sha256") != manifest_sha:
        raise ValueError("eval metrics manifest hash mismatch")
    # Format-invalid generations are legitimate completed predictions and must be
    # scored as incorrect, not mistaken for an interrupted evaluation.
    if (
        metrics.get("samples") != 39670
        or metrics.get("completed") != 39670
        or metrics.get("expected_samples") != 39670
    ):
        raise ValueError("eval metrics completion contract failed")
    if attestation.get("adapter_weights_sha256") != "ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed":
        raise ValueError("FREB-CAVER adapter hash mismatch")

    payload = {
        "schema_version": "judo-freb-caver-mmad-full-analysis-v1",
        "status": "complete",
        "samples": len(joined),
        "unique_sample_ids": len(manifest_by_id),
        "unique_query_images": len({str(row["image"]) for row in joined}),
        "unique_reference_images": len({str(row["template_image"]) for row in joined}),
        "unique_assets": len(
            {str(row["image"]) for row in joined} | {str(row["template_image"]) for row in joined}
        ),
        "manifest_sha256": manifest_sha,
        "predictions_sha256": sha256_file(args.predictions),
        "adapter_weights_sha256": attestation["adapter_weights_sha256"],
        "model_fingerprint": metrics.get("model_fingerprint"),
        "strict_format": {
            "valid": sum(bool(row["strict_format"]) for row in joined),
            "invalid": sum(not bool(row["strict_format"]) for row in joined),
        },
        "fallback_parser_metrics": score(joined, "gpt_answer"),
        "strict_parser_metrics": score(joined, "strict_gpt_answer"),
        "legacy_judo_eval_metrics": {
            "mmad_official_overall_accuracy": metrics.get("mmad_official_overall_accuracy"),
            "mmad_official_task_average": metrics.get("mmad_official_task_average"),
            "mmad_official_source_average": metrics.get("mmad_official_source_average"),
        },
        "runtime": {
            "gpu_name": metrics.get("gpu_name"),
            "runtime_seconds": metrics.get("runtime_seconds"),
            "inference_seconds": metrics.get("inference_seconds"),
            "mean_recorded_inference_seconds": metrics.get("mean_recorded_inference_seconds"),
            "peak_vram_allocated_mib": metrics.get("peak_vram_allocated_mib"),
            "peak_vram_reserved_mib": metrics.get("peak_vram_reserved_mib"),
        },
        "population": {
            "source_counts": dict(sorted(Counter(str(row["source"]) for row in joined).items())),
            "task_counts": dict(sorted(Counter(str(row["question_type"]) for row in joined).items())),
            "ad_label_counts": dict(
                sorted(
                    Counter(
                        str(row["label"])
                        for row in joined
                        if row["question_type"] == "Anomaly Detection"
                    ).items()
                )
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
