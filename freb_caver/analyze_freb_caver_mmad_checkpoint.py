#!/usr/bin/env python3
"""Audit and score an append-resumable prefix checkpoint of the MMAD run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ADAPTER_SHA256 = "ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accuracy(rows: list[dict[str, Any]], field: str) -> float:
    return sum(str(row.get(field, "X")) == str(row["correct_answer"]) for row in rows) / len(rows)


def grouped(rows: list[dict[str, Any]], field: str, key: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in sorted({str(row[key]) for row in rows}):
        subset = [row for row in rows if str(row[key]) == value]
        result[value] = {
            "accuracy": accuracy(subset, field),
            "correct": sum(str(row.get(field, "X")) == str(row["correct_answer"]) for row in subset),
            "total": len(subset),
        }
    return result


def ad_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    ad = [row for row in rows if row["question_type"] == "Anomaly Detection"]
    labels = grouped(ad, field, "label")
    if set(labels) != {"anomaly", "normal"}:
        raise ValueError("checkpoint AD rows must contain anomaly and normal labels")
    recall = labels["anomaly"]["accuracy"]
    specificity = labels["normal"]["accuracy"]
    return {
        "accuracy": accuracy(ad, field),
        "balanced_accuracy": (recall + specificity) / 2,
        "anomaly_recall": recall,
        "normal_specificity": specificity,
        "normal_false_positive_rate": 1 - specificity,
        "by_label": labels,
        "total": len(ad),
    }


def score(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    cell_metrics: dict[str, Any] = {}
    for source, task in sorted({(str(row["source"]), str(row["question_type"])) for row in rows}):
        subset = [row for row in rows if row["source"] == source and row["question_type"] == task]
        value = (
            ad_metrics(subset, field)["balanced_accuracy"]
            if task == "Anomaly Detection"
            else accuracy(subset, field)
        )
        cell_metrics[f"{source}|{task}"] = {"accuracy": value, "total": len(subset)}
    return {
        "answer_field": field,
        "micro_accuracy": accuracy(rows, field),
        "observed_cell_macro_accuracy": sum(x["accuracy"] for x in cell_metrics.values()) / len(cell_metrics),
        "observed_cell_count": len(cell_metrics),
        "per_source_micro_accuracy": grouped(rows, field, "source"),
        "per_task_micro_accuracy": grouped(rows, field, "question_type"),
        "per_observed_cell_accuracy": cell_metrics,
        "anomaly_detection": ad_metrics(rows, field),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--eval-metrics", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    predictions = read_jsonl(args.predictions)
    expected_rows = manifest[: args.expected_samples]
    if len(manifest) != 39670 or len(predictions) != args.expected_samples:
        raise ValueError(
            f"checkpoint cardinality mismatch: manifest={len(manifest)} predictions={len(predictions)}"
        )
    expected_by_id = {str(row["sample_id"]): row for row in expected_rows}
    predictions_by_id = {str(row["sample_id"]): row for row in predictions}
    if len(expected_by_id) != args.expected_samples or set(expected_by_id) != set(predictions_by_id):
        raise ValueError("checkpoint predictions are not the exact frozen-manifest prefix")

    joined: list[dict[str, Any]] = []
    for source in expected_rows:
        sample_id = str(source["sample_id"])
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

    metrics = json.loads(args.eval_metrics.read_text(encoding="utf-8"))
    attestation = json.loads(args.attestation.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(args.manifest)
    if metrics.get("manifest_sha256") != manifest_sha:
        raise ValueError("eval metrics manifest hash mismatch")
    if metrics.get("samples") != args.expected_samples or metrics.get("valid_predictions") != args.expected_samples:
        raise ValueError("eval metrics completion contract failed")
    if attestation.get("adapter_weights_sha256") != ADAPTER_SHA256:
        raise ValueError("FREB-CAVER adapter hash mismatch")

    payload = {
        "schema_version": "judo-freb-caver-mmad-prefix-checkpoint-analysis-v1",
        "status": "half-complete" if args.expected_samples * 2 == len(manifest) else "prefix-complete",
        "samples": len(joined),
        "remaining_samples": len(manifest) - len(joined),
        "completion_fraction": len(joined) / len(manifest),
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
        "runtime": {
            "gpu_name": metrics.get("gpu_name"),
            "runtime_seconds": metrics.get("runtime_seconds"),
            "inference_seconds": metrics.get("inference_seconds"),
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
        "scope_warning": "This prefix checkpoint is not a representative estimate of all 28 source-task cells.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
