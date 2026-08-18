#!/usr/bin/env python3
"""Paired pilot analysis for one or more refdiff candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def correctness(prediction: dict[str, Any]) -> bool:
    return bool(prediction.get("strict_format")) and prediction.get("strict_gpt_answer") == prediction.get("correct_answer")


def summarize(rows: list[dict[str, Any]], correct: dict[str, bool]) -> dict[str, Any]:
    by_task: dict[str, list[bool]] = defaultdict(list)
    by_source: dict[str, list[bool]] = defaultdict(list)
    by_cell: dict[str, list[bool]] = defaultdict(list)
    ad: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        value = correct[str(row["sample_id"])]
        task, source = str(row["question_type"]), str(row["source"])
        by_task[task].append(value)
        by_source[source].append(value)
        by_cell[f"{source}|{task}"].append(value)
        if task == "Anomaly Detection":
            ad[str(row["label"])].append(value)
    mean = lambda values: sum(values) / len(values)
    return {
        "samples": len(rows),
        "accuracy": mean(list(correct.values())),
        "source_task_macro_accuracy": mean([mean(values) for values in by_cell.values()]),
        "per_task_accuracy": {key: mean(value) for key, value in sorted(by_task.items())},
        "per_source_accuracy": {key: mean(value) for key, value in sorted(by_source.items())},
        "per_cell_accuracy": {key: mean(value) for key, value in sorted(by_cell.items())},
        "ad_anomaly_recall": mean(ad["anomaly"]),
        "ad_normal_specificity": mean(ad["normal"]),
        "ad_balanced_accuracy": (mean(ad["anomaly"]) + mean(ad["normal"])) / 2,
    }


def bootstrap(
    rows: list[dict[str, Any]],
    baseline: dict[str, bool],
    candidate: dict[str, bool],
    replicates: int,
    seed: int,
    cluster_unit: str,
) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if cluster_unit == "query":
        for row in rows:
            clusters[str(row["image"])].append(row)
    elif cluster_unit == "query-reference-component":
        parent: dict[str, str] = {}

        def find(value: str) -> str:
            parent.setdefault(value, value)
            if parent[value] != value:
                parent[value] = find(parent[value])
            return parent[value]

        def union(left: str, right: str) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent[max(a, b)] = min(a, b)

        for row in rows:
            union(str(row["image"]), str(row["template_image"]))
        for row in rows:
            clusters[find(str(row["image"]))].append(row)
    else:
        raise ValueError(f"unknown bootstrap cluster unit: {cluster_unit}")
    keys = sorted(clusters)
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sampled = [rng.choice(keys) for _ in keys]
        expanded = [row for key in sampled for row in clusters[key]]
        b = sum(baseline[str(row["sample_id"])] for row in expanded) / len(expanded)
        c = sum(candidate[str(row["sample_id"])] for row in expanded) / len(expanded)
        values.append(c - b)
    values.sort()
    low = values[int(0.025 * len(values))]
    high = values[min(len(values) - 1, int(0.975 * len(values)))]
    return {
        "cluster_unit": cluster_unit,
        "clusters": len(keys),
        "replicates": replicates,
        "seed": seed,
        "ci95": [low, high],
        "mean": sum(values) / len(values),
    }


def exact_mcnemar(rescues: int, regressions: int) -> dict[str, Any]:
    """Two-sided exact paired test over discordant predictions."""
    discordant = rescues + regressions
    if discordant == 0:
        return {"discordant": 0, "two_sided_p": 1.0}
    tail = sum(math.comb(discordant, index) for index in range(min(rescues, regressions) + 1)) / 2**discordant
    return {"discordant": discordant, "two_sided_p": min(1.0, 2.0 * tail)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--candidate-name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument(
        "--bootstrap-cluster-unit",
        choices=("query", "query-reference-component"),
        default="query-reference-component",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if len(args.candidate) != len(args.candidate_name):
        raise ValueError("candidate paths and names must have equal length")

    rows = read_jsonl(args.manifest)
    ids = {str(row["sample_id"]) for row in rows}
    baseline_rows = {str(row["sample_id"]): row for row in read_jsonl(args.baseline)}
    if set(baseline_rows) != ids:
        raise ValueError("baseline prediction IDs do not match the pilot manifest")
    baseline_correct = {key: correctness(value) for key, value in baseline_rows.items()}
    result = {
        "schema_version": "judo-refdiff-pilot-analysis-v1",
        "samples": len(rows),
        "baseline": summarize(rows, baseline_correct),
        "candidates": {},
    }
    for name, path in zip(args.candidate_name, args.candidate):
        predictions = {str(row["sample_id"]): row for row in read_jsonl(path)}
        if set(predictions) != ids:
            raise ValueError(f"candidate {name} IDs do not match the pilot manifest")
        candidate_correct = {key: correctness(value) for key, value in predictions.items()}
        contingency = Counter()
        for key in ids:
            b, c = baseline_correct[key], candidate_correct[key]
            contingency["both_correct" if b and c else "regressions" if b else "rescues" if c else "both_wrong"] += 1
        summary = summarize(rows, candidate_correct)
        summary["delta_accuracy"] = summary["accuracy"] - result["baseline"]["accuracy"]
        summary["delta_source_task_macro"] = summary["source_task_macro_accuracy"] - result["baseline"]["source_task_macro_accuracy"]
        summary["paired_contingency"] = dict(contingency) | {"net_rescues": contingency["rescues"] - contingency["regressions"]}
        summary["exact_mcnemar"] = exact_mcnemar(contingency["rescues"], contingency["regressions"])
        summary["cluster_bootstrap"] = bootstrap(
            rows, baseline_correct, candidate_correct, args.bootstrap_replicates,
            args.seed, args.bootstrap_cluster_unit,
        )
        result["candidates"][name] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
