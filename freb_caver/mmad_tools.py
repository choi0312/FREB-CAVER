#!/usr/bin/env python3
"""Deterministic MMAD subset, evaluation-manifest, EDA, and asset checks.

The released JUDO evaluator randomizes both the normal template and option
order at evaluation time.  This utility freezes those choices once so paired
baseline/Ours runs consume byte-identical evaluation inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable


SCHEMA_VERSION = "judo-ablation-mmad-v1"
DEFAULT_SEED = 42


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(seed: int, *parts: object) -> str:
    material = "\0".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def raw_source(image_path: str) -> str:
    return image_path.replace("\\", "/").split("/", 1)[0]


def canonical_source(image_path: str) -> str:
    source = raw_source(image_path)
    return "MVTec-AD" if source in {"DS-MVTec", "MVTec-AD"} else source


def category(image_path: str) -> str:
    parts = image_path.replace("\\", "/").split("/")
    return parts[1] if len(parts) > 1 else "unknown"


def path_label(image_path: str) -> str:
    # This intentionally matches the released JUDO/MMAD evaluator semantics.
    # Keep the check case-sensitive so the dataset root ``GoodsAD`` is not
    # itself mistaken for the lower-case defect folder name ``good``.
    return "normal" if "good" in image_path else "anomaly"


def normalize_task(task: str) -> str:
    if task in {"Object Structure", "Object Details"}:
        return "Object Analysis"
    return task


def conversations(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    for key, value in metadata.items():
        if key.startswith("conversation"):
            if not isinstance(value, list):
                raise ValueError(f"{key} is not a list")
            return value
    return []


def task_signature(metadata: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    counts = Counter(str(row.get("type", "unknown")) for row in conversations(metadata))
    return tuple(sorted(counts.items()))


def hamilton_allocation(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if not 0 <= target <= total:
        raise ValueError(f"invalid Hamilton target {target} for population {total}")
    if total == 0:
        return {key: 0 for key in counts}
    exact = {key: target * count / total for key, count in counts.items()}
    allocated = {key: math.floor(value) for key, value in exact.items()}
    remainder = target - sum(allocated.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - allocated[key]), str(key)))
    for key in order[:remainder]:
        allocated[key] += 1
    return allocated


def stratified_image_keys(
    dataset: dict[str, dict[str, Any]], fraction: float, seed: int
) -> list[str]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    target = round(len(dataset) * fraction)
    entries = list(dataset.items())
    levels: list[Callable[[str, dict[str, Any]], object]] = [
        lambda image, _: canonical_source(image),
        lambda image, _: raw_source(image),
        lambda image, _: category(image),
        lambda image, _: path_label(image),
        lambda _, metadata: task_signature(metadata),
    ]

    def select(
        members: list[tuple[str, dict[str, Any]]], level: int, quota: int
    ) -> list[str]:
        if quota == 0:
            return []
        if quota >= len(members):
            return [image for image, _ in members]
        if level >= len(levels):
            ranked = sorted(members, key=lambda item: stable_digest(seed, item[0]))
            return [image for image, _ in ranked[:quota]]
        groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        key_fn = levels[level]
        for image, metadata in members:
            groups[repr(key_fn(image, metadata))].append((image, metadata))
        allocations = hamilton_allocation(
            {key: len(group) for key, group in groups.items()}, quota
        )
        chosen: list[str] = []
        for key in sorted(groups):
            chosen.extend(select(groups[key], level + 1, allocations[key]))
        return chosen

    selected = select(entries, 0, target)
    if len(selected) != target or len(set(selected)) != target:
        raise AssertionError("stratified selection did not produce the exact unique target")
    return sorted(selected)


def image_counts(dataset: dict[str, dict[str, Any]], field: str) -> Counter[str]:
    functions = {
        "source": canonical_source,
        "raw_source": raw_source,
        "category": category,
        "label": path_label,
    }
    function = functions[field]
    return Counter(function(image) for image in dataset)


def question_counts(dataset: dict[str, dict[str, Any]], field: str) -> Counter[str]:
    output: Counter[str] = Counter()
    for image, metadata in dataset.items():
        for question in conversations(metadata):
            if field == "task":
                value = normalize_task(str(question.get("type", "unknown")))
            elif field == "raw_task":
                value = str(question.get("type", "unknown"))
            elif field == "source":
                value = canonical_source(image)
            elif field == "raw_source":
                value = raw_source(image)
            elif field == "category":
                value = category(image)
            elif field == "label":
                value = path_label(image)
            else:
                raise KeyError(field)
            output[value] += 1
    return output


def count_payload(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def max_share_error_pp(full: Counter[str], subset: Counter[str]) -> float:
    full_total = sum(full.values())
    subset_total = sum(subset.values())
    keys = set(full) | set(subset)
    if not full_total or not subset_total:
        return 0.0
    return max(
        abs(full.get(key, 0) / full_total - subset.get(key, 0) / subset_total) * 100
        for key in keys
    )


def distribution_report(
    full: dict[str, dict[str, Any]], subset: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "full_images": len(full),
        "subset_images": len(subset),
        "image_fraction": len(subset) / len(full),
        "full_questions": sum(len(conversations(value)) for value in full.values()),
        "subset_questions": sum(len(conversations(value)) for value in subset.values()),
    }
    report["question_fraction"] = report["subset_questions"] / report["full_questions"]
    report["images"] = {}
    report["questions"] = {}
    for field in ("source", "raw_source", "category", "label"):
        full_counts = image_counts(full, field)
        subset_counts = image_counts(subset, field)
        report["images"][field] = {
            "full": count_payload(full_counts),
            "subset": count_payload(subset_counts),
            "max_share_error_pp": max_share_error_pp(full_counts, subset_counts),
        }
    for field in ("task", "raw_task", "source", "raw_source", "category", "label"):
        full_counts = question_counts(full, field)
        subset_counts = question_counts(subset, field)
        report["questions"][field] = {
            "full": count_payload(full_counts),
            "subset": count_payload(subset_counts),
            "max_share_error_pp": max_share_error_pp(full_counts, subset_counts),
        }
    return report


def build_subset(args: argparse.Namespace) -> None:
    source_path = Path(args.input).resolve()
    dataset = read_json(source_path)
    if not isinstance(dataset, dict):
        raise ValueError("MMAD JSON must be an image-keyed object")
    selected = set(stratified_image_keys(dataset, args.fraction, args.seed))
    subset = {key: value for key, value in dataset.items() if key in selected}
    write_json(Path(args.output).resolve(), subset)
    if args.keys_output:
        keys_path = Path(args.keys_output).resolve()
        keys_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = keys_path.with_suffix(keys_path.suffix + ".tmp")
        temporary.write_text("".join(f"{key}\n" for key in sorted(selected)), encoding="utf-8")
        temporary.replace(keys_path)
    report = distribution_report(dataset, subset)
    report.update(
        {
            "schema_version": SCHEMA_VERSION,
            "seed": args.seed,
            "fraction_requested": args.fraction,
            "input_path": str(source_path),
            "input_sha256": sha256_file(source_path),
        }
    )
    write_json(Path(args.report).resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def deterministic_rng(seed: int, *parts: object) -> random.Random:
    integer = int(stable_digest(seed, *parts), 16)
    return random.Random(integer)


def build_manifest_rows(
    dataset: dict[str, dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    rows = []
    for image, metadata in dataset.items():
        templates = metadata.get("random_templates") or []
        if not templates:
            raise ValueError(f"no random_templates for {image}")
        template_rng = deterministic_rng(seed, "template", image)
        template = templates[template_rng.randrange(len(templates))]
        for question_index, question in enumerate(conversations(metadata)):
            option_items = list(question["Options"].items())
            option_rng = deterministic_rng(seed, "options", image, question_index)
            option_rng.shuffle(option_items)
            remapped: dict[str, str] = {}
            correct_answer = None
            for index, (original_label, text) in enumerate(option_items):
                label = chr(ord("A") + index)
                remapped[label] = text
                if original_label == question["Answer"]:
                    correct_answer = label
            if correct_answer is None:
                raise ValueError(f"answer is absent from options: {image} question {question_index}")
            options_text = "".join(f"{label}. {text}\n" for label, text in remapped.items())
            identity = stable_digest(
                seed,
                "sample",
                image,
                question_index,
                question["Question"],
                template,
                json.dumps(remapped, ensure_ascii=False, sort_keys=False),
                correct_answer,
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": identity[:24],
                    "seed": seed,
                    "image": image,
                    "template_image": template,
                    "question_index": question_index,
                    "question": question["Question"],
                    "question_type_raw": question.get("type", "unknown"),
                    "question_type": normalize_task(question.get("type", "unknown")),
                    "options": remapped,
                    "options_text": options_text,
                    "correct_answer": correct_answer,
                    "source_raw": raw_source(image),
                    "source": canonical_source(image),
                    "category": category(image),
                    "label": path_label(image),
                }
            )
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise AssertionError("evaluation manifest contains duplicate sample IDs")
    return rows


def build_manifest(args: argparse.Namespace) -> None:
    source_path = Path(args.input).resolve()
    dataset = read_json(source_path)
    rows = build_manifest_rows(dataset, args.seed)
    selected_task = getattr(args, "task", None)
    if selected_task:
        rows = [row for row in rows if row["question_type"] == selected_task]
        if not rows:
            raise ValueError(f"manifest task filter matched no rows: {selected_task}")
    output_path = Path(args.output).resolve()
    write_jsonl(output_path, rows)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "manifest_path": str(output_path),
        "manifest_sha256": sha256_file(output_path),
        "samples": len(rows),
        "images": len({row["image"] for row in rows}),
        "task": selected_task,
    }
    write_json(Path(args.metadata).resolve(), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


def filter_manifest(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    selector_path = Path(args.ids_from).resolve()
    rows = read_jsonl(input_path)
    selectors = read_jsonl(selector_path)
    selected_ids = [str(row.get(args.id_field, "")) for row in selectors]
    if not selected_ids or any(not value for value in selected_ids):
        raise ValueError(f"selector contains an empty {args.id_field}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selector contains duplicate sample IDs")
    requested = set(selected_ids)
    selected = [row for row in rows if str(row.get("sample_id", "")) in requested]
    found = {str(row.get("sample_id", "")) for row in selected}
    missing = requested - found
    if missing or len(selected) != len(requested):
        raise ValueError(
            f"manifest selector mismatch: requested={len(requested)} "
            f"selected={len(selected)} missing={len(missing)}"
        )
    output_path = Path(args.output).resolve()
    write_jsonl(output_path, selected)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(input_path),
        "source_manifest_sha256": sha256_file(input_path),
        "selector": str(selector_path),
        "selector_sha256": sha256_file(selector_path),
        "selector_field": args.id_field,
        "manifest_path": str(output_path),
        "manifest_sha256": sha256_file(output_path),
        "samples": len(selected),
        "images": len({row["image"] for row in selected}),
        "tasks": count_payload(Counter(str(row["question_type"]) for row in selected)),
    }
    write_json(Path(args.metadata).resolve(), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


def summarize_stage_jsonl(path: Path, stage: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    images = [str(row.get("image") or row.get("image_path") or "") for row in rows]
    report: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "samples": len(rows),
        "unique_images": len(set(images)),
    }
    if stage == "stage1":
        def stage1_mmad_path(image: str) -> str:
            return image.split("/", 1)[1] if image.startswith("mmad/") else image

        report["source"] = count_payload(
            Counter(
                canonical_source(stage1_mmad_path(image))
                if image.startswith("mmad/")
                else "Real-IAD"
                for image in images
            )
        )
        report["category"] = count_payload(
            Counter(
                category(stage1_mmad_path(image))
                if str(row.get("item_type")) == "mmad"
                else str(row.get("item_type") or category(image))
                for row, image in zip(rows, images)
            )
        )
        report["label"] = count_payload(
            Counter("anomaly" if row.get("mask_path") else "normal" for row in rows)
        )
    elif stage == "stage2":
        report["source"] = count_payload(Counter(raw_source(image) for image in images))
        report["category"] = count_payload(Counter(category(image) for image in images))
        report["label"] = {"normal": len(rows)}
    elif stage == "stage3":
        report["source"] = count_payload(Counter(raw_source(image) for image in images))
        report["category"] = count_payload(Counter(category(image) for image in images))
        report["label"] = count_payload(
            Counter("anomaly" if row.get("mask_path") else "normal" for row in rows)
        )
        report["task_raw"] = count_payload(Counter(str(row.get("type", "unknown")) for row in rows))
        report["task"] = count_payload(
            Counter(normalize_task(str(row.get("type", "unknown"))) for row in rows)
        )
    return report


def eda(args: argparse.Namespace) -> None:
    full_path = Path(args.mmad).resolve()
    full = read_json(full_path)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mmad": {
            "path": str(full_path),
            "sha256": sha256_file(full_path),
            "images": len(full),
            "questions": sum(len(conversations(value)) for value in full.values()),
            "images_by_source": count_payload(image_counts(full, "source")),
            "images_by_raw_source": count_payload(image_counts(full, "raw_source")),
            "images_by_category": count_payload(image_counts(full, "category")),
            "images_by_label": count_payload(image_counts(full, "label")),
            "questions_by_task": count_payload(question_counts(full, "task")),
            "questions_by_raw_task": count_payload(question_counts(full, "raw_task")),
            "questions_by_source": count_payload(question_counts(full, "source")),
            "questions_by_raw_source": count_payload(question_counts(full, "raw_source")),
            "questions_by_category": count_payload(question_counts(full, "category")),
            "questions_by_label": count_payload(question_counts(full, "label")),
        },
    }
    if args.fast:
        fast = read_json(Path(args.fast).resolve())
        report["fast25"] = distribution_report(full, fast)
    stages = {}
    for name in ("stage1", "stage2", "stage3"):
        value = getattr(args, name)
        if value:
            stages[name] = summarize_stage_jsonl(Path(value).resolve(), name)
    report["training"] = stages

    if args.stage3:
        stage3_rows = read_jsonl(Path(args.stage3).resolve())
        full_pairs = {
            (image, str(question.get("Question", "")).strip())
            for image, metadata in full.items()
            for question in conversations(metadata)
        }
        full_images = set(full)
        overlap_pairs = 0
        overlap_images = set()
        for row in stage3_rows:
            image = str(row.get("image", ""))
            question = str(row.get("problem", "")).split("\n Choice:", 1)[0].strip()
            if (image, question) in full_pairs:
                overlap_pairs += 1
            if image in full_images:
                overlap_images.add(image)
        report["training_eval_overlap"] = {
            "stage3_rows_matching_mmad_image_and_question": overlap_pairs,
            "stage3_rows": len(stage3_rows),
            "stage3_unique_images_in_mmad": len(overlap_images),
            "stage3_unique_images": len({str(row.get("image", "")) for row in stage3_rows}),
        }

    write_json(Path(args.output).resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def preflight(args: argparse.Namespace) -> None:
    image_root = Path(args.image_root).resolve()
    missing: Counter[str] = Counter()
    checked = 0
    if args.stage3:
        for row in read_jsonl(Path(args.stage3).resolve()):
            for field in ("image", "template_image"):
                relative = row.get(field)
                checked += 1
                if not relative or not (image_root / relative).is_file():
                    missing[f"{field}:{relative}"] += 1
            mask = row.get("mask_path")
            if mask:
                checked += 1
                if not (image_root / mask).is_file():
                    missing[f"mask_path:{mask}"] += 1
    if args.manifest:
        for row in read_jsonl(Path(args.manifest).resolve()):
            for field in ("image", "template_image"):
                relative = row.get(field)
                checked += 1
                if not relative or not (image_root / relative).is_file():
                    missing[f"{field}:{relative}"] += 1
    payload = {
        "schema_version": SCHEMA_VERSION,
        "image_root": str(image_root),
        "paths_checked": checked,
        "missing_unique": len(missing),
        "missing_occurrences": sum(missing.values()),
        "missing": dict(sorted(missing.items())),
        "ok": not missing,
    }
    if args.output:
        write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    subset = commands.add_parser("subset", help="Create the fixed image-stratified MMAD subset")
    subset.add_argument("--input", required=True)
    subset.add_argument("--output", required=True)
    subset.add_argument("--report", required=True)
    subset.add_argument("--keys-output")
    subset.add_argument("--fraction", type=float, default=0.25)
    subset.add_argument("--seed", type=int, default=DEFAULT_SEED)
    subset.set_defaults(function=build_subset)

    manifest = commands.add_parser("manifest", help="Freeze templates and option permutations")
    manifest.add_argument("--input", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--metadata", required=True)
    manifest.add_argument("--seed", type=int, default=DEFAULT_SEED)
    manifest.add_argument(
        "--task",
        help="Optionally keep one normalized task, for example 'Anomaly Detection'.",
    )
    manifest.set_defaults(function=build_manifest)

    filter_parser = commands.add_parser(
        "filter-manifest", help="Select an exact sample-ID set from a frozen manifest"
    )
    filter_parser.add_argument("--input", required=True)
    filter_parser.add_argument("--ids-from", required=True)
    filter_parser.add_argument("--id-field", default="sample_id")
    filter_parser.add_argument("--output", required=True)
    filter_parser.add_argument("--metadata", required=True)
    filter_parser.set_defaults(function=filter_manifest)

    eda_parser = commands.add_parser("eda", help="Summarize training/evaluation data")
    eda_parser.add_argument("--mmad", required=True)
    eda_parser.add_argument("--fast")
    eda_parser.add_argument("--stage1")
    eda_parser.add_argument("--stage2")
    eda_parser.add_argument("--stage3")
    eda_parser.add_argument("--output", required=True)
    eda_parser.set_defaults(function=eda)

    check = commands.add_parser("preflight", help="Fail if any required image is absent")
    check.add_argument("--image-root", required=True)
    check.add_argument("--stage3")
    check.add_argument("--manifest")
    check.add_argument("--output")
    check.set_defaults(function=preflight)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
