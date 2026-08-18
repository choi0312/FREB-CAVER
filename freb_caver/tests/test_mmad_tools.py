from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))
tools = importlib.import_module("mmad_tools")


def test_stratified_selection_is_exact_and_deterministic() -> None:
    dataset = {
        f"GoodsAD/widget/{'good' if index % 2 == 0 else 'bad'}/{index}.png": {
            "conversation": [
                {"type": "Anomaly Detection"},
                {"type": "Object Analysis"},
            ]
        }
        for index in range(12)
    }
    first = tools.stratified_image_keys(dataset, 0.5, 42)
    second = tools.stratified_image_keys(dataset, 0.5, 42)
    assert first == second
    assert len(first) == 6
    assert len(set(first)) == 6


def test_task_and_source_normalization() -> None:
    assert tools.normalize_task("Object Structure") == "Object Analysis"
    assert tools.normalize_task("Object Details") == "Object Analysis"
    assert tools.canonical_source("DS-MVTec/cable/good/0.png") == "MVTec-AD"
    assert tools.path_label("GoodsAD/widget/good/0.png") == "normal"
    assert tools.path_label("GoodsAD/widget/bent/0.png") == "anomaly"


def test_filter_manifest_uses_the_exact_requested_id_set(tmp_path: Path) -> None:
    source = [
        {"sample_id": "a", "image": "a.png", "question_type": "Anomaly Detection"},
        {"sample_id": "b", "image": "b.png", "question_type": "Anomaly Detection"},
        {"sample_id": "c", "image": "c.png", "question_type": "Object Analysis"},
    ]
    selector = [{"sample_id": "c"}, {"sample_id": "a"}]
    source_path = tmp_path / "source.jsonl"
    selector_path = tmp_path / "selector.jsonl"
    tools.write_jsonl(source_path, source)
    tools.write_jsonl(selector_path, selector)
    args = argparse.Namespace(
        input=str(source_path),
        ids_from=str(selector_path),
        id_field="sample_id",
        output=str(tmp_path / "selected.jsonl"),
        metadata=str(tmp_path / "selected.meta.json"),
    )

    tools.filter_manifest(args)

    selected = tools.read_jsonl(tmp_path / "selected.jsonl")
    assert [row["sample_id"] for row in selected] == ["a", "c"]
    assert tools.read_json(tmp_path / "selected.meta.json")["samples"] == 2
