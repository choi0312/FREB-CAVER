#!/usr/bin/env python3
"""Create and compare immutable input locks for paired JUDO ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "judo-ablation-input-lock-v1"
MODEL_METADATA = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(root: Path) -> list[Path]:
    files = [root / name for name in MODEL_METADATA if (root / name).is_file()]
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        files.extend(root / name for name in shard_names)
    else:
        files.extend(sorted(root.glob("*.safetensors")))
    files.extend(sorted(root.glob("*.model")))
    return sorted(set(files))


def checkpoint_fingerprint(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"not a local model checkpoint: {root}")
    files = checkpoint_files(root)
    if not any(path.suffix == ".safetensors" for path in files):
        raise FileNotFoundError(f"no safetensors weights found under {root}")
    records = []
    identity = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        record = {"path": relative, "bytes": path.stat().st_size, "sha256": digest}
        records.append(record)
        identity.update(json.dumps(record, sort_keys=True).encode("utf-8"))
    return {
        "resolved_path": str(root),
        "fingerprint": identity.hexdigest(),
        "files": records,
    }


def canonical_payload(args: argparse.Namespace) -> dict[str, Any]:
    settings = {}
    for item in args.setting:
        if "=" not in item:
            raise ValueError(f"setting must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        settings[key] = value
    checkpoint = checkpoint_fingerprint(args.checkpoint)
    inputs = {}
    for label, path in (
        ("stage3", args.stage3),
        ("fast_manifest", args.fast_manifest),
        ("full_manifest", args.full_manifest),
    ):
        if path:
            resolved = path.resolve()
            inputs[label] = {
                "path": str(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_role": args.checkpoint_role,
        "checkpoint": checkpoint,
        "inputs": inputs,
        "settings": dict(sorted(settings.items())),
    }
    identity_copy = json.loads(json.dumps(payload))
    identity_copy["checkpoint"].pop("resolved_path", None)
    for value in identity_copy["inputs"].values():
        value.pop("path", None)
    payload["input_lock_sha256"] = hashlib.sha256(
        json.dumps(identity_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def lock(args: argparse.Namespace) -> None:
    payload = canonical_payload(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("input_lock_sha256") != payload["input_lock_sha256"]:
            raise ValueError("refusing to resume: current inputs differ from the saved run lock")
    else:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def compare(args: argparse.Namespace) -> None:
    left = json.loads(args.left.read_text(encoding="utf-8"))
    right = json.loads(args.right.read_text(encoding="utf-8"))
    equal = left.get("input_lock_sha256") == right.get("input_lock_sha256")
    print(json.dumps({"equal": equal, "left": left.get("input_lock_sha256"), "right": right.get("input_lock_sha256")}))
    if not equal:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("lock")
    make.add_argument("--checkpoint", type=Path, required=True)
    make.add_argument("--checkpoint-role", choices=("stage2", "post-grpo", "other"), required=True)
    make.add_argument("--stage3", type=Path)
    make.add_argument("--fast-manifest", type=Path)
    make.add_argument("--full-manifest", type=Path)
    make.add_argument("--setting", action="append", default=[])
    make.add_argument("--output", type=Path, required=True)
    make.set_defaults(function=lock)
    check = commands.add_parser("compare")
    check.add_argument("left", type=Path)
    check.add_argument("right", type=Path)
    check.set_defaults(function=compare)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
