#!/usr/bin/env python3
"""Download the pinned MMAD release and/or an explicitly selected checkpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

from huggingface_hub import hf_hub_download, snapshot_download
import requests


MMAD_REPO = "jiang-cc/MMAD"
MMAD_REVISION = "c4ed190dcb530f2f673ab293e575ad32054bb3cf"
MMAD_JSON_SHA256 = "639343b491bc67b2abb3c5d719f221ce27f83b2ed97948f4e88055aaa31f1c1e"
MMAD_FILES = (
    "mmad.json",
    "metadata.csv",
    "domain_knowledge.json",
    "DS-MVTec.zip",
    "MVTec-AD.zip",
    "MVTec-LOCO.zip",
    "VisA.zip",
    "GoodsAD.zip",
)
MMAD_ARCHIVES = {
    "DS-MVTec.zip": (1663225174, "1b0ff7b548021e977a755b02167c069dc03e701718d4436e635071f7e1f899f3"),
    "GoodsAD.zip": (13338776535, "d80d03b2f4c9d8ab8f54322d0bcd48ef407e27bca8199c4abdd3a4e3112c29ef"),
    "MVTec-AD.zip": (5273925193, "0a4ff072fafcfc4bccce68c0657b5d3890cb3c01737c66ab36259f806438faf2"),
    "MVTec-LOCO.zip": (6131998488, "e2bbcb234fcb7478199dde32a861642eadade7c10ea762958bc7d762f3cd82e9"),
    "VisA.zip": (1916719940, "f8a0511b7c2231fbfc16d59f1e00fc20c6eafcaff262b3f64805273e404a2e24"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
        zipped.extractall(destination)


def resolve_range_url(filename: str, expected_size: int) -> str:
    url = (
        f"https://huggingface.co/datasets/{MMAD_REPO}/resolve/"
        f"{MMAD_REVISION}/{filename}?download=true"
    )
    response = requests.get(
        url,
        headers={"Range": "bytes=0-0"},
        stream=True,
        allow_redirects=True,
        timeout=120,
    )
    try:
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get("content-range") != f"bytes 0-0/{expected_size}":
            raise ValueError(f"range probe failed for {filename}: {response.status_code} {response.headers}")
        return response.url
    finally:
        response.close()


def download_range_segment(url: str, destination: Path, start: int, end: int) -> None:
    position = start
    descriptor = os.open(destination, os.O_WRONLY)
    try:
        failures = 0
        while position <= end:
            try:
                response = requests.get(
                    url,
                    headers={"Range": f"bytes={position}-{end}"},
                    stream=True,
                    timeout=(30, 180),
                )
                with response:
                    response.raise_for_status()
                    expected_prefix = f"bytes {position}-"
                    if response.status_code != 206 or not response.headers.get("content-range", "").startswith(expected_prefix):
                        raise ValueError(
                            f"range response mismatch: {response.status_code} {response.headers.get('content-range')}"
                        )
                    for block in response.iter_content(8 * 1024 * 1024):
                        if not block:
                            continue
                        if position + len(block) - 1 > end:
                            raise ValueError("range response exceeded its declared segment")
                        written = os.pwrite(descriptor, block, position)
                        if written != len(block):
                            raise OSError("short positional write")
                        position += written
                failures = 0
            except (OSError, requests.RequestException, ValueError):
                failures += 1
                if failures > 8:
                    raise
                time.sleep(min(60, 2**failures))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ranged_archive_download(filename: str, directory: Path, connections: int) -> Path:
    expected_size, expected_sha = MMAD_ARCHIVES[filename]
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as handle:
        handle.truncate(expected_size)
    url = resolve_range_url(filename, expected_size)
    segments = []
    for index in range(connections):
        start = expected_size * index // connections
        end = expected_size * (index + 1) // connections - 1
        segments.append((url, temporary, start, end))
    with ThreadPoolExecutor(max_workers=connections) as pool:
        list(pool.map(lambda args: download_range_segment(*args), segments))
    if temporary.stat().st_size != expected_size or sha256(temporary) != expected_sha:
        raise ValueError(f"pinned archive integrity mismatch: {filename}")
    temporary.replace(destination)
    return destination


def prepare_mmad(
    destination: Path,
    metadata_only: bool,
    download_workers: int,
    range_connections: int,
    archive_names: list[str] | None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    selected_archives = list(MMAD_ARCHIVES) if archive_names is None else archive_names
    unknown = set(selected_archives) - set(MMAD_ARCHIVES)
    if unknown:
        raise ValueError(f"unknown MMAD archives: {sorted(unknown)}")
    names = list(MMAD_FILES[:3]) if metadata_only else [*MMAD_FILES[:3], *selected_archives]

    def download(name: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=MMAD_REPO,
                repo_type="dataset",
                revision=MMAD_REVISION,
                filename=name,
            )
        )

    if download_workers < 1:
        raise ValueError("download_workers must be positive")
    hub_names = names if range_connections == 0 else list(MMAD_FILES[:3])
    if download_workers == 1:
        hub_cached = [download(name) for name in hub_names]
    else:
        with ThreadPoolExecutor(max_workers=min(download_workers, len(hub_names))) as pool:
            hub_cached = list(pool.map(download, hub_names))
    cached_files = dict(zip(hub_names, hub_cached, strict=True))
    if range_connections:
        archive_root = destination / ".pinned_archives"
        with ThreadPoolExecutor(max_workers=len(selected_archives)) as pool:
            archive_paths = list(
                pool.map(
                    lambda name: ranged_archive_download(name, archive_root, range_connections),
                    selected_archives,
                )
            )
        cached_files.update(zip(selected_archives, archive_paths, strict=True))
    for name in names:
        cached = cached_files[name]
        target = destination / name
        if name.endswith(".zip"):
            safe_extract(cached, destination)
            if range_connections:
                cached.unlink()
        else:
            shutil.copy2(cached, target)
    if range_connections:
        (destination / ".pinned_archives").rmdir()
    metadata = destination / "mmad.json"
    actual = sha256(metadata)
    if actual != MMAD_JSON_SHA256:
        raise ValueError(f"unexpected MMAD metadata SHA-256: {actual}")
    print(json.dumps({"mmad_root": str(destination.resolve()), "mmad_json_sha256": actual}))


def prepare_checkpoint(repo_id: str, revision: str, destination: Path, role: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=destination,
    )
    provenance = {
        "repo_id": repo_id,
        "revision": revision,
        "role": role,
    }
    (destination / "judo_checkpoint_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checkpoint": str(destination.resolve()), **provenance}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmad-root", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--range-connections", type=int, default=0)
    parser.add_argument("--archive", action="append", dest="archive_names")
    parser.add_argument("--checkpoint-repo")
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument(
        "--checkpoint-role",
        choices=("stage2", "post-grpo", "other"),
        default="other",
    )
    args = parser.parse_args()
    if not args.mmad_root and not args.checkpoint_repo:
        parser.error("select --mmad-root and/or --checkpoint-repo")
    if args.mmad_root:
        if args.range_connections < 0:
            parser.error("--range-connections cannot be negative")
        prepare_mmad(
            args.mmad_root,
            args.metadata_only,
            args.download_workers,
            args.range_connections,
            args.archive_names,
        )
    if args.checkpoint_repo:
        if not args.checkpoint_revision or not args.checkpoint_output:
            parser.error("checkpoint download also needs --checkpoint-revision and --checkpoint-output")
        prepare_checkpoint(
            args.checkpoint_repo,
            args.checkpoint_revision,
            args.checkpoint_output,
            args.checkpoint_role,
        )


if __name__ == "__main__":
    main()
