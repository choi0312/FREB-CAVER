#!/usr/bin/env python3
"""Extract only the pinned MMAD archive members referenced by a manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path, PurePosixPath
import struct
import time
import zlib
import zipfile

import requests

MMAD_REPO = "jiang-cc/MMAD"
MMAD_REVISION = "c4ed190dcb530f2f673ab293e575ad32054bb3cf"
MMAD_ARCHIVES = {
    "DS-MVTec.zip": (1663225174, "1b0ff7b548021e977a755b02167c069dc03e701718d4436e635071f7e1f899f3"),
    "GoodsAD.zip": (13338776535, "d80d03b2f4c9d8ab8f54322d0bcd48ef407e27bca8199c4abdd3a4e3112c29ef"),
    "MVTec-AD.zip": (5273925193, "0a4ff072fafcfc4bccce68c0657b5d3890cb3c01737c66ab36259f806438faf2"),
    "MVTec-LOCO.zip": (6131998488, "e2bbcb234fcb7478199dde32a861642eadade7c10ea762958bc7d762f3cd82e9"),
    "VisA.zip": (1916719940, "f8a0511b7c2231fbfc16d59f1e00fc20c6eafcaff262b3f64805273e404a2e24"),
}
ROOT_TO_ARCHIVE = {
    "DS-MVTec": "DS-MVTec.zip",
    "GoodsAD": "GoodsAD.zip",
    "MVTec-AD": "MVTec-AD.zip",
    "MVTec-LOCO": "MVTec-LOCO.zip",
    "VisA": "VisA.zip",
}
LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50


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


def get_range(url: str, start: int, end: int, *, retries: int = 8) -> bytes:
    if start < 0 or end < start:
        raise ValueError(f"invalid byte range {start}-{end}")
    expected = end - start + 1
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(30, 240),
            )
            with response:
                response.raise_for_status()
                content_range = response.headers.get("content-range", "")
                if response.status_code != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise ValueError(f"unexpected range response: {response.status_code} {content_range}")
                payload = response.content
            if len(payload) != expected:
                raise ValueError(f"short range response: {len(payload)} != {expected}")
            return payload
        except (requests.RequestException, ValueError):
            if attempt == retries:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unknown whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        end = min(self.size, self.position + size) - 1
        payload = get_range(self.url, self.position, end)
        self.position += len(payload)
        return payload


@dataclass(frozen=True)
class DownloadGroup:
    start: int
    end: int
    members: tuple[zipfile.ZipInfo, ...]


def read_manifest_paths(manifest: Path) -> dict[str, set[str]]:
    required = {archive: set() for archive in MMAD_ARCHIVES}
    rows = 0
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        rows += 1
        for field in ("image", "template_image"):
            value = str(row[field]).replace("\\", "/").lstrip("./")
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
                raise ValueError(f"unsafe MMAD path in {field}: {value}")
            archive = ROOT_TO_ARCHIVE.get(path.parts[0])
            if archive is None:
                raise ValueError(f"unknown MMAD archive root: {path.parts[0]}")
            required[archive].add(path.as_posix())
    if rows == 0:
        raise ValueError("empty manifest")
    return {archive: paths for archive, paths in required.items() if paths}


def crc32_file(path: Path) -> tuple[int, int]:
    checksum = 0
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            size += len(block)
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF, size


def is_valid_existing(destination: Path, info: zipfile.ZipInfo) -> bool:
    if not destination.is_file() or destination.stat().st_size != info.file_size:
        return False
    checksum, size = crc32_file(destination)
    return size == info.file_size and checksum == info.CRC


def resolve_members(url: str, archive_size: int, wanted: set[str]) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(HTTPRangeReader(url, archive_size)) as zipped:
        files = [item for item in zipped.infolist() if not item.is_dir()]
    exact = {item.filename.replace("\\", "/").lstrip("./"): item for item in files}
    folded: dict[str, zipfile.ZipInfo | None] = {}
    for name, info in exact.items():
        key = name.casefold()
        folded[key] = info if key not in folded else None
    selected = []
    missing = []
    for name in sorted(wanted):
        info = exact.get(name)
        if info is None:
            info = folded.get(name.casefold())
        if info is None:
            missing.append(name)
        else:
            selected.append(info)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} archive members missing; first: {preview}")
    return selected


def build_groups(
    members: list[zipfile.ZipInfo],
    archive_size: int,
    max_gap_bytes: int,
    max_group_bytes: int,
) -> list[DownloadGroup]:
    # The range begins at each local header, so the exact filename/extra lengths can
    # be parsed after download. 131 KiB is the ZIP-format upper bound for both fields.
    header_slack = LOCAL_HEADER.size + 2 * 65535
    groups: list[DownloadGroup] = []
    current_start = -1
    current_end = -1
    current_members: list[zipfile.ZipInfo] = []
    for info in sorted(members, key=lambda item: item.header_offset):
        if info.flag_bits & 0x1:
            raise ValueError(f"encrypted archive member is unsupported: {info.filename}")
        start = info.header_offset
        end = min(archive_size - 1, start + header_slack + info.compress_size - 1)
        proposed_end = max(current_end, end)
        can_merge = (
            current_members
            and start <= current_end + max_gap_bytes
            and proposed_end - current_start + 1 <= max_group_bytes
        )
        if current_members and not can_merge:
            groups.append(DownloadGroup(current_start, current_end, tuple(current_members)))
            current_start, current_end, current_members = start, end, [info]
        elif current_members:
            current_end = proposed_end
            current_members.append(info)
        else:
            current_start, current_end, current_members = start, end, [info]
    if current_members:
        groups.append(DownloadGroup(current_start, current_end, tuple(current_members)))
    return groups


def extract_member(payload: bytes, group: DownloadGroup, info: zipfile.ZipInfo, root: Path) -> str:
    relative_header = info.header_offset - group.start
    fields = LOCAL_HEADER.unpack_from(payload, relative_header)
    signature, flag_bits, method = fields[0], fields[2], fields[3]
    filename_length, extra_length = fields[9], fields[10]
    if signature != LOCAL_FILE_HEADER_SIGNATURE:
        raise ValueError(f"bad local header signature for {info.filename}")
    if flag_bits & 0x1 or method != info.compress_type:
        raise ValueError(f"unsupported local header for {info.filename}")
    data_start = relative_header + LOCAL_HEADER.size + filename_length + extra_length
    data_end = data_start + info.compress_size
    compressed = payload[data_start:data_end]
    if len(compressed) != info.compress_size:
        raise ValueError(f"truncated member payload for {info.filename}")
    if method == zipfile.ZIP_STORED:
        data = compressed
    elif method == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"unsupported compression method {method}: {info.filename}")
    if len(data) != info.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
        raise ValueError(f"CRC/size mismatch for {info.filename}")
    relative = PurePosixPath(info.filename.replace("\\", "/").lstrip("./"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive member path: {info.filename}")
    destination = root.joinpath(*relative.parts)
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()
    if resolved_root not in resolved_destination.parents:
        raise ValueError(f"archive member escaped destination: {info.filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_bytes(data)
    os.replace(temporary, destination)
    return relative.as_posix()


def extract_group(url: str, group: DownloadGroup, root: Path) -> list[str]:
    payload = get_range(url, group.start, group.end)
    return [extract_member(payload, group, info, root) for info in group.members]


def process_archive(
    archive: str,
    wanted: set[str],
    root: Path,
    workers: int,
    max_gap_bytes: int,
    max_group_bytes: int,
    plan_only: bool,
) -> dict[str, object]:
    archive_size, archive_sha256 = MMAD_ARCHIVES[archive]
    url = resolve_range_url(archive, archive_size)
    members = resolve_members(url, archive_size, wanted)
    pending = []
    reused = []
    for info in members:
        destination = root.joinpath(*PurePosixPath(info.filename.replace("\\", "/").lstrip("./")).parts)
        if is_valid_existing(destination, info):
            reused.append(info.filename)
        else:
            pending.append(info)
    groups = build_groups(pending, archive_size, max_gap_bytes, max_group_bytes)
    downloaded_bytes = sum(group.end - group.start + 1 for group in groups)
    extracted: list[str] = []
    if not plan_only and groups:
        with ThreadPoolExecutor(max_workers=min(workers, len(groups))) as pool:
            for paths in pool.map(lambda group: extract_group(url, group, root), groups):
                extracted.extend(paths)
    return {
        "archive": archive,
        "archive_expected_size": archive_size,
        "archive_expected_sha256": archive_sha256,
        "required_members": len(members),
        "reused_members": len(reused),
        "extracted_members": len(extracted),
        "range_groups": len(groups),
        "range_bytes": downloaded_bytes,
        "plan_only": plan_only,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mmad-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--archive-workers", type=int, default=1)
    parser.add_argument("--max-gap-mib", type=int, default=8)
    parser.add_argument("--max-group-mib", type=int, default=256)
    parser.add_argument("--archive", action="append", dest="archive_names")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.archive_workers < 1 or args.max_gap_mib < 0 or args.max_group_mib < 1:
        parser.error("invalid worker/range settings")
    required = read_manifest_paths(args.manifest)
    if args.archive_names:
        unknown = set(args.archive_names) - set(MMAD_ARCHIVES)
        if unknown:
            parser.error(f"unknown archives: {sorted(unknown)}")
        required = {name: paths for name, paths in required.items() if name in set(args.archive_names)}
        if not required:
            parser.error("selected archives contain no manifest members")
    args.mmad_root.mkdir(parents=True, exist_ok=True)
    items = sorted(required.items())

    def run(item: tuple[str, set[str]]) -> dict[str, object]:
        archive, wanted = item
        result = process_archive(
            archive,
            wanted,
            args.mmad_root,
            args.workers,
            args.max_gap_mib * 1024 * 1024,
            args.max_group_mib * 1024 * 1024,
            args.plan_only,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return result

    if args.archive_workers == 1:
        results = [run(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=min(args.archive_workers, len(items))) as pool:
            results = list(pool.map(run, items))
    ledger = {
        "schema_version": "judo-mmad-manifest-subset-v1",
        "repo_id": MMAD_REPO,
        "revision": MMAD_REVISION,
        "manifest": str(args.manifest.resolve()),
        "required_members": sum(len(paths) for paths in required.values()),
        "plan_only": args.plan_only,
        "archives": results,
    }
    if args.ledger:
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        args.ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(ledger, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
