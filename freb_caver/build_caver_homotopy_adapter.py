#!/usr/bin/env python3
"""Build one model-native checkpoint on the GRAFT-to-CAVER residual path.

This is a diagnostic transformation, not an inference ensemble.  Parameters
shared with GRAFT are kept exactly, except for the trained visual-verdict head
which is interpolated.  CAVER's zero-initialized output writers are scaled by
the same fixed alpha, while their learned feature extractors remain intact.
The result is one ordinary CAVER safetensors checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


GRAFT_ARCHITECTURE = "grounded-reference-anchored-faithful-tuning-v1"
CAVER_ARCHITECTURE = "causal-anomaly-visual-evidence-recurrent-binding-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensors(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="np") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graft-dir", type=Path, required=True)
    parser.add_argument("--caver-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")

    graft_identity_path = args.graft_dir / "native_deep_residual_identity.json"
    caver_identity_path = args.caver_dir / "native_deep_residual_identity.json"
    graft_weights_path = args.graft_dir / "native_deep_residual.safetensors"
    caver_weights_path = args.caver_dir / "native_deep_residual.safetensors"
    graft_identity = json.loads(graft_identity_path.read_text(encoding="utf-8"))
    caver_identity = json.loads(caver_identity_path.read_text(encoding="utf-8"))
    if graft_identity["statistics"]["architecture"] != GRAFT_ARCHITECTURE:
        raise ValueError("invalid GRAFT source architecture")
    if caver_identity["statistics"]["architecture"] != CAVER_ARCHITECTURE:
        raise ValueError("invalid CAVER source architecture")
    if graft_identity["weights_sha256"] != sha256(graft_weights_path):
        raise ValueError("GRAFT source identity mismatch")
    if caver_identity["weights_sha256"] != sha256(caver_weights_path):
        raise ValueError("CAVER source identity mismatch")
    if caver_identity.get("graft_seed_weights_sha256") != graft_identity["weights_sha256"]:
        raise ValueError("CAVER was not trained from the supplied GRAFT seed")

    graft, caver = load_tensors(graft_weights_path), load_tensors(caver_weights_path)
    if not set(graft).issubset(caver):
        raise ValueError("CAVER is missing GRAFT parameters")
    output: dict[str, np.ndarray] = {}
    changed_common: list[str] = []
    for name, value in caver.items():
        if name.startswith("visual_verdict."):
            if name not in graft:
                raise ValueError(f"visual verdict tensor missing in GRAFT: {name}")
            output[name] = np.ascontiguousarray(
                graft[name] + args.alpha * (value - graft[name])
            )
            changed_common.append(name)
        elif name.startswith("belief_up."):
            # belief_up was initialized to zero, so this scales the actual
            # recurrent residual without destroying the learned evidence basis.
            output[name] = np.ascontiguousarray(args.alpha * value)
        elif name in graft:
            if not np.array_equal(value, graft[name]):
                raise ValueError(f"unexpected trained GRAFT parameter: {name}")
            output[name] = np.ascontiguousarray(graft[name])
        else:
            output[name] = np.ascontiguousarray(value)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "native_deep_residual.safetensors"
    temporary = weights_path.with_suffix(".tmp")
    save_file(
        output,
        temporary,
        metadata={
            "format": "pt",
            "schema": str(caver_identity["schema_version"]),
            "transformation": "graft-caver-residual-homotopy-v1",
        },
    )
    temporary.replace(weights_path)
    identity = json.loads(json.dumps(caver_identity))
    identity["weights_sha256"] = sha256(weights_path)
    identity["parameter_count"] = int(sum(value.size for value in output.values()))
    identity["screen_pass"] = False
    identity["homotopy"] = {
        "schema_version": "graft-caver-residual-homotopy-v1",
        "alpha": args.alpha,
        "graft_weights_sha256": graft_identity["weights_sha256"],
        "caver_weights_sha256": caver_identity["weights_sha256"],
        "scaled_parameter_groups": ["visual_verdict_delta", "belief_up"],
        "preserved_parameter_groups": ["graft_backbone", "caver_evidence_basis"],
        "external_router": False,
        "inference_threshold": False,
    }
    identity["statistics"]["homotopy_alpha"] = args.alpha
    identity["statistics"]["homotopy_transformation"] = (
        "graft-caver-residual-homotopy-v1"
    )
    identity_path = args.output_dir / "native_deep_residual_identity.json"
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "schema_version": "graft-caver-residual-homotopy-build-v1",
                "alpha": args.alpha,
                "weights_sha256": identity["weights_sha256"],
                "parameter_count": identity["parameter_count"],
                "changed_common_tensors": changed_common,
                "output_dir": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
