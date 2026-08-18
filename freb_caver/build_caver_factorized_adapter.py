#!/usr/bin/env python3
"""Factor visual-estimator drift from recurrent-writer strength in CAVER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from build_caver_homotopy_adapter import (
    CAVER_ARCHITECTURE,
    GRAFT_ARCHITECTURE,
    load_tensors,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graft-dir", type=Path, required=True)
    parser.add_argument("--caver-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-alpha", type=float, required=True)
    parser.add_argument("--writer-alpha", type=float, required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.visual_alpha <= 1.0:
        raise ValueError("visual alpha must be in [0,1]")
    if not 0.0 <= args.writer_alpha <= 1.0:
        raise ValueError("writer alpha must be in [0,1]")

    gi_path=args.graft_dir/"native_deep_residual_identity.json"
    ci_path=args.caver_dir/"native_deep_residual_identity.json"
    gw_path=args.graft_dir/"native_deep_residual.safetensors"
    cw_path=args.caver_dir/"native_deep_residual.safetensors"
    gi=json.loads(gi_path.read_text(encoding="utf-8"))
    ci=json.loads(ci_path.read_text(encoding="utf-8"))
    if gi["statistics"]["architecture"]!=GRAFT_ARCHITECTURE:
        raise ValueError("invalid GRAFT source")
    if ci["statistics"]["architecture"]!=CAVER_ARCHITECTURE:
        raise ValueError("invalid CAVER source")
    if gi["weights_sha256"]!=sha256(gw_path) or ci["weights_sha256"]!=sha256(cw_path):
        raise ValueError("source identity mismatch")
    if ci.get("graft_seed_weights_sha256")!=gi["weights_sha256"]:
        raise ValueError("CAVER/GRAFT lineage mismatch")
    graft,caver=load_tensors(gw_path),load_tensors(cw_path)
    if not set(graft).issubset(caver):
        raise ValueError("CAVER is missing GRAFT tensors")

    output={}
    for name,value in caver.items():
        if name.startswith("visual_verdict."):
            output[name]=np.ascontiguousarray(
                graft[name]+args.visual_alpha*(value-graft[name]))
        elif name.startswith("belief_up."):
            output[name]=np.ascontiguousarray(args.writer_alpha*value)
        elif name in graft:
            if not np.array_equal(value,graft[name]):
                raise ValueError(f"unexpected trained common tensor: {name}")
            output[name]=np.ascontiguousarray(graft[name])
        else:
            output[name]=np.ascontiguousarray(value)

    args.output_dir.mkdir(parents=True,exist_ok=True)
    weights=args.output_dir/"native_deep_residual.safetensors"
    temporary=weights.with_suffix(".tmp")
    save_file(output,temporary,metadata={
        "format":"pt","schema":str(ci["schema_version"]),
        "transformation":"graft-caver-factorized-residual-v1"})
    temporary.replace(weights)
    identity=json.loads(json.dumps(ci))
    identity["weights_sha256"]=sha256(weights)
    identity["parameter_count"]=int(sum(value.size for value in output.values()))
    identity["screen_pass"]=False
    identity["homotopy"]={
        "schema_version":"graft-caver-residual-homotopy-v1",
        "factorized_schema_version":"graft-caver-factorized-residual-v1",
        "alpha":{"visual":args.visual_alpha,"writer":args.writer_alpha},
        "variant":args.variant,
        "graft_weights_sha256":gi["weights_sha256"],
        "caver_weights_sha256":ci["weights_sha256"],
        "external_router":False,"inference_threshold":False,
    }
    identity["statistics"]["factorized_variant"]=args.variant
    identity["statistics"]["visual_alpha"]=args.visual_alpha
    identity["statistics"]["writer_alpha"]=args.writer_alpha
    identity_path=args.output_dir/"native_deep_residual_identity.json"
    identity_path.write_text(json.dumps(identity,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"variant":args.variant,"visual_alpha":args.visual_alpha,
      "writer_alpha":args.writer_alpha,"weights_sha256":identity["weights_sha256"],
      "parameter_count":identity["parameter_count"]},sort_keys=True))


if __name__=="__main__":
    main()
