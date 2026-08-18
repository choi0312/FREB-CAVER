# Reproducibility

## Fixed identities

- Base: `woodavid31/JUDO`
- Base revision: `b308d1cb07130d265a51147183cf94607fa07116`
- Final adapter SHA-256: `ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed`
- Independent holdout SHA-256: `ba0cd97dbc9c17395bd0d471db17dcf1ec83428eb9ad5ae9b8555f703233f7ec`
- Full-evaluation frozen manifest SHA-256: `01239440d51538450d50ccbc73866fd3685b0dfd106198921df44cf9ced547e8`

The source hashes inside `configs/` identify the sealed execution bundle used
for the reported experiments. This public repository preserves the scientific
model, trainer, and CAVER evaluator sources, while reorganizing them under
`freb_caver/`. One packaging-only change was made to `eval_manifest.py`: it
discovers the separately cloned upstream evaluator through `JUDO_REPO` instead
of assuming that upstream JUDO code is copied into this repository. Unit tests
that depended on non-redistributable MMAD fixtures were replaced by synthetic,
data-free invariant tests. The release tag and Git commit are the identities of
the public package; the preregistration hashes remain the identities of the
historical run bundle.

## Environment

The full run used one NVIDIA RTX PRO 6000 Blackwell Server Edition exposed by a
Google/Colab G4 session. Peak allocation was 17,769 MiB and recorded inference
time was 7:19:52 for 39,670 questions. End-to-end elapsed time including asset
preparation and packaging was approximately 8:19.

Install a GPU-compatible PyTorch build first. The recorded Python dependency
contract is in `requirements.txt`; the model path was evaluated with BF16 and
SDPA.

## Data

This repository does not redistribute MMAD. Prepare the dataset under the path
layout expected by its official release. The utilities in `freb_caver/` can
validate a frozen JSONL manifest and selectively materialize required archive
members. Keep query, reference, option order, labels, and sample IDs unchanged.

Training and validation source rows are intentionally absent because they
contain dataset paths and teacher traces derived from restricted assets. Their
counts and hashes are sealed in `configs/caver_stage4b_preregistration.json`.

## Evaluation

Clone the upstream JUDO repository and expose its official evaluator:

```bash
git clone https://github.com/woodavid31/JUDO third_party/JUDO
export JUDO_REPO="$PWD/third_party/JUDO"
```

Download the base model and the release adapter, then set:

```bash
export JUDO_CAVER_ADAPTER_DIR="$PWD/checkpoints/FREB-CAVER"
```

Run `freb_caver/eval_judo_caver.py` with a frozen manifest. Evaluation writes
append-safe predictions, batch timing segments, metrics, run configuration,
and a CAVER runtime attestation. An interrupted run resumes only after checking
sample-ID uniqueness and repairing a torn final JSONL record.

## Acceptance checks

Before a scientific run, verify:

1. base, adapter, manifest, and asset hashes;
2. exact zero-function identity for a fresh adapter;
3. nonzero gradients in the visual-verdict and recurrent writer paths;
4. zero gradients in the frozen JUDO backbone;
5. phase masks exclude the system prompt;
6. one saved optimizer checkpoint after the first valid step;
7. strict and fallback parsers are both reported.

The local unit suite covers the architecture-level invariants. A full
autoregressive reproduction additionally requires the JUDO checkpoint, MMAD,
and a CUDA GPU.
