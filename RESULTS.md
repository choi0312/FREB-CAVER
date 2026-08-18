# Results

## Primary: independent asset-disjoint paired holdout

The candidate was fixed after a three-variant validation screen, then evaluated
once on a balanced 1,120-question holdout with 28 source-task cells and 1,601
unique assets. The public JUDO baseline and FREB-CAVER used the same manifest,
generation configuration, and official CoT answer contract.

| Metric | JUDO | FREB-CAVER | Change |
|---|---:|---:|---:|
| Overall / 28-cell macro | 80.0893% | **81.0714%** | **+0.9821 pp** |
| Anomaly Detection | 65.6250% | 65.6250% | 0.0000 pp |
| Defect Analysis | 87.5000% | **88.7500%** | +1.2500 pp |
| Defect Classification | 70.6250% | **71.2500%** | +0.6250 pp |
| Defect Description | **86.8750%** | 86.2500% | -0.6250 pp |
| Defect Localization | 70.0000% | **73.1250%** | +3.1250 pp |
| Object Analysis | 85.0000% | **87.5000%** | +2.5000 pp |
| Object Classification | 95.0000% | 95.0000% | 0.0000 pp |

Paired outcomes: 18 rescues, 7 regressions, and 11 net rescues. Exact McNemar
used 25 discordant pairs and returned `p=0.0432852507`. A 10,000-replicate
query-reference-component bootstrap returned a mean change of `+0.9828 pp` and
a 95% interval of `+0.0908` to `+1.8683 pp`.

AD anomaly recall and normal specificity were both unchanged at 86.25% and
45.00%, respectively. Under the sealed rule requiring both overall and AD
balanced improvement, the joint claim status is **not confirmed**.

## Secondary: full-MMAD behavior audit

| Metric | FREB-CAVER |
|---|---:|
| Questions | 39,670 / 39,670 |
| Normalized 28-cell macro | 81.9063% |
| Micro accuracy | 81.2150% |
| Strict-format 28-cell macro | 81.8989% |
| Strict-format invalid generations | 9 / 39,670 |

### Source task-macro

| Source | Accuracy |
|---|---:|
| GoodsAD | 79.1896% |
| MVTec-AD | 89.6923% |
| MVTec-LOCO | 77.1433% |
| VisA | 81.6000% |

### Task source-macro

| Task | Accuracy |
|---|---:|
| Anomaly Detection | 65.4343% |
| Defect Analysis | 89.7165% |
| Defect Classification | 75.2367% |
| Defect Description | 85.0844% |
| Defect Localization | 74.1183% |
| Object Analysis | 88.2244% |
| Object Classification | 95.5294% |

### AD diagnostic

The raw AD accuracy was 69.2781% over 8,297 questions. Anomaly recall was high
at 84.7632%, but normal specificity was only 45.2308% (false-positive rate
54.7692%), giving 64.9970% balanced accuracy. Over-detection therefore remains
the principal AD failure.

## Interpretation limits

The full-MMAD score is not an independent generalization estimate. Every one of
the 5,600 training sample IDs and 560 validation sample IDs occurs in the full
39,670-question manifest: 6,160 overlapping rows, or 15.5281%. It is retained
as a completeness, parser, runtime, and behavior audit. The 1,120-row
asset-disjoint paired holdout is the appropriate inferential result.

The published JUDO paper reports 81.20% average performance. FREB-CAVER's full
audit is descriptively +0.71 pp above that figure, but protocol and overlap
differences prevent treating the comparison as a clean head-to-head claim.

Machine-readable values are preserved in `results/`.
