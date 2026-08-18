# Method

## Problem

JUDO can produce useful localization and reasoning traces while still making a
binary anomaly decision inconsistent with the visual comparison. FREB-CAVER
targets that representation-to-decision gap without replacing the base model,
adding a post-hoc router, or tuning an inference threshold.

## 1. Frozen native visual tokens

The query and normal reference pass through the same frozen JUDO vision tower.
FREB-CAVER consumes the resulting native patch sequences, avoiding a separate
comparison encoder whose feature geometry may not match the language decoder.

## 2. Defect-preserving partial transport

A learned low-rank similarity aligns query patches to reference patches. A
dustbin/unmatched path allows query mass to remain unmatched instead of forcing
every defect patch onto a normal patch. The transport module is reference-order
equivariant and yields three evidence types:

- signed query-minus-reference difference;
- mismatch magnitude/energy;
- matched agreement.

## 3. GRAFT dual evidence

Grounded Reference-Anchored Faithful Tuning (GRAFT) keeps unmatched defect
evidence and matched normal-agreement evidence in separate branches. A visual
verdict head maps their summaries to normal/anomaly logits. The two branches
are replayed only after the final assistant `<seg>` state and at the answer
decision boundary.

## 4. CAVER recurrent binding

Causal Anomaly Visual-Evidence Recurrent Binding (CAVER) converts the visual
verdict into a signed belief. At decoder layers 18, 20, 22, 24, and 26, a
prompt-conditioned low-rank writer injects this belief into grounded CoT states
and the answer state. Per-site relative-RMS trust regions cap the recurrent CoT
residual at 0.02 and the answer residual at 0.08.

The output projections are initialized to zero, so installation changes no
base-model output before continuation training. Token-phase masks prevent the
system prompt from receiving the replay signal.

## 5. Interventional training

Training uses same-anchor triplets:

1. an anomalous query;
2. a hard normal query;
3. a normal-null intervention where the query is replaced by the normal
   reference while the question and option order are preserved.

Direct, partial-CoT, and full-CoT views are optimized jointly with:

- answer cross-entropy;
- direct-to-partial-to-full margin persistence;
- anomaly > hard-normal > normal-null ordinal ranking;
- visual-belief supervision;
- semantic answer/belief binding;
- periodic frozen-JUDO KL preservation on non-AD tasks.

## 6. Final factorization

The validation screen compared three sealed variants. The released
`frozen_writer050` checkpoint freezes the learned visual estimator and applies
half of the recurrent writer displacement. This isolates the improvement to
recurrent evidence binding rather than continued visual-head fitting.

## Scientific boundary

FREB-CAVER is a model-native adapter and can be packaged as a standard
checkpoint artifact. The current evidence supports a modest overall paired
MMAD improvement. It does not support the stronger claim that anomaly
over-detection has been solved; the independent AD balanced-accuracy endpoint
was unchanged.
