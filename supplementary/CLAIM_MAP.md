# Main-paper claim map for the supplement

This map aligns the author manuscript `acd_v1.pdf` with the material that can
currently appear in a reproducible supplement.  It is an editorial working
document, not a paper section to upload unchanged.

| Manuscript topic | Supplement evidence | Current strength | Recommended paper wording |
| --- | --- | --- | --- |
| Classifier parameterization changes KD supervision | S2/S3 in `ASSETS.md`, fixed-direction and fixed-checkpoint probes | Local diagnostic evidence | Parameterization can change output distributions and local KD gradients under the documented interventions. |
| Output discrepancy exceeds scalar confidence | S1 and the oracle subsection in `supplementary.tex` | One frozen CIFAR-100 endpoint, all test images | Per-image nonnegative scale calibration does not fully explain the discrepancy for this endpoint. |
| ACI changes the KD outcome | Table S1: affine KD 74.76, cosine KD 77.30 | One matched pair and seed | In the matched ResNet32x4-to-ResNet8x4 run, replacing the affine head with a cosine head increases endpoint Top-1 by 2.54 points. |
| SICD contributes after ACI | Table S1: cosine KD/SICD and ETF variants | One matched pair and seed; mixed sign by head | SICD has an initialization-dependent effect in the completed comparison; the best endpoint is 0.20 points above cosine KD. |
| ETF improves or preserves an optimal geometry during training | ETF-init and rotating-ETF rows in Table S1; implementation section | Geometry implementation is verified; performance result is one seed | ETF initialization and rotating ETF are distinct controls. Do not state that free ETF initialization preserves the ETF Gram or that ETF is uniformly beneficial. |
| ACD is state of the art or consistently improves KD | No complete, protocol-aligned multi-pair/multi-seed comparison in the current release | Not supported by the new supplement evidence | Restrict to reported settings until aligned multi-seed comparisons are available. |
| ACD is a universal drop-in replacement | Two historical compatibility records, with mixed effects; no current detection endpoint | Limited historical evidence | Report the particular compatible pipelines and their protocols; do not generalize to arbitrary distillation pipelines. |
| No additional inference cost | Cosine head changes only the classifier computation; gradient matching runs in training | Implementation statement | State that gradient matching is training-only and report measured training cost separately; do not infer equal training cost. |

## Terminology to keep consistent

- **ACI**: bias-free cosine student classifier with fixed scale and trainable
  directions.
- **ETF initialization**: an initial simplex configuration followed by freely
  trainable cosine directions.
- **Rotating ETF**: a distinct constrained classifier that preserves simplex
  geometry through a learned Stiefel frame.
- **SICD**: dual-side standardized, teacher-mass-weighted conditional
  non-target KL. In the released seven-arm study, full ACD also includes
  gradient-norm matching to standardized full KL.
- **ACD**: the selected combination of ACI and SICD; state the ETF variant
  explicitly instead of treating it as an implicit part of the name.

## Tables that should remain separate

Table S1 is a matched factorial control for one seed and one pair. Table S2
is a set of released frozen ACD endpoints for evaluation and reproducibility.
Historical multi-run manuscript rows, seed-3201 reruns, and new seed-3301
controls use different provenance and should not be averaged or ranked as a
single benchmark without a source/configuration alignment audit.
