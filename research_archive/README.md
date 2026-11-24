# Curated research archive

This directory preserves the compact, source-level ACD material originally
developed alongside the larger `KD` workspace.  It is included for inspection,
figure regeneration, and protocol provenance; the maintained implementation is
the repository root ([`acd/`](../acd/), [`train.py`](../train.py), and
[`configs/`](../configs/)).

## Contents and evidence status

- [`figures/`](figures/) contains editable figure scripts and rendered assets.
  It deliberately excludes disposable analysis caches and recovered model
  checkpoints.  The ImageNet Grad-CAM visual is a CE-versus-KD baseline
  diagnostic, **not** an ACD result.
- [`tables/main_results_20261116/`](tables/main_results_20261116/) contains the
  paper-table sources, historical result records, and release-recovery metadata.
  Rows marked historical or user-reported are not substitutes for a verified
  checkpoint/configuration pair.
- [`reproductions/cifar100_acd_rerun_20261116/`](reproductions/cifar100_acd_rerun_20261116/)
  is a frozen reconstruction used for the CIFAR-100 rerun.  It is not claimed
  to be a bitwise recovery of the historical training source.
- [`reproductions/imagenet1k_preflight_20261113/`](reproductions/imagenet1k_preflight_20261113/)
  contains the ImageNet protocol and preflight evidence, without ImageNet data,
  environments, or model weights.
- [`reports/`](reports/) preserves dated research notes and gap analyses.  They
  are context, not current release claims.

Raw datasets, downloaded archives, environments, tensor caches, and `.pth` / `.pt`
weights are intentionally omitted.  Portable, checksum-manifested models are
published through the GitHub Releases listed in
[`docs/model_guide.md`](../docs/model_guide.md).
