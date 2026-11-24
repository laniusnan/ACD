# Code, evidence, and historical-material map

The active ACD implementation is [`acd/`](../acd/), with the CLI in
[`train.py`](../train.py), documented workflows in [`usage.md`](usage.md), and
versioned experimental settings in [`configs/`](../configs/).  This page keeps
the status of code, models, and inherited paper material explicit.

## Reusable, checksum-backed assets

Two published CIFAR-100 Releases are the supported model entry points:

- [`cifar100-rerun-20261116`](https://github.com/laniusnan/ACD/releases/tag/cifar100-rerun-20261116):
  frozen-rerun teachers and ACD student weights, with the manifest and
  verification record in [`release_publication.json`](release_publication.json).
- [`cifar100-head-controls-20261031`](https://github.com/laniusnan/ACD/releases/tag/cifar100-head-controls-20261031):
  ordinary-KD, cosine, and ETF head controls; see
  [`head_controls_publication.json`](head_controls_publication.json).
- [`cifar100-seven-arms-res32x4-res8x4-20261118`](https://github.com/laniusnan/ACD/releases/tag/cifar100-seven-arms-res32x4-res8x4-20261118):
  a fully verified, matched seven-arm ResNet32×4 → ResNet8×4 comparison
  (KD, cosine/ETF heads, and SICD); see its
  [release note](releases/cifar100-seven-arms-res32x4-res8x4-20261118.md)
  and [`seven_arm_release_publication_20261124.json`](seven_arm_release_publication_20261124.json).

Each release includes portable evaluation weights, SHA-256 checksums, source
checkpoint provenance, and archived test metrics.  Use
[`tools/download_models.py`](../tools/download_models.py) and
[`tools/verify_release.py`](../tools/verify_release.py) rather than treating a
training checkpoint as a release artifact.

## Current formal training evidence

The seven-arm CIFAR-100 study is defined in
[`configs/studies/cifar100_seven_arms_20261118.json`](../configs/studies/cifar100_seven_arms_20261118.json)
and described in [`seven_arm_training_20261118.md`](seven_arm_training_20261118.md).
Its completed results, selection records, and checkpoints remain in the ignored
`outputs/seven_arms_20261118/` working area while the queue continues.  The
checked-in validation summary is
[`seven_arm_training_validation_20261118.json`](seven_arm_training_validation_20261118.json).
Do not combine these new single-seed runs with older paper rows as though they
were the same recipe.

## Historical archive

[`research_archive/`](../research_archive/) is a curated migration from the KD
workspace: editable figures, table sources, frozen CIFAR reconstruction,
ImageNet preflight material, and dated research notes.  It excludes raw data,
checkpoint blobs, downloaded release tarballs, and analysis caches.  Its
[`README`](../research_archive/README.md) records scope and caveats.

In particular, historical results or user-transcribed rows without a complete
seed/configuration/checkpoint trail remain record-only evidence.  The
checkpoint audit ([`checkpoint_audit_20261118.md`](checkpoint_audit_20261118.md))
and manuscript review ([`manuscript_review_20261119.md`](manuscript_review_20261119.md))
should be consulted before making paper claims.
