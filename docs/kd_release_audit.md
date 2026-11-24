# KD Release audit for the ACD model collection

Audit date: **November 18, 2026**. Source account: **laniusnan**. Scope: the three published releases of `laniusnan/KD`, their 177 attached assets, model manifests, recovered completion records and selected weight files. This is not a claim about unpublished files or later experiments elsewhere.

Published supplement: [cifar100-head-controls-20261031](https://github.com/laniusnan/ACD/releases/tag/cifar100-head-controls-20261031). [Publication and public-download checks](head_controls_publication.json).

For the subsequent **11 pairs × 7 training settings** audit, including local files, rotating ETF checkpoints and missing training weights, see [the complete checkpoint coverage](checkpoint_audit_20261118.md).

## Findings

| Source Release | Assets | Relevant content | Decision |
|---|---:|---|---|
| `model-backup-20261027` | 162 | Pretrained teachers and early classifier experiments | Add missing VGG13 teacher and selected final linear/cosine/ETF controls |
| `kd-head-models-20261031-completed` | 8 | Completed ordinary-KD classifier studies and early snapshots | Add selected cosine/ETF pairs from final epoch 240 |
| `kd-head-models-20261031-final` | 7 | Incremental completions, overlapping metadata, rotating-ETF controls | Use completion evidence; exclude duplicate/rotating variants from this supplement |

The release manifests and recovered study records contain **no newly verified complete SICD/ACD checkpoint**. They predate the November ACD studies. A cosine head or ETF initialization by itself does not make an ordinary-KD model a complete ACD model.

The earlier metadata recovery found 111 distinct completed result records, including 25 later linear confirmations, but a result record does not establish that its weight is attached to a release. The later linear confirmation weights were not located in these releases and are not republished. Thirty overlapping records are not additional independent runs.

## Selected supplement

The new release `cifar100-head-controls-20261031` contains **11 historical student controls + 1 teacher**, approximately 120 MB:

- ResNet32×4 → ResNet8×4: affine, cosine and trainable ETF-initialized cosine, seed 2051.
- VGG13 → VGG8: affine and cosine, seed 2051.
- ResNet50 → MobileNetV1: cosine and trainable ETF-initialized cosine, seed 2101.
- VGG13 → MobileNetV1: cosine and trainable ETF-initialized cosine, seed 2101.
- WRN40-2 → ShuffleNetV1: cosine and trainable ETF-initialized cosine, seed 2101.
- Affine VGG13 teacher, shared by both VGG-teacher configurations.

This selection provides a complete three-way head example, a second affine/cosine example, compact-student initialization examples, and the missing teacher architecture. Within each selected study the earliest available seed is used consistently; final epoch 240 is selected without searching for the highest test score. Remaining repeated seeds and other pairs stay available in KD's original archives.

ResNet32×4, ResNet50 and WRN40-2 teachers already exist in `cifar100-rerun-20261116`; the supplement references those assets rather than uploading duplicate teacher files. It does not replace the seven full ACD reruns or change their scores.

## Validation and delivery

Selected files are recovered from verified archives, checked against original file size and SHA256, strictly loaded and exported as portable evaluation weights. Original tensors are compared exactly with the exports. All twelve new models were evaluated on the complete official CIFAR-100 test split. All eleven student scores match the completed native records; the VGG13 teacher reaches 74.64% Top-1 and 92.60% Top-5. All 33 uploaded assets match the local SHA256 digests, and an unauthenticated download check passed for a new student, the new teacher, a reused teacher and their configs.

The supplement includes a per-model evaluation config, fine-label order, model guide, manifest, archive receipt, sanitized source evidence, evaluation report, standalone checksum-verifying downloader and checksums. The VGG13 teacher evaluation config is separate from the ACD training configs. The release tag targets the existing public source commit; it does not silently publish pending local source changes.

See [model selection, scores and usage](head_controls_guide.md), [model manifest](head_controls_manifest.json), [evaluation report](head_controls_verification.json), and [release notes](releases/cifar100-head-controls-20261031.md).

## What remains missing

- Complete ACD runs for the four additional CIFAR-100 configurations, including the VGG pairs, remain unverified. Historical ordinary-KD controls cannot fill those benchmark cells.
- No verified ImageNet ACD model or detector was found in the audited releases.
- Later paired linear baselines require their actual checkpoints before they can be distributed with the cosine/ETF comparisons.
- The historical controls use two recipes and different seeds from the existing ACD reruns. New paired ablations are needed for direct causal comparisons with ACD.

No existing KD release or model file was removed or modified.
