# Historical classifier controls and VGG13 teacher

Release: **`cifar100-head-controls-20261031`**, exported on November 18, 2026 from the October KD archives. The tag date identifies the historical study, not the export date. These weights supplement `cifar100-rerun-20261116`.

This release contains **11 student controls and one VGG13 teacher**, approximately **120 MB** of weights. All students use **ordinary temperature-scaled KL distillation**. The cosine variants isolate the classifier interface; they do **not** use SICD standardization, target-conditioned transfer or gradient matching. Use the earlier ACD release for complete ACD students.

## Which model to use

| Teacher → student | Seed / recipe | Affine KD | Cosine + KD | ETF-initialized cosine + KD |
|---|---|---:|---:|---:|
| ResNet32×4 → ResNet8×4 | 2051 / early_double | 75.18 | 76.99 | 77.42 |
| VGG13 → VGG8 | 2051 / early_double | 74.45 | 75.13 | — |
| ResNet50 → MobileNetV1 | 2101 / compact_double | — | 67.88 | 67.45 |
| VGG13 → MobileNetV1 | 2101 / compact_double | — | 66.39 | 66.60 |
| WRN40-2 → ShuffleNetV1 | 2101 / compact_double | — | 77.82 | 77.60 |

Scores are final-epoch Top-1 percentages on the official CIFAR-100 test split. A dash means no weight for that combination is included; it does not mean an accuracy of zero. The first pair gives a compact three-way classifier comparison, the VGG pair supplies an additional affine/cosine comparison, and the final three pairs cover compact students with trainable ETF initialization. The first seed of each selected study is used consistently, without ranking checkpoints by test accuracy. This is a representative collection, not an exhaustive copy of the historical studies.

Filename prefixes identify the training variant:

- `kd_`: ordinary KD with an affine student classifier, including its bias; config `method=kd`.
- `ci_kd_`: ordinary KD with a bias-free cosine student classifier; config `method=angular_kd, etf=false`.
- `ci_etf_kd_`: the same cosine interface initialized from an ETF; config `method=angular_kd, etf=true`. Classifier directions remain trainable, so the final directions need not remain an ETF.
- `teacher_vgg13_cifar100.pth`: pretrained affine VGG13. It fills the teacher dependency for `vgg13_vgg8` and `vgg13_mv1`, including new ACD training runs.

For example, `ci_etf_kd_res32x4_res8x4_s2051_e240.pth` uses seed 2051 and the final checkpoint at epoch 240. It is not a fixed-ETF or rotating-ETF model. The JSON file with the same stem specifies the exact inference architecture and head.

## Download and evaluate

Install the repository as described in the README. From this release, download a `.pth` file and its matching `.json` into `checkpoints/`. The following works with the release's source commit `cd7bc8a964991de677eb664756da35f2eaccbc4a`:

```bash
python train.py --config checkpoints/ci_kd_res32x4_res8x4_s2051_e240.json \
  --data-root data --download --workers 0 --device cpu \
  --evaluate checkpoints/ci_kd_res32x4_res8x4_s2051_e240.pth
```

Expected Top-1: **76.99%**, 10,000 samples. Change `--device cpu` to your CUDA device for GPU evaluation. A teacher is not needed to evaluate a student. The source at the release tag constructs both CIFAR-100 splits; the subsequent training/evaluation workflow update supports test-only loading.

The standalone `download_models.py` attachment uses only the Python standard library (Python 3.11+). Download it and `model_manifest.json` into one directory, then run:

```bash
python download_models.py --manifest model_manifest.json --list
python download_models.py --manifest model_manifest.json \
  --model ci_kd_res32x4_res8x4_s2051_e240.pth --student-only --output checkpoints
```

It checks SHA256, downloads the matching JSON, and reuses an existing file only if its hash matches. Omit `--student-only` to include the teacher. `--pair res32x4_res8x4` selects all three controls for that pair; `--model` selects one exact filename. `--all` selects all twelve new models plus the three referenced teachers. The `teacher_dependencies` entries point to the existing ACD release, so those teachers are not duplicated here.

Once the workflow update is available in your checkout, the equivalent command is:

```bash
python -m tools.download_models --manifest docs/head_controls_manifest.json \
  --model ci_kd_res32x4_res8x4_s2051_e240.pth --student-only
```

## VGG13 teacher for ACD training

Download the new teacher with its evaluation config:

```bash
python download_models.py --manifest model_manifest.json \
  --model teacher_vgg13_cifar100.pth --output checkpoints
python train.py --config checkpoints/teacher_vgg13_cifar100.json \
  --data-root data --download --workers 0 --device cpu \
  --evaluate checkpoints/teacher_vgg13_cifar100.pth
```

Fresh VGG13 teacher evaluation: **74.64% Top-1**, **92.60% Top-5**, 10,000 images.

The evaluation config places VGG13 in the model slot and uses an affine head. It is not the teacher's original training recipe. For new ACD training, use the repository's ACD pair config instead:

```bash
python train.py --config configs/vgg13_vgg8.json \
  --data-root data --download \
  --teacher-checkpoint checkpoints/teacher_vgg13_cifar100.pth \
  --output outputs/vgg13_vgg8_acd --device cuda:0
```

The same teacher works with `configs/vgg13_mv1.json`. Releasing this teacher and historical controls does not establish a completed ACD benchmark for either pair.

## Inputs and label order

Models accept RGB CIFAR-100 images with shape `[N, 3, 32, 32]`. Convert pixels to `[0, 1]`, then normalize using mean `(0.5071, 0.4867, 0.4408)` and standard deviation `(0.2675, 0.2565, 0.2761)`. Test evaluation uses no augmentation. Outputs follow the official 100 fine-label order, supplied in `cifar100_classes.json`.

Cosine student logits use fixed scale **30**. Distillation temperature **4** is part of the training objective; do not divide inference logits by 4 when reporting classification accuracy or CE. The teacher remains an affine classifier.

## Historical training recipes and comparison limits

| Property | early_double | compact_double |
|---|---|---|
| Source study | head_transfer_20261021 | accepted_kd_heads_20261030 |
| Selected seed | 2051 | 2101 |
| Initial learning rate | 0.05 | 0.01 |
| CE / KD weights | 0.5 / 0.5 | 0.1 / 0.9 |
| Epochs / batch size | 240 / 64 | 240 / 64 |
| Optimizer | SGD, momentum 0.9, no Nesterov | Same |
| Weight decay | 0.0005 | Same |
| LR decay | ×0.1 at zero-based epochs 150, 180, 210 | Same |
| Augmentation | 50,000 weak + 50,000 strong views per epoch | Same |
| KD temperature / cosine scale | 4 / 30 | Same |

The loss is `ce_weight * CE + kd_weight * T² * KL(teacher || student)`. Weak views use crop and horizontal flip; strong views additionally use the historical RandAugment implementation. ETF initialization uses historical ETF seed 1993. Both models receive the same image within each KD example.

The JSON configs preserve evaluation identity and the main recipe values. Historical pairing, RNG handling and ETF construction differ from the refactored trainer. These configs are **evaluation configs and recipe references**, not a promise of bit-for-bit retraining. The exported checkpoints do not contain optimizer, scheduler or RNG state and **cannot resume a historical training run**.

The later study followed recipe development and baseline acceptance using official-test feedback. These are historical diagnostic controls, not an independent unbiased benchmark. Do not mix the two recipe groups, compare them as paired seed-3201 ACD ablations, report a single seed as a multi-seed mean, or infer that ETF always helps. For a controlled ACD comparison, run all methods under the same new protocol and seeds.

## Provenance and verification

Original model files come from `laniusnan/KD` releases `model-backup-20261027` and `kd-head-models-20261031-completed`. Some completion records are recovered from the later overlapping `kd-head-models-20261031-final` metadata snapshot. Original checkpoint SHA256 ties those records to the exact weights. The name `final` does not mean that release contains all earlier weights.

Each original archive and selected checkpoint was checked against the GitHub asset digest or release checksums and the original model manifest before loading. Export strict-loads every state dictionary, preserves all original tensors, and reloads the portable result with `weights_only=True`. Missing legacy BatchNorm counter buffers in the VGG13 teacher are filled by PyTorch and listed in its manifest entry. Teacher state hashes also match the six selected compact-double students' recorded teacher hashes.

`verification.json` records fresh evaluation of all twelve new exports on all 10,000 official test images. The eleven student Top-1 values must match their native completion records before publication. `source_compatibility.json` records strict loading and exact sample-logit agreement for all twelve models with the original release-tag code; the documented student command was also checked on the complete test split. Reused teachers retain the verification report in `cifar100-rerun-20261116`. `model_manifest.json` contains exported hashes, original hashes, config identities and archive paths; `source_evidence.json` preserves sanitized recipe and completion evidence; `restoration_receipt.json` identifies the archived files.

For all assets in one directory, run `sha256sum -c SHA256SUMS`. For a selected subset, run `sha256sum --ignore-missing -c SHA256SUMS` and verify that each downloaded file is checked. Optimizer state, early-epoch snapshots, duplicate seeds, fixed/rotating-ETF variants and unrelated models are intentionally omitted.
