# Historical KD classifier controls and VGG13 teacher

This supplement adds **11 final-epoch student controls and the missing VGG13 teacher**, approximately **120 MB** of portable weights recovered from `laniusnan/KD` releases. They support classifier geometry/initialization diagnostics and new VGG-teacher training runs.

**The students use ordinary temperature-scaled KL distillation, not complete SICD/ACD.** For the seven complete ACD reruns, use `cifar100-rerun-20261116`. The date in this supplement's tag identifies the historical study; export and verification were performed on November 18, 2026.

| Teacher → student | Seed | Affine KD | Cosine + KD | ETF-initialized cosine + KD |
|---|---:|---:|---:|---:|
| ResNet32×4 → ResNet8×4 | 2051 | 75.18 | 76.99 | 77.42 |
| VGG13 → VGG8 | 2051 | 74.45 | 75.13 | — |
| ResNet50 → MobileNetV1 | 2101 | — | 67.88 | 67.45 |
| VGG13 → MobileNetV1 | 2101 | — | 66.39 | 66.60 |
| WRN40-2 → ShuffleNetV1 | 2101 | — | 77.82 | 77.60 |

Scores are Top-1 percentages on all 10,000 official CIFAR-100 test images. All eleven student scores were re-evaluated and match their original completion records. Every student is the final epoch-240 checkpoint. `—` means not included.

Read **`MODEL_GUIDE.md`** before comparing methods. The seed-2051 and seed-2101 groups use different recipes, and later recipe development used official-test feedback. These historical controls are not paired seed-3201 ACD ablations or unbiased multi-seed benchmarks. ETF denotes initialization followed by trainable classifier directions, not a fixed or rotating ETF constraint.

## Download and evaluate

Install the repository, then download `download_models.py` and `model_manifest.json` from this release. The attached downloader needs only Python 3.11+ and verifies the weights and matching configuration by SHA256:

```bash
python download_models.py --manifest model_manifest.json \
  --model ci_kd_res32x4_res8x4_s2051_e240.pth --student-only --output checkpoints
python train.py --config checkpoints/ci_kd_res32x4_res8x4_s2051_e240.json \
  --data-root data --download --workers 0 --device cpu \
  --evaluate checkpoints/ci_kd_res32x4_res8x4_s2051_e240.pth
```

Expected result: **76.99% Top-1 / 10,000 images**. No teacher checkpoint is needed for student evaluation. The tag targets the existing public source commit `cd7bc8a964991de677eb664756da35f2eaccbc4a`; pending local workflow changes are not required for this command.

The VGG13 teacher re-evaluates to **74.64% Top-1 / 92.60% Top-5** on the same 10,000 images.

Use `--model teacher_vgg13_cifar100.pth` to get the new teacher and its evaluation config. For training, pass that teacher to the repository's `configs/vgg13_vgg8.json` or `configs/vgg13_mv1.json` with `--teacher-checkpoint`. See the model guide for full commands.

## Contents

- 11 student `.pth` files: `kd_`, `ci_kd_`, or `ci_etf_kd_` prefix.
- `teacher_vgg13_cifar100.pth` and 12 matching evaluation JSON configs.
- `MODEL_GUIDE.md`, `model_manifest.json`, `verification.json`, `source_compatibility.json`, `source_evidence.json`, `restoration_receipt.json`, `cifar100_classes.json`, `download_models.py`, `SHA256SUMS`.

ResNet32×4, ResNet50 and WRN40-2 teachers are referenced from `cifar100-rerun-20261116` rather than duplicated. Omit `--student-only` to have the downloader retrieve the required teacher as well.

Exported weights load with `weights_only=True`. Original tensors were checked exactly, and all twelve new weights were independently evaluated. They contain no optimizer, scheduler or RNG state and **cannot resume a historical training run**. Configs provide evaluation identity and recipe reference; exact historical training replay is not claimed.

For all assets in one directory, run `sha256sum -c SHA256SUMS`. For a selected subset, run `sha256sum --ignore-missing -c SHA256SUMS` and check that each downloaded file is covered.
