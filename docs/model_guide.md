# CIFAR-100 Release model guide

Release: `cifar100-rerun-20261116` · Repository: `laniusnan/ACD` · Prepared: 2026-11-18.

This release provides **seven ACD students and five distinct affine teachers**, approximately 170 MB of model files. Students are the final epoch 240 checkpoints from the seed 3201 rerun. They use a cosine classifier with scale 30, trainable directions and **ETF disabled**. They are single-run results, separate from the manuscript's historical aggregate results.

## Which model should I download?

Choose the pair matching the student you want to evaluate or train. `res56_res20` is a small starting example. `res32x4_sv2` has the highest Top-1 among these seven archived runs; this does not establish a general architecture ranking.

| Pair / config filename stem | Teacher → student | Test Top-1 (%) | Student size (MB) |
|---|---|---:|---:|
| `res110_res32` | `resnet110` → `resnet32` | 73.51 | 1.97 |
| `res32x4_sv1` | `resnet32x4` → `shufflenetv1` | 78.74 | 3.99 |
| `res32x4_sv2` | `resnet32x4` → `shufflenetv2` | 80.07 | 5.59 |
| `res50_mv1` | `resnet50` → `mobilenetv1` | 69.45 | 6.76 |
| `res56_res20` | `resnet56` → `resnet20` | 71.08 | 1.16 |
| `wrn402_wrn162` | `wrn_40_2` → `wrn_16_2` | 75.67 | 2.85 |
| `wrn402_wrn401` | `wrn_40_2` → `wrn_40_1` | 74.67 | 2.37 |

Student files use `acd_<pair>_s3201_e240.pth`. Teacher files use `teacher_<architecture>_cifar100.pth`. Pairing details, sizes, model identities and hashes are in `model_manifest.json`. Teachers are deduplicated: WRN40-2 is shared by two pairs, and ResNet32×4 is shared by two pairs.

All models are CIFAR-100 variants. In particular, `resnet50` is the included CIFAR ResNet50, not torchvision's ImageNet ResNet50. Use `acd.models.build` from this repository; a same-named external architecture may have incompatible keys or strides.

## Teacher reference scores

Full-test scores of the exported affine teachers:

| Teacher | Top-1 (%) | Top-5 (%) | Size (MB) |
|---|---:|---:|---:|
| `resnet110` | 74.31 | 92.92 | 7.20 |
| `resnet32x4` | 79.42 | 94.58 | 29.85 |
| `resnet50` | 79.34 | 94.81 | 95.14 |
| `resnet56` | 72.41 | 92.50 | 3.58 |
| `wrn_40_2` | 75.61 | 93.63 | 9.12 |

## Download and verify

In a checkout containing the repository utilities:

```bash
python -m tools.download_models --list
python -m tools.download_models --pair res56_res20
```

Add `--student-only` for evaluation-only downloads, or use `--all` for all twelve models. Each download is SHA256-checked. Existing matching files are reused; mismatched existing files are not overwritten.

The release is also usable with the existing source commit `cd7bc8a964991de677eb664756da35f2eaccbc4a`, without the new helper scripts. Install that source, then download assets from the GitHub Releases page or use GitHub CLI:

```bash
gh release download cifar100-rerun-20261116 --repo laniusnan/ACD \
  --dir checkpoints --pattern 'acd_res56_res20_s3201_e240.pth' \
  --pattern 'teacher_resnet56_cifar100.pth' --pattern 'res56_res20.json' \
  --pattern 'model_manifest.json' --pattern 'SHA256SUMS'
```

`SHA256SUMS` covers the model weights, configs, model guide, class names, manifest and verification report. For a full asset download, run `sha256sum -c SHA256SUMS` in the download directory. When downloading only selected assets, verify their entries in the manifest or use `sha256sum --ignore-missing -c SHA256SUMS`.

## Evaluate a released student

This command works with the source commit named above and with the updated trainer:

```bash
python train.py --config checkpoints/res56_res20.json \
  --data-root data --download \
  --evaluate checkpoints/acd_res56_res20_s3201_e240.pth \
  --device cpu --workers 0
```

The updated trainer additionally accepts `--metrics-json outputs/student_eval.json`, `--eval-batch-size 128` and `--evaluate-teacher`. The model weights themselves do not depend on these additions.

Use the official 10,000-image CIFAR-100 test set. Preprocessing: 32×32 RGB, `ToTensor()`, normalization mean `(0.5071, 0.4867, 0.4408)` and standard deviation `(0.2675, 0.2565, 0.2761)`. Do not resize, crop randomly or apply training augmentation for evaluation. Logit column order is the official CIFAR-100 fine-label order, provided in `cifar100_classes.json`.

All twelve exported files are strictly loadable with weights-only loading. `verification.json` reports their full-test Top-1, Top-5, mean CE and sample count. Every student Top-1 matches its archived final score. Verification uses CPU float32, batch size 128, two threads, PyTorch 2.9.0; it is an inference check, not a new training run.

## Train a new student with a released teacher

```bash
python train.py --config checkpoints/res56_res20.json \
  --data-root data --download \
  --teacher-checkpoint checkpoints/teacher_resnet56_cifar100.pth \
  --output outputs/res56_res20_acd --device cuda:0
```

The config identifies the required teacher and student. Do not initialize a new student from the released student when trying to reproduce training from random initialization. The teacher stays frozen.

Training recipe: 240 epochs, batch size 64, SGD momentum 0.9, weight decay 5e-4; LR milestones 150/180/210 with factor 0.1. Homogeneous pairs use LR 0.05 and CE/KD weights 0.5/0.5; heterogeneous pairs use LR 0.01 and 0.1/0.9. Temperature is 4; cosine scale is 30. Each epoch uses 50,000 weak and 50,000 strong views. Matching config files are attached individually to this release.

## Checkpoint format and limitations

```python
import torch
from acd.models import build

model, _ = build("resnet20", dataset="cifar100", angular=True, scale=30.)
checkpoint = torch.load("checkpoints/acd_res56_res20_s3201_e240.pth",
                        map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()
```

The portable payload contains `format`, `state_dict`, `model_config` and `metadata`. Metadata records role, source SHA256, and student pair/epoch/seed where applicable. Tensor values are retained; optimizer state, scheduler state and runtime objects are omitted. **These are not resumable training checkpoints.** For future runs, resume with `latest.pth` produced by the current trainer, using the exact same teacher file and effective config.

The source checkpoint hash and exported file hash differ because the file structure changed. Use the export hash for download integrity and the source hash to trace the archived experiment. The `model_manifest.json` contains both. The teacher metadata also names its original backup release and archive path.

## Provenance and interpretation

Sources: `laniusnan/KD/reproductions/cifar100_acd_rerun_20261116/runs/<pair>__s3201__acd/latest.pth` and `assets/teachers/<architecture>.pth`. Student originals were verified against `results/cifar100_rerun.json`; teacher originals were verified against the archive's `artifacts/assets.json` before export. Teachers trace to the KD backup release `model-backup-20261027`. Source hashes are the immutable identity for these files.

This release includes no ImageNet model, detector, incomplete run or layout-placeholder result. No test-set best-checkpoint selection was used for these seven students. Numerical agreement of exported inference does not establish that the refactored trainer reproduces an archived full training trajectory bit for bit. No historical multi-seed confidence interval is inferred from this single seed.

Upstream code attribution remains in the source files and source manifest.

## Additional historical controls and VGG13 teacher

The separate `cifar100-head-controls-20261031` release provides 11 ordinary-KD head controls and the missing VGG13 teacher. See the [supplement guide](head_controls_guide.md). These files do not add complete ACD benchmark results or replace any model above.
