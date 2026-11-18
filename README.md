# ACD: Angular Conditional Distillation

PyTorch implementation of the Angular Coordinate Interface (ACI) and Scale-Invariant Conditional Distillation (SICD).

ACI replaces the student's affine classifier with a bias-free cosine head. SICD standardizes teacher and student logits, transfers the teacher-mass-weighted conditional distribution over non-target classes, and optionally matches the gradient strength of standardized full KL:

```text
z[k] = s * cos(theta[k])
SICD = tau^2 * (1 - p[y]) * KL(p[-y] || q[-y])
r = stop_gradient(||grad_backbone(full standardized KL)|| / ||grad_backbone(SICD)||)
ACD = CE + lambda * r * SICD
```

The teacher is frozen. `p` and `q` are computed after per-sample, all-class standardization using sample standard deviation (correction=1). The shared cosine scale is fixed; classifier directions remain trainable. ETF is optional initialization, not a fixed classifier constraint.

## Installation

Use Python 3.11 or newer and install a matching PyTorch/torchvision pair for your hardware, then:

```bash
pip install -e .
python -m unittest discover -s tests -v
```

Verified locally with Python 3.12, PyTorch 2.9.0, torchvision 0.24.0 and NumPy 1.26.4. GPU training uses float32 and disables TF32. No dataset or checkpoint is bundled.

## CIFAR-100

Provide a trusted teacher checkpoint (plain state dictionary, or a dictionary containing `student` or `state_dict`). Training and inference load model weights strictly.

```bash
python train.py --config configs/res56_res20.json \
  --data-root data --download \
  --teacher-checkpoint checkpoints/resnet56.pth \
  --output outputs/res56_res20_acd --device cuda:0
```

`--download` downloads official CIFAR-100 via torchvision. Each epoch uses 50,000 weak and 50,000 strong views; teacher and student see exactly the same augmented input. The retained historical RandAugment implementation uses its original magnitude sampling; its constructor's `m=10` argument is ignored. See [reproduction notes](docs/reproduction.md).

Resume at an epoch boundary:

```bash
python train.py --config configs/res56_res20.json --data-root data \
  --teacher-checkpoint checkpoints/resnet56.pth \
  --resume outputs/res56_res20_acd/latest.pth \
  --output outputs/res56_res20_acd
```

Evaluate the saved final student:

```bash
python train.py --config configs/res56_res20.json --data-root data \
  --evaluate outputs/res56_res20_acd/latest.pth --device cuda:0
```

Outputs contain the resolved configuration, per-epoch metrics, resumable checkpoint and final checkpoint SHA256. Use `--max-steps 1 --device cpu --workers 0` for a short execution check; its output is explicitly an incomplete smoke checkpoint.

## Ablations

Use `--method kd` for an affine student with ordinary KL; `--method angular_kd` for ACI with ordinary KL; `--etf` enables ETF initialization for angular students; `--method standardized_full` retains full standardized KL; `--method conditional` omits gradient matching; the default `--method acd` includes gradient matching. Keep the same seed and configuration when comparing methods.

The implementation accepts `ce_weight` and `kd_weight`. The archived rerun configs use CE/KD weights 0.5/0.5 (homogeneous) or 0.1/0.9 (heterogeneous). Set `ce_weight=1` and `kd_weight=lambda` to match the displayed paper objective directly. Do not silently equate configurations with different weights.

## ImageNet

The shared trainer supports torchvision ResNet and MobileNetV2 models, with 224-pixel inputs and ImageNet normalization. Supply extracted official `train/<WNID>/...` and `val/<WNID>/...` folders and a teacher state dictionary.

```bash
python train.py --config configs/imagenet_resnet34_resnet18.json \
  --data-root /path/to/imagenet \
  --teacher-checkpoint checkpoints/resnet34.pth \
  --output outputs/imagenet_r34_r18_acd
```

ImageNet configs are implementation examples, not verified reproductions of the manuscript results. The existing MobileNetV2 experiments must not be identified as MobileNetV1 results.

## Results and scope

[Recorded CIFAR-100 reruns](results/cifar100_rerun.json) contain seven completed seed3201 ACD runs from the existing experiment archive. They are separate from the manuscript's supplied main-table and VGG ablation numbers. [Reproduction status](docs/reproduction.md) distinguishes completed runs, supported but unverified configurations, and missing experiment packages.

MS-COCO/Faster R-CNN-FPN integration, ImageNet MobileNetV1, and verified checkpoints for the manuscript's VGG ablation table are not included. This repository does not claim to reproduce those results. The core loss can be integrated into other trainers, but detector classifier replacement requires a task-specific design.

## Source attribution

CIFAR backbone and augmentation files were extracted from the existing KD archive. Original source comments are preserved; [source manifest](docs/source_manifest.json) lists paths and hashes. Several backbone definitions identify mdistiller as their upstream source. Redistribution licensing of these inherited files must be reviewed before a public release; no blanket license is assigned here.

Paper title, publication link and BibTeX will be added when finalized.
