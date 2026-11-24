# Training and evaluation guide

Run commands from the repository root after `pip install -e .`. Each config defines the dataset, architectures, seed, optimizer, loss weights and training schedule. Use a separate output directory for each experiment.

## Choose a configuration

`configs/<pair>.json` is the configuration for that pair. The seven released pairs are listed in the README and model guide. Four additional CIFAR pairs have model/config support but no released full run: `res32x4_res8x4`, `vgg13_vgg8`, `vgg13_mv1`, `wrn402_sv1`. Supply their own matching trained teacher checkpoints.

ImageNet examples are `imagenet_resnet34_resnet18.json` and `imagenet_resnet50_mobilenet_v2.json`. Their learning rate remains 0.1; these examples have no verified full ACD result. CIFAR MobileNetV1 and torchvision ImageNet MobileNetV2 are different architectures and their weights are not interchangeable.

## Data layout

For CIFAR-100, `--data-root data --download` obtains the official torchvision archive. Without `--download`, place `cifar-100-python/` under the data root. The archive has both training and test files because torchvision checks its integrity, even when only the test loader is used.

For ImageNet, arrange data as:

```text
data/imagenet/
  train/<class-id>/*.JPEG
  val/<class-id>/*.JPEG
```

Training checks that the 1,000 class folder names match in both splits. Evaluation only reads `val/`. Keep the original class order; arbitrary folder renaming silently changes label meaning. Evaluation uses bilinear resize to 256, center crop to 224, then ImageNet normalization. CIFAR evaluation uses the original 32×32 RGB images, tensor conversion and normalization without resize or augmentation.

## Training and short checks

```bash
python -m tools.download_models --pair res56_res20
python train.py --config configs/res56_res20.json --data-root data --download \
  --teacher-checkpoint checkpoints/teacher_resnet56_cifar100.pth \
  --output outputs/res56_res20_acd --device cuda:0
```

For a short check, use a fresh output directory and append `--max-steps 1 --device cpu --workers 0`. This stops after one optimizer update and writes `smoke.pth`; it is not an accuracy result or a resumable epoch checkpoint.

The reference CIFAR configs use 50,000 weak plus 50,000 strong views per epoch. Teacher and student receive the same augmented tensors. The historical RandAugment magnitude sampling is preserved; its constructor argument `m=10` is unused by that implementation.

## Outputs and resume

| File | Meaning |
|---|---|
| `config.json` | Effective configuration, including CLI overrides |
| `history.jsonl` | Epoch, training loss and evaluation Top-1/Top-5/CE/sample count |
| `latest.pth` | Student, optimizer, scheduler, completed epoch, config, teacher hash and metrics |
| `final.json` | Completion status, final metrics and final checkpoint SHA256 |
| `smoke.pth` | Incomplete weights from a short check |

Resume with the same config, `--method` / `--etf` / worker overrides, teacher file and output directory. `--resume` restores the end of the last saved epoch. A crash inside an epoch reruns that epoch. Fresh runs reject existing run outputs to avoid mixing logs. `latest.pth` is the final completed epoch, not the best test-set checkpoint; CIFAR's test split is used for monitoring and should not be used to select tuned hyperparameters.

```bash
python train.py --config configs/res56_res20.json --data-root data \
  --teacher-checkpoint checkpoints/teacher_resnet56_cifar100.pth \
  --resume outputs/res56_res20_acd/latest.pth --output outputs/res56_res20_acd
```

Release weights and the original archived rerun files cannot resume this refactored trainer. Exporting a teacher changes its file hash even when tensors are identical: continue an existing run with the same teacher file it started with.

## Evaluate students and teachers

```bash
python train.py --config configs/res56_res20.json --data-root data \
  --evaluate checkpoints/acd_res56_res20_s3201_e240.pth \
  --metrics-json outputs/student_eval.json --device cpu --workers 0

python train.py --config configs/res56_res20.json --data-root data \
  --evaluate checkpoints/teacher_resnet56_cifar100.pth --evaluate-teacher \
  --metrics-json outputs/teacher_eval.json --device cpu --workers 0
```

For a student trained with an ablation, pass the saved `outputs/<run>/config.json` so the head and method match. Evaluation uses no training loader, optimizer or distillation teacher. `--eval-batch-size 128` reduces evaluation memory. Percentages are in [0,100]; CE is mean cross-entropy; `samples` should be 10,000 for the full CIFAR test split. The JSON also includes architecture, role, dataset and checkpoint SHA256.

## Ablations

Append `--method ce`, `kd`, `angular_kd`, `standardized_full`, `conditional`, or `acd`. CE requires no teacher. Other methods require one. Keep the same seed, data recipe and intended CE/KD weights when comparing methods; overriding the method leaves those weights unchanged.

Use `--etf` only with cosine heads of sufficient dimension. A 100-class simplex ETF needs at least 99 features; CIFAR ResNet20/32 and WRN40-1 expose only 64. ImageNet ResNet18 exposes 512 features for 1,000 classes. No projection or approximate ETF is silently added. For supported models, ETF is initialization only and is not required to load a saved cosine head.

## Export and plot

```bash
python -m tools.export_weights --config outputs/res56_res20_acd/config.json \
  --checkpoint outputs/res56_res20_acd/latest.pth \
  --output checkpoints/my_resnet20_weights.pth

pip install -e '.[figures]'
python -m tools.plot_history --history outputs/res56_res20_acd/history.jsonl \
  --output outputs/res56_res20_acd/curves
```

Export strips training state and includes model identity and the source checksum. Use `--role teacher` when exporting an affine teacher. A filename that already exists is rejected. Plots are written as PNG, SVG and PDF; keep `history.jsonl` alongside them for reproducibility.

## Common errors

| Message / symptom | Action |
|---|---|
| CUDA unavailable | Use `--device cpu --workers 0`, or install the correct CUDA-enabled torch pair |
| Missing/unexpected state keys | Use the correct architecture and saved config; CIFAR and ImageNet weights differ |
| Checkpoint model/dataset/head differs | Check the config, student/teacher role and cosine scale |
| Resume configuration / teacher differs | Reuse the saved config and original teacher file |
| Resume requires an epoch-boundary checkpoint | Use this trainer's `latest.pth`, not an exported or smoke file |
| Output already contains a run | Resume it or choose a fresh directory |
| Legacy Python objects | Only for a trusted historical file, pass `--allow-legacy-checkpoint`; released weights do not need it |
| Download hash mismatch | Keep the manifest and asset from the same release; download again into a fresh directory |
| Unknown released pair | Run `python -m tools.download_models --list`; configs without a completed run have no release weights |

## Maintainer release checks

```bash
python -m tools.prepare_release --archive-root /path/to/cifar100_acd_rerun_20261116 \
  --output outputs/new-release
python -m tools.verify_release --release-dir outputs/new-release --data-root data --device cpu
```

Preparation verifies original student and teacher hashes and exports twelve weights. Verification evaluates all twelve on the complete official test split, checks exported hashes and checks students against their archived Top-1. Attach configs, the model guide, manifest, verification report and checksums to the release. Source and export checksums have different meanings; keep both.
