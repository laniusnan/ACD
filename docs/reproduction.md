# Reproduction scope

The code was extracted and refactored from `laniusnan/KD`. The local seven-pair rerun is a new implementation based on the equations, not a recovered copy of the historical SICD training source. Its numerical outputs must not be merged into the manuscript's historical means.

| Experiment | Code/config | Evidence |
|---|---|---|
| Seven CIFAR rerun pairs | Included | Completed, seed3201, 240 epochs; results JSON |
| ResNet32x4→ResNet8x4, VGG13→VGG8/MobileNetV1, WRN40-2→ShuffleNetV1 | Included | Model/config support; paper numbers not independently reproduced |
| Optional ETF | Included | Mathematical invariance/Gram tests; historical ablation numbers not regenerated |
| ImageNet ResNet34→ResNet18, ResNet50→MobileNetV2 | Included | Example ACD configs only; full ACD endpoints unavailable |
| ImageNet ResNet50→MobileNetV1 | Missing | Do not substitute MobileNetV2 |
| MS-COCO detection | Missing | Requires detection training/config/checkpoints |
| Grad-CAM++ manuscript comparison | Not included | Prior SKD/Ours-labelled figure was a disclosed layout placeholder |

## Implementation details

- Class standardization uses sample SD, epsilon 1e-6; no separate target exclusion during standardization.
- SICD masks the ground-truth class only when forming conditional distributions; teacher non-target mass remains in the weighting.
- Gradient matching uses all trainable backbone parameters, excluding the head; ratio is detached and is not clipped. Denominator floor: 1e-12.
- No AMP/DDP or gradient accumulation is used in the reference trainer. Adding these requires auditing the meaning of gradient-strength matching.
- All CIFAR reference models return `(logits, features)`; the adapter also supports torchvision tensor outputs.
- Historical configs use 100,000 independently augmented views per epoch. RandAugment is copied unchanged, including its unused magnitude argument.
- Epoch-boundary resume validates configuration and teacher hash. Smoke checkpoints cannot be resumed as complete epoch checkpoints.
- Core numerical behavior is validated by KL decomposition, positive-rescaling invariance, ETF geometry, finite backward gradients and strict architecture loading. A short real-data CPU check is an engineering check, not a new benchmark result.

## Checkpoint provenance

Teacher filenames and exact hashes in the source archive can be checked against the original experiment identities. Checkpoints are not redistributed here. The final SHA256 values in `results/cifar100_rerun.json` refer to the original archive, not newly produced checkpoints from the refactored trainer.
