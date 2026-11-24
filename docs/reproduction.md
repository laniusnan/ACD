# Reproduction scope

The code was extracted and refactored from `laniusnan/KD`. The seven-pair CIFAR rerun is a frozen reconstruction, not recovered historical SICD training source. Its outputs are tracked separately from the manuscript's historical means.

| Experiment | Code/config | Evidence |
|---|---|---|
| Seven CIFAR rerun pairs | Included | Seed 3201, 240 epochs; released final weights, full test verification |
| ResNet32×4→ResNet8×4, VGG13→VGG8/MobileNetV1, WRN40-2→ShuffleNetV1 | Included | Model/config support; no released complete run |
| Optional ETF | Included | Geometric and dtype tests; released reruns have ETF off |
| ImageNet ResNet34→ResNet18, ResNet50→MobileNetV2 | Included | Example configs only; no verified complete ACD endpoint |
| ImageNet ResNet50→MobileNetV1 | Not included | MobileNetV2 is not a substitute for this experiment |
| MS-COCO detection | Not included | Requires a separate detection training integration |

## Implementation conventions

- ACI uses normalized features and classifier directions, no bias, fixed scale 30 by default. Directions remain trainable.
- Per-sample standardization uses all classes, sample standard deviation (`correction=1`) and epsilon 1e-6. The target logit remains involved in standardization.
- SICD masks the ground-truth class when forming conditional distributions and retains the teacher non-target mass as its weight. Temperature is 4 by default.
- Gradient matching uses all trainable backbone parameters, excluding the classifier. Its reference is full KL on the same standardized logits. The ratio is detached, not clipped, with denominator floor 1e-12. The floor can prevent exact norm equality near zero.
- Optional ETF requires feature dimension at least C−1. It preserves dtype/device; decomposition uses float32 for lower-precision inputs. Impossible dimensions and affine-head ETF requests are rejected.
- Historical CIFAR configs use 100,000 augmented views per epoch. The copied RandAugment preserves its original magnitude sampling, including its unused `m` argument.
- Homogeneous rerun pairs use learning rate 0.05 and CE/KD weights 0.5/0.5. Heterogeneous pairs use learning rate 0.01 and weights 0.1/0.9. ImageNet example learning rates remain 0.1 and are unverified.
- Epoch-boundary resume validates effective configuration and teacher hash. The current trainer's RNG scheme is epoch-derived. Smoke and exported weight files are not resumable training states.
- AMP, DDP and gradient accumulation are not implemented. Adding them requires checking the meaning of gradient-norm matching.

## Evidence and provenance

[Source manifest](source_manifest.json) records copied source hashes. [Paper provenance](paper_source.json) records the unchanged author PDF. [Model guide](model_guide.md) explains the released assets; [model manifest](model_manifest.json) records both original and exported hashes. [Release verification](release_verification.json) records complete test evaluation of the portable weights.

The SHA256 values in `results/cifar100_rerun.json` identify original archive checkpoints. Exported files contain identical model tensors with new metadata and omit optimizer state, so their file hashes differ. Loading exported models and reproducing their inference scores does not establish exact equivalence of a new full training run.
