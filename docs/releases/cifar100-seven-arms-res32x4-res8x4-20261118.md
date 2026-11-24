# CIFAR-100 seven-arm controls: ResNet32×4 → ResNet8×4

Tag: `cifar100-seven-arms-res32x4-res8x4-20261118`

This release packages the complete seven-arm, single-seed formal comparison for
ResNet32×4 teacher to ResNet8×4 student.  Every student was trained from scratch
for 240 epochs with seed 3301; all seven share the pair-selected recipe
(`lr=0.01`, CE/KD weights `0.5/0.5`).  The release contains portable evaluation
weights, per-arm configurations, source checkpoint hashes, final test records,
the selected-recipe record, a model manifest, and `SHA256SUMS`.

| Arm | Official CIFAR-100 top-1 (%) |
| --- | ---: |
| KD | 74.76 |
| cosine KD | 77.30 |
| cosine ETF-init KD | 77.07 |
| rotating ETF KD | 76.72 |
| cosine SICD | 76.58 |
| cosine ETF-init SICD | 77.50 |
| rotating ETF SICD | 77.37 |

These are single-run endpoint measurements, not multi-seed claims.  The exact
selection procedure and shared settings are documented in
[`docs/seven_arm_training_20261118.md`](../seven_arm_training_20261118.md).
`verify_release.py` evaluates every asset against the full official 10,000-image
test split and checks archived top-1 to tolerance.
