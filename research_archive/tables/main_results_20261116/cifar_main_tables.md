# CIFAR-100: six homogeneous and five heterogeneous pairs

已找回覆盖全部 11 个配对的 52 条早期完成记录，详见 [历史结果与逐配对覆盖](historical_results.md)。早期普通 KD／余弦头实验与后期 ACD 分开列出；主表空白不代表没有跑过该配对。

## Homogeneous CIFAR-100 benchmark

| Method / Teacher | ResNet56 | ResNet110 | ResNet32×4 | WRN-40-2 | WRN-40-2 | VGG13 |
| --- | --- | --- | --- | --- | --- | --- |
| Teacher Top-1 (reference) | 72.40 | 74.31 | 79.42 | 75.59 | 75.59 | 74.64 |
| Student | ResNet20 | ResNet32 | ResNet8×4 | WRN-16-2 | WRN-40-1 | VGG8 |
| Student Top-1 (reference) | 65.79 | 67.92 | 72.48 | 71.12 | 69.54 | 68.19 |
| KD† | — | — | — | — | — | 74.39 |
| DKD† | — | — | 77.40 | — | — | — |
| NormKD | — | — | — | — | — | — |
| CRD | — | — | — | — | — | — |
| ReviewKD | — | — | — | — | — | — |
| ACD† | — | — | 78.36 | — | — | 75.48 |
| ACD runs | — | — | 3 | — | — | 2 |
| Reported relation-only* | — | — | — | — | — | 80.22 |
| Reported +ETF init.* | — | — | — | — | — | 80.52 |
| Reported ETF delta (pp)* | — | — | — | — | — | +0.30 |
| Recovered historical head-study runs | 10 | 5 | 15 | 4 | 4 | 2 |

Teacher/student reference accuracies are supplied by the user and are not local checkpoint re-evaluations. † Local experimental records are provisionally placed by architecture pair; protocol compatibility with the reference table must be checked before ranking. Dashes denote entries unavailable for this method/protocol, not evidence that a pair was never trained. 52 completed ordinary-KD head-study records cover all 11 pairs; these are summarized separately in the historical recovery tables, not relabeled as ACD. Feature-pipeline combinations are excluded from standalone ACD cells. * User-transcribed historical screenshot values: evaluation split, checkpoint selection, run count, recipe and equivalence to ACD remain unverified. They are retained separately from archived ACD means and are not used for ranking.

## Heterogeneous CIFAR-100 benchmark

| Method / Teacher | ResNet32×4 | WRN-40-2 | VGG13 | ResNet50 | ResNet32×4 |
| --- | --- | --- | --- | --- | --- |
| Teacher Top-1 (reference) | 79.42 | 75.59 | 74.64 | 79.33 | 79.42 |
| Student | ShuffleNetV1 | ShuffleNetV1 | MobileNetV1 | MobileNetV1 | ShuffleNetV2 |
| Student Top-1 (reference) | 69.43 | 69.43 | 60.09 | 60.09 | 73.50 |
| KD† | — | — | 67.78 | — | — |
| DKD† | — | — | — | — | — |
| NormKD | — | — | — | — | — |
| CRD | — | — | — | — | — |
| ReviewKD | — | — | — | — | — |
| ACD† | — | 78.86 | 69.71 | — | — |
| ACD runs | — | 2 | 2 | — | — |
| Reported relation-only* | 81.46 | 81.38 | — | 73.30 | 82.28 |
| Reported +ETF init.* | 81.36 | 81.62 | — | 72.80 | 83.12 |
| Reported ETF delta (pp)* | -0.10 | +0.24 | — | -0.50 | +0.84 |
| Recovered historical head-study runs | 4 | 2 | 2 | 2 | 2 |

Teacher/student reference accuracies are supplied by the user and are not local checkpoint re-evaluations. † Local experimental records are provisionally placed by architecture pair; protocol compatibility with the reference table must be checked before ranking. Dashes denote entries unavailable for this method/protocol, not evidence that a pair was never trained. 52 completed ordinary-KD head-study records cover all 11 pairs; these are summarized separately in the historical recovery tables, not relabeled as ACD. Feature-pipeline combinations are excluded from standalone ACD cells. * User-transcribed historical screenshot values: evaluation split, checkpoint selection, run count, recipe and equivalence to ACD remain unverified. They are retained separately from archived ACD means and are not used for ranking.

