# Main-results tables — retained experimental evidence

Top-1/Top-5 are percentages; gains are percentage points (pp).

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

## 用户补充的历史 ETF 初始化对照（待核实协议）

| 教师 → 学生 | relation-only* | +ETF 初始化* | 差值 (pp) |
| --- | --- | --- | --- |
| ResNet50 → MobileNetV1 | 73.30 | 72.80 | -0.50 |
| WRN-40-2 → ShuffleNetV1 | 81.38 | 81.62 | +0.24 |
| ResNet32×4 → ShuffleNetV1 | 81.46 | 81.36 | -0.10 |
| ResNet32×4 → ShuffleNetV2 | 82.28 | 83.12 | +0.84 |
| VGG13 → VGG8 | 80.22 | 80.52 | +0.30 |

* 为用户提供的历史截图文字转录。评测划分、最终/最佳 checkpoint、seed 数、配方及 relation-only 与当前 ACD 实现的对应关系待核实；未知项不默认等于当前重跑设置。账本列来自之前保留的四组 ACD 记录，不能与截图列取最大值合并或混算均值。准确率单位为 %，差值单位为百分点（pp）；— 表示缺失。

5 组中 3 组提升、2 组下降，等权平均差值 +0.16 pp（未舍入值 +0.156 pp）。这支持将 ETF 作为效果依赖配对的初始化消融；不能据此声称所有配对都受益、统计显著，或已证明弱敏感类别保留最优角度的机制。

## Controlled ablation: paired CIFAR-100 results

| Method | VGG13 → VGG8 | VGG13 → MobileNetV1 |
| --- | --- | --- |
| KD (affine) | 74.39 ± 0.43 | 67.78 ± 0.40 |
| ACI + KL | 74.80 ± 0.19 | 66.60 ± 0.16 |
| ACI + standardized full KL | 75.55 ± 0.05 | 68.98 ± 0.02 |
| ACD (ACI + SICD) | 75.48 ± 0.02 | 69.71 ± 0.01 |

Mean ± sample SD over seeds 2102/2103; final epoch 240, official 10,000-image test. Only the displayed four-way comparisons are ranked.

## 2. Additional CIFAR-100 archived means

| Pair | Runs | DKD | ACI + std. full | ACD |
| --- | --- | --- | --- | --- |
| ResNet32×4 → ResNet8×4 | 3 | 77.40 | 77.98 | 78.36 |
| WRN40-2 → ShuffleNetV1 | 2 | — | 78.68 | 78.86 |

Only means are retained locally; no SD is reconstructed. The ledger reports a ResNet relation-minus-full gain of +0.39 pp, whereas subtraction of its rounded means gives +0.38 pp; use the endpoint means without asserting a more precise gain.

## 3. ImageNet-1K evidence snapshot

| Model/method | Top-1 | Top-5 | Status |
| --- | --- | --- | --- |
| Teacher (ResNet50) | 76.146 | 92.872 | revalidated official pretrained teacher |
| Student (CE) | 68.662 | 88.798 | completed final-checkpoint evaluation |
| KD | — | — | no completed evaluation available at snapshot |
| ACD | — | — | no verified full-run ACD result; I7 pilot is excluded |

ResNet50 → MobileNetV2; full 50,000-image validation set. Teacher weights are official pretrained weights. Student results are single-seed final-checkpoint evaluations; pilot/probe scores are excluded.

## 4. Replacement in feature-distillation pipelines

| Pipeline | Pair | Affine + KL | ACI + KL | ACI + SICD | Δ vs ACI + KL | Δ vs affine + KL |
| --- | --- | --- | --- | --- | --- | --- |
| AT + KL | ResNet32×4 → ShuffleNetV1 | 77.30 | 77.60 | 78.97 | +1.37 | +1.67 |
| ReviewKD-style + KL | WRN40-2 → WRN40-1 | 75.74 | 75.54 | 75.60 | +0.06 | -0.14 |

Seed 2101 only. The first gain isolates output-loss replacement on the angular head; the second compares the joint head/loss change with the affine baseline. ReviewKD-style is an adapted pipeline, not a claim of original-author recipe reproduction.

## 5. Archived training-cost diagnostic

| Objective | Median ms/update | Relative to ACI + KL |
| --- | --- | --- |
| ACI + KL | 15.09 | 1.00× |
| DKD | 15.55 | 1.03× |
| ACI + standardized full KL | 15.51 | 1.03× |
| ACD matched relation | 30.04 | 1.99× |

Transcribed from the historical same-checkpoint/GPU/batch/loader benchmark. Full benchmark metadata is not locally restored; these are not ImageNet throughput measurements or a general efficiency ranking.

