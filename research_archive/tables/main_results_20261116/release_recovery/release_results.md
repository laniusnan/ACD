# 从 GitHub Release 找回的实验结果

已用 laniusnan 登录读取全部 3 个 Release，下载并核验 9 份清单附件及 4 个必要归档。恢复 **706 个文件路径**（两个时间点快照有重复），确认 **111 条不重复的完成实验记录**：早期 52 条、后期分类头 34 条、后期线性 KD confirmation 25 条。这 111 条属于历史普通 KD／分类头实验，不是 111 条完整 ACD。

原始数据保存在 [restored](restored)，[逐运行 TSV](release_runs.tsv)、[汇总 JSON](release_results.json)和[逐文件恢复凭据](restoration_receipt.json)记录 seed、checkpoint 哈希与来源。52 条早期状态文件的 SHA256 和端点读数全部与先前目录快照一致。

## 后期异构线性 KD：五 seed 结果已恢复

以下均为 compact_double、seed2101–2105、epoch240、官方 10,000 图测试集 Top-1（%）；± 为样本 SD。

| 配对 | 逐 seed Top-1 | 均值 ± SD |
| --- | --- | --- |
| ResNet32×4 → ShuffleNetV1 | 77.28, 76.75, 76.85, 76.80, 76.43 | 76.82 ± 0.30 |
| WRN-40-2 → ShuffleNetV1 | 76.92, 76.99, 77.32, 77.68, 77.64 | 77.31 ± 0.35 |
| VGG13 → MobileNetV1 | 67.79, 68.06, 67.50, 67.43, 68.40 | 67.84 ± 0.40 |
| ResNet50 → MobileNetV1 | 68.80, 68.52, 68.31, 68.59, 67.89 | 68.42 ± 0.35 |
| ResNet32×4 → ShuffleNetV2 | 76.59, 76.70, 76.45, 76.55, 75.92 | 76.44 ± 0.31 |

## 后期分类头对照：只使用共同完成的 seed

| 配对 | 共同 seed | 线性 KD | 余弦头 KD | ETF 初始化余弦头 |
| --- | --- | --- | --- | --- |
| ResNet32×4 → ShuffleNetV1 | 2101 | 77.28 | 76.97 | — |
| WRN-40-2 → ShuffleNetV1 | 2101, 2102, 2103, 2104 | 77.23 ± 0.35 | 77.61 ± 0.43 | 77.76 ± 0.24 |
| VGG13 → MobileNetV1 | 2101, 2102, 2103, 2104, 2105 | 67.84 ± 0.40 | 66.67 ± 0.23 | 66.70 ± 0.15 |
| ResNet50 → MobileNetV1 | 2101, 2102, 2103, 2104, 2105 | 68.42 ± 0.35 | 67.53 ± 0.37 | 67.39 ± 0.34 |
| ResNet32×4 → ShuffleNetV2 | 2101, 2102, 2103, 2104, 2105 | 76.44 ± 0.31 | — | — |

WRN→ShuffleNetV1 的三臂共同完成 seed 只有 2101–2104，因此这里的线性均值与上表五 seed 均值不同。ResNet32×4→ShuffleNetV1 只有 seed2101 的线性/余弦对照；ShuffleNetV2 只恢复了线性基线，其余两臂在当时快照中因基线门槛未通过而跳过。所有方法各自的完整 seed 数组另见 JSON，不把不同 seed 数量混为配对比较。

## 旋转 ETF：同 seed2101 比较

| 配对 | 线性 KD | 普通余弦 | 旋转 ETF |
| --- | --- | --- | --- |
| ResNet32×4 → ShuffleNetV1 | 77.28 | 76.97 | 76.90 |
| WRN-40-2 → ShuffleNetV1 | 76.92 | 77.82 | 77.45 |
| VGG13 → MobileNetV1 | 67.79 | 66.39 | 66.63 |
| ResNet50 → MobileNetV1 | 68.80 | 67.88 | 67.57 |

这是四个单 seed pilot；不能宣称普遍正收益，也不能与五 seed 均值直接计算配对增益。

## 配方与论文使用边界

恢复的 compact_double 锁定配置为 lr=0.01、CE/KD 权重=0.1/0.9、T=4、batch64、240 epochs、每轮 100,000 弱/强视图；余弦尺度 s=30。早期 head_transfer 则为 lr=0.05、CE/KD=0.5/0.5。这解释了同一配对历史数值不同的一个明确来源，不能跨配方择优填表。

本批分类头训练调用普通 KD loss；不含 SICD 的标准化条件关系目标。记录还明确披露曾依据官方测试反馈选择基线配方，这些恢复结果可支持历史结果核验与头部消融，不能直接当作无测试集调参的统一 SOTA benchmark。

## 是否找到了缺失的完整 ACD

在这 3 个 Release 的全部 manifest 路径中，未发现 `homogeneous_lift_etf_20261103`、`homogeneous_relation_etf_registration_20261103`、`cvpr_*`、`acd_*` 或 `sicd`；资产最后更新时间为 2026-10-31T18:13:40Z。结果快照属于 10 月 27/31 日，未找回 11 月的 SICD/ACD 研究包。

因此，先前能对应到 standalone ACD 的四组记录保持不变，其余七组不能用这些普通 KD/ETF 数值替代。仍需从原运行机器或其他备份寻找后期研究包；本结论不等于那些实验从未运行。
