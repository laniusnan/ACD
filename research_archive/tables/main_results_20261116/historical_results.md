# CIFAR-100 历史结果找回记录（2026-11-16）

之前确实跑过这 11 个配对。上一次表格整理只覆盖了后期 ACD 记录，漏查了旧分类头实验快照；不能把主表空白解释为没有运行实验。

## 找回了什么

从已纳入 Git 的 [2026-10-27 目录快照](../../../reports/research_catalog_20261027/native_statuses.tsv)找回 **52 条完成至 epoch 240 的记录，覆盖全部 11 个配对**。逐项与同次扫描的 `cross_model.tsv` 对账一致；这两个文件不是两次独立实验。原工作路径的 52 个 `status.json` 当时均未纳入 Git。现已从 Release 恢复全部 52 个原始文件至独立目录，逐个 SHA256 与快照一致。

**Release 恢复已完成：** [结果报告](release_recovery/release_results.md)还补齐后期 34 条头部实验与 25 条线性基线记录；共核对 111 条不重复的完成记录，尚未找到 11 月 SICD/ACD 研究包。

[逐运行 TSV](historical_runs.tsv)包含原始路径、完成时间、seed 和当时的文件哈希；[完整 JSON](historical_results.json)包含各方法全部 seed、样本 SD、配对差和来源哈希。

## 早期分类头对照：按共同完成的 seed 汇总

以下为官方 CIFAR-100 10,000 张测试集的 epoch-240 重载端点 Top-1（%）。只比较同配对、同 seed 的线性 KD 和余弦头 KD；均值 ± 样本 SD，单 seed 不补造 SD。快照截至 2026-10-27，运行数量不是截至今天的最终总数。

### 同构

| 教师 → 学生 | 共同 seed | 线性 KD | 余弦头 KD | 差值 pp |
| --- | --- | --- | --- | --- |
| ResNet56 → ResNet20 | 2051, 2052, 2053, 2054, 2055 | 72.07 ± 0.23 | 72.45 ± 0.09 | +0.39 |
| ResNet110 → ResNet32 | 2051, 2052 | 74.39 ± 0.48 | 74.37 ± 0.25 | -0.02 |
| ResNet32×4 → ResNet8×4 | 2051, 2052, 2053, 2054, 2055 | 74.99 ± 0.24 | 77.25 ± 0.19 | +2.27 |
| WRN-40-2 → WRN-16-2 | 2051, 2052 | 76.54 ± 0.40 | 77.21 ± 0.23 | +0.67 |
| WRN-40-2 → WRN-40-1 | 2051, 2052 | 75.22 ± 0.10 | 75.50 ± 0.39 | +0.28 |
| VGG13 → VGG8 | 2051 | 74.45 | 75.13 | +0.68 |

### 异构

| 教师 → 学生 | 共同 seed | 线性 KD | 余弦头 KD | 差值 pp |
| --- | --- | --- | --- | --- |
| ResNet32×4 → ShuffleNetV1 | 2051, 2052 | 73.55 ± 0.91 | 75.80 ± 0.25 | +2.25 |
| WRN-40-2 → ShuffleNetV1 | 2051 | 73.48 | 75.41 | +1.93 |
| VGG13 → MobileNetV1 | 2051 | 59.12 | 58.24 | -0.88 |
| ResNet50 → MobileNetV1 | 2051 | 58.53 | 58.72 | +0.19 |
| ResNet32×4 → ShuffleNetV2 | 2051 | 72.60 | 74.58 | +1.98 |

额外找回 ResNet32×4→ResNet8×4 的 ETF 初始化余弦头 5 个 seed：77.42, 77.18, 77.33, 77.52, 77.28，均值 **77.35 ± 0.13**。它仍使用普通 KL，不是完整 ACD。

ResNet110→ResNet32 另有线性 KD seed2053=74.31，因缺少该 seed 的余弦完成值，未计入配对均值。另有 ResNet50→MobileNetV1 线性 seed2052 在旧快照中为 running，未作为完成结果；不能由这个历史状态推断现在仍在跑。

## 为什么不能把这些数值直接写进 ACD 行

保留的 [study.json](../../../reproductions/head_transfer_20261021/study.json) 与目录快照中的 SHA256 一致。[损失实现](../../../reproductions/head_transfer_20261021/methods.py)使用 `0.5 CE + 0.5 T² KL`，T=4；没有双侧标准化或 conditional relation-only。余弦头尺度 s=30，ETF 仅用于原 ResNet32×4→ResNet8×4 配对。

该批训练使用 SGD、240 epochs、batch64、lr0.05、momentum0.9、weight decay0.0005，150/180/210 轮衰减为原来的 0.1；double 增强每轮 100,000 视图。它支持分类头对照，不能替代 SICD/ACD 完整方法结果。

## 之后还跑过其他配方：单独保留

[11 月 2 日周报](../../../reports/HETEROGENEOUS_KD_WEEK_20261102.md)还保存了后续 `compact_double` 的五 seed 官方测试均值：

| 配对 | 线性 KD | 余弦头 KD | ETF 初始化余弦头 |
| --- | --- | --- | --- |
| VGG13 → MobileNetV1 | 67.836 | 66.670 | 66.702 |
| ResNet50 → MobileNetV1 | 68.422 | 67.532 | 67.394 |
| WRN-40-2 → ShuffleNetV1 | 77.310 | 未恢复汇总 | 未恢复汇总 |
| ResNet32×4 → ShuffleNetV1 | 76.822 | 未恢复汇总 | 未恢复汇总 |
| ResNet32×4 → ShuffleNetV2 | 76.442 | 未恢复汇总 | 未恢复汇总 |

这些是另一批实验，现已从 Release 恢复逐 seed 数组，见 [后期结果报告](release_recovery/release_results.md)。不能与上面早期配方混算均值或择优填表。例如 VGG13→MobileNetV1 的线性 KD 从早期表的 59.12 到后续配方的 67.836，已经足以说明配方匹配不可省略。

## 完整 ACD 的现有覆盖

| 配对 | 已找回的早期完成记录数 | 后期 standalone ACD |
| --- | --- | --- |
| ResNet56 → ResNet20 | 10 | 当前缺少可归入该行的终点记录 |
| ResNet110 → ResNet32 | 5 | 当前缺少可归入该行的终点记录 |
| ResNet32×4 → ResNet8×4 | 15 | 78.36（n=3） |
| WRN-40-2 → WRN-16-2 | 4 | 当前缺少可归入该行的终点记录 |
| WRN-40-2 → WRN-40-1 | 4 | 当前缺少可归入该行的终点记录 |
| VGG13 → VGG8 | 2 | 75.48（n=2） |
| ResNet32×4 → ShuffleNetV1 | 4 | 当前缺少可归入该行的终点记录 |
| WRN-40-2 → ShuffleNetV1 | 2 | 78.86（n=2） |
| VGG13 → MobileNetV1 | 2 | 69.71（n=2） |
| ResNet50 → MobileNetV1 | 2 | 当前缺少可归入该行的终点记录 |
| ResNet32×4 → ShuffleNetV2 | 2 | 当前缺少可归入该行的终点记录 |

后期 4 个 ACD 数值及其证据级别详见 [主表说明](README.md)。AT+ACD=78.97 和 ReviewKD-style+ACD=75.60 是组合实验，不能替代相应配对的 standalone ACD。

## 缺失发生在哪里，下一步需要恢复什么

- 早期状态文件：已从 Release 恢复全部 52 个，并验证与目录快照哈希及端点读数一致；恢复目录独立于训练工作目录。
- 11 月 3 日的 `homogeneous_lift_etf_20261103` 和 `homogeneous_relation_etf_registration_20261103`：根 README 保留了登记/部署说明，当前目录缺失，本地完整 Git 历史中也未找到。前者最后保留的说明是 9 个任务等待资源，后者是登记通过/部分维度阻塞；这些说明都不是最终完成状态。
- 10 月 31 日两个私有 Release 按 README 归档了 92+18 个模型，并声明当时保留指标、历史、预测、日志、配置和源码。模型归档不等于指标被删除；也不能据此断言这些 Release 包含 11 月后的 ACD 结果。
- 应从原运行机器或备份恢复后期研究包的 `status.json`、`state.json`、`summary.json`、逐 epoch history、配置和任务登记；先恢复这些小文件即可判断最终结果和配方，不必先下载全部模型。需要重新评测时再恢复对应 checkpoint。

当前结论是：实验确实跑过，部分结果已经找回；其余 ACD 终点记录在本机尚未恢复，不能据此断言从未运行或永久丢失。
