# ACD 主结果正文与表格（2026-11-16）

本目录按“CIFAR-100 主结果 → ImageNet-1K → 现有蒸馏管线的输出损失替换”整理实测证据。
这是已有记录的数值对账和写作产物，本次没有训练、重新评测模型或推送 Git。

**新增用户截图记录：** 已补入用户在对话中转录的五组 relation-only / +ETF 初始化成绩。
见[当前 11 配对结果台账](current_acd_records.md)、[ETF 对照表](user_reported_etf_table.md)、
[独立 LaTeX 表](user_reported_etf_table.tex)及[原始转录和待核实字段](user_reported_etf_20261116.json)。
两张 CIFAR 主表使用独立的 `Reported ...*` 行展示这些数据，不覆盖已有账本均值。
此前四组账本加上本次五组截图，共覆盖七个不同配对；这表示数值记录覆盖率，
不等于七组已通过统一协议核验。截图列不默认是官方测试集、最终 epoch 或单 seed。
WRN→ShuffleNetV1 的 78.86/81.38 和 VGG13→VGG8 的 75.48/80.22 均保留来源。
ETF 在五组中三升两降，等权平均差值为 +0.156 pp（显示 +0.16 pp）；
不能宣称一致提升，也不能把五个配对当作五个随机 seed 做显著性解释。
当前七组单 seed ACD 重跑保持原调度；截图补录不改变冻结配方。

**结果找回修正：之前确实跑过全部 11 个配对。** 本次补查旧目录快照，找回 52 条完成至 epoch240 的
线性 KD／余弦头 KD／ETF 初始化对照记录。此前只整理后期 ACD 记录，漏掉了这些历史实验。
详见 [11 个配对的历史结果、配方与缺失原因](historical_results.md)、[逐运行 TSV](historical_runs.tsv)
和 [历史分类头对照 LaTeX 表](historical_head_tables.tex)。这些实验使用普通 KL，不能直接改名为完整 ACD。
**Release 已恢复：** 3 个 Release 共恢复 706 个文件路径，核对 111 条不重复的完成记录，
其中 52 条早期状态文件与目录快照哈希一致，另补出 34 条后期分类头和 25 条线性 KD confirmation。
详见 [Release 结果表与逐 seed 数据](release_recovery/release_results.md)。这些归档不包含 11 月 SICD/ACD 研究包，完整 ACD 的四组已知成绩保持不变。

- [英文正文 main_results.tex](main_results.tex)：可放入论文的段落草稿，ImageNet 部分明确保留未完成状态。
- [CIFAR 两张主表](cifar_main_tables.tex) / [主表预览](cifar_main_tables.md)：按用户给定的 6 组同构、5 组异构配对排列。
- [全部 LaTeX 表格 tables.tex](tables.tex)：八张表，需要 `booktabs`；包括两张主表、截图 ETF 对照、受控消融、补充摘要、ImageNet、替换与效率。
- [Markdown 表格预览](tables.md)：便于阅读和复制。
- [机器可读结果与来源 SHA256](results.json)：逐 seed 数据、均值、样本 SD、配对增益、来源和缺失项。
- [生成脚本](build_tables.py)：使用标准库重新计算表格，不导入模型或发起训练。

## CIFAR-100 主表范围修正

两张 CIFAR 主表固定为用户给定的 11 个配对，顺序如下：

- 同构：ResNet56→ResNet20、ResNet110→ResNet32、ResNet32×4→ResNet8×4、WRN-40-2→WRN-16-2、WRN-40-2→WRN-40-1、VGG13→VGG8。
- 异构：ResNet32×4→ShuffleNetV1、WRN-40-2→ShuffleNetV1、VGG13→MobileNetV1、ResNet50→MobileNetV1、ResNet32×4→ShuffleNetV2。

架构和教师／学生参考精度逐项保存在 [cifar_benchmark_layout.json](cifar_benchmark_layout.json)。
这些精度是用户提供的参考表头，不等于本项目同配方的重新评测结果；其文献来源尚未提供。
表中 † 标注的本地结果只按架构对应暂填，参考表的教师身份、增强、损失权重和评测协议
仍须核对后才能据此排名。NormKD、CRD、ReviewKD 等缺少完整对照数值的行保留空位。

当前能按保留账本归入主表 ACD 行的有 4 个配对，其余 7 个暂缺同级来源的读数。
用户截图为其中 3 个配对补出了 relation-only 和 ETF 数值，暂列截图行，待核实协议及方法身份。
主表新增“Recovered historical head-study runs”一行，显示全部 11 个配对均有旧实验完成记录。
ACD 的空位表示该方法终点记录尚未恢复，不是断言相应配对从未运行。
ResNet32×4→ShuffleNetV1 的 78.97 是 AT+ACD；WRN-40-2→WRN-40-1 的 75.60
是 ReviewKD-style+ACD，均不能填入 standalone ACD 行。

此前两组 VGG 表保留为 `tab:cifar_paired` 受控消融表，不再代替完整主表。
新主表标签为 `tab:cifar_homo` 与 `tab:cifar_hetero`，正文已改为先同构、再异构、
再解释受控配对，最后讨论 ImageNet 和输出损失替换。

## 核心数值

| 比较 | VGG13→VGG8 | VGG13→MobileNetV1 |
|---|---:|---:|
| ACD Top-1，均值 ± 样本 SD | 75.48 ± 0.02 | 69.71 ± 0.01 |
| ACD − affine KD | +1.09 pp | +1.93 pp |
| ACD − ACI+KL | +0.68 pp | +3.12 pp |
| ACD − ACI+standardized-full | −0.07 pp | +0.74 pp |

以上使用 seed2102/2103。均值和配对差从逐 seed 读数先计算、再按两位小数四舍五入。
因此，不能用两个已经四舍五入的均值相减来检查最后一位。例如 MobileNet 的关系项增益
为 `(0.74+0.73)/2=0.735`，显示为 +0.74 pp，而 69.71−68.98=0.73。
SD 使用 `n−1` 分母，表示独立训练 seed 间的离散程度；不是置信区间，也不是显著性检验。

其余保留的 CIFAR 摘要：ResNet32×4→ResNet8×4，ACD=78.36、DKD=77.40、
standardized-full=77.98（三 seed）；WRN40-2→ShuffleNetV1，ACD=78.86、
standardized-full=78.68（两个已归档 seed）。本地没有其逐 seed 源数组，因此不补造 SD。
11 月 11 日文档里的“正在训练”仅为当日状态，不能据此宣称任务现在仍在运行。

## 应修改的原稿表述

| 原表述 | 问题 | 本稿处理 |
|---|---|---|
| consistently achieves state-of-the-art across all pairs | 未建立完整统一协议的最近邻/feature-KD 对照；VGG 上 full KL 略高 | 写清具体配对和具体增益，保留反例 |
| a significant margin (X.X%) | 没有相应显著性检验；准确率之差应为 percentage points | 使用 +1.09 / +1.93 pp，不用 statistically significant |
| bridging geometry mismatch is demonstrated by accuracy | 准确率改善本身不能证明唯一因果机制 | 准确率表负责有效性，机制解释引用独立探针 |
| much more efficient than heavy feature methods | 缺少同硬件的完整 feature-method 成本对照；匹配本身增加训练成本 | 报告已记录的训练开销；不作泛化效率排名 |
| ImageNet ACD outperforms baselines | 当前仅有完整 CE 学生结果，正式 ACD 结果未恢复/完成 | 列出基线和缺失单元格，保持段落为进度快照 |
| Universal Drop-in Replacement | 两个单 seed pilot 不足以支持普适性 | 标题改为 ACD in Existing Distillation Pipelines |
| ReviewKD and CRD original versions are consistently improved | 可核对的 loss-replacement 是 AT 与 ReviewKD-style；没有这一组 CRD 替换结果 | 表中使用实际管线，并列 affine/ACI 两种 KL 对照 |
| no inference-time overhead | 必须说明相对哪个分类头 | 仅就同一角度头上的 KL→SICD 替换，学生推理模型不变 |

特别注意：ReviewKD-style 的 loss-only 提升是 75.60−75.54=+0.06 pp，
但相对 affine+KL 是 75.60−75.74=−0.14 pp。不能把前者描述成对原线性版本的提升。
AT 对应两种差值分别是 +1.37 和 +1.67 pp。

## 来源级别与已发现的不一致

1. **两组 VGG 配对数据**取自
   [保存的数值数组](../../acd_figures/mechanism_figures/acd_evidence_figure_data.json)
   的 `ablation` 部分；seed 语义和配对来源由
   [证据账本](../../../reports/ACD_EVIDENCE_LEDGER_AND_NEXT_PLAN_20261111.md)
   与[审稿审计](../../../reports/ACD_CVPR_REVIEWER_AUDIT_20261111.md)交叉确认。
   本地没有对应的封存研究包和完整检查点，因此本次不是重新评测或完整配置一致性复核。
2. **旧数组存在明确误标**：`external.ACD=[75.51,75.58]` 实际与
   `ablation.vgg13_vgg8.cosine_standardized_full` 一致；真正的
   `cosine_relation_matched` 为 `[75.49,75.46]`。新表使用后者，旧历史输入不覆盖。
   外部 DKD/NCKD/LSKD 点来自不同注册，未混入这两组严格配对表。
3. **扩展配对、替换和效率**使用账本与审计中的汇总。ResNet 行账本记载
   relation-minus-full=+0.39 pp，但已四舍五入的 78.36−77.98=+0.38 pp。
   未恢复原始 seed 数组前，正文仅并列端点均值，不为此差值声称额外精度。
4. **ImageNet**直接读取当前完整 CE 的 `final.json` 和教师全验证集预检记录，
   存储文件哈希。普通 KD 的最新读取进度记录在 `results.json`，尚无 `final.json`；
   表格因此留空。I7 的 210 步方程重建预检不能充当正式 ACD 准确率。
5. 此处 ACD 按现有结果命名定义为 `ACI + gradient-matched conditional loss`。
   这些读数不单独验证 ETF 初始化收益，也不应宣称已证明包含 ETF 的每项机制。

## 表格使用建议

正文以同构主表（`tab:cifar_homo`）与异构主表（`tab:cifar_hetero`）组织 benchmark，
配对表（`tab:cifar_paired`）用于受控消融，替换表（`tab:dropin`）用于组合性。
主表缺失项及本地记录与参考表的协议对应补齐后，才能作为完整的正式方法排名。
当前不能仅挑多个协议中较高的读数填入同一列并标为公平比较。

ImageNet 表目前属于进度快照。正式投稿前至少需要完整 KD、ACD 以及主张涉及的公平
对照，才能把本段改成“验证大规模迁移”的结果段落。没有完成的单元格是缺失，不是 0。

效率表保留已知训练开销：matched relation / ACI+KL = 30.04/15.09，约 1.99 倍。
它不是 ImageNet 每步时延，也不是所有架构上的固定倍数。原始 GPU、batch、模型和测量
样本信息恢复前，不建议将它作为独立的最终效率主表。

## 复现

在 `KD/` 目录运行：

```bash
.venv/bin/python paper_assets/acd_tables/main_results_20261116/build_tables.py
```

脚本生成 `cifar_main_tables.tex/md`、`tables.tex`、`tables.md` 和 `results.json`，保留缺失项并显式检查旧 ACD
误标，避免静默使用错误序列。若以后新增完整 I7 结果，脚本会要求先核对新结果的实验协议。
同时调用 [历史结果恢复脚本](recover_historical_results.py)，生成 `historical_results.md/json`、
`historical_runs.tsv` 和 `historical_head_tables.tex`；逐项对账目录快照，按共同完成 seed 计算统计量，
并核对旧训练配置与扫描时保存的 SHA256。全部八张表与两张历史表分开保留，避免混合不同配方。
同一生成脚本读取用户截图转录，校验五组差值，并生成 `current_acd_records.md` 与
`user_reported_etf_table.md/tex`。独立 ETF 表已包含在 `tables.tex` 中，勿重复引入。
`main_results.tex` 为人工编写的快照段落；数值或训练状态更新后须同步检查正文。

LaTeX 主文件导言区需要 `\usepackage{booktabs}`；按需分别 `\input` 正文和表格。
不必把全部七张表都放入主文；只需两张 CIFAR 主表时可单独引用 `cifar_main_tables.tex`。
不要同时引入这个文件与包含它们的 `tables.tex`，以免标签重复。
