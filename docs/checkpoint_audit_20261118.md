# 11 对 CIFAR-100 模型 × 7 种设置：checkpoint 盘点

**审查日期：2026-11-18。结论：77 个组合中，38 个组合至少有一个完整历史 checkpoint；39 个组合在本次范围内未找到。** 这是资产覆盖率，不代表已经完成同 seed、同实现、同配方的七臂比较。

范围包括本机 `KD/`、`ACD/` 工作区的权重与缓存归档、KD 的 3 个 Release（177 个附件）、ACD 的 2 个 Release（57 个附件），以及两个仓库默认分支的完整文件树。默认分支均没有直接提交的 `.pth/.pt/.ckpt/.safetensors` 文件。原 `/lanius/kd` 路径在本机不存在；未把旧机器或未检查的外部备份推断为已永久丢失。

本次只整理资产和缺口，没有启动训练、修改队列、上传模型或推送代码。

## 1. 模型与方法口径

11 个配对逐项对应本地作者 PDF `acd_v1.pdf` 的 Table 1（6 对同构）和 Table 2（5 对异构）。

| 简写 | 用户要求 | 盘点中的明确含义 |
|---|---|---|
| KD | kd | 原始 affine/linear 分类头 + 普通温度 KL |
| C | 余弦头 | 通用初始化、分类方向可自由学习的余弦头 + 普通 KD |
| C+I | 余弦头 + ETF 初始化 | 只用 ETF 初始化，之后自由学习方向 + 普通 KD |
| E | 可训练 ETF | 按历史代码解释为 Stiefel/旋转 ETF；全程保持 ETF Gram，只学习 frame 的朝向 + 普通 KD |
| C+S | 余弦头 + SICD | 余弦头 + 双侧标准化、带教师非真类质量权重的条件 KL、骨干梯度范数匹配 |
| C+I+S | 余弦头 + ETF 初始化 + SICD | 初始化后方向自由学习 + 同一 SICD |
| E+S | 可训练 ETF + SICD | 始终保持 ETF 几何的旋转头 + 同一 SICD |

`C+I` 的参数也可训练，但不保持 ETF 几何；因此它与 `E` 是两个不同实验。固定 ETF、固定轴向 ETF、教师方向头、教师置乱、CE-only、标准化 full-KL、feature-KD 组合和图像诊断缓存均不冒充上述七臂。若“可训练 ETF”有不同于历史旋转 ETF 的定义，E/E+S 两列需要重新分类。

## 2. 77 格覆盖矩阵

**✓：已找到本地权重或有当前 Release 清单支持的最终权重；缺：未找到；维：未找到且原始特征维度不足；记：存在历史成绩记录但未找到对应权重。** ✓ 不表示已经重测所有历史文件。

| 类型 | 教师 → 学生 | KD | C | C+I | E | C+S | C+I+S | E+S |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 同构 | ResNet56 → ResNet20 | ✓ | ✓ | 维 | 维 | ✓ | 维 | 维 |
| 同构 | ResNet110 → ResNet32 | ✓ | ✓ | 维 | 维 | ✓ | 维 | 维 |
| 同构 | ResNet32×4 → ResNet8×4 | ✓ | ✓ | ✓ | ✓ | 记 | 缺 | 缺 |
| 同构 | WRN40-2 → WRN16-2 | ✓ | ✓ | 缺 | 缺 | ✓ | 缺 | 缺 |
| 同构 | WRN40-2 → WRN40-1 | ✓ | ✓ | 维 | 维 | ✓ | 维 | 维 |
| 同构 | VGG13 → VGG8 | ✓ | ✓ | 缺 | 缺 | 记 | 记 | 缺 |
| 异构 | ResNet32×4 → ShuffleNetV1 | ✓ | ✓ | 缺 | ✓ | ✓ | 记 | 缺 |
| 异构 | WRN40-2 → ShuffleNetV1 | ✓ | ✓ | ✓ | ✓ | 记 | 记 | 缺 |
| 异构 | VGG13 → MobileNetV1 | ✓ | ✓ | ✓ | ✓ | 记 | 缺 | 缺 |
| 异构 | ResNet50 → MobileNetV1 | ✓ | ✓ | ✓ | ✓ | ✓ | 记 | 缺 |
| 异构 | ResNet32×4 → ShuffleNetV2 | ✓ | ✓ | 缺 | 缺 | ✓ | 记 | 缺 |

| 设置 | 有 checkpoint 的配对数 | 未找到 checkpoint 的配对数 |
|---|---:|---:|
| KD | 11 | 0 |
| C | 11 | 0 |
| C+I | 4 | 7 |
| E | 5 | 6 |
| C+S | 7 | 4 |
| C+I+S | 0 | 11 |
| E+S | 0 | 11 |

## 3. 需要补齐哪些组合

以下是**39 个资产缺口**，每格只按至少一个完整训练端点计数；尚未指定新增训练 seed。先查找遗失的历史权重，不能把“未找到”写成“从未训练”。

| 设置 | 缺少 checkpoint 的配对 | 数量 |
|---|---|---:|
| C+I | ResNet56 → ResNet20；ResNet110 → ResNet32；WRN40-2 → WRN16-2；WRN40-2 → WRN40-1；VGG13 → VGG8；ResNet32×4 → ShuffleNetV1；ResNet32×4 → ShuffleNetV2 | 7 |
| E | ResNet56 → ResNet20；ResNet110 → ResNet32；WRN40-2 → WRN16-2；WRN40-2 → WRN40-1；VGG13 → VGG8；ResNet32×4 → ShuffleNetV2 | 6 |
| C+S | ResNet32×4 → ResNet8×4；VGG13 → VGG8；WRN40-2 → ShuffleNetV1；VGG13 → MobileNetV1 | 4 |
| C+I+S | 全部 11 对 | 11 |
| E+S | 全部 11 对 | 11 |

按当前 ACD 代码是否能够执行，可再分成三组：

- **16 格：已有对应训练路径，但无 checkpoint。** C+I 的 WRN40-2→WRN16-2、VGG13→VGG8、ResNet32×4→ShuffleNetV1/V2 共 4 格；C+S 的 ResNet32×4→ResNet8×4、VGG13→VGG8、WRN40-2→ShuffleNetV1、VGG13→MobileNetV1 共 4 格；C+I+S 的 8 个非 64 维学生配对。
- **11 格：维度可行，但需要接入旋转 ETF 训练。** E 的 WRN40-2→WRN16-2、VGG13→VGG8、ResNet32×4→ShuffleNetV2 共 3 格；E+S 的 8 个非 64 维学生配对。当前 `ACD/train.py` 的 `--etf` 仅代表初始化，不能执行 E/E+S。
- **12 格：先解决 64 维分类接口。** ResNet56→ResNet20、ResNet110→ResNet32、WRN40-2→WRN40-1，各自的 C+I、E、C+I+S、E+S 共 4 格。

当前 100 类 simplex ETF 至少需要 99 维；历史 Stiefel 实现采用 100 列 frame。三种学生的原始维度是 64，直接打开 `--etf` 会失败。旧 README 登记过 64→128 非线性升维及其配对对照，但当前工作区和 Release 没有该研究的 checkpoint。升维改变了头部结构，需要同时设置相同升维模块的非 ETF 控制，才能解释 ETF 的贡献；这些控制可能增加最终训练数量。

## 4. 有历史权重，并不等于现配方已经齐全

**异构 5 对的后期 `compact_double` 线性 KD：有 25 条完成记录（每对 seed2101–2105），但当前本地和所有 Release 清单中都找不到对应 checkpoint SHA256。** 这些逐条列在 `record_only.csv`，不能用早期文件覆盖其完成记录，也不能据此再声称后期权重仍可下载。

矩阵中的异构 KD“✓”来自更早的 `early_double`：lr=0.05、CE/KD=0.5/0.5。后期异构 head 对照与 ACD rerun 使用 lr=0.01、CE/KD=0.1/0.9。若下一轮采用后期配方，这 5 个 KD 格还需恢复或重训。ResNet32×4→ShuffleNetV2 的普通余弦头也只找到早期配方，后期同配方 C 格尚无权重。这 **6 个额外的配方缺口** 单列在 `recipe_alignment_gaps.csv`，不与 39 个纯资产缺口混算。

此外，已发布 C+S 是 seed3201；多数普通头/ETF 对照来自 seed2051–2055 或 seed2101–2105。当前没有任一配对凑齐同 seed、同实现的七臂。因此 **39 是“缺少文件”的下界，不是完整严格配对实验所需新增训练数**。训练前应固定 teacher、数据划分、增强、输入顺序、seed、优化器、头部结构和 SICD 数值约定。

## 5. 已有模型具体在哪里

| 来源 | 可用内容 | 提取/使用方式 |
|---|---|---|
| KD `model-backup-20261027` | 11 对早期 KD/C；ResNet32×4→ResNet8×4 的 C+I；该对 seed2051/2052 的旋转 ETF；旧诊断模型与教师 | 按 `runs.csv` 的 archive + 内部 path + SHA256 恢复，不能只看压缩包名称 |
| KD `kd-head-models-20261031-completed` + `...-final` | 后期 C/C+I，以及 4 对异构旋转 ETF（seed2101） | 两个 Release 是增量关系，final 不是完整替代包 |
| ACD `cifar100-rerun-20261116` | 7 个 C+S（seed3201，无 ETF）和 5 个教师 | 已有逐模型便携权重、匹配配置和完整测试验证 |
| ACD `cifar100-head-controls-20261031` | 11 个 KD/C/C+I 对照与 VGG13 教师 | 是从 KD 抽取的便携导出，不是额外的训练 seed |
| 本地 KD rerun runs / ACD outputs | 原始训练权重、上述导出副本及缓存归档 | 已逐个计算 SHA256；路径列在 `runs.csv` 和 JSON 中 |

去重后，主索引有 **95 条已定位 checkpoint 的完成运行**；它们覆盖 38 格。25 条后期 KD 完成记录仅有证据、未定位权重。另有更早的 `kdhead_*` 模型单列在 `additional_legacy_assets.csv`：其中名字叫 `cosine` 的旧实现实际上由 ETF 初始化，且部分 scale=10，不能作为普通初始化、scale=30 的 C 对照。早期/最佳快照、同一文件的重复备份及 ACD 导出均不当作独立 seed。

当前有便携 ACD Release 权重的格子是 18 格，能直接找到解包后的本地权重是 19 格，其余格子需要从 KD 归档恢复；“不在 ACD Release”不等于“没有 checkpoint”。六种教师架构均已具备可下载权重。

值得优先恢复的遗漏资产是 **可训练旋转 ETF**：ResNet32×4→ResNet8×4（seed2051、2052）以及 ResNet32×4→ShuffleNetV1、WRN40-2→ShuffleNetV1、VGG13→MobileNetV1、ResNet50→MobileNetV1（各 seed2101）。这 5 个格子已经有 KD Release 权重，不应作为完全没训练的空白重跑。它们目前没有单独导入 ACD Release。

## 6. 只有成绩而缺少权重的 SICD 记录

历史证据账本的四组 C+S：ResNet32×4→ResNet8×4 78.36（3 seeds）、VGG13→VGG8 75.48（2 seeds）、WRN40-2→ShuffleNetV1 78.86（2 seeds）、VGG13→MobileNetV1 69.71（2 seeds），在当前范围都没有找回对应 endpoint。论文、账本、用户截图和图表 JSON 中的值不能替代 checkpoint。

五组用户转录的 ETF+relation-only 结果也仅作为待核实历史记录保留；seed、划分、最佳/最终选择及与当前 SICD 的一致性未核实。它们不会把 C+I+S 的 0/11 覆盖率改成“已完成”。

## 7. 可直接使用的盘点文件

- [覆盖矩阵 CSV](../results/checkpoint_audit_20261118/coverage.csv)：77 格、可用位置、历史 seed、维度约束和下一步。
- [39 格缺权重清单](../results/checkpoint_audit_20261118/missing_cells.csv)：只列资产缺口。
- [待补训练设置](../results/checkpoint_audit_20261118/training_gaps.csv)：按可直接支持／旋转头集成／维度设计分组，全部 `not_scheduled`。
- [已有/缺失逐运行索引](../results/checkpoint_audit_20261118/runs.csv)：逐个 seed 的原始 SHA256、本地路径、KD archive/path、ACD 导出。
- [后期 KD 的 25 条缺权重记录](../results/checkpoint_audit_20261118/record_only.csv)。
- [6 个额外配方缺口](../results/checkpoint_audit_20261118/recipe_alignment_gaps.csv)。
- [早期诊断模型附件](../results/checkpoint_audit_20261118/additional_legacy_assets.csv)：避免误把旧 ETF 初始化头当普通余弦头。
- [完整机器可读索引](../results/checkpoint_audit_20261118/inventory.json)：包含来源、教师、历史读数和审查边界。

验证方式：本地权重与缓存归档计算 SHA256；Release 清单与当前 GitHub 附件大小/摘要核对；完成记录与源 checkpoint hash 交叉匹配；ACD 导出通过 source_sha256 去重。此次没有重新下载、加载或重测全部历史模型；Release-only 项的证据级别是“清单确认文件及归档附件存在”，不是新一轮精度复核。

主要本地依据：`KD/reports/ACD_EVIDENCE_LEDGER_AND_NEXT_PLAN_20261111.md`、`KD/reports/research_catalog_20261027/{native_statuses,source_files}.tsv`、`KD/paper_assets/acd_tables/main_results_20261116/release_recovery/`、`KD/reproductions/head_mechanism_20261022/methods.py`、`KD/README.md` 中 2026-11-03 升维登记，以及当前 ACD 配置和模型定义。
