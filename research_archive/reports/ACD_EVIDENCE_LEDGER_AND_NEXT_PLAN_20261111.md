# ACD 实验证据总账本、去重核验与下一步计划

**更新时间：2026-11-11。** 本文是 ACD 论文唯一的训练证据账本：按论文问题和
`(teacher→student, seed, arm, protocol)` 组织，而不是沿用多个历史 dispatcher 的
`pending/running` 标签。每一个“完成”均指 epoch-240 checkpoint reload 后的一次 official
CIFAR-100 test；仅有训练日志、预检或旧 ledger 条目均不算结果。

## 0. 最重要的去重结论

11-08 的“三 seed 扩展”原计划新增 46 臂；随后 priority interruption、独立 interaction
实验和 completion recovery 把其中若干科学单元分散到不同注册中。**旧队列中仍显示为
pending/running 的条目不等于缺失实验。**

| 原扩展工作包 | 原计划 | 经核验后的实际覆盖 | 处理决定 |
|---|---:|---|---|
| WRN40-2→ShuffleNetV1 四臂 | 8 | seed2101、2102完成；seed2103四臂正在训练 | 保留当前四 worker；不新建重复臂。 |
| ViT-T→ResNet8×4 七臂 | 14 | 2101 已完成；2102/2103 的 KD/full/relation 由 interaction/recovery 组合覆盖；DKD 与 linear-relation 的一部分已由 completion audit 核验 | **冻结旧 extension/recovery 的 12 个 pending 和 4 个陈旧 running 标签**，先做配置同一性审计，绝不自动释放。 |
| VGG13→MobileNetV1 主机制 | 18 | 2102/2103 的核心 linear/cosine、full/relation 和两个 linear-full 已完成；部分 affine/profile 扩展已降级或另有已登记 controls | 核心 factorial 已闭环；不再按原九臂盲目补种子。 |
| VGG13→MobileNetV1 RKD 组合 | 6 | 2102/2103 三臂均已完成 | 已闭环；不增加 RKD。 |

旧 ViT recovery 的四个“failed”记录在 epoch240 后才因 seal 校验报错，但 prediction 文件已
写入。`acd_priority_completion_recovery_20261110` 对其 checkpoint、epoch history、保存的
10k prediction 与标签顺序作了只读核验，确认均已有一次有效 official test；它们不是应当
续训或重跑的失败训练。

## 1. 论文要回答的四个问题

1. **ACI 是否对 KD 有特异价值？** 应看 CE×KD 的差分中的差分，而非单看 cosine CE。
2. **SICD 是否在 ACI 上选择了更有用的监督？** 应比较 cosine KD、standardized-full 和
   matched conditional relation，并保留负例。
3. **优势是否只是某个 logit baseline 或某个 pair 的偶然现象？** 需要公平 DKD/NCKD/SICD
   同协议对照与未见 pair 检验。
4. **能否替换 feature-KD 中的输出 KL？** 必须保持 feature loss、adapter、初始化和预算不变。

以下所有表均为 Top-1 (%)；除另有说明外为 full-50k、240 epoch、final-only official test。

## 2. 已完成的同构与异构主实验

### 2.1 ACI：严格配对的 CE×KD interaction

`acd_cnn_interaction_20261109` 使用相同教师、初始骨干、数据顺序、优化器和训练预算；
仅改变头和训练损失。表中为 seed2102/2103 均值，
\(\Delta_{ACI}=(C_{KD}-C_{CE})-(L_{KD}-L_{CE})\)。

| Pair | 类型 | linear CE | cosine CE | linear KD | cosine KD | \(\Delta_{ACI}\) | 能够支持的结论 |
|---|---|---:|---:|---:|---:|---:|---|
| VGG13→VGG8 | 同构 | 72.88 | 71.33 | 74.39 | 74.80 | **+1.96** | 余弦头的 KD 净效应高于其 CE 净效应。 |
| VGG13→MobileNetV1 | 异构 | 65.39 | 63.14 | 67.78 | 66.60 | **+1.07** | 即使余弦头使 CE 和绝对 KD 更低，KD 特异差分仍为正。 |

这回答的是 ACI“改变 KD 作用”的问题，而不是“余弦头一定提高最终准确率”。异构 pair 中
linear KD 仍比 cosine KD 高 1.18pp；该负面事实必须保留。

### 2.2 SICD：同一余弦接口上的 full 与 relation

| Pair / 可合并 seed | standardized-full | SICD relation | relation−full | 判读 |
|---|---:|---:|---:|---|
| VGG13→MobileNetV1，2102/2103 | 68.98 | 69.71 | **+0.74** | 两 seed 均约 +0.73pp，是当前最清楚的异构正例。 |
| VGG13→VGG8，2102/2103 | 75.55 | 75.48 | **−0.07** | 两 seed 都不支持 relation 额外收益（−0.02、−0.12）。 |
| ResNet32×4→ResNet8×4，2101/2102/2103 | 77.98 | 78.36 | +0.39 | 三 seed 支持性证据；relation 并非每 seed 最优。 |
| WRN40-2→ShuffleNetV1，2101/2102 | 78.68 | 78.86 | +0.18 | 2101 为 −0.13、2102 为 +0.49；2103仍在训练，不能宣称三 seed 结论。 |

因此 SICD 的准确表述应是“**pair-dependent improvement**”，而不是稳定普适改进。尤其同构
VGG13→VGG8 是必须保留的反例。

### 2.3 异构 MobileNet 的三-seed factorial 覆盖

三个 seed 的核心 `(linear/cosine) × (full/relation)` endpoint 已由 supplementary、
interaction、preflight-recovery 和 interrupted-recovery 四个 sealed study 覆盖。每个
同 seed 差分的初始化和训练配方已在原注册中配对；但不同 seed 来自不同注册，因此在形成
一张正式三-seed 主表前仍须完成 source/config/input-hash 的跨登记一致性审计。

| seed | linear full | linear relation | cosine full | cosine relation |
|---|---:|---:|---:|---:|
| 2101 | 70.19 | 70.54 | 69.32 | 69.81 |
| 2102 | 70.26 | 70.27 | 68.96 | 69.70 |
| 2103 | 69.46 | 70.49 | 68.99 | 69.72 |

观察上，linear relation 也不弱于 cosine relation；故“角度接口是 SICD 必要条件”不成立。
这张表更适合回答“标准化条件关系的 pair/头部边界”，不应被包装为 ACI 必然性证明。

### 2.4 ViT-T→ResNet8×4：覆盖已分散，暂不可直接总平均

seed2101 的七臂来自 `cvpr_supplement_queue`；2102/2103 的核心项分布在原 recovery 和
后续 angular interaction 中。例如：2102 的 linear/cosine KD 已作为 interaction 的
外部核验 reference；2103 的 KD、full/relation 则由 interaction 完成。四个其后标 failed
的 ViT 单元由 completion audit 验证为有效保存终点：

| 经只读核验的端点 | official Top-1 |
|---|---:|
| 2102 linear relation | 65.44 |
| 2103 linear DKD | 57.80 |
| 2103 cosine DKD | 58.04 |
| 2103 linear relation | 65.24 |

这些数值说明该 weak-Teacher ViT→CNN 设定的 KD/ACI 故事不稳定（例如 cosine CE 约 75.5%，
而 cosine KD 约 61），不能被列为性能主表或用作跨架构 SOTA 证据。它仍可作为“不同
classifier/training setting 会出现反常行为”的分析反例。下一步不是重跑旧 pending，而是
先审计所有 arms 是否具有相同 teacher、recipe、initialization 和 input-hash 语义；不一致则
只保留逐注册描述，不合成三-seed 均值。

## 3. 与其他 KD 方法、组合性和效率

### 3.1 DKD / NCKD

ResNet32×4→ResNet8×4 的三 seed 中 relation 平均 78.36，高于 DKD 77.40；但这不能推广。
VGG13→VGG8 和 VGG13→MobileNetV1 的 DKD/NCKD 历史结果含不同 registration、初始化或输入
轨迹，不能与 interaction 的 SICD 直接排名。当前结论是“有竞争力”，而非“优于 DKD/NCKD”。

### 3.2 feature-KD 的 output-loss replacement

| 固定 feature pipeline，单 seed | linear+KL | cosine+KL | cosine+SICD | SICD−cosine KL |
|---|---:|---:|---:|---:|
| ReviewKD-style + KL，WRN40-2→WRN40-1 | 75.74 | 75.54 | 75.60 | +0.06 |
| AT + KL，ResNet32×4→ShuffleNetV1 | 77.30 | 77.60 | 78.97 | **+1.37** |

AT 是一组干净的正向替换 pilot；ReviewKD 的改善接近噪声量级。另有 VGG13→MobileNetV1
ReviewKD-style reuse audit（linear+KL 68.48、cosine+KL 67.48、cosine+SICD 70.43），但这是
另一注册/实现，不应与 WRN 数值合并。现有证据只支持“可组合、部分 pipeline 有收益”。

RKD 组合两 seed 已完成：linear RKD 65.50、cosine RKD 65.02、cosine RKD+relation 66.31。
它没有 KL output term，因此只能作为附加组合证据，不能作为 KL replacement 论据。

### 3.3 效率

严格同硬件/loader benchmark 已完成：cosine KD 15.09 ms/update、DKD 15.55、
standardized-full 15.51、matched relation 30.04。ACD 不改变 inference graph 且不增加
推理参数；但当前 relation 实现为匹配骨干梯度范数而额外查询梯度，训练 step 约为 ordinary
KD 两倍。论文必须报告此训练开销，不能写“extreme training efficiency”。

## 4. 已完成但仅限分析/机制的工作

- 分类器分解、正仿射不变性与 finite-temperature KL 分解：是数学/代码一致性证据，不是
  准确率定理。
- 固定 checkpoint 的 distribution deformation、KD force、trajectory 图：说明尺度/范数/
  bias 可改变局部监督与梯度方向；不能代替 end-to-end 因果或泛化证据。
- teacher scale/shift 控制、content permutation、binary relation component：可作为
  supplementary 的选择性机制材料。uniform shift 的 softmax 不变性不值得继续投入长训练。
- ETF、rotating ETF、原型方向：结果混合且非核心，不再扩大。

## 5. 当前真实运行和已注册等待项

1. **正在训练：** WRN40-2→ShuffleNetV1 seed2103 的 cosine KD、DKD、full、relation 共 4 臂。
2. **已注册并等待 WRN 完成：** 8 个 follow-up controls：两 pair 的 cosine-DKD 重复 4 臂，
   以及 MobileNet 的 teacher scale/temperature/relation controls 4 臂。
3. **明确不自动启动：** 旧 ViT extension/recovery 残留 12 pending、4 stale running；
   它们与上面的 endpoint/interaction 记录重叠，须先做跨注册一致性审计。

## 6. 下一步计划（以信息增益排序）

### 阶段 A：完成并收口既有队列

1. 完成 WRN seed2103 和 8 个 follow-up，逐 seed 输出完整表、均值、样本 SD、同 seed 差。
2. 对 ViT 三-seed coverage 生成只读 manifest：按 `(seed, arm)` 列出唯一 canonical
   checkpoint、teacher/source/config/input/initial hash、official-test provenance、以及能否与
   同 seed comparator 合并。旧 pending 全部保持冻结。
3. 对 MobileNet 三-seed factorial 也完成同样 manifest；只有审计通过才写三-seed主表。

### 阶段 B：补最能改变审稿判断的新训练

先澄清命名：**ACI** 是 student 的角度分类器接口，**SICD** 是标准化条件非真类损失，
完整方法才是 **ACD = ACI + SICD**。因此，主方法对比表不能把 “SICD” 写成与 KD/DKD
并列的方法名；它应写作 **ACD**。将所有外部 output objectives 放在同一个 ACI 上，是为
了隔离“loss 选择”的问题，不能误称为这些方法的原生发表配方复现。

1. **公平 closest-logit baseline（优先）：** 在一个异构正例和一个同构/边界例上，各以
   一个 fresh seed 固定 trainer/teacher/init/data/epoch/T 和 ACI，运行以下六臂：
   **ACI+KL**（ordinary KD 基线）、**ACI+DKD**（target/non-target decoupling）、
   **ACI+SKD**（Streamlined KD，若作者实现可在同一训练器中锁定）、**ACI+NCKD**
   （最邻近的 non-target 对照，前提是其精确定义/权重预注册）、**ACI+standardized-full**
   （标准化而不筛选信息）与 **ACD = ACI+SICD**。这不是无目的地堆方法：每一项分别
   排除普通 KL、解耦、已有 streamlined/non-target 设计，以及“仅标准化”四种替代解释。
   若 SKD 或 NCKD 无法在同一 recipe 下忠实实现，明确删去该臂，而不是使用不同作者配方
   的历史结果凑排名。

   另在**消融表**中保留 `linear+KL`、`ACI+KL`、`linear+SICD`、`ACD` 与
   `ACI+standardized-full`；前两项和 CE-only 对照识别 ACI 的 KD 特异效应，后两项识别
   SICD 是否必须依赖 ACI。方法排名与模块消融不得混为同一张表。
2. **未见异构 pair 的 ACI×SICD 因子验证：** 优先有 source-lock 的
   ResNet50→MobileNetV1；一 fresh seed 跑 linear/cosine CE-only、linear/cosine KD、
   cosine full、**ACD（ACI+SICD）** 六臂。它同时测 ACI difference-in-differences 和 SICD，优先级
   高于再跑 RKD/ETF。
3. **feature replacement 的第二个复现：** 在第二个 AT+KL pair 运行 linear+KL、
   cosine+KL、**ACD output loss（ACI+SICD）** 三臂。只有 AT 正例可复现后，才值得由用户决定是否扩 seed。

### 阶段 C：只在阶段 B 支持时做

- 对最有判别力的一个正例及一个边界例，为 SICD/KL/DKD 独立增加两 seed；预先固定所有
  recipe，不可基于首轮结果选 pair、选超参或挑 checkpoint。
- ImageNet-1K 迁移到具备数据、冻结教师、评测代码和 checkpoint 空间的平台后，另行注册；
  当前本地不可伪造等待队列。

## 7. 当前论文应如何表述

可表述：ACI 改变了两个受控 pair 的 KD 特异交互；SICD 在部分 pair/pipeline 上优于
matched standardized-full/KL，并保持反例与训练成本。

不可表述：SICD 或 ACD 普遍优于 DKD/feature-KD、余弦头是必要条件、ACD 训练开销等同
ordinary KD、或已达到 SOTA。

## 8. 可追溯来源

- 三-seed原计划：`reports/CVPR_THREE_SEED_EXPANSION_20261108.md`。
- interaction：`reproductions/acd_cnn_interaction_20261109/artifacts/` 与
  `reproductions/cvpr_angular_interface_interaction_20261109/artifacts/`。
- ViT completion audit：`reproductions/acd_priority_completion_recovery_20261110/artifacts/completion_audit.json`。
- WRN：`reproductions/cvpr_geometry_falsification_preflight_recovery4_20261108/artifacts/` 与
  `reproductions/cvpr_wrn_three_seed_extension_20261108/artifacts/`。
- feature replacement：`reproductions/acd_at_res32x4_sv1_priority_20261111/artifacts/` 和
  `reproductions/acd_reviewkd_wrn_priority_20261110/artifacts/`。

