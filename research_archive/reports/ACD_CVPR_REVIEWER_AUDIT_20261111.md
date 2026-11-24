# ACD：以严格 CVPR 审稿标准审计现有证据与补充计划

**审计日期：2026-11-11。** 本文只使用训练结束后重载 epoch-240 并进行一次
official CIFAR-100 test 的状态记录；正在训练、协议不同的历史结果和机制探针均不混入
主性能结论。本文按论文问题组织，而非按运行批次组织。

## 审稿结论先行

当前工作已经有一个有价值、但必须收窄的故事：在两组严格配对的 CIFAR-100
interaction 试验中，余弦头的 **KD 特异增益** 为正；SICD 在异构
VGG13→MobileNetV1 上稳定优于同一余弦头的 standardized-full，而在同构
VGG13→VGG8 上没有额外增益。另有 AT+KL 的一组正向 loss-replacement pilot。

这不足以支撑 “SOTA”、“universal” 或 “consistently significant”的摘要表述。最重要的
反例是：同构 VGG 对上 relation 相对 full 的收益约为零；WRN 未见 pair 的两个已完成
seed 一正一负；ReviewKD+KL 的增益只有 +0.06pp。论文应定位为**有明确适用边界的
classifier-coordinate / conditional-relation 分析驱动方法**，而不是所有 KD 的替代品。

## 1. ACI 是否有 KD 特异作用？——目前最干净的证据

`acd_cnn_interaction_20261109` 对每个 pair 固定教师、学生初始化、数据顺序、优化器和
240 epoch 预算；每个数值是 seed2102/2103 的均值（Top-1，%）。记
\(\Delta_{ACI}=(C_{KD}-C_{CE})-(L_{KD}-L_{CE})\)。这是比“余弦 CE 更好/更差”更恰当的
ACI 检验。

| Pair | linear CE | cosine CE | linear KD | cosine KD | \(\Delta_{ACI}\) | 审稿判断 |
|---|---:|---:|---:|---:|---:|---|
| VGG13→MobileNetV1（异构） | 65.39 | 63.14 | 67.78 | 66.60 | **+1.07** | 余弦头本身降低 CE，但 KD 下的相对收益更大；支持 KD 特异交互。 |
| VGG13→VGG8（同构） | 72.88 | 71.33 | 74.39 | 74.80 | **+1.96** | 同样为正，且并非由余弦 CE 提升造成。 |

这两行足以拒绝“只看 cosine CE 准确率就证明 ACI”的错误论证，也支持更窄的说法：
**在这两个固定配方的 pair 上，角度学生接口改变了 KD 的净作用。**

局限：仅两 seed、两个 CIFAR pair；它没有证明几何失配是唯一因果机制，也不证明余弦头
总体优于线性头（异构 pair 的 linear KD 仍高于 cosine KD）。

## 2. SICD 是否始终优于 full standardized KD？——答案是否定的

| 严格同协议组 | cosine standardized-full | cosine SICD/relation | relation − full | 判断 |
|---|---:|---:|---:|---|
| VGG13→MobileNetV1，seed2102/2103 均值 | 68.98 | 69.71 | **+0.74** | 两个 seed 均为正（+0.74/+0.73）；是目前最强正例。 |
| VGG13→VGG8，seed2102/2103 均值 | 75.55 | 75.48 | **−0.07** | 两个 seed 的差为 −0.02/−0.12；full 略高，不能声称稳定提升。 |
| ResNet32×4→ResNet8×4，三 seed | 77.98 | 78.36 | +0.39 | 支持性补充，但没有同一批 CE factor，因此不能作为 ACI 主证据。 |
| WRN40-2→ShuffleNetV1，已完成 seed2101/2102 | 78.68 | 78.86 | +0.18 | 单 seed 差分别为 −0.13/+0.49；seed2103 尚在训练，不得提前平均或下结论。 |

结论是条件性的：SICD 对 VGG→MobileNet 的增益可靠地出现，但在 VGG→VGG8 上未出现。
这恰好是论文应保留的反证边界；删去它会使审稿人合理怀疑选择性汇报。

## 3. 相对 DKD / 其他 logit 方法，证据是否足够？

还不够。ResNet32×4→ResNet8×4 的同一三-seed 矩阵中，relation 均值 78.36 高于 DKD
77.40；但 VGG13→VGG8 的 DKD 记录与 interaction 记录来自不同独立注册，不能作为公平
方法排名。VGG13→MobileNetV1 也存在不同初始化或输入历史的 DKD/NCKD/LSKD 记录。

因此目前只能说“在部分严格配对设置中，SICD 是有竞争力的输出目标”；不能写“优于
DKD/LSKD”等跨方法 SOTA 结论。协议混合的外部方法点图只能留在内部诊断或补充说明中，
不能进入主表排名。

## 4. 能否作为 feature-KD 的 output-loss replacement？

这部分是有希望、但仍是 pilot 的。

| pipeline（均单 seed） | linear + KL | cosine + KL | cosine + SICD | SICD 相对 cosine KL |
|---|---:|---:|---:|---:|
| ReviewKD-style + KL，WRN40-2→WRN40-1 | 75.74 | 75.54 | 75.60 | +0.06 |
| Attention Transfer + KL，ResNet32×4→ShuffleNetV1 | 77.30 | 77.60 | 78.97 | **+1.37** |

AT 是更干净的替换例子：feature attention loss 在所有臂保持不变，替换的正是输出 KL。
ReviewKD 的正增益极小，且其 ABF/HCL adapter 更复杂。两者共同支持“**可插入且不必然
破坏 feature KD**”，不支持“普遍且显著改进所有 pipeline”。此前 VGG13→MobileNetV1 的
ReviewKD-style 三臂复用审计也给出 cosine+SICD 70.43、linear+KL 68.48；它是一个
支持性异构例，但与上述 WRN pilot 应分别报告。

RKD 不应再用来证明 KL replacement：它原本是 CE+relation feature loss，没有 KL output
项。其 VGG13→MobileNetV1 两-seed composition 结果从 cosine RKD 65.02 到
RKD+SICD 66.31（+1.30），仅能支持“额外组合可以工作”。

## 5. 效率、机制和报告边界

同 checkpoint/GPU/batch/loader 的已完成 benchmark 显示 median update time：cosine KD
15.09 ms、DKD 15.55 ms、standardized-full 15.51 ms、matched relation 30.04 ms。
SICD 没有推理时额外模块或参数，但训练并非“extreme efficiency”：其梯度范数匹配会引入
额外梯度查询，当前实现约为 ordinary KD 的两倍 update 时间。论文应同时报告这两个事实。

现有概率、方向力和 checkpoint 图是固定 checkpoint 的机制诊断，能说明参数化改变输出
分布及局部 KD 力；它们不能单独证明长程训练冲突或泛化因果。它们应服务于动机/分析，
不替代端到端对照。

## 6. 当前进行中与已登记任务

1. WRN40-2→ShuffleNetV1 seed2103 的 cosine KD、DKD、full、relation 正在训练；完成前
   不汇总 WRN 三 seed，也不使用其作方法排名。
2. 已登记的 8 个 follow-up controls 在 WRN 结束后自动调度：四个 cosine-DKD 重复及
   VGG13→MobileNetV1 的 shift/scale/temperature/relation affine controls。
3. 两个 `linear standardized-full` MobileNet factorial cells 曾在 epoch0 的 preflight
   失败。它们不是可续训工作；若仍要作为主交互结论的一部分，必须先定位失败根因，再做
   独立 fresh registration。

## 7. 待补充实验：按审稿价值排序

### P0：先完成和整理已登记证据（不新增科学变量）

- 完成 WRN seed2103 与 8 个已登记 follow-up；逐 seed 报告而非仅平均。
- 对失败的两个 linear-standardized-full cells 做只读失败审计。若是工程/审计构造问题而
  非科学配置错误，重新建立**两个完整 240-epoch fresh arms**。这补齐 MobileNet 的
  linear/cosine × {full, relation} 因子表；不得从 epoch0 或中断 checkpoint 续训。
- 用已有 paired checkpoints 扩展 raw-KD 与 ACD 的 epoch 1/30/120/240 固定 probe 到另一
  seed；这是分析复核，不需要新 240-epoch 训练。

### P1：一个真正公平的 closest-method 对照（最高新增训练价值）

完整方法名是 **ACD = ACI（cosine interface）+ SICD**，不是单独的 “SICD”。在一个
异构和一个同构 pair 上各做一个 fresh seed；固定 ACI、训练器、初始化和数据次序，比较
ACI+KL、ACI+DKD、ACI+SKD、ACI+NCKD、ACI+standardized-full 与 ACD。六臂分别覆盖
ordinary KD、解耦、streamlined、non-target、仅标准化和完整方法。SKD/NCKD 若不能在同一
recipe 下忠实锁定则删除，而非混用作者配方。另以 linear+KL、ACI+KL、linear+SICD、ACD
及 CE-only 构成独立模块消融。不能复用不同注册的历史 DKD 点，也不能在结果后改变温度、
权重或选择 best epoch；若 ACD 不胜出，保留结果并将贡献写为条件性选择原则，而不是 SOTA。

### P2：一个未见异构 pair 的完整 ACI × SICD 因子矩阵

选择已有 source-lock 和可用 linear-KD recipe 的 pair（优先 ResNet50→MobileNetV1），
在一个 fresh seed 上跑 linear/cosine × {CE-only, ordinary KD}，再加 cosine full 与
cosine SICD，共 6 臂。该试验同时检验 ACI 的 difference-in-differences 和 SICD 的增益，
比继续堆积 ETF、RKD 或 scale sweep 更有判别力。首轮不自动扩为多 seed。

### P3：composition 的可复现性，而非“普适性”口号

在第二个可审计 AT+KL pair 上运行 linear+KL、cosine+KL、cosine+SICD 三臂（一个 fresh
seed）。若 AT 的 +1.37pp 无法复现，应如实把结论降为单-pair pilot；若复现，再由用户
决定是否授权独立 seed，而非自动扩展。

### 不建议现在投入 GPU 的项目

- ETF/rotating-ETF/原型方向继续扫描：与核心 SICD 主张的因果距离远且已有混合结果。
- 更多 RKD：不能回答 KL replacement，信息增益低。
- 多 seed affine 长训练：目前的 loss-level 不变性与少量控制足以说明机制边界，不能换来
  几何因果定理。
- ImageNet-1K：本地数据、教师来源和磁盘条件未满足；应迁移到具备冻结数据/教师/空间的
  平台后独立注册。

## 投稿前的可接受表述

可写：ACD constructs an angular student interface and selectively transfers standardized
conditional non-target relations; controlled CIFAR results show positive but pair-dependent
effects, including a clean AT+KL replacement pilot.

不可写：ACD is universally superior, consistently improves feature KD, is state of the art,
or retains the same training efficiency as ordinary logit KD. 这些表述均超出当前证据。

## 可追溯性

核心严格配对结果：`reproductions/acd_cnn_interaction_20261109/artifacts/`；WRN：
`reproductions/cvpr_wrn_three_seed_extension_20261108/artifacts/`；AT 与 ReviewKD：
`reproductions/acd_at_res32x4_sv1_priority_20261111/artifacts/`、
`reproductions/acd_reviewkd_wrn_priority_20261110/artifacts/`。协议边界与等待队列见
`reports/ACD_EXECUTION_QUEUE_20261109.md`。每项数值均可由各 job 的 `status.json` 复核。
