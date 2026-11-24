# ACD 论文论证梳理与补充实验缺口

**日期：2026-11-13。状态：分析与建议，不是实验注册或训练授权。**

本报告整理本次论文审阅，依据本地 `84c9ab2` 快照中的论证稿、证据账本、绘图代码和结果摘要；未重新验证训练权重，未确认历史服务器队列的实时状态。未发现完整的 LaTeX 稿，审阅对象是论文叙事和证据链，而不是完整投稿稿件。

**用户当前优先事项是 ImageNet-1K 补充实验。** 执行准备与实验排序以配套的 [ImageNet-1K 可行性和实验方案](IMAGENET1K_FEASIBILITY_AND_EXPERIMENT_PLAN_20261113.md) 为准；不要求先完成下面的 CIFAR 机制筛查，也不自动扩大任何既有队列。

## 1. 总体判断与论文定位

当前最需要回答的不是“还能在哪些模型上涨点”，而是：

> 在输出蒸馏中，分类器坐标、监督内容和监督强度分别贡献多少？标准化条件关系在什么条件下比完整分布匹配更有效？

建议将论文定位为**有明确适用边界、由受控分析支持的输出蒸馏方法**，而不是几何失配的普适因果定理或所有 KD 方法的替代品。

| 章节 | 应回答的问题 | 不能跳过的证据边界 |
|---|---|---|
| Introduction | 输出监督仍有什么未解决的问题？ | 关系建模、归一化和非真类蒸馏本身不是新概念 |
| Background | logit 中有哪些参数化因素？ | 普通 KL 比较概率，已经忽略共同加性偏移，不是直接 raw-logit L2 |
| Analysis | 参数化如何影响分布和局部梯度？ | 参数分解、固定 checkpoint 探针不等于端到端准确率因果 |
| Method | ACI、SICD、梯度匹配各改变什么？ | 梯度匹配属于实际训练方法，不能隐藏为无关细节 |
| Experiments | 收益来自接口、关系筛选还是强度？ | 方法排名、模块消融、作者配方复现必须分表 |
| Discussion | 哪些配对不受益，训练代价多大？ | 保留反例，区分单 seed 先导与确认性结果 |

## 2. 必须先统一方法定义与证据命名

以 2026-11-11 证据账本为准：

- **ACI**：Angular Coordinate Interface，学生侧可学习方向的余弦分类头。
- **SICD**：Scale-Invariant Conditional Distillation，标准化后的加权条件非真类关系，以及当前使用的 detached 梯度强度匹配。
- **ACD = ACI + SICD**。
- **ACI + standardized-full** 是独立消融，不是完整 ACD。

当前材料仍有命名漂移：

| 位置 | 发现 | 应如何处理 |
|---|---|---|
| `paper_assets/acd_figures/extract_raw_acd_trajectory.py:10` | `acd` 映射到 `cosine_standardized_full` | 现有轨迹重标为 ACI+Full；有真实 relation 中间权重时再提取 ACD 轨迹 |
| `paper_assets/acd_figures/render_acd_evidence_figures.py:44` | 旧外部比较数据将 full-loss 数值命名为 ACD | 核对旧脚本、JSON、图例和图注，不只改显示名称 |
| `paper_assets/acd_figures/extract_acd_phase_plane_trajectories.py:109` | 较新的相图已区分 full 与 SICD | 保留正确区分，避免再次用旧脚本覆盖 |

这不是说全部近期图表都有错，而是旧产物、生成脚本和新方法定义尚未完全一致。**本次只记录问题，没有修改图表或科学结果。**

还要注意，2026-11-11 审稿审计曾建议重跑的两个 MobileNet linear-full 单元，后续证据账本已登记完成。应先做跨注册一致性审计，不按旧“failed/pending”标签重复训练。

## 3. ACI 的作用尚不能写成必要性

### 3.1 已有数据不支持“没有 ACI 就没有 SICD 收益”

根据证据账本的 MobileNet factorial 表，逐 seed 计算得到：

| Seed | cosine relation − linear relation，pp | `(cosine relation − cosine full) − (linear relation − linear full)`，pp |
|---|---:|---:|
| 2101 | -0.73 | +0.14 |
| 2102 | -0.57 | +0.73 |
| 2103 | -0.77 | -0.30 |

这些是对报告读数的算术整理，不是本次重新评测。不同 seed 的来源注册尚须审计，不能据此直接合并显著性检验。但现有证据不能支持 ACI 是 SICD 必要条件，也不能把 ordinary-KD 的 CE×KD 交互自动外推为 ACI×SICD 协同。

来源：`reports/ACD_EVIDENCE_LEDGER_AND_NEXT_PLAN_20261111.md` §2.1–2.3。

### 3.2 一个应明确写入分析的标准化恒等式

设逐样本标准化为 `N(z)=(z-mean(z))/std(z)`，权重逐行归一化为 `W_hat`，则 ACI 输出为：

```text
z_s = gamma * W_hat * h_s / ||h_s||
N(z_s) = N(W_hat * h_s)
```

第二个等式成立的条件是正尺度、非零特征、非退化 logit 方差以及数值截断没有触发。这是由当前公式推出的性质，不是新增训练发现。

因此，特征幅度归一化和固定 `gamma` 在**标准化 KD 分支**中抵消；但权重行归一化、去偏置以及原输出上的 CE 仍然改变训练。不能据此断言 ACI 整体无作用，也不能把该等式扩展到数值保护触发的样本。

低成本检查：固定张量比较两种表达的输出、KD loss 和非 head 梯度；记录 clamp 触发比例；再用受控 head 消融区分 CE 尺度与 KD 结构变化。无需先增加长训练。

## 4. 监督选择与梯度强度仍有混杂

当前方法图记录的关系项和训练目标为：

```text
p = softmax(N(z_teacher) / T)
q = softmax(N(z_student) / T)
L_relation = T^2 * batch_mean[(1 - p_y) * KL(p_non_target || q_non_target)]
a = stopgrad(||grad_backbone L_full_standardized|| / ||grad_backbone L_relation||)
L = lambda_ce * CE(z_student, y) + lambda_kd * a * L_relation
```

精确的 epsilon、归约、空梯度处理及是否存在 clipping，必须从封存训练源码恢复，不应凭图示补猜。当前本地缺少部分 ACD 原生研究包。

已有单 seed、45k/5k 留出实验记录：

| 变体 | Top-1，% | 梯度参考 |
|---|---:|---|
| standardized-full | 71.78 | 无额外匹配 |
| conditional relation | 72.58 | 同状态 standardized-full |
| full + norm control | 73.24 | 同状态 ordinary KD |

最后一项更高，但匹配参考不同，所以不能将它与 relation 视为干净的单变量干预。教师见过这些留出图像；这也不是官方测试结果。后来的 full-50k 结果应独立报告。

“删除显式二元置信度项”比“完全删除置信度”更准确：全类别标准化与教师非真类质量 `1-p_y` 仍携带目标类相关信息。

来源：`paper_assets/acd_figures/README.md` 的 “条件监督” 与 “ACD 实现流程” 两节。

## 5. 最近邻基线与创新边界

必须正面对比的操作包括：

1. **Logit Standardization**：逐样本 Z-score，解除共同幅度匹配约束。
2. **DKD / NCKD**：目标类与非目标类监督分解。
3. **Logit Standardization + DKD / NCKD**：标准化与非真类蒸馏的直接组合；标准化论文的官方仓库已包含 DKD 接口。
4. **Spherical KD、DIST**：用于相关工作中区分幅度归一化、关系度量和当前条件目标，不必首批穷举所有长训练。

SICD 与标准化 NCKD 的关键差异应按操作拆开验证：`1-p_y` 样本权重、梯度匹配、分类头及训练配方。不能只对普通 KD 获胜，就认为这些替代解释已经排除。

对照分两类：统一 head/优化器/输入序列的**机制表**，以及保留合理原生设置、具有可比验证预算的**方法表**。把所有方法放到 ACI 上并固定同一个温度，并不等于复现了它们的发表配方。

## 6. 补充实验设计

### P0：不增加训练的整理

- 统一 arm ID、方法名、checkpoint、评估数据集、最终轮数和生成脚本。
- 核对既有 MobileNet/ViT source、config、initial/input hash，先去重再决定是否缺失。
- 使用真正 relation arm 提取已有轨迹；不足时明确记录，不把 full 曲线当 ACD。
- 数值检查标准化恒等式、KL 分解及标准化 NCKD 与 SICD 的损失/梯度差异。
- 对历史运行状态作只读核验；历史文档不能代表现在的空闲卡或服务状态。

### P1：监督内容 × 梯度强度

原建议是在 CIFAR 的预指定正例 VGG13→MobileNetV1 和边界例 VGG13→VGG8，各做一个 fresh seed 的六臂筛查：

| 臂 | 监督内容 | 强度规则 |
|---|---|---|
| A | standardized-full | 固定系数 |
| B | standardized weighted relation | 固定系数 |
| C | standardized-full | 匹配同状态 ordinary KD |
| D | standardized weighted relation | 与 C 同参考 |
| E | standardized weighted relation | 匹配 standardized-full，即当前方法规则 |
| F | standardized unweighted NCKD | 与 D 同参考 |

`D-C` 检查强度受控后的内容差异，`C-A` / `D-B` 检查强度作用，`D-F` 检查样本权重，`E-D` 检查匹配参考。所有梯度使用相同参数集合和归约规则，系数必须 detach。

**12 臂只是原先的机制筛查建议，未排队。用户已优先选择 ImageNet-1K；同一辨析逻辑已迁移到配套方案，不必先支付这 12 条 CIFAR 训练成本。**

### P2：完整的 head × objective 与最近邻基线

若 ACI 仍是核心贡献，新配对应覆盖：

```text
{linear, cosine} × {CE, KD, standardized-full, SICD}
```

共八臂，而不是缺少 linear-full / linear-SICD 的六臂。补 DKD、标准化 DKD 等最近邻对照时必须与同 seed 配对，不能直接借用协议不同的旧排名。

### P3：确认、组合与跨数据集

- 关键比较预先指定后补独立 seed，保留正例和边界例，不只扩首轮赢家。
- 报告逐 seed 配对差、均值、样本标准差；三 seed 不自动等于显著。
- AT output-loss replacement 若作为主贡献，应补第二配对；否则作为先导结果放补充材料。
- 新数据集应以预冻结方法迁移为主，不能在官方评估集上不断选择温度/权重。
- 记录完整训练耗时、峰值显存和吞吐；不能因没有额外推理参数就宣称训练开销相同。

## 7. 接受负结果的决策规则

| 后续结果 | 论文应如何调整 |
|---|---|
| relation 在相同强度规则下仍优于 full | 可以加强“监督选择”主张，但限定数据集和配对 |
| 优势主要由 norm control 解释 | 将梯度强度列为核心因素，不把改善全部归因给条件关系 |
| linear+SICD 不弱于 ACD | 不宣称 ACI 必要；考虑作为可选接口或分析变量 |
| 标准化 NCKD 与 SICD 相当 | 明确与已有组合的关系，收窄独立方法创新主张 |
| ImageNet-1K 不复现 CIFAR 收益 | 报告迁移边界，不用新的测试集调参掩盖原失败 |

## 8. 来源与状态

本地主要来源：

- [ACD 写作论证](ACD_WRITING_ARGUMENT_20261110.md)
- [ACD 证据账本](ACD_EVIDENCE_LEDGER_AND_NEXT_PLAN_20261111.md)
- [历史审稿审计](ACD_CVPR_REVIEWER_AUDIT_20261111.md)，其中队列建议须与后续账本去重
- [图表证据指南](../paper_assets/acd_figures/README.md)

外部原始资料入口：

- Logit Standardization，CVPR 2024：`https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html`
- DKD，CVPR 2022：`https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html`
- 标准化论文作者代码：`https://github.com/sunshangquan/logit-standardization-KD`，实际使用前须固定 commit，而非浮动分支。
- Spherical KD：`https://arxiv.org/abs/2010.07485`

本报告没有更改历史读数、完成状态、训练代码或已有图表；所有新增实验均为建议。
