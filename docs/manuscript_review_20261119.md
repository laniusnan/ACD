# ACD 论文与项目实验证据核对（2026-11-19）

核对对象：[作者稿 acd_v1.pdf](../acd_v1.pdf)、本地代码、历史资产审查，以及 `seven_arms_20261118` 已完成的正式训练。PDF 来源和 SHA256 见 [paper_source.json](paper_source.json)。原始 LaTeX 未提供，本文是修改建议，不是已修改的论文。

核心建议：论文应更清楚地区分“分类头参数化的敏感性”“余弦头的训练收益”“条件监督及初始化的交互作用”。当前实验不支持把所有增益解释成各模块稳定、独立地叠加，也不能用一个新配对的结果替换全部历史主表。

## 1. 新实验能支持什么

本次快照只有 ResNet32×4 → ResNet8×4 完成全部七组正式训练；其余配对尚不能形成完整比较。以下均为 CIFAR-100 全训练集、同一教师、共同配方、seed 3301、第 240 轮、官方 10,000 张测试集的 Top-1。不是多 seed 均值。SICD 在此表示标准化、加权非目标条件 KL 和梯度范数匹配的完整组合。

| 分类头 | 普通 KD | SICD 组合 | 同分类头下的变化（百分点） |
|---|---:|---:|---:|
| 通用初始化余弦头 | 77.30 | 76.58 | −0.72 |
| ETF 初始化、随后自由学习 | 77.07 | 77.50 | +0.43 |
| 保持 ETF 几何、学习旋转 | 76.72 | 77.37 | +0.65 |

线性头 KD 为 **74.76**。因此：

- 余弦头 + 普通 KD 相对线性头为 **+2.54**。
- 最好组合相对线性头为 **+2.74**，但相对余弦头 + 普通 KD 仅 **+0.20**；不能把前一个数全部归因给 SICD。
- ETF 初始化在普通 KD 下为 **−0.23**，在 SICD 下为 **+0.92**；描述性的差中差为 **+1.15**。这提示初始化与损失存在交互，尚不是统计显著性或机制证明。
- 保持 ETF 几何的 SICD 组合比只做 ETF 初始化低 **0.13**，不足以据此宣称严格保持 ETF 更好或更差。
- 这些结果需要与其他配对及多个 seed 一起判断。当前不能写“普遍提升”“显著提升”或“ETF 必需”。

机器可读结果、配方、端点记录和算术核对见 [manuscript_review_evidence_20261119.json](manuscript_review_evidence_20261119.json)。这次审阅未重新运行测试集评估，也未重新计算权重文件摘要；精度和 checkpoint 摘要来自完成运行的 `final.json`。

## 2. 摘要、引言与贡献归因：优先修改

涉及摘要、§1、§4.2 和 §5.3。

| 当前表述或逻辑 | 建议 |
|---|---|
| 将范数、偏置及置信度笼统称为不可迁移的 artifacts | 改成模型参数化相关的因素。固定状态干预证明它们能改变监督信号，并未证明其中所有信息都无用或不可迁移。 |
| Oracle 残差“confirms”普通 KD 强迫学生复制参数化伪影 | 改成：该 checkpoint 的差异不能由单个非负标量完全解释。残差也可能包含预测差异、容量限制和训练不足。 |
| “pure/true dark knowledge”“exclusively transfers” | 写清楚转移的是标准化后的、教师质量加权的非目标条件分布，省略显式二元目标质量 KL。不要写成与目标类或教师置信度完全无关。 |
| “optimal angular prior”“perfectly aligns”及不敏感类别持续保持最优夹角 | ETF 仅提供初始对称配置；自由训练不保持 ETF Gram。类别敏感性叙事目前缺乏逐类轨迹证据，建议删除。 |
| “universal drop-in”“consistently improves” | 限定为已评估设置。§5.4 本身已承认仅两项单 seed 兼容性案例，摘要应与此一致。 |
| 将总提升全部归因于 SICD | 同时报告线性 KD、余弦 KD 和完整组合，明确相对强基线的增益。 |

建议保留论文的核心问题，但将主线改成：**分类器参数化会影响分布和局部 KD 梯度；余弦读出提供一种可控的学生接口；标准化条件监督的作用需要在该接口和不同初始化下实证检验。**

可用于新实验讨论的英文段落（只能明确标为一对、一个 seed 的初步结果）：

> Under a matched training recipe on ResNet32×4→ResNet8×4, replacing the affine classifier with a cosine classifier increases Top-1 accuracy from 74.76% to 77.30%. The effect of the complete SICD objective depends on the classifier initialization and constraints: it changes accuracy by −0.72, +0.43, and +0.65 percentage points for the default cosine, ETF-initialized cosine, and rotating-ETF heads, respectively. The best configuration reaches 77.50%, exceeding cosine KD by 0.20 points. These single-seed results suggest an initialization-dependent interaction rather than a uniformly additive benefit from each component; replication across seeds and architecture pairs is needed to assess its robustness.

## 3. 先统一 ACD 定义，再整理主表

当前方法正文和 Table 5 将 ACI 写成 cosine + ETF，但结构图写着 ETF optional，Table 6 的文字又描述 cosine + SICD。建议明确：

- **CI / ACI**：选定一个名称，固定表示 bias-free cosine interface。
- **ETF init**：只改变初始权重，之后自由学习。
- **Rotating ETF**：训练中保持 ETF 几何的单独对照。
- **SICD**：明确是否包含标准化和 GM；本文新七组实验的 SICD 包含二者。
- **ACD 主结果行**：明确一套方法定义。如果 ETF 是可选的验证超参数，必须事先说明候选范围、验证规则和预算，而不能按测试成绩逐列取较大值。

必须优先核对的历史来源是 [用户转录记录](../../KD/paper_assets/acd_tables/main_results_20261116/user_reported_etf_20261116.json)：

| 配对 | 历史 relation-only | 历史 +ETF init | 当前主表 ACD |
|---|---:|---:|---:|
| ResNet50 → MobileNetV1 | 73.30 | 72.80 | 73.30 |
| WRN40-2 → ShuffleNetV1 | 81.38 | 81.62 | 81.62 |
| ResNet32×4 → ShuffleNetV1 | 81.46 | 81.36 | 81.46 |
| ResNet32×4 → ShuffleNetV2 | 82.28 | 83.12 | 83.12 |
| VGG13 → VGG8 | 80.22 | 80.52 | 80.52 |

这五个主表值分别与转录记录两种设置中的较大值相同。这是需要核对方法来源的线索，**不能仅凭相同数字断定主表实际采用了测试集择优**。转录记录缺少 seed、split、checkpoint 和完整配方，不能代替原始结果证明。VGG13→MobileNetV1 的 73.30 也需要独立来源，不能因与另一个配对相同而推断来源。

每个表格单元至少关联：teacher/student、head、ETF 模式、loss、GM、特征维度、数据划分、增强、配方、seed、选点规则、评估样本数、checkpoint SHA256。多 seed 的 `mean ± std` 要能追溯每个 seed；Table 5 的 ±0.01/±0.02 尤其应核对，不应由单 seed 或不同协议的均值拼出。

新实验的 77.50 与原稿 ResNet32×4→ResNet8×4 的 78.33 不应直接解释成复现失败；先对齐方法和协议。旧 Release 中的 seed3201 结果同样不能与新的 seed3301 结果拼成配对实验或多 seed 均值。

## 4. 消融表改为交叉对照，并补齐真正缺失的隔离实验

Table 5 已有 standardized full-KL 行，并且正文诚实注明最后一行同时改变 conditional supervision 与 GM。需要进一步解决的是归因，不是再把已有的 DS 行当成未做过。

建议主消融以第 1 节的三种头 × 两种监督展示，额外列出线性 KD。这能显示交互作用，避免只展示不断增加模块的单一路径。

七组实验仍然不能分别识别标准化、条件化和 GM。完成当前训练后，优先在一个同构和一个异构配对上固定分类头，补充：

| 固定分类头下的监督 | 要回答的问题 |
|---|---|
| 普通 full KL | 共同基线 |
| 标准化 full KL | 标准化带来多少变化 |
| 标准化 conditional KL，无 GM | 删除显式二元目标项带来多少变化 |
| 标准化 conditional KL，有 GM | 梯度范数匹配带来多少变化 |
| conditional KL 去掉教师质量权重（可选） | 保留 `(1−p_y)` 是否有必要，是否保留有益/有害的置信度加权 |

如要声称“余弦接口是 KD 特有优势”或“条件监督必须依赖余弦头”，再加入 affine CE / cosine CE，以及 affine SICD 对照。历史账本中已有记录应先找回原始资产，避免无依据地声称这些从未做过。

最近邻比较建议包括同头同配方的 DKD 和 logit-standardization baseline：DKD 已研究目标/非目标项分离，Logit Standardization 已研究 Z-score KD。创新应落在经验证的接口与监督组合、交互和分析上，不能把 KL 链式分解或 Z-score 本身写成新发明。相关原始论文：Zhao et al., *Decoupled Knowledge Distillation*, CVPR 2022, arXiv:2203.08679；Sun et al., *Logit Standardization in Knowledge Distillation*, CVPR 2024, arXiv:2403.01427。

对强基线差值仅 0.20pp 的设置，建议至少补 3 个共同 seed 并报告配对差值、均值和标准差；若波动较大再增加样本。3 个 seed 并不自动保证统计显著。本节为建议，未增加训练任务。

## 5. 方法公式与实现需要直接对齐

依据：[acd/losses.py](../acd/losses.py)、[acd/heads.py](../acd/heads.py)、本轮冻结源码及配置。

1. **Eq. 6 的尺度。** 当前 cosine scale 是固定 buffer，值为 30，不是学习参数；教师保留原始分类头。标准化可消去学生的正尺度，但 CE 仍使用未标准化 logits，不能据此说尺度对训练无影响。
2. **Eq. 9 的标准化。** 实现按全部 C 类使用 sample std，分母为 `max(std, 1e-6)`。它保留标准化后的间隔，不仅保留排名。正仿射不变性需要说明标准差下限未触发。
3. **Eq. 10 的目标依赖。** 保留 `(1−p_y)`；全类标准差还依赖学生目标 logit。因此省略显式目标质量 KL 并不等价于目标 logit 梯度恒为零或完全消除置信度影响。
4. **Eq. 11 的参考损失。** 应写 `L_full^std`，即在相同标准化 logits 上的 full KL。当前 `L_KD` 容易被解释成 Eq. 4 的原始 logits KD，与代码不符。
5. **梯度范围和数值细节。** 对骨干参数求范数，排除最后分类器；若有共同特征升维层则纳入。ratio detach，分母 floor=1e−12，无额外 clipping。范数匹配只控制强度，不保证方向与 CE 一致，也不直接证明优化稳定性。
6. **Eq. 12 的权重。** 写成 `w_CE L_CE + w_KD r L_SICD`，并逐配对报告权重。本轮分别存在 0.5/0.5 和 0.1/0.9。若只把 CE 权重除掉而保持优化器不变，会改变整体梯度规模，不能假装是同一训练过程。

建议的梯度匹配定义：

```latex
\begin{align}
L_{\mathrm{full}}^{\mathrm{std}}
 &= \frac{\tau^2}{B}\sum_{i=1}^{B}D_{\mathrm{KL}}(p_i\Vert q_i),\\
r &= \operatorname{stopgrad}\!\left(
\frac{\|\nabla_{\theta_b}L_{\mathrm{full}}^{\mathrm{std}}\|_2}
{\max(\|\nabla_{\theta_b}L_{\mathrm{SICD}}\|_2,\epsilon_g)}\right),\\
L &= w_{\mathrm{CE}}L_{\mathrm{CE}}(z_s,y)
   + w_{\mathrm{KD}}rL_{\mathrm{SICD}}.
\end{align}
```

其中 p、q 均由标准化后的 logits 与同一个温度计算。

## 6. ETF 维度条件和“无额外成本”必须说清

§4.2.2 的原构造 `U∈R^{d×C}, UᵀU=I_C` 要求 d≥C。严格 simplex ETF 的最小可行维度是 d≥C−1，当前新实现使用 C−1 列 frame。可统一用 Gram 条件写初始化：

```latex
\widehat W(0)^\top\widehat W(0)
=\frac{C}{C-1}\left(I_C-\frac{1}{C}\mathbf1\mathbf1^\top\right),
\qquad \widehat W\in\mathbb R^{d\times C},\quad d\ge C-1.
```

右侧秩为 C−1。低于这一维度不可能得到该严格 Gram。

- CIFAR-100 的 ResNet20、ResNet32、WRN40-1 原始特征为 64 维。本轮所有七组及配方搜索共同加入 `Linear(64,128) → ReLU`；应标为升维架构，且 KD 基线也有同一层。
- 这保证该配对内部对照可比，但不能把它们不加标记地与原始 64 维学生的外部结果比较；应说明结构及新增参数/计算。
- ImageNet ResNet18 的常见 512 维特征不足以容纳 1000 类严格 ETF。原稿 ImageNet 实验是否关闭 ETF、升维或采用其他构造，需要原配置确认。
- ETF init 仅要求 t=0 的 Gram；rotating ETF 则训练全程保持 Gram，不能统称“可训练 ETF”。新旧旋转实现不同，历史结果也应记录实现版本。

摘要可改成“无需 teacher–student feature alignment”；若保留“无额外推理成本”，需要逐设置测量支持并排除升维例外。普通余弦归一化也有计算，不宜笼统说成本严格为零。

当前已完成配对的 SICD 整轮训练用时约 4.46–4.53 小时，普通余弦 KD 约 2.42 小时。这是共享运行环境下的作业耗时，不是受控吞吐基准；足以提醒应补训练时间/显存报告，不能据此宣称精确通用倍数。

## 7. Oracle 和图的直接修正

| 位置 | 已核实问题 | 修改建议 |
|---|---|---|
| Eq. 1、Algorithm 1 第 5 行、§3.1、Fig. 2 | 1.2198→0.5575 来自 `8 KL, T=4`；正文/算法写 `T² KL`，即 16 KL | 保留现有图和数值，统一定义 `ell_i(alpha)=c KL`，注明 c=8、T=4。正乘数不影响最优 alpha。若选 T² 则全部数值和横轴乘 2。 |
| 引言中的 oracle 解释 | 将剩余差异直接认定为参数化伪影 | 限定结论为单标量不足以解释当前 checkpoint 的差异。45.7% 是两个均值之比，不是逐样本残差比例的均值。 |
| §3.2 | 引用 Fig. 4(a/b/c)，实际参数化三联图是 Fig. 3；Fig. 4 只有 a/b | 使用对应 `\ref`，检查最终编译。 |
| Fig. 6 | 只有四个完整 ACD 端点的条件 TV，没有匹配 baseline | 保留“端点关系一致性”的描述；若要声称更好，添加相同图像/配方的 KD、cosine KD 对照。 |
| Fig. 7 | 固定 ImageNet CE checkpoint 的误差向量/Jacobian 交叉诊断 | 保留固定状态与诊断构造限定，不写成训练准确率机制已被证明。 |
| Fig. 8 | 列名 Input / SKD / Ours，但正文说明 CE / KD MobileNetV2，且明确未评估 ACD | 核验图像来源后更名 Input / CE / KD 并移到诊断附录，或用实际 SKD / ACD checkpoint 重画。不能只换文字冒充新实验。 |

Oracle 原始记录见 [fig_oracle_calib_seed2051.json](../../KD/paper_assets/acd_figures/analysis_20261116/fig_oracle_calib_seed2051.json)。同一冻结 checkpoint 替换 angular readout 后的校准残差为 0.5601，affine 为 0.5575；这个诊断本身没有显示 angular 会消除残差，也不等同于训练过的 angular 模型表现。

§3.1 可直接采用以下措辞，并在算法中保持 c=8 一致：

> Following Alg.~\ref{alg:oracle_calibration}, we freeze both models and optimize one nonnegative scalar per image to minimize the KL divergence to the fixed teacher distribution. We report the diagnostic as $8D_{\mathrm{KL}}$ at $T=4$. Across all 10,000 CIFAR-100 test images, mean divergence decreases from 1.2198 to 0.5575, leaving 45.7\% of the original mean discrepancy. Thus, for this checkpoint, image-specific scalar calibration does not fully explain the output mismatch. The residual does not by itself identify which part of the discrepancy is caused by classifier parameterization.

## 8. 实验协议、跨任务结论与主表发布条件

§5.1 建议补一个逐配对配置表，完整说明学习率、CE/KD 权重、温度、scale、增广、seed、教师来源及模型维度。不同配对可以不同，但同一配对的方法比较要共用协议。

本轮实际协议见 [seven_arm_training_20261118.md](seven_arm_training_20261118.md)：

- 训练集每类留出 50 张，45k/5k 搜索；四候选为 LR 0.01/0.05 × CE/KD 0.1/0.9 或 0.5/0.5。
- 以线性 KD 搜索共同配方，60 轮筛选两个候选再续训至 240 轮，按末 5 轮平均验证精度选择；不是为每个方法独立调参，也不是全局最优搜索。
- 搜索 seed3300，划分 seed20261118；正式七组重新训练 full50k、seed3301、第240轮取点。每图弱/强两个视图，即每轮100k训练视图。
- 教师在完整训练集预训练，因此 5k 仅是学生训练留出集，不是教师未见过的数据。官方测试集不参与配方选择。
- 当前只完成一个 seed，不能填写多 seed 的 ±std；后续复现实验应保持配对 seed。

Table 3 ImageNet 和 Table 4 detection 需要另行找回原训练配置、各 seed 端点、评估记录及方法接入实现。当前代码支持范围或本地权重缺口不等于实验从未发生，但不足以独立验证这些表。

Table 4 中 ACD 的三组 AP 为 36.91、40.17、33.98；SKD 为 36.74、40.62、34.25。后两项 ACD 较低，因此适合写“competitive”，不能写全任务一致领先。正文已有一定限定，摘要也应同步。

Table 6 只支持所测 AT 与 ReviewKD-style 案例的兼容性，不代表所有 feature-distillation pipelines。

建议顺序：先修复数值单位、引用、图例和公式；同时完成主表方法/结果来源核对；待11对七组训练齐全后，用统一口径更新主表与交叉消融；再补关键配对多 seed 及 SICD 内部隔离实验。当前不能用未完成训练的中途最好测试值填充缺项。

本次仅新增本文和结果证据快照，未更改训练代码、实验队列或 PDF，也未提交或推送仓库。
