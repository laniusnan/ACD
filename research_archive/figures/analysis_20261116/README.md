# Analysis 图件与证据说明（2026-11-16）

**独立面板导出：**[calib 左图（PDF）](fig_oracle_calib_seed2051.pdf) / [PNG](fig_oracle_calib_seed2051.png) / [SVG](fig_oracle_calib_seed2051.svg) 已删除顶部标题及底部两行说明，Mean/Residual 右对齐，横轴使用统一字体并移除 `; symlog` 字样；坐标变换与数据不变。[输出分数密度右图（PDF）](fig_output_mismatch_right_cifar100_seed2051.pdf) / [PNG](fig_output_mismatch_right_cifar100_seed2051.png) / [SVG](fig_output_mismatch_right_cifar100_seed2051.svg) 单独保留坐标轴、色条和一致性虚线。右图沿用双面板图的全部 100 万个分数对、分箱和坐标范围。

仅从已保存数组重绘独立图：`.venv/bin/python -B paper_assets/acd_figures/analysis_20261116/render_oracle_calib_seed.py --calib-only`；`.venv/bin/python -B paper_assets/acd_figures/analysis_20261116/render_output_mismatch_right.py`。每张图的同名 JSON 记录数据来源与输出哈希，不进行训练或新推理。

**第一段推荐图：**[师生输出 mismatch 双面板图（PDF）](fig_output_mismatch_cifar100_seed2051.pdf) / [PNG](fig_output_mismatch_cifar100_seed2051.png)。只使用同一个 Linear KD 端点：左侧是校准前后 ECDF，右侧是校准后全部 100 万个图像—类别对的中心化分数密度图。没有 Cosine/ACI/SICD 方法排名，没有标准差归一化或尾部裁剪。它支持“整体尺度不能消除输出差异”，不单独归因于分类器几何。[对应英文段落与双栏图注](output_mismatch.tex)。复现绘图：`.venv/bin/python -B paper_assets/acd_figures/analysis_20261116/render_output_mismatch.py`，只读取已保存数组，不重新推理。

**后续补充（同日）：**已恢复历史 seed 2051 权重并重新推理全部 CIFAR-100 测试图，生成 [两曲线 calib](fig_oracle_calib_seed2051.pdf) 和 [四曲线＋配对差值 same-state](fig_oracle_same_state_cifar100_seed2051.pdf)。原始 affine 校准残差为 0.5575，事后 angular 为 0.5601，没有降低均值。详见 [旧图复用审查、指标解释与复现](../../../docs/CLASSIFIER_FIGURE_REUSE_20261116.md) 和 [论文段落及图注](oracle_cifar100.tex)。下文“只重绘历史均值、未恢复逐图数据”描述的是最初 10 组图的生成状态，不适用于本次新 CIFAR 图；原 10 组图不变。

已生成 **10 组实测图件**，每组包含 PDF、SVG、PNG，并提供 [总览](overview.png)、[LaTeX 图注](figures.tex)、[数值汇总](results_summary.json) 和 [来源与校验清单](evidence_manifest.json)。本次新增探针在 CPU 上运行，没有训练更新；历史 oracle 图由封存汇总重绘。

**当前结果不能支持原稿全部判断。** CIFAR-100 历史余弦模型具有较低校准残差；ImageNet 同一 CE 模型直接换角度头没有降低平均残差；本次同批次梯度未出现负余弦；210 步模型尚不能验证完整 ACI/SICD 的表示收益。论文应按下面的证据边界选图和修改措辞。

## 图件索引

PDF 适合论文排版，SVG 可继续编辑，PNG 用于预览。点击文件名可直接查看。

| 对应问题 | PDF | PNG 预览 | 证据类型与用途 |
|---|---|---|---|
| 你提出的 oracle calibration | [fig_oracle_calib.pdf](fig_oracle_calib.pdf) | [预览](fig_oracle_calib.png) | CIFAR-100 两个 seed 的历史全测试集均值，展示 scalar-removable 部分与残差 |
| 补充：固定同一模型校准 | [fig_oracle_same_state.pdf](fig_oracle_same_state.pdf) | [预览](fig_oracle_same_state.png) | ImageNet 完整 CE 模型，原头与事后角度头；保留反例 |
| 你提出的 t-SNE | [fig_tsne_pilot.pdf](fig_tsne_pilot.pdf) | [预览](fig_tsne_pilot.png) | 两个真实 210 步预检模型，原特征与 L2 归一化特征分别可视化 |
| 补充：表示的定量检验 | [fig_representation_pilot.pdf](fig_representation_pilot.pdf) | [预览](fig_representation_pilot.png) | 高维原空间内的类内角离散度、最近类中心余弦、cosine silhouette |
| 补充：t-SNE 敏感性 | [fig_tsne_sensitivity.pdf](fig_tsne_sensitivity.pdf) | [预览](fig_tsne_sensitivity.png) | 额外报告三组 perplexity/seed，不挑选最有利投影 |
| 你提出的 CE–KD 梯度冲突 | [fig_gradient_conflict.pdf](fig_gradient_conflict.pdf) | [预览](fig_gradient_conflict.png) | 完整 CE 模型与其事后角度头；同一状态比较四种蒸馏目标 |
| 补充：预检模型梯度 | [fig_gradient_conflict_pilot.pdf](fig_gradient_conflict_pilot.pdf) | [预览](fig_gradient_conflict_pilot.png) | 在两个 210 步预检状态复查，不能当作训练全程趋势 |
| 建议：参数化因素干预 | [fig_parameterization.pdf](fig_parameterization.pdf) | [预览](fig_parameterization.png) | 分别改变整体尺度、类别权重范数、类别偏置，固定特征/方向/教师 |
| 建议：梯度来源拆解 | [fig_gradient_sources.pdf](fig_gradient_sources.pdf) | [预览](fig_gradient_sources.png) | 概率误差 × 分类器 Jacobian 的 2×2 交换诊断 |
| 建议：条件损失的目标类依赖 | [fig_conditional_dependency.pdf](fig_conditional_dependency.pdf) | [预览](fig_conditional_dependency.png) | 单独改变目标 logit，检查标准化分布、教师权重与关系损失 |

所有 SVG 使用与 PDF 相同的文件名前缀，例如 [fig_oracle_calib.svg](fig_oracle_calib.svg)。

## 实测结果及论文可以写到哪里

### 1. Scalar calibration：支持“仍有残差”，不支持“只能靠角度坐标解决”

历史 CIFAR-100 记录来自每个 seed 的全部 10,000 张测试图，目标为 **8 KL，T=4**：

| 模型 | seed 2051：校准前 → 后 | seed 2052：校准前 → 后 |
|---|---:|---:|
| Linear KD | 1.2198 → 0.5575 | 1.1987 → 0.5561 |
| Cosine KD | 0.7081 → 0.4116 | 0.7066 → 0.4123 |
| Sample-amplitude restored | 1.1137 → 0.5290 | 1.1259 → 0.5276 |

两个 seed 的余弦模型平均残差分别比线性模型低约 26.2% 和 25.9%。这些是分别训练后的端点比较，并非同一模型的因果干预，也不是 ETF 专项实验。历史记录中，差异主要集中在教师正确的样本；教师错误样本的余弦减线性残差仅为 −0.0037 / −0.0008。不能据此宣称普遍或统计显著的收益。

新 ImageNet 探针使用 **16 KL，T=4**，固定同一个完整 CE 模型和同样 256 张训练集图像：

| 分类头 | 校准前平均 KL | 校准后平均 KL |
|---|---:|---:|
| 原始 affine | 2.2026 | 1.4742 |
| 事后 angular，s=30 | 3.4382 | 1.5305 |

角度头的平均校准残差高 **0.0563（约 3.8%）**。这说明，直接归一化已由 affine 目标训练的权重/特征，不保证更接近教师。它不否定从头训练 ACI 的可能收益，但不支持原稿的 “residual can only be significantly reduced by switching …”。两幅图的数据集、训练状态、损失系数不同，不能横向比较绝对损失。

每张图像的 oracle 独立优化整个学生 logit 向量的乘数（包括偏置），定义域为 α≥0；α=0 表示严格正数域的边界极限。新探针两种头均没有取到零边界。历史逐图 NPZ 和原检查点未在本地恢复，本次只重绘保存的均值，未伪造逐图 ECDF、误差条或 bootstrap CI。

### 2. 表示空间：先看高维量化，再解释 t-SNE

两个模型均只训练 **210 个 minibatch 更新**。检查点内 `epoch=1` 是恢复演练标记，不表示已完成一整轮训练。线性+KD 与角度+SICD 方程重建同时改变分类头和损失，不能隔离 ACI 的贡献；当前角度头也没有 ETF 初始化。

| 指标 | Linear + KD | Angular + SICD reconstruction |
|---|---:|---:|
| 类内角离散度，中位数，越低越紧凑 | 0.2025 | 0.1604 |
| 最近其他类中心余弦，中位数，越低越分离 | 0.9499 | 0.9497 |
| 原空间 cosine silhouette，越高越好 | −0.0830 | −0.0996 |

结果是混合的：类内方向更紧，但类中心仍高度相似，silhouette 反而较低。不能写“更均匀且与教师语义更一致”；本次没有测量教师表示对齐。t-SNE 独立拟合后的旋转、尺度、二维簇间空隙不能直接比较。主图对 raw/L2 features 使用相同设置；敏感性图报告额外设置，但不是完整的参数网格。

误差线为图像或类别的 IQR，不是独立训练 seed 的置信区间。类内离散度使用本次样本的归一化类中心，属于描述性、样本内指标。

### 3. 梯度：当前没有“严重冲突被消除”的证据

每一状态在相同 8 个 batch（每批 8 图）上，分别计算 CE、Raw KL、双侧标准化 full KL、teacher-mass-weighted conditional KL 及其正数范数匹配版本。梯度为全部 backbone 可训练参数的拼接，排除分类头；使用 eval 模式冻结 BN、关闭 dropout，没有优化器更新。

| 固定学生状态 | Raw KL 与 CE 平均 cosine | Std. full | Conditional | 各目标的负余弦 batch |
|---|---:|---:|---:|---|
| CE 完整端点，原 affine | 0.1905 | 0.2819 | 0.2515 | 均为 0/8 |
| 同一端点，事后 angular | 0.2279 | 0.3097 | 0.2702 | 均为 0/8 |
| 210 步 KD 预检 | 0.6749 | 0.6020 | 0.5571 | 均为 0/8 |
| 210 步 SICD 重建预检 | 0.7017 | 0.6176 | 0.5539 | 均为 0/8 |

条件目标没有一致优于标准化 full KL；预检状态中 Raw KL 的 CE 对齐反而更高。正数、停止梯度的范数匹配保持 conditional 梯度方向，仅改变长度，这也在实测中成立。匹配参考是 **Std. full 的 backbone 梯度范数**，不是 CE 范数。

这些是固定状态、eval-mode、小 batch 的诊断，不能代表训练全程或实际 batch=256 的冲突率。0/8 也不意味着所有样本绝无冲突。未来若做训练期结论，应预先固定多个 checkpoint、同批数据和参数集合，报告分布与独立 seed。

### 4. 参数化干预与梯度来源：可作为机制段落主图

`fig_parameterization_compact.{pdf,svg,png}` 将相同测量排为分布、排序、梯度三个子图。紧凑版去掉底部两行说明与预留空白，并在子图空白处用加粗浅底框标注 “Same angles, shifted distributions”、“Rank changes beyond scaling” 和 “KD gradients rotate”；数据、均值曲线和四分位区间不变。移出的实验条件仍须在论文图注中说明：固定 ImageNet CE epoch100 检查点、256 张训练图像，曲线为均值、阴影为图像间 IQR；TV 与梯度轴采用 symlog（低于 $10^{-4}$ 为线性段），不同因素的干预幅度未匹配。这些标注描述固定状态下的变化，不表示变化普遍有害或证明最终准确率提升。

三个子图的横轴标题已移除，并直接裁紧底部，保留数字刻度与原有绘图区尺寸；论文图注需说明横轴为干预参数、原始状态为 1。

整体正尺度保持非目标类别排序，但会改变分布与 KD 梯度方向；把类别权重范数全部置为几何平均值（ρ=0）后，**26.56%** 图像的最高非目标类改变，条件 TV 均值为 **0.02415**；去除中心化类别偏置后，该比例为 **1.17%**，TV 为 **0.002226**。

这里的 common scale 是对整组 logits 乘数值，不等同于只放大特征（存在偏置时尤其不同）。各因素干预范围不在统一扰动范数上匹配，不能据曲线高度断言某一因素普遍占主导。参数化改变关系不等于证明被去掉的关系都是无用的“伪影”。

对 `g = Jᵀ T(q−p)` 交换概率误差和 Jacobian，平均方向偏离如下（相对 raw/raw）：

| Jacobian / probability error | Raw error | Angular error |
|---|---:|---:|
| Raw Jacobian | 0 | 0.2926 |
| Angular Jacobian | 0.0622 | 0.2827 |

两条路径都能改变局部特征梯度。交叉格子是受控构造的诊断向量，不是实际训练算法的端点或性能。它们的差值不能被解释为可加的因果贡献。

### 5. 条件监督：去掉显式 confidence 项不等于完全 target-independent

令 `p=softmax(std(t)/T)`、`q=softmax(std(z)/T)`，有限温度 KL 精确分解为二分类目标质量项和 `(1−p_y) KL(p_non-target || q_non-target)`。删除第一项后，若仍用全类别标准差，学生目标 logit 会通过标准差改变条件分布；教师目标 logit 还会改变 `(1−p_y)`。

单独将目标 logit 增加原标准差的 8 倍时，未标准化的非目标条件 TV 始终为 **0**，标准化后的 TV 均值则为学生侧 **0.00716**、教师侧 **0.00824**。教师非目标权重均值由 **0.99363 降到 0.96710**。这一组图测量范数匹配之前的关系目标，不能称作完整匹配后训练损失。

建议把 SICD 动机写为“移除显式的目标置信度匹配项、保留非目标条件关系监督”，避免 “exclusively pure dark knowledge / entirely confidence-free”。正仿射尺度不变性与目标 logit 独立性是两个不同性质。

## 推荐在 Analysis 中怎样选图

1. **问题是否超出整体尺度？** 使用历史 oracle 图，配合参数化因素干预；补充材料同时保留 ImageNet 事后换头反例。
2. **角度坐标具体改变什么？** 使用概率误差/Jacobian 拆解，解释局部优化路径；目前不以 210 步 t-SNE 宣称 ACI 有效。
3. **条件目标保留什么、去掉什么？** 使用目标类干预与有限温度 KL 分解，明确标准化和 teacher weight 的残余耦合。
4. **训练证据是否支持机制？** 梯度图目前呈现诊断而非消除冲突的结论；完整 ACI/SICD、ETF 对照、多 seed 表示和准确率仍需后续实验。

可直接使用的英文表述与逐图实验条件在 [figures.tex](figures.tex)。主文是否纳入 pilot 图需按篇幅决定；即使放补充材料，也应保留 “210 updates” 标签。

## 数据、计算与复现

- 新探针：ImageNet-1K **训练集**，固定 seed=20261116，先随机选择 128 类各 2 图供 256 图诊断；其中先选出的 12 类各扩到 20 图供 t-SNE。合并为 472 张独立图片，选择不依赖模型输出。另从诊断图确定性抽取 64 张供 8×8 梯度探针。
- 使用 Resize(256)/CenterCrop(224) 的干净变换。教师与学生可能见过训练图片；这些指标不是验证准确率，抽样也不是全 ImageNet 分布的完整估计。
- 教师：官方 ResNet50 ImageNet1K V1；学生：MobileNetV2。完整 CE 端点为 epoch 100，checkpoint SHA256 为 `3519ca20e94e375f27b00703eb8e51c2bfcd86a1a222faf6293a8e42ead4bc56`。
- I2/I7 预检模型来自 2026-11-14 保存的稳定文件，配对初始化与预检报告有哈希记录。I7 是 **SICD 方程重建**，`SICD_REFERENCE_VERIFIED=False`；不是已核验原生实现，也不包含完整 ACD/ETF 的训练验证。
- 新探针使用 CPU 4 线程，约 64 秒；绘图单独运行。分类头数学探针用 float64，网络前向和 backbone 反向用 FP32。KL 分解最大绝对误差 `2.70e-14`，正仿射标准化检查 `1.33e-14`，角度归一化消去检查 `2.84e-14`。标准化的尺度不变性要求标准差未触发 epsilon 下限。
- 原始样本路径、标签、图片 SHA256、checkpoint/teacher 来源、逐图数组、逐 batch 梯度指标位于本地 `artifacts/`。该目录由本包 `.gitignore` 排除；可追踪的汇总与输入/输出 SHA256 清单保存在本目录。
- 只读取稳定检查点，没有加载仍在更新的正式 KD `latest.pth`，没有修改训练源码、协议或模型参数。backbone 探针前后检查了完整模型状态哈希一致。

从 `KD/` 目录运行。当前文件均已生成，仅重绘无需再次进行网络推理：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python paper_assets/acd_figures/analysis_20261116/render.py
```

复查探针并复用现有特征缓存：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python paper_assets/acd_figures/analysis_20261116/probe.py --resume-cache
```

只在尚无特征缓存的新环境中，首次提取时省略 `--resume-cache`。脚本依赖上述本地 ImageNet、教师权重、三个稳定检查点、数据全量审计文件与历史 CIFAR 汇总，脚本本身不会下载这些输入。

绘图库安装在本地 `artifacts/plot_deps`，没有写入训练虚拟环境。当前环境的 pip 也单独位于 `artifacts/pip_bootstrap`。已有该 bootstrap 时可按锁定版本重建绘图依赖：

```bash
PYTHONPATH=paper_assets/acd_figures/analysis_20261116/artifacts/pip_bootstrap .venv/bin/python -m pip install --target paper_assets/acd_figures/analysis_20261116/artifacts/plot_deps -r paper_assets/acd_figures/analysis_20261116/plot_requirements.lock.txt
```

没有 bootstrap 的环境先准备 Python 3.11 可用的 pip，使用同样的 `--target` 安装命令即可。训练环境依赖和数据准备见 [ImageNet 实验包](../../../reproductions/imagenet1k_acd_20261113/README.md)。
