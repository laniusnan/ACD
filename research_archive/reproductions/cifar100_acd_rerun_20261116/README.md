# CIFAR-100 ACD 七个缺失配对重跑

用户在2026-11-16确认后期实验无法继续恢复，要求重新运行，并进一步明确：**先只跑七个配对的ACD，每对一个seed，沿用已整理的配方**。
原 ACD 训练包没有恢复，不能声称本实现与历史源码逐字节或数值一致；历史四组 ACD 结果另列，不与本批直接合并均值。

**启动状态：2026-11-16 10:56 UTC 已开始正式训练。** 全部预检通过，GPU0 首条任务为 ResNet56→ResNet20 / seed3201 / ACD。
首个epoch用时56.44秒，首条任务粗估约4小时。当前训练继续；仅ACD的总计算量见 [配方沿用核验](control/recipe_reuse_audit.json)。
GPU1继续运行既有ImageNet KD。本段是启动快照，后续以 `artifacts/queue.json` 和 `runs/*/status.json` 为准。

## 范围与配方

| 配对 | 学习率 | CE/KD 权重 |
| --- | --- | --- |
| ResNet56→ResNet20 | 0.05 | 0.5/0.5 |
| ResNet110→ResNet32 | 0.05 | 0.5/0.5 |
| WRN40-2→WRN16-2 | 0.05 | 0.5/0.5 |
| WRN40-2→WRN40-1 | 0.05 | 0.5/0.5 |
| ResNet32×4→ShuffleNetV1 | 0.01 | 0.1/0.9 |
| ResNet50→MobileNetV1 | 0.01 | 0.1/0.9 |
| ResNet32×4→ShuffleNetV2 | 0.01 | 0.1/0.9 |

**当前范围：每对只使用seed3201，只运行ACD，合计7个训练任务**。
21条线性KD/角度KD/角度standardized-full对照暂不运行；seed3202/3203的56个任务已取消。
七配对的学习率、CE/KD权重、轮数、batch、momentum、weight decay、T及衰减计划已与先前恢复的配方逐项核对一致。
ACD尺度与标准化/梯度匹配参数沿用本批开跑前已封存定义，未作调整；缺失的历史SICD训练源码仍不能声称已恢复。
单seed结果不报告跨seed标准差或多次独立重复结论。
本轮补齐ACD单独成绩；由于暂不运行配对对照，本轮自身不构成完整消融或SOTA方法排名。

240 epochs，SGD，batch64，momentum0.9，weight decay5e-4，150/180/210 完成后学习率乘0.1；FP32，关闭 TF32。
CIFAR-100 为完整50k训练图、10k官方测试图。每轮50k弱增强与50k强增强，使用冻结仓库实现 `RandAugment(2,10)`。
该历史实现实际忽略 M，随机采样幅度并加 Cutout；不能将其写作 torchvision 固定幅度 M=10。

## 明确定义的新 ACD 实现

- 角度头：学生特征与权重逐向量归一化，固定 s=30、无 bias，初始权重方向来自同 seed 线性头；类别方向可训练。
- 不增加 ETF 或升维模块。三个学生仅64维，不能构造100类严格 simplex ETF；本批验证 ACI+SICD，不验证 ETF 贡献。
- T=4；师生 logits 在全部100类上逐样本中心化，样本标准差 correction=1，分母下限1e-6。
- 非目标条件 KL 保留标准化教师的非目标概率质量权重，对应有限温度 KL 分解中的关系项。
- 梯度范数参考为同批 standardized-full KL；对全部非分类头、可训练学生参数求范数。匹配比例 detach，分母下限1e-12，不裁剪；发现未使用梯度或非有限数值即失败并停止队列。
- 原 ImageNet SICD 历史核验门槛不变；这里是明确的新 CIFAR 协议，不能伪称恢复旧实现。

以上参数在 [protocol.json](protocol.json) 固定，开跑后用 source/config/runtime/data/teacher 哈希封存。
其中seeds/arms列表保留原始登记作为训练身份历史，**后续运行范围以 [control/schedule.json](control/schedule.json) 的seed3201、acd限制为准**。
不改已封存的源码/配方，避免让正在训练的检查点失去可恢复性；更新后的调度器只向原训练入口传入seed3201、acd。

**调度切换状态：自动接管已部署，无需额外操作。** 用户已经确认调整，但自动审批仍拒绝终止等待进程，要求让它自然退出。
本次采用自然退出方案，不发送终止信号：新控制器通过 `--wait-for-prior-controller` 等待旧控制器释放排他锁。
当前ACD训练PID1442200继续；STOP保持为旧控制器不识别的新标记，旧队列和旧控制器都会在当前任务结束后退出，不会放行对照臂。
新控制器取得锁后消费匹配标记、跳过已完成ACD，继续剩余六个ACD；其他STOP不会被移除，失败任务不会自动重试。
11:29 UTC核验时新进程PID1829577处于 `waiting_for_prior_controller_natural_exit`，当前训练已到第36轮，封存源代码与配方哈希保持一致。
权威接管状态为 `artifacts/acd_only_handoff.json`；旧 `artifacts/queue.json` 在交接前仍可能显示28，新调度取得锁后更新为7。

## 数据和教师来源

教师来自 laniusnan/KD 的 `model-backup-20261027`，按 Release digest 与 MODEL_MANIFEST 的 SHA256 核验。
模型结构从当前仓库 CIFAR-100/Models 冻结，先严格加载 state_dict，再在官方10k测试集复验教师，要求距历史参考不超过2pp。
数据从官方多伦多大学源下载，固定 CIFAR-100 MD5；分段下载结果先合并核验，再由 torchvision 检查解包数据。
准备记录存于 `artifacts/assets.json`；这里不会重训教师或使用部分数据集。

## 预检、运行和结果保存

CPU 检查覆盖有限温度 KL 数值/梯度分解、正仿射不变性、常量输入稳定性、匹配范数与教师冻结、角度缩放不变性。
GPU 预检对全部七配对四臂进行真实批次更新，验证配对骨干/初始方向与增强输入、有限梯度，以及实际保存/恢复后单步完全重放。
预检通过才生成 `artifacts/seal.json` 并启动队列。只使用 GPU0，GPU1 留给正在运行的 ImageNet KD；不抢占已有进程。

每个 seed/epoch 使用固定独立随机数流，逐批哈希图片、标签和样本索引；同 seed 四臂每轮校验一致。
每 epoch 原子保存 `latest.pth`，含模型、SGD、完成 epoch、历史与身份；续训从最近完整 epoch 后开始并重置该 epoch 的确定性随机数流。
官方学生测试只在 epoch240 重载 checkpoint 后进行，输出 `final.json` 和完整预测；不用 test-best 选择。

每任务输出在 `runs/<pair>__s<seed>__<arm>/`：`status.json`、`history.json`、`config.json`、`job.log`、`latest.pth`，完成后另有 `final.json`、`predictions.json`。
队列状态为 `artifacts/queue.json`，总体日志为 `artifacts/launcher.log`。失败时停止，不自动改参数、重试或跳过失败任务。
数据/权重/运行目录由 .gitignore 排除；每个任务完成后自动将最终结果、逐epoch历史、配置身份和预测哈希写入可纳入 Git 的 `results/<job>.json`。
这些小型结果文件备份前不做任何模型或运行文件清理，避免再次只保留权重而遗漏论文读数。

```bash
.venv/bin/python reproductions/cifar100_acd_rerun_20261116/control/dispatch_single_seed.py --check
# 重启前先核对已有后台控制器，避免重复运行：
CUDA_VISIBLE_DEVICES=0 .venv/bin/python reproductions/cifar100_acd_rerun_20261116/control/dispatch_single_seed.py --wait-for-prior-controller
```

后台使用 tmux socket `kd-cifar-rerun`、session `acd-only-handoff`；旧 `single-seed-controller` 会自然退出。旧 `launch.sh` / `run.py dispatch` 保留为封存历史，不再作为新队列入口。
用户另行创建的本目录 `STOP` 文件只停止当前任务之后的新任务调度，不中止正在训练的任务。
