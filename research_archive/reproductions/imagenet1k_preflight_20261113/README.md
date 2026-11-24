# ImageNet-1K ACD 开跑准备

**2026-11-14：完整数据审计、教师复验和四臂真实预检已通过，05:28 UTC 已启动 CE/KD 基线，各 100 epoch；尚无完整训练结果。**

[执行状态与实测](../../reports/IMAGENET1K_PREPARATION_STATUS_20261113.md) / [源代码、环境与 GPU 预检证据](../../reports/IMAGENET1K_PREPARATION_EVIDENCE_20261113.json)。

**当前进度：** [完整下载与审计已通过](../../reports/IMAGENET1K_DOWNLOAD_STATUS_20261113.md)：04:18 UTC 下载全部 1,331,167 张图片，05:01 UTC 全量审计成功；05:20 UTC 完整教师验证 top-1=76.146%、top-5=92.872%。新增真实预检工具位于 `tools/preflight_imagenet1k.py`，通过后生成独立基线协议，保留原候选协议与本目录 `.py` 源码不变。最新状态、正式启动范围和监控命令见[实验启动记录](../../reports/IMAGENET1K_EXPERIMENT_START_20261114.md)。

先读[环境与实验方案](../../reports/IMAGENET1K_FEASIBILITY_AND_EXPERIMENT_PLAN_20261113.md)。本目录不修改旧 ImageNet/CIFAR 训练路径，不以 ImageNet-100 或合成图片替代正式实验。

## 当前硬门槛

1. ImageNet-1K 授权、完整下载与全量审计均已通过，数据目录为 `/fastdata/users/iaii2026/datasets/imagenet1k`。正式入口仍会核对文件清单，不能以子集替换该目录。
2. `reproductions/acd_cnn_interaction_20261109` 等原生训练包在本地和本地 Git 历史均未找到。`methods.py` 中 SICD 是依据论文图示和分析脚本的**方程级重建**，不是已核对的历史实现。
3. 原实现的 epsilon、梯度集合、空梯度处理、裁剪、CE/KD 系数等仍需核对。`SICD_REFERENCE_VERIFIED=False` 阻止 I6/I7 正式训练；不能只翻转布尔值，须恢复封存源码并补数值/梯度对拍。
4. 原 `protocol.json` 保留为候选；独立真实预检通过后才允许生成新基线冻结协议。该协议不会绕过 I6/I7 的 SICD 核验门槛。使用原候选文件直接训练仍会被阻止。

## 实验矩阵

| ID | 头 | 监督 |
|---|---|---|
| I0 | linear | CE |
| I1 | cosine | CE |
| I2 | linear | CE + ordinary KD |
| I3 | cosine | CE + ordinary KD |
| I4 | linear | CE + standardized-full |
| I5 | cosine | CE + standardized-full |
| I6 | linear | CE + SICD，目前仅工程重建 |
| I7 | cosine | CE + SICD，目前仅工程重建 |

主配对为官方 torchvision `ResNet50 → MobileNetV2`，边界配对为 `ResNet34 → ResNet18`。不是仓内被误命名为 `mobilenetv2` 的 MobileNetV1。学生随机初始化，教师使用明确的 `IMAGENET1K_V1` 权重。余弦头不走 ETF，不要求类别数小于特征维度。

## 环境和权重

从仓库根目录 `KD/` 执行。当前解释器是 `.venv/bin/python`，由用户已有的独立 CPython 3.12.16 创建；不是修改系统 Python 或借用其他项目的已安装依赖。若外部基础解释器被移除，须按锁文件重建环境。

`requirements.lock.txt` 记录本次实际安装的全部包版本；CUDA 运行时来自 PyTorch wheel，并不要求系统另装 CUDA toolkit。驱动兼容性以本机真实 GPU 测试为准。

```bash
.venv/bin/python reproductions/imagenet1k_acd_20261113/run.py prepare
.venv/bin/python reproductions/imagenet1k_acd_20261113/run.py prepare --pair resnet34_resnet18
.venv/bin/python -m unittest discover -s reproductions/imagenet1k_acd_20261113 -p 'test_*.py' -v
```

`prepare` 只从官方站点下载教师权重和公开类别索引，不下载 ImageNet 图片。权重检查官方文件名中的哈希前缀并记录完整 SHA256，类别索引固定完整 SHA256。`artifacts/` 和 `runs/` 不入 Git。

## 数据接入与审计

要求已解包的 `ROOT/train/nXXXXXXXX/*.JPEG`、`ROOT/val/nXXXXXXXX/*.JPEG`。验证集必须按官方 ground truth 正确归类，不能随意按文件顺序分配标签。新增的 `tools/fetch_imagenet1k.py` 可将授权后的固定发布版本 parquet 按标签物化成这一布局；本训练入口本身不会下载或解压原始 tar 包。

```bash
.venv/bin/python reproductions/imagenet1k_acd_20261113/run.py audit \
  --data-root /path/to/authorized/imagenet1k
```

审计检查 1000 个 WNID、1281167 张训练图、50000 张验证图、每类 50 张验证图，逐图 RGB 解码和内容 SHA256，记录有序内容摘要及路径/大小/mtime 摘要。训练启动再次检查文件清单；内容变化后应重新审计。数据本身的授权和来源仍需用户确认，计数/哈希不是授权证明。

## 合成 GPU 冒烟

```bash
.venv/bin/python reproductions/imagenet1k_acd_20261113/run.py smoke \
  --device cuda:0 --batch-size 4 --steps 3 \
  --output reproductions/imagenet1k_acd_20261113/artifacts/new_smoke
```

使用真实官方教师、随机学生、随机合成输入；跑前向、反向、优化和保存/恢复重放，记录梯度、匹配系数、峰值显存与纯计算吞吐。测试结果明确标为 `synthetic_engineering_only_not_imagenet_accuracy`，**不是准确率、真实数据吞吐或论文实验**。

可用 `--arms I7 --batch-size 64` 等做显存阶梯测试，也可加 `--pair resnet34_resnet18 --device cuda:1`。每次使用新输出目录，防止覆盖已有证据。此脚本不启用 BF16、TF32、DDP 或 `torch.compile`。

## 正式训练与恢复

新基线协议 `protocol_baselines_20261114.json` 已经真实预检并冻结，I0/I2 使用该协议在独立后台会话运行。日志分别为 `runs/I0_seed3101_baseline_20261114/job.log` 和 `runs/I2_seed3101_baseline_20261114/job.log`，参见[运行记录](../../reports/IMAGENET1K_EXPERIMENT_START_20261114.md)。原候选协议直接训练仍被阻止；下列 I7 命令仍因缺少原生 SICD 核验而返回 exit 2。

```bash
.venv/bin/python reproductions/imagenet1k_acd_20261113/run.py train \
  --data-root /path/to/authorized/imagenet1k --arm I7 --device cuda:0 \
  --output reproductions/imagenet1k_acd_20261113/runs/I7_seed3101
```

先确认没有其他进程占用所选 GPU，再一张卡运行一个独立臂；当前没有自动 GPU 调度器或后台等待下载/开跑服务。协议一次冻结后，所有臂共享 batch、学习率、增强、训练轮数、初始 backbone 和线性权重。CE 对照不做无用教师前向。

每 epoch 原子保存 `latest.pth`，包含模型、SGD、scheduler、CPU/CUDA/Python/NumPy RNG、完成 epoch、数据/源代码/教师身份。使用相同命令加 `--resume` 从最近完整 epoch 恢复；不声称支持任意 minibatch 精确续训。恢复拒绝源码、协议、数据或教师变化。

正式入口先检查完整 50k 验证集上的教师 top-1 是否在官方 V1 参考值 ±2 pp 内，用于发现标签/预处理错误，不调学生超参数。学生只在固定最终 epoch 重载 `latest.pth` 后评估完整验证集，写入 `final.json`；没有按 val-best 选择 checkpoint 的逻辑。若任务在 epoch 中途被中断，从上一个完整 epoch 重跑，不能把未完成结果计为正式终点。

## 实现边界

- 当前为 FP32 单卡版；合成冒烟不能替代真实 DataLoader 吞吐、数据增强配对、完整训练和验证耗时测试。
- SICD 候选：全类别 sample std，epsilon=1e-6；保留教师非真类概率质量；以同批 standardized-full 的非 head 参数梯度范数做 detached 匹配，分母 epsilon=1e-12，不裁剪比值。以上保护值**不是已恢复的历史设置**。
- 新基线协议经真实预检冻结为 100 epoch、SGD lr=0.05、batch=256、CE/KD 各 1；历史 SICD 设置仍待核对，不能写成 CIFAR 原配方零调参迁移。
- 这里只实现首批八臂。DKD、Logit Standardization+DKD 和统一梯度参考对照仍按实验方案第二阶段安排，未实现、未启动。
