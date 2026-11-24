# Release 结果恢复完成（2026-11-16）

已使用 **laniusnan** 登录读取全部 3 个 Release、177 个附件的清单，下载并核验 9 份清单附件和 4 个必要归档，恢复 **706 个文件路径**。
两个发布时点的快照存在重复；去重后核对 **111 条完成实验记录**：52 条早期分类头、34 条后期余弦/ETF、25 条后期线性 KD confirmation。

- [恢复结果表与论文使用边界](release_results.md)
- [111 条逐运行结果 TSV](release_runs.tsv) / [统计、来源与配置 JSON](release_results.json)
- [全部 Release 附件清单](release_inventory.md) / [API 元数据](release_inventory.json)
- [逐文件恢复凭据与哈希](restoration_receipt.json)
- [原始恢复目录](restored)

## 已读取的归档

| Release | 本次恢复范围 |
| --- | --- |
| [model-backup-20261027](https://github.com/laniusnan/KD/releases/tag/model-backup-20261027) | 两卷 head_transfer_20261021 中的 209 份 metadata/protocol，包括 52 组 config/history/predictions/status；未提取其中的模型权重 |
| [kd-head-models-20261031-completed](https://github.com/laniusnan/KD/releases/tag/kd-head-models-20261031-completed) | results-and-provenance 的 240 个文件路径，快照含 30 条完成头部实验 |
| [kd-head-models-20261031-final](https://github.com/laniusnan/KD/releases/tag/kd-head-models-20261031-final) | results-and-provenance 的 257 个文件路径，快照含 34 条完成头部实验及线性基线状态；前述 30 条均在其中 |

所有归档均核对 GitHub digest 和 SHA256SUMS；提取对象逐个核对 manifest 的字节数与 SHA256。
52 个早期 status 文件还与先前 Git 目录快照的哈希和端点准确率逐项匹配。
恢复目录独立于训练工作目录，没有加载 checkpoint、执行归档训练代码或启动训练。

## 对缺失 ACD 结果的结论

归档确实包含历史实验结果，用户关于 Release 的记忆正确。恢复内容还能补齐后期五 seed 基线的标准差与部分分类头对照。
但三个 manifest 中没有 11 月的 `homogeneous_lift_etf_20261103`、`homogeneous_relation_etf_registration_20261103`、`cvpr_*`、`acd_*` 或 `sicd` 研究路径。
177 个附件中最后更新时间为 **2026-10-31T18:13:40Z**。这些结果不能替代完整 SICD/ACD 的缺失行。

完整 ACD 仍有之前四组可对应的记录，其余七组需从原运行机器或其他备份找回；不能据此认定它们从未运行。

## 恢复命令

本次授权保存在独立的 `/tmp/kd-gh-laniusnan` 配置目录，避免切换原有账号。
此前 API 404 来自另一个项目的 xtNie 登录配置；现已通过 laniusnan 设备授权解决，不再受此问题阻塞。

在 `KD/` 目录运行：

```bash
.venv/bin/python paper_assets/acd_tables/main_results_20261116/release_recovery/inspect_releases.py
.venv/bin/python paper_assets/acd_tables/main_results_20261116/release_recovery/restore_result_packages.py --include-head-study
.venv/bin/python paper_assets/acd_tables/main_results_20261116/release_recovery/summarize_release_results.py
```

脚本仅读取远端。已经下载或恢复的文件先校验再复用；下载包在 `downloads/`，被本目录 `.gitignore` 排除。
归档内的文件通过内容哈希映射至 `restored/`，不使用任意路径解包，不覆盖活动训练目录。
API 身份必须为 laniusnan；若独立配置以后不存在或授权失效，再执行：

```bash
GH_CONFIG_DIR=/tmp/kd-gh-laniusnan /home/iaii2026/data/nxt/tools/github-cli/gh_2.104.0_linux_amd64/bin/gh auth login --hostname github.com --git-protocol ssh --web --skip-ssh-key
```
