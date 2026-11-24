"""Reconcile restored result snapshots without executing archived training code."""

import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from recover_historical_results import stats, fmt, md_table


def main():
    inventory = json.loads((HERE / "release_inventory.json").read_text())
    receipt = json.loads((HERE / "restoration_receipt.json").read_text())
    receipts = {r["path"]: r for r in receipt["files"]}
    sources = {}

    def read(path):
        relative = str(path.relative_to(HERE))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == receipts[relative]["sha256"]
        sources[relative] = digest
        return json.loads(path.read_text())

    layout = json.loads((HERE.parent / "cifar_benchmark_layout.json").read_text())
    names = {c["key"]: c["teacher"] + " → " + c["student"]
             for group in ("homogeneous", "heterogeneous") for c in layout[group]}
    older = json.loads((HERE.parent / "historical_results.json").read_text())
    records = []
    for row in older["completed_records"]:
        path = HERE / "restored/model-backup-20261027" / row["source"]
        status = read(path)
        assert status["status"] == "completed" and status["epoch"] == 240
        assert status["reloaded_test"]["samples"] == 10000
        assert status["reloaded_test"]["top1"] == row["completed_reloaded_top1"]
        assert sources[str(path.relative_to(HERE))] == row["original_status_sha256_at_scan"]
        records.append({"study": "head_transfer_20261021", "name": status["name"], "pair": status["pair"],
                        "head": status["head"], "seed": status["seed"], "top1": status["reloaded_test"]["top1"],
                        "epoch": 240, "protocol": "early_double", "checkpoint_sha256": status["checkpoint_sha256"],
                        "source": str(path.relative_to(HERE)), "source_kind": "native status"})
    final = HERE / "restored/kd-head-models-20261031-final"
    latest = read(final / "RESULTS.json")
    earlier = read(HERE / "restored/kd-head-models-20261031-completed/RESULTS.json")
    index = {(j["study"], j["job"]): j for j in latest["jobs"]}
    assert len(index) == 34 and len(earlier["jobs"]) == 30
    for job in earlier["jobs"]:
        newer = index[(job["study"], job["job"])]
        assert job["test"] == newer["test"] and job["checkpoint_sha256"] == newer["checkpoint_sha256"]
    for (study, name), row in sorted(index.items()):
        path = final / "reproductions" / study / "artifacts" / name / "status.json"
        status = read(path)
        assert status["status"] == "completed" and status["epoch"] == 240 and status["reload_verified"]
        assert status["test"] == row["test"] and status["test"]["samples"] == 10000
        assert status["checkpoint_sha256"] == row["checkpoint_sha256"]
        records.append({"study": study, "name": name, "pair": status["pair"], "head": status["head"],
                        "seed": status["seed"], "top1": status["test"]["top1"], "epoch": 240,
                        "protocol": "compact_double", "checkpoint_sha256": status["checkpoint_sha256"],
                        "source": str(path.relative_to(HERE)), "source_kind": "native status"})
    linear_path = final / "reproductions/linear_kd_single_seed_20261029/artifacts/state.json"
    linear = read(linear_path)
    seeds = read(final / "reproductions/linear_kd_single_seed_20261029/protocol.json")["confirmation_seeds"]
    for job in linear["jobs"].values():
        if job.get("phase") != "confirmation" or job["status"] != "completed":
            continue
        status = job["result"]
        assert status["epoch"] == 240 and status["reload_verified"] and status["test"]["samples"] == 10000
        records.append({"study": "linear_kd_single_seed_20261029", "name": status["name"], "pair": status["pair"],
                        "head": "linear", "seed": status["seed"], "top1": status["test"]["top1"],
                        "epoch": 240, "protocol": "compact_double", "checkpoint_sha256": status["checkpoint_sha256"],
                        "source": str(linear_path.relative_to(HERE)), "source_kind": "state.json completed result"})
    assert len(records) == 111 and len({(r["study"], r["name"]) for r in records}) == 111
    heads_path = final / "reproductions/accepted_kd_heads_20261030/artifacts/state.json"
    head_state = read(heads_path)
    paired, all_methods, training = {}, {}, {}
    for column in layout["heterogeneous"]:
        pair = column["key"]
        values = {head: {r["seed"]: r["top1"] for r in records
                         if r["protocol"] == "compact_double" and r["pair"] == pair and r["head"] == head}
                  for head in ("linear", "cosine", "cosine_etf", "rotating_etf")}
        assert sorted(values["linear"]) == sorted(seeds)
        assert [values["linear"][s] for s in seeds] == head_state["pairs"][pair]["baseline"]["test_top1"]
        all_methods[pair] = {h: {"seeds": sorted(v), **stats([v[s] for s in sorted(v)])}
                             for h, v in values.items() if v}
        comparison = [h for h in ("linear", "cosine", "cosine_etf") if values[h]]
        common = sorted(set.intersection(*(set(values[h]) for h in comparison)))
        paired[pair] = {"seeds": common, "methods": {h: stats([values[h][s] for s in common]) for h in comparison}}
        lock = read(final / f"reproductions/linear_kd_single_seed_20261029/artifacts/locks/{pair}.json")
        training[pair] = lock["training_config"]["training"]
        for key, expected in {"lr": .01, "ce_weight": .1, "kd_weight": .9, "temperature": 4,
                              "batch_size": 64, "epochs": 240, "examples_per_epoch": 100000}.items():
            assert training[pair][key] == expected
    manifests = [json.loads(p.read_text()) for p in (HERE / "metadata").glob("*/*_MODEL_MANIFEST.json")]
    needles = ("homogeneous_lift_etf_20261103", "homogeneous_relation_etf_registration_20261103", "cvpr_", "acd_", "sicd")
    missing = {term: [f["path"] for m in manifests for f in m["files"] if term in f["path"].lower()] for term in needles}
    result = {"authenticated_login": inventory["authenticated_login"], "release_count": len(inventory["releases"]),
              "restored_file_count": len(receipt["files"]), "records": records,
              "counts": {"early_native_runs": 52, "later_native_head_runs": 34, "later_linear_confirmations": 25,
                         "unique_completed_runs": 111, "duplicate_snapshot_runs_not_recounted": 30},
              "later_all_available_methods": all_methods, "later_paired_head_comparisons": paired,
              "later_training": training, "unlocated_study_path_matches": missing,
              "latest_release_asset_update": max(a["updated_at"] for r in inventory["releases"] for a in r["assets"]),
              "source_sha256": sources,
              "limits": ["These are historical ordinary-KD/head studies, not SICD/ACD results.",
                         "Adaptive official-test recipe selection limits benchmark interpretation.",
                         "No checkpoint was loaded or re-evaluated; counts concern recovered historical records.",
                         "Earlier completed/final release snapshots overlap; duplicate records are not independent seeds."]}
    (HERE / "release_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    with (HERE / "release_runs.tsv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(records)
    rows = []
    for pair, data in paired.items():
        cells = []
        for head in ("linear", "cosine", "cosine_etf"):
            s = data["methods"].get(head)
            cells.append("—" if not s else s["mean_display"] + (" ± " + s["sd_display"] if s["sd_display"] else ""))
        rows.append([names[pair], ", ".join(map(str, data["seeds"])), *cells])
    baseline_rows = [[names[p], ", ".join(fmt(v) for v in d["linear"]["values"]),
                      d["linear"]["mean_display"] + " ± " + d["linear"]["sd_display"]] for p, d in all_methods.items()]
    rotating_rows = []
    for pair, data in all_methods.items():
        if "rotating_etf" not in data: continue
        vals = {h: next(r["top1"] for r in records if r["pair"] == pair and r["head"] == h
                       and r["protocol"] == "compact_double" and r["seed"] == 2101)
                for h in ("linear", "cosine", "rotating_etf")}
        rotating_rows.append([names[pair], *[fmt(vals[h]) for h in ("linear", "cosine", "rotating_etf")]])
    md = ["# 从 GitHub Release 找回的实验结果", "",
          "已用 laniusnan 登录读取全部 3 个 Release，下载并核验 9 份清单附件及 4 个必要归档。"
          "恢复 **706 个文件路径**（两个时间点快照有重复），确认 **111 条不重复的完成实验记录**：早期 52 条、后期分类头 34 条、后期线性 KD confirmation 25 条。"
          "这 111 条属于历史普通 KD／分类头实验，不是 111 条完整 ACD。", "",
          "原始数据保存在 [restored](restored)，[逐运行 TSV](release_runs.tsv)、[汇总 JSON](release_results.json)和"
          "[逐文件恢复凭据](restoration_receipt.json)记录 seed、checkpoint 哈希与来源。52 条早期状态文件的 SHA256 和端点读数全部与先前目录快照一致。", "",
          "## 后期异构线性 KD：五 seed 结果已恢复", "",
          "以下均为 compact_double、seed2101–2105、epoch240、官方 10,000 图测试集 Top-1（%）；± 为样本 SD。", "",
          md_table(["配对", "逐 seed Top-1", "均值 ± SD"], baseline_rows), "",
          "## 后期分类头对照：只使用共同完成的 seed", "",
          md_table(["配对", "共同 seed", "线性 KD", "余弦头 KD", "ETF 初始化余弦头"], rows), "",
          "WRN→ShuffleNetV1 的三臂共同完成 seed 只有 2101–2104，因此这里的线性均值与上表五 seed 均值不同。"
          "ResNet32×4→ShuffleNetV1 只有 seed2101 的线性/余弦对照；ShuffleNetV2 只恢复了线性基线，其余两臂在当时快照中因基线门槛未通过而跳过。"
          "所有方法各自的完整 seed 数组另见 JSON，不把不同 seed 数量混为配对比较。", "",
          "## 旋转 ETF：同 seed2101 比较", "",
          md_table(["配对", "线性 KD", "普通余弦", "旋转 ETF"], rotating_rows), "",
          "这是四个单 seed pilot；不能宣称普遍正收益，也不能与五 seed 均值直接计算配对增益。", "",
          "## 配方与论文使用边界", "",
          "恢复的 compact_double 锁定配置为 lr=0.01、CE/KD 权重=0.1/0.9、T=4、batch64、240 epochs、"
          "每轮 100,000 弱/强视图；余弦尺度 s=30。早期 head_transfer 则为 lr=0.05、CE/KD=0.5/0.5。"
          "这解释了同一配对历史数值不同的一个明确来源，不能跨配方择优填表。", "",
          "本批分类头训练调用普通 KD loss；不含 SICD 的标准化条件关系目标。记录还明确披露曾依据官方测试反馈选择基线配方，"
          "这些恢复结果可支持历史结果核验与头部消融，不能直接当作无测试集调参的统一 SOTA benchmark。", "",
          "## 是否找到了缺失的完整 ACD", "",
          "在这 3 个 Release 的全部 manifest 路径中，未发现 `homogeneous_lift_etf_20261103`、"
          "`homogeneous_relation_etf_registration_20261103`、`cvpr_*`、`acd_*` 或 `sicd`；资产最后更新时间为 "
          + result["latest_release_asset_update"] + "。结果快照属于 10 月 27/31 日，未找回 11 月的 SICD/ACD 研究包。", "",
          "因此，先前能对应到 standalone ACD 的四组记录保持不变，其余七组不能用这些普通 KD/ETF 数值替代。"
          "仍需从原运行机器或其他备份寻找后期研究包；本结论不等于那些实验从未运行。", ""]
    (HERE / "release_results.md").write_text("\n".join(md))
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
