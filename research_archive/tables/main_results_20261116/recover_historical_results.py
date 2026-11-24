"""Recover historical CIFAR records from a tracked catalog, without model access."""

import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CATALOG = ROOT / "reports/research_catalog_20261027"
STUDY = ROOT / "reproductions/head_transfer_20261021"


def read_tsv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fmt(value):
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def stats(values):
    values = [Decimal(str(v)) for v in values]
    mean = sum(values) / len(values)
    sd = (sum((v - mean)**2 for v in values)/(len(values)-1)).sqrt() if len(values)>1 else None
    return {"values": [float(v) for v in values], "n": len(values), "mean": float(mean),
            "sample_sd": None if sd is None else float(sd), "mean_display": fmt(mean),
            "sd_display": None if sd is None else fmt(sd)}


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def recover(acd_cells):
    layout_path = HERE / "cifar_benchmark_layout.json"
    layout = json.loads(layout_path.read_text())
    catalog_path = CATALOG / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    metadata = {r["path"]: r for r in read_tsv(CATALOG / "source_files.tsv")}
    config_path = STUDY / "study.json"
    config = json.loads(config_path.read_text())
    # The retained protocol must match the protocol scanned with the results.
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert config_hash == metadata[str(config_path.relative_to(ROOT))]["read_sha256"]
    rows = [r for r in read_tsv(CATALOG / "native_statuses.tsv")
            if r["study"] == "head_transfer_20261021"]
    completed = [r for r in rows if r["status"] == "completed" and r["epoch"] == "240"
                 and r["completed_reloaded_top1"] not in ("", "null")]
    assert len(completed) == 52 and len({r["name"] for r in completed}) == 52
    cross = {}
    for row in read_tsv(CATALOG / "cross_model.tsv"):
        for head in ("linear", "cosine", "cosine_etf"):
            item = json.loads(row[head])
            if item is not None and "name" in item:
                cross[item["name"]] = item
    records = []
    for row in completed:
        for key in ("source", "name", "pair", "status", "finished_at"):
            assert row[key] == cross[row["name"]][key], (row["name"], key)
        for key in ("seed", "epoch", "completed_reloaded_top1"):
            assert Decimal(row[key]) == Decimal(str(cross[row["name"]][key]))
        source = metadata[row["source"]]
        records.append({**row, "seed": int(row["seed"]), "epoch": int(row["epoch"]),
                        "head": row["name"].split("__")[-1],
                        "completed_reloaded_top1": float(row["completed_reloaded_top1"]),
                        "original_status_available_locally": (ROOT / row["source"]).is_file(),
                        "original_status_tracked_at_scan": source["tracked_at_scan"] == "True",
                        "original_status_sha256_at_scan": source["read_sha256"]})
        restored = HERE / "release_recovery/restored/model-backup-20261027" / row["source"]
        if restored.is_file():
            assert hashlib.sha256(restored.read_bytes()).hexdigest() == source["read_sha256"]
            records[-1]["restored_status_path"] = str(restored.relative_to(HERE))
    restored_count = sum("restored_status_path" in r for r in records)
    release_results_path = HERE / "release_recovery/release_results.json"
    release_results = json.loads(release_results_path.read_text()) if release_results_path.is_file() else None
    pairs = {}
    for kind in ("homogeneous", "heterogeneous"):
        for col in layout[kind]:
            key = col["key"]
            items = [r for r in records if r["pair"] == key]
            heads = {head: {r["seed"]: r["completed_reloaded_top1"]
                            for r in items if r["head"] == head}
                     for head in ("linear", "cosine", "cosine_etf")}
            seeds = sorted(heads["linear"].keys() & heads["cosine"].keys())
            assert seeds
            pairs[key] = {"pair": col["teacher"] + " → " + col["student"], "group": kind,
                          "completed_record_count": len(items), "paired_seeds": seeds,
                          "paired_methods": {h: stats([heads[h][s] for s in seeds])
                                             for h in ("linear", "cosine")},
                          "paired_delta_pp": stats([Decimal(str(heads["cosine"][s])) -
                                                    Decimal(str(heads["linear"][s])) for s in seeds]),
                          "all_completed_by_head": {h: {"seeds": sorted(v),
                              **stats([v[s] for s in sorted(v)])} for h, v in heads.items() if v},
                          "unpaired_records": [r["name"] for r in items
                                               if r["head"] != "cosine_etf" and r["seed"] not in seeds],
                          "retained_later_acd": acd_cells.get(key)}
    assert len(pairs) == 11 and sum(len(p["paired_seeds"]) for p in pairs.values()) == 23

    weekly_path = ROOT / "reports/HETEROGENEOUS_KD_WEEK_20261102.md"
    weekly = weekly_path.read_text()
    later_baselines = [
        {"pair": "vgg13_mv1", "linear": 67.836, "cosine": 66.670, "cosine_etf": 66.702},
        {"pair": "res50_mv1", "linear": 68.422, "cosine": 67.532, "cosine_etf": 67.394},
        {"pair": "wrn402_sv1", "linear": 77.310},
        {"pair": "res32x4_sv1", "linear": 76.822},
        {"pair": "res32x4_sv2", "linear": 76.442},
    ]
    for item in later_baselines:
        for key, value in item.items():
            if key != "pair":
                assert f"{value:.3f}" in weekly
        item.update(n=5, protocol="compact_double", evidence="weekly report means; seed arrays absent locally",
                    source=str(weekly_path.relative_to(ROOT)))
        if release_results:
            item["restored_methods"] = release_results["later_all_available_methods"][item["pair"]]
            item["evidence"] = "Release results restored; see release_recovery/release_results.json for per-seed records"
    inputs = [CATALOG / f for f in ("native_statuses.tsv", "cross_model.tsv", "source_files.tsv", "catalog.json")]
    inputs += [config_path, STUDY / "methods.py", layout_path, weekly_path]
    if release_results:
        inputs.append(release_results_path)
    result = {"study": "head_transfer_20261021", "recovered_on": "2026-11-16",
              "snapshot_started_at": catalog["scan_started_at"],
              "snapshot_finished_at": catalog["scan_finished_at"],
              "completed_records": records, "completed_record_count": len(records), "pairs": pairs,
              "excluded_nonfinal_snapshot_records": [r for r in rows if r not in completed],
              "cross_model_catalog_check": "All 52 completed records match cross_model.tsv; same scan, not independent reruns.",
              "original_status_files_available_locally": sum(r["original_status_available_locally"] for r in records),
              "original_status_files_tracked_at_scan": sum(r["original_status_tracked_at_scan"] for r in records),
              "original_status_files_restored_from_release": restored_count,
              "training": config["training"], "heads": config["heads"],
              "method_classification": "Ordinary CE + temperature-scaled KL; no standardization or conditional distillation. Not full ACD.",
              "later_compact_double_means": later_baselines,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (HERE / "historical_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    fields = ["pair", "head", "seed", "status", "epoch", "completed_reloaded_top1", "finished_at",
              "name", "source", "original_status_available_locally", "original_status_tracked_at_scan",
              "original_status_sha256_at_scan", "restored_status_path"]
    with (HERE / "historical_runs.tsv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    md = ["# CIFAR-100 历史结果找回记录（2026-11-16）", "",
          "之前确实跑过这 11 个配对。上一次表格整理只覆盖了后期 ACD 记录，漏查了旧分类头实验快照；不能把主表空白解释为没有运行实验。", "",
          "## 找回了什么", "",
          "从已纳入 Git 的 [2026-10-27 目录快照](../../../reports/research_catalog_20261027/native_statuses.tsv)找回 **52 条完成至 epoch 240 的记录，覆盖全部 11 个配对**。"
          "逐项与同次扫描的 `cross_model.tsv` 对账一致；这两个文件不是两次独立实验。"
          "原工作路径的 52 个 `status.json` 当时均未纳入 Git。"
          + ("现已从 Release 恢复全部 52 个原始文件至独立目录，逐个 SHA256 与快照一致。" if restored_count == 52
             else "当前副本缺失，完成时间和重载端点精度仍保存在快照中。"), "",
          ("**Release 恢复已完成：** [结果报告](release_recovery/release_results.md)还补齐后期 34 条头部实验与 25 条线性基线记录；"
           "共核对 111 条不重复的完成记录，尚未找到 11 月 SICD/ACD 研究包。" if release_results else ""), "",
          "[逐运行 TSV](historical_runs.tsv)包含原始路径、完成时间、seed 和当时的文件哈希；"
          "[完整 JSON](historical_results.json)包含各方法全部 seed、样本 SD、配对差和来源哈希。", "",
          "## 早期分类头对照：按共同完成的 seed 汇总", "",
          "以下为官方 CIFAR-100 10,000 张测试集的 epoch-240 重载端点 Top-1（%）。只比较同配对、同 seed 的线性 KD 和余弦头 KD；"
          "均值 ± 样本 SD，单 seed 不补造 SD。快照截至 2026-10-27，运行数量不是截至今天的最终总数。", ""]
    latex = ["% Recovered historical ordinary-KD head study; not full ACD. Requires booktabs."]
    for kind, title in (("homogeneous", "同构"), ("heterogeneous", "异构")):
        table_rows = []
        tex_rows = []
        for col in layout[kind]:
            p = pairs[col["key"]]
            values = [p["paired_methods"][h] for h in ("linear", "cosine")]
            displays = [v["mean_display"] + (" ± " + v["sd_display"] if v["sd_display"] else "") for v in values]
            delta = p["paired_delta_pp"]["mean"]
            gain = ("+" if delta >= 0 else "") + fmt(delta)
            table_rows.append([p["pair"], ", ".join(map(str, p["paired_seeds"])), *displays, gain])
            name = p["pair"].replace("×", r"$\times$").replace("→", r"$\to$")
            tex_values = ["$" + v["mean_display"] + (r"\pm" + v["sd_display"] if v["sd_display"] else "") + "$" for v in values]
            tex_rows.append(" & ".join([name, str(len(p["paired_seeds"])), *tex_values, gain])+r" \\")
        md += ["### " + title, "", md_table(["教师 → 学生", "共同 seed", "线性 KD", "余弦头 KD", "差值 pp"], table_rows), ""]
        latex += [r"\begin{table*}[t]", r"\centering\small",
                  r"\caption{Recovered " + kind + r" CIFAR-100 head-study results (Top-1, \%).}",
                  r"\label{tab:historical_" + kind + "}", r"\begin{tabular}{lrrrr}", r"\toprule",
                  r"Teacher $\to$ student & Paired seeds & Affine KD & Cosine-head KD & $\Delta$ (pp) \\",
                  r"\midrule", *tex_rows, r"\bottomrule", r"\end{tabular}",
                  r"\par\smallskip\begin{minipage}{0.95\textwidth}\footnotesize",
                  r"Mean $\pm$ sample SD over common completed seeds in the 2026-10-27 snapshot; no SD for a single seed. "
                  r"All heads use ordinary KL, not SICD. Results use the historical double-view recipe and must not be ranked against later ACD runs from other recipes.",
                  r"\end{minipage}", r"\end{table*}", ""]
    etf = pairs["res32x4_res8x4"]["all_completed_by_head"]["cosine_etf"]
    md += ["额外找回 ResNet32×4→ResNet8×4 的 ETF 初始化余弦头 5 个 seed："
           + ", ".join(fmt(v) for v in etf["values"]) + "，均值 **" + etf["mean_display"] + " ± " + etf["sd_display"] + "**。"
           "它仍使用普通 KL，不是完整 ACD。", "",
           "ResNet110→ResNet32 另有线性 KD seed2053=74.31，因缺少该 seed 的余弦完成值，未计入配对均值。"
           "另有 ResNet50→MobileNetV1 线性 seed2052 在旧快照中为 running，未作为完成结果；不能由这个历史状态推断现在仍在跑。", "",
           "## 为什么不能把这些数值直接写进 ACD 行", "",
           "保留的 [study.json](../../../reproductions/head_transfer_20261021/study.json) 与目录快照中的 SHA256 一致。"
           "[损失实现](../../../reproductions/head_transfer_20261021/methods.py)使用 `0.5 CE + 0.5 T² KL`，T=4；"
           "没有双侧标准化或 conditional relation-only。余弦头尺度 s=30，ETF 仅用于原 ResNet32×4→ResNet8×4 配对。", "",
           "该批训练使用 SGD、240 epochs、batch64、lr0.05、momentum0.9、weight decay0.0005，"
           "150/180/210 轮衰减为原来的 0.1；double 增强每轮 100,000 视图。它支持分类头对照，不能替代 SICD/ACD 完整方法结果。", "",
           "## 之后还跑过其他配方：单独保留", "",
           "[11 月 2 日周报](../../../reports/HETEROGENEOUS_KD_WEEK_20261102.md)还保存了后续 `compact_double` 的五 seed 官方测试均值：", "",
           md_table(["配对", "线性 KD", "余弦头 KD", "ETF 初始化余弦头"],
                    [[pairs[r["pair"]]["pair"], *[f"{r[h]:.3f}" if h in r else "未恢复汇总" for h in ("linear", "cosine", "cosine_etf")]] for r in later_baselines]), "",
           ("这些是另一批实验，现已从 Release 恢复逐 seed 数组，见 [后期结果报告](release_recovery/release_results.md)。" if release_results
            else "这些是另一批实验，当前只有周报均值，没有完整逐 seed 数组。")
           + "不能与上面早期配方混算均值或择优填表。"
           "例如 VGG13→MobileNetV1 的线性 KD 从早期表的 59.12 到后续配方的 67.836，已经足以说明配方匹配不可省略。", "",
           "## 完整 ACD 的现有覆盖", "",
           md_table(["配对", "已找回的早期完成记录数", "后期 standalone ACD"],
                    [[p["pair"], p["completed_record_count"],
                      (p["retained_later_acd"]["ACD"] + "（n=" + str(p["retained_later_acd"]["acd_runs"]) + "）")
                      if p["retained_later_acd"] else "当前缺少可归入该行的终点记录"] for p in pairs.values()]), "",
           "后期 4 个 ACD 数值及其证据级别详见 [主表说明](README.md)。AT+ACD=78.97 和 ReviewKD-style+ACD=75.60 是组合实验，不能替代相应配对的 standalone ACD。", "",
           "## 缺失发生在哪里，下一步需要恢复什么", "",
           ("- 早期状态文件：已从 Release 恢复全部 52 个，并验证与目录快照哈希及端点读数一致；恢复目录独立于训练工作目录。" if restored_count == 52
            else "- 早期状态文件：快照明确标记为未跟踪，当前副本缺失；已用快照恢复其数值，并未伪造原始 status 文件。"),
           "- 11 月 3 日的 `homogeneous_lift_etf_20261103` 和 `homogeneous_relation_etf_registration_20261103`："
           "根 README 保留了登记/部署说明，当前目录缺失，本地完整 Git 历史中也未找到。前者最后保留的说明是 9 个任务等待资源，后者是登记通过/部分维度阻塞；这些说明都不是最终完成状态。",
           "- 10 月 31 日两个私有 Release 按 README 归档了 92+18 个模型，并声明当时保留指标、历史、预测、日志、配置和源码。"
           "模型归档不等于指标被删除；也不能据此断言这些 Release 包含 11 月后的 ACD 结果。",
           "- 应从原运行机器或备份恢复后期研究包的 `status.json`、`state.json`、`summary.json`、逐 epoch history、配置和任务登记；"
           "先恢复这些小文件即可判断最终结果和配方，不必先下载全部模型。需要重新评测时再恢复对应 checkpoint。", "",
           "当前结论是：实验确实跑过，部分结果已经找回；其余 ACD 终点记录在本机尚未恢复，不能据此断言从未运行或永久丢失。", ""]
    (HERE / "historical_results.md").write_text("\n".join(md))
    (HERE / "historical_head_tables.tex").write_text("\n".join(latex)+"\n")
    return result


if __name__ == "__main__":
    cells = json.loads((HERE / "results.json").read_text())["cifar_provisional_local_cells"]
    result = recover(cells)
    print(json.dumps({"completed_records": result["completed_record_count"], "pairs": len(result["pairs"])}))
