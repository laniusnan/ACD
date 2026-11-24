"""Build manuscript tables from retained results; never run model evaluation."""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

from recover_historical_results import recover

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEDGER = ROOT / "reports/ACD_EVIDENCE_LEDGER_AND_NEXT_PLAN_20261111.md"
AUDIT = ROOT / "reports/ACD_CVPR_REVIEWER_AUDIT_20261111.md"
FIGURE_DATA = ROOT / "paper_assets/acd_figures/mechanism_figures/acd_evidence_figure_data.json"
LAYOUT = HERE / "cifar_benchmark_layout.json"
SCREENSHOT = HERE / "user_reported_etf_20261116.json"
IMAGENET = ROOT / "reproductions/imagenet1k_acd_20261113"
PAIRS = [("vgg13_vgg8", "VGG13 → VGG8"), ("vgg13_mv1", "VGG13 → MobileNetV1")]
METHODS = [("linear_kd", "KD (affine)"), ("cosine_kd", "ACI + KL"),
           ("cosine_standardized_full", "ACI + standardized full KL"),
           ("cosine_relation_matched", "ACD (ACI + SICD)")]


def dec(value):
    return Decimal(str(value))


def fmt(value, places=2):
    return str(dec(value).quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP))


def statistics(values):
    values = [dec(v) for v in values]
    mean = sum(values) / len(values)
    sd = (sum((v-mean)**2 for v in values)/(len(values)-1)).sqrt() if len(values)>1 else None
    return {"values": [float(v) for v in values], "n": len(values), "mean": float(mean),
            "sample_sd": None if sd is None else float(sd), "mean_display": fmt(mean),
            "sd_display": None if sd is None else fmt(sd)}


def ledger_cells(prefix):
    matches = [line for line in LEDGER.read_text().splitlines() if line.startswith("| "+prefix)]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous ledger row: {prefix}")
    return [cell.strip() for cell in matches[0].strip("|").split("|")]


def markdown_table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"]+
                      ["| "+" | ".join(map(str,row))+" |" for row in rows])


def tex_table(label, caption, columns, headers, rows, note):
    return "\n".join([r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
        "\\caption{"+caption+"}", "\\label{"+label+"}", "\\begin{tabular}{"+columns+"}",
        r"\toprule", " & ".join(headers)+r" \\", r"\midrule"]+
        [" & ".join(row)+r" \\" for row in rows]+[r"\bottomrule", r"\end{tabular}",
        r"\par\smallskip", r"\begin{minipage}{0.96\textwidth}\footnotesize", note,
        r"\end{minipage}", r"\end{table*}", ""])


def screenshot_records():
    data = json.loads(SCREENSHOT.read_text())
    rows = data["rows"]
    assert len(rows) == len({r["pair_key"] for r in rows}) == 5
    for row in rows:
        delta = dec(row["plus_etf_init"]) - dec(row["relation_only"])
        assert delta == dec(row["reported_delta_pp"])
        assert all(0 <= dec(row[k]) <= 100 for k in ("relation_only", "plus_etf_init"))
        row["delta_pp"] = float(delta)
        row["delta_display"] = ("+" if delta >= 0 else "") + fmt(delta)
    deltas = [dec(r["delta_pp"]) for r in rows]
    data["descriptive_summary"] = {
        "pairs": len(rows), "positive_pairs": sum(d > 0 for d in deltas),
        "negative_pairs": sum(d < 0 for d in deltas),
        "unweighted_mean_delta_pp": float(sum(deltas) / len(deltas)),
        "mean_delta_display": fmt(sum(deltas) / len(deltas)),
        "interpretation": "Descriptive difference across five user-reported pairs, not a seed average or significance test."
    }
    return data


def reported_tables(layout, mapped, reported):
    lookup = {r["pair_key"]: r for r in reported["rows"]}
    columns = {r["key"]: r for kind in ("homogeneous", "heterogeneous") for r in layout[kind]}
    assert set(lookup) <= set(columns)
    note = ("* 为用户提供的历史截图文字转录。评测划分、最终/最佳 checkpoint、seed 数、"
            "配方及 relation-only 与当前 ACD 实现的对应关系待核实；未知项不默认等于当前重跑设置。"
            "账本列来自之前保留的四组 ACD 记录，不能与截图列取最大值合并或混算均值。"
            "准确率单位为 %，差值单位为百分点（pp）；— 表示缺失。")
    md = ["# 当前 11 配对结果台账", "", note, ""]
    for kind, title in (("homogeneous", "同构配对"), ("heterogeneous", "异构配对")):
        rows = []
        for col in layout[kind]:
            old = mapped.get(col["key"], {})
            screenshot = lookup.get(col["key"], {})
            old_value = old.get("ACD", "—")
            if "acd_runs" in old:
                old_value += f" (n={old['acd_runs']})"
            rows.append([col["teacher"] + " → " + col["student"], old_value,
                         screenshot.get("relation_only", "—"), screenshot.get("plus_etf_init", "—"),
                         screenshot.get("delta_display", "—")])
        md += ["## " + title, "", markdown_table(
            ["教师 → 学生", "已保留账本 ACD", "截图 relation-only*", "截图 +ETF 初始化*", "ETF 差值*"], rows), ""]
    summary = reported["descriptive_summary"]
    md += [f"历史账本覆盖 {len(mapped)} 个配对，截图覆盖 {len(lookup)} 个；合并记录覆盖 {len(set(mapped) | set(lookup))}/11 个配对。"
           "这表示有数值记录的覆盖率，不等于统一协议下已完成主表的覆盖率。", "",
           "WRN-40-2→ShuffleNetV1 的账本值 78.86 与截图 relation-only 81.38，"
           "以及 VGG13→VGG8 的账本值 75.48 与截图 80.22，属于尚未核对身份的不同记录，均予保留。", "",
           "新跑批次仍为七配对 × seed3201 × ACD，未含 ETF；本次补表不修改训练或调度。"
           "新增运行必须有完成后的 final.json，且核对协议后，才可作为独立的新结果列纳入。", ""]
    (HERE / "current_acd_records.md").write_text("\n".join(md))
    erows = []
    for row in reported["rows"]:
        col = columns[row["pair_key"]]
        erows.append([col["teacher"] + " → " + col["student"], row["relation_only"],
                      row["plus_etf_init"], row["delta_display"]])
    etf_md = "\n".join(["## 用户补充的历史 ETF 初始化对照（待核实协议）", "",
        markdown_table(["教师 → 学生", "relation-only*", "+ETF 初始化*", "差值 (pp)"], erows), "", note, "",
        f"5 组中 {summary['positive_pairs']} 组提升、{summary['negative_pairs']} 组下降，"
        f"等权平均差值 +{summary['mean_delta_display']} pp（未舍入值 +0.156 pp）。"
        "这支持将 ETF 作为效果依赖配对的初始化消融；不能据此声称所有配对都受益、统计显著，"
        "或已证明弱敏感类别保留最优角度的机制。", ""])
    etf_tex = tex_table("tab:etf_user_reported",
        "User-reported historical ETF-initialization comparison (accuracy, \\%; protocol pending verification).",
        "lrrr", ["Teacher $\\to$ student", "Relation-only", "+ ETF init.", r"$\Delta$ (pp)"],
        [[r[0].replace("×", r"$\times$").replace("→", r"$\to$"), *r[1:]] for r in erows],
        "Values were transcribed by the user from a historical screenshot. Evaluation split, checkpoint selection, "
        "run count, recipe, and equivalence to ACD remain unverified. Three of five differences are positive; "
        "the unweighted mean is +0.16 pp. This is descriptive evidence, not a significance test or a uniformly "
        "matched benchmark. These entries do not replace archived ACD means.")
    (HERE / "user_reported_etf_table.md").write_text(etf_md)
    (HERE / "user_reported_etf_table.tex").write_text(etf_tex)
    return etf_md, etf_tex


def benchmark_tables(paired, extended, reported):
    layout = json.loads(LAYOUT.read_text())
    assert len(layout["homogeneous"]) == 6 and len(layout["heterogeneous"]) == 5
    mapped = {}
    for key, _ in PAIRS:
        value = paired[key]
        mapped[key] = {"KD": value["methods"]["linear_kd"]["mean_display"],
                       "ACD": value["methods"]["cosine_relation_matched"]["mean_display"],
                       "acd_runs": len(value["seeds"]), "source": "paired_cifar."+key}
    for key, row in zip(("res32x4_res8x4", "wrn402_sv1"), extended):
        mapped[key] = {"ACD": fmt(row["acd"]), "acd_runs": row["n"],
                       "source": "extended_cifar_archived_means: "+row["pair"]}
        if row["dkd"] is not None:
            mapped[key]["DKD"] = fmt(row["dkd"])
    historical = recover(mapped)
    reported_map = {r["pair_key"]: r for r in reported["rows"]}
    markdown=[]; latex=[]
    for kind, title in (("homogeneous", "Homogeneous"), ("heterogeneous", "Heterogeneous")):
        columns = layout[kind]
        rows = [["Teacher Top-1 (reference)"]+[fmt(col["teacher_top1"]) for col in columns],
                ["Student"]+[col["student"] for col in columns],
                ["Student Top-1 (reference)"]+[fmt(col["student_top1"]) for col in columns]]
        for method in ("KD", "DKD", "NormKD", "CRD", "ReviewKD", "ACD"):
            label = method+"†" if method in {"KD","DKD","ACD"} else method
            rows.append([label]+[mapped.get(col["key"],{}).get(method,"—") for col in columns])
        rows.append(["ACD runs"]+[str(mapped[col["key"]]["acd_runs"]) if col["key"] in mapped else "—" for col in columns])
        for label, key in (("Reported relation-only*", "relation_only"),
                           ("Reported +ETF init.*", "plus_etf_init"), ("Reported ETF delta (pp)*", "delta_display")):
            rows.append([label]+[reported_map.get(col["key"], {}).get(key, "—") for col in columns])
        rows.append(["Recovered historical head-study runs"]+
                    [str(historical["pairs"][col["key"]]["completed_record_count"]) for col in columns])
        heads=["Method / Teacher"]+[col["teacher"] for col in columns]
        note=("Teacher/student reference accuracies are supplied by the user and are not local checkpoint re-evaluations. "
              "† Local experimental records are provisionally placed by architecture pair; protocol compatibility with the reference table must be checked before ranking. "
              "Dashes denote entries unavailable for this method/protocol, not evidence that a pair was never trained. "
              "52 completed ordinary-KD head-study records cover all 11 pairs; these are summarized separately in the historical recovery tables, not relabeled as ACD. "
              "Feature-pipeline combinations are excluded from standalone ACD cells. "
              "* User-transcribed historical screenshot values: evaluation split, checkpoint selection, run count, recipe and equivalence to ACD remain unverified. "
              "They are retained separately from archived ACD means and are not used for ranking.")
        markdown += ["## "+title+" CIFAR-100 benchmark", "",markdown_table(heads,rows),"",note,""]
        def tex_cell(cell):
            return cell.replace("×",r"$\times$").replace("†",r"$^{\dagger}$").replace("—",r"\textemdash")
        table=tex_table("tab:cifar_"+("homo" if kind=="homogeneous" else "hetero"),
                        title+" CIFAR-100 benchmark layout with provisional local results (Top-1, \\%).",
                        "l"+"c"*len(columns),[tex_cell(s) for s in heads],
                        [[tex_cell(s) for s in row] for row in rows],tex_cell(note))
        # Fit the six architecture columns without shrinking the entire table.
        table=table.replace(r"\small"+"\n",r"\footnotesize"+"\n",1).replace("{5pt}","{3pt}")
        latex.append(table)
    return layout, mapped, "\n".join(markdown), "\n".join(latex), historical


def main():
    reported = screenshot_records()
    figure = json.loads(FIGURE_DATA.read_text())
    paired = {}
    for key, title in PAIRS:
        source = figure["ablation"][key]
        paired[key] = {"pair": title, "seeds": [2102,2103],
                       "methods": {method: statistics(source[method]) for method,_ in METHODS}}
        acd = source["cosine_relation_matched"]
        paired[key]["paired_gains"] = {method: statistics([dec(a)-dec(b) for a,b in zip(acd,source[method])])
                                       for method,_ in METHODS if method != "cosine_relation_matched"}
    # The external ACD series is mislabeled: it equals standardized-full, not conditional ACD.
    assert figure["external"]["ACD"]["values"] == figure["ablation"]["vgg13_vgg8"]["cosine_standardized_full"]
    assert figure["external"]["ACD"]["values"] != figure["ablation"]["vgg13_vgg8"]["cosine_relation_matched"]
    extended = []
    for prefix, title, n, dkd in (("ResNet32×4→ResNet8×4，2101/2102/2103", "ResNet32×4 → ResNet8×4",3,77.40),
                                 ("WRN40-2→ShuffleNetV1，2101/2102", "WRN40-2 → ShuffleNetV1",2,None)):
        cells = ledger_cells(prefix)
        extended.append({"pair": title, "n": n, "standardized_full": float(cells[1]),
                         "acd": float(cells[2]), "dkd": dkd,
                         "source_level": "2026-11-11 ledger means only; per-seed data unavailable locally",
                         "source_reported_gain": cells[3]})
    plugin = []
    for prefix, title, pair in (("Attention Transfer + KL", "AT + KL", "ResNet32×4 → ShuffleNetV1"),
                               ("ReviewKD-style + KL", "ReviewKD-style + KL", "WRN40-2 → WRN40-1")):
        ledger_prefix = "AT + KL" if title=="AT + KL" else prefix
        cells = ledger_cells(ledger_prefix)
        linear, angular, acd = [dec(v) for v in cells[1:4]]
        plugin.append({"pipeline":title,"pair":pair,"seed":2101,"n":1,
                       "linear_kl":float(linear),"angular_kl":float(angular),"acd":float(acd),
                       "gain_vs_angular_kl":float(acd-angular),"gain_vs_linear_kl":float(acd-linear)})
    preflight_path = IMAGENET/"artifacts/real_preflight_20261114_gpu0/preflight.json"
    preflight = json.loads(preflight_path.read_text())
    inputs = [LEDGER,AUDIT,FIGURE_DATA,preflight_path,LAYOUT,SCREENSHOT]
    imagenet = [{"method":"Teacher (ResNet50)","n":None,"epoch":None,"status":"revalidated official pretrained teacher",
                 **preflight["teacher_validation"]}]
    for arm, method in (("I0","Student (CE)"),("I2","KD")):
        path = IMAGENET/f"runs/{arm}_seed3101_baseline_20261114/final.json"
        row={"method":method,"n":1,"seed":3101,"top1":None,"top5":None,"epoch":None}
        if path.exists():
            data=json.loads(path.read_text())
            assert data["status"]=="completed" and data["epoch"]==100 and data["validation"]["count"]==50000
            row.update(data["validation"],epoch=100,status="completed final-checkpoint evaluation",
                       checkpoint_sha256=data["reloaded_checkpoint_sha256"])
            inputs.append(path)
        else:
            row.update(status="no completed evaluation available at snapshot")
            log=path.parent/"job.log"
            if log.exists():
                with log.open("rb") as handle:
                    handle.seek(max(0,log.stat().st_size-8192))
                    tail=handle.read().decode(errors="replace")
                entries=[]
                for line in tail.splitlines():
                    try:
                        entry=json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "epoch" in entry: entries.append(entry)
                if entries:
                    row["last_observed_training"]={k:entries[-1][k] for k in ("epoch","batch") if k in entries[-1]}
        imagenet.append(row)
    if list((IMAGENET/"runs").glob("I7*/final.json")):
        raise ValueError("A new ACD endpoint exists; audit its protocol before updating this manuscript.")
    imagenet.append({"method":"ACD","n":None,"epoch":None,"top1":None,"top5":None,
                     "status":"no verified full-run ACD result; I7 pilot is excluded"})
    efficiency=[{"method":name,"median_ms":ms,"relative_to_angular_kl":float(dec(ms)/dec(15.09))}
                for name,ms in (("ACI + KL",15.09),("DKD",15.55),("ACI + standardized full KL",15.51),("ACD matched relation",30.04))]
    layout, mapped, benchmark_md, benchmark_tex, historical = benchmark_tables(paired, extended, reported)
    reported_md, reported_tex = reported_tables(layout, mapped, reported)
    result={"created_at":datetime.now(timezone.utc).isoformat(),"paired_cifar":paired,
            "cifar_benchmark_layout":layout,"cifar_provisional_local_cells":mapped,
            "user_reported_historical_etf": reported,
            "historical_head_study_recovery":{"path":"historical_results.json",
                "completed_records":historical["completed_record_count"], "pairs":len(historical["pairs"]),
                "snapshot_finished_at":historical["snapshot_finished_at"],
                "method":"Ordinary-KD head study; not full ACD"},
            "extended_cifar_archived_means":extended,"imagenet_snapshot":imagenet,"plugin":plugin,
            "efficiency_archived_summary":efficiency,
            "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            "source_correction":"Ignored external.ACD=[75.51,75.58] in figure_data; these are full-KL values. Conditional ACD=[75.49,75.46]. Historical input was not edited.",
            "limits":["Historical CIFAR endpoint checkpoints/registrations are not locally restored; these tables reconcile retained records, not rerun evaluations.",
                      "ACD denotes angular head plus matched conditional loss; ETF gains are not established by these rows.",
                      "Standard deviations describe two training seeds and are not confidence intervals or significance tests.",
                      "Missing cells are unavailable results, never zeros.",
                      "No training or checkpoint evaluation was performed to generate these tables."]}
    (HERE/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    recovery_note=("已找回覆盖全部 11 个配对的 52 条早期完成记录，详见 [历史结果与逐配对覆盖](historical_results.md)。"
                   "早期普通 KD／余弦头实验与后期 ACD 分开列出；主表空白不代表没有跑过该配对。")
    md=["# Main-results tables — retained experimental evidence", "", "Top-1/Top-5 are percentages; gains are percentage points (pp).", "", recovery_note, ""]
    tex=["% Generated by build_tables.py. Requires \\usepackage{booktabs}.\n% CIFAR historical results are reconciled retained records, not fresh evaluations.\n"]
    md += [benchmark_md, reported_md]
    tex += [benchmark_tex, reported_tex]
    (HERE/"cifar_main_tables.md").write_text("# CIFAR-100: six homogeneous and five heterogeneous pairs\n\n"+recovery_note+"\n\n"+benchmark_md+"\n")
    (HERE/"cifar_main_tables.tex").write_text("% Requires booktabs. Reference headers are user supplied; local cells remain provisional.\n"+benchmark_tex+"\n")
    paired_rows=[]; paired_tex=[]
    best={p:max(paired[p]["methods"][m]["mean"] for m,_ in METHODS) for p,_ in PAIRS}
    for method,label in METHODS:
        cells=[]; tcells=[]
        for p,_ in PAIRS:
            stat=paired[p]["methods"][method]
            value=stat["mean_display"]+" ± "+stat["sd_display"]
            cells.append(value)
            val=stat["mean_display"]+r"\pm"+stat["sd_display"]
            tcells.append(r"$\mathbf{"+val+"}$" if stat["mean"]==best[p] else "$"+val+"$")
        paired_rows.append([label]+cells); paired_tex.append([label]+tcells)
    md += ["## Controlled ablation: paired CIFAR-100 results", "", markdown_table(["Method"]+[title for _,title in PAIRS],paired_rows),
           "", "Mean ± sample SD over seeds 2102/2103; final epoch 240, official 10,000-image test. Only the displayed four-way comparisons are ranked.", ""]
    tex.append(tex_table("tab:cifar_paired", "Paired CIFAR-100 ablation results (Top-1, \\%).", "lcc",
       ["Method",r"VGG13 $\to$ VGG8",r"VGG13 $\to$ MobileNetV1"],paired_tex,
       "Mean $\\pm$ sample standard deviation over seeds 2102/2103, evaluated at epoch 240 on the official test set. Bold marks the highest mean within each displayed comparison; it does not indicate statistical significance. These are reconciled historical records, not a complete SOTA comparison."))
    erows=[[r["pair"],str(r["n"]),"—" if r["dkd"] is None else fmt(r["dkd"]),fmt(r["standardized_full"]),fmt(r["acd"])] for r in extended]
    md += ["## 2. Additional CIFAR-100 archived means", "",markdown_table(["Pair","Runs","DKD","ACI + std. full","ACD"],erows),"",
           "Only means are retained locally; no SD is reconstructed. The ledger reports a ResNet relation-minus-full gain of +0.39 pp, whereas subtraction of its rounded means gives +0.38 pp; use the endpoint means without asserting a more precise gain.", ""]
    tex.append(tex_table("tab:cifar_extended", "Additional CIFAR-100 means retained in the experiment ledger (Top-1, \\%).", "lrrrr",
       ["Teacher $\\to$ student","Runs","DKD","ACI + std. full","ACD"],
       [[r[0].replace("×",r"$\times$").replace("→",r"$\to$"),*r[1:]] for r in erows],
       "Means are transcribed from the 2026-11-11 ledger. Per-seed values are not available locally, so no standard deviations are inferred. A dash denotes an unavailable result; the WRN entry covers only the two archived completed seeds."))
    irows=[]; itex=[]
    for r in imagenet:
        one="—" if r["top1"] is None else fmt(r["top1"],3)
        five="—" if r["top5"] is None else fmt(r["top5"],3)
        status="Available" if r["top1"] is not None else "Pending"
        irows.append([r["method"],one,five,r["status"]]); itex.append([r["method"],one,five,status])
    md += ["## 3. ImageNet-1K evidence snapshot", "",markdown_table(["Model/method","Top-1","Top-5","Status"],irows),"",
           "ResNet50 → MobileNetV2; full 50,000-image validation set. Teacher weights are official pretrained weights. Student results are single-seed final-checkpoint evaluations; pilot/probe scores are excluded.",""]
    tex.append(tex_table("tab:imagenet", "ImageNet-1K baseline evidence snapshot (accuracy, \\%).", "lrrl",
       ["Model / method","Top-1","Top-5","Result status"],itex,
       "ResNet50 teacher and MobileNetV2 student, evaluated on all 50,000 validation images. Student baselines use seed 3101 and 100 epochs. Pending denotes a missing final evaluation, not zero accuracy. No verified full-run ACD result is available in this snapshot; this table does not establish ACD scalability."))
    prows=[]; ptex=[]
    for r in plugin:
        vals=[fmt(r[k]) for k in ("linear_kl","angular_kl","acd")]
        gains=[("+" if r[k]>=0 else "")+fmt(r[k]) for k in ("gain_vs_angular_kl","gain_vs_linear_kl")]
        prows.append([r["pipeline"],r["pair"],*vals,*gains])
        ptex.append([r["pipeline"],r["pair"].replace("×",r"$\times$").replace("→",r"$\to$"),*vals,*gains])
    md += ["## 4. Replacement in feature-distillation pipelines", "",markdown_table(
       ["Pipeline","Pair","Affine + KL","ACI + KL","ACI + SICD","Δ vs ACI + KL","Δ vs affine + KL"],prows),"",
       "Seed 2101 only. The first gain isolates output-loss replacement on the angular head; the second compares the joint head/loss change with the affine baseline. ReviewKD-style is an adapted pipeline, not a claim of original-author recipe reproduction.",""]
    tex.append(tex_table("tab:dropin", "Output-loss replacement in fixed feature-distillation pipelines on CIFAR-100 (Top-1, \\%).", "llrrrrr",
       ["Pipeline","Teacher $\\to$ student","Affine + KL","ACI + KL","ACI + SICD",r"$\Delta_{\rm loss}$",r"$\Delta_{\rm joint}$"],ptex,
       "Single-seed pilots (seed 2101). Feature objectives and adapters are fixed within each pipeline. $\\Delta_{\\rm loss}$ compares ACI+SICD with ACI+KL; $\\Delta_{\\rm joint}$ compares ACI+SICD with affine+KL. Gains are in percentage points. The loss-only replacement preserves the angular student's inference model. These results do not establish universal improvements."))
    frows=[[r["method"],fmt(r["median_ms"]),fmt(r["relative_to_angular_kl"])+"×"] for r in efficiency]
    md += ["## 5. Archived training-cost diagnostic", "",markdown_table(["Objective","Median ms/update","Relative to ACI + KL"],frows),"",
           "Transcribed from the historical same-checkpoint/GPU/batch/loader benchmark. Full benchmark metadata is not locally restored; these are not ImageNet throughput measurements or a general efficiency ranking.",""]
    tex.append(tex_table("tab:training_cost", "Archived training-update cost under a shared benchmark setup.", "lrr",
       ["Objective","Median ms/update","Relative cost"],[[r[0],r[1],r[2].replace("×",r"$\times$")] for r in frows],
       "Historical controlled benchmark, not ImageNet training throughput. The original checkpoint/GPU/batch/loader were held fixed; full benchmark metadata remains to be restored before using this as a standalone efficiency table. Gradient-norm matching adds training computation."))
    (HERE/"tables.md").write_text("\n".join(md)+"\n")
    (HERE/"tables.tex").write_text("\n".join(tex)+"\n")
    print(json.dumps({"status":"built","tables":8,"cifar_pairs":11,"mapped_acd_pairs":len(mapped),
                      "user_reported_pairs":len(reported["rows"])},ensure_ascii=False))


if __name__ == "__main__":
    main()
