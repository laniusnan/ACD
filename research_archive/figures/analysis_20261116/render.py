"""Render measured diagnostics to PDF, SVG, PNG and a contact sheet."""

import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACTS = HERE / "artifacts"
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / "mplconfig"))
sys.path.insert(0, str(ARTIFACTS / "plot_deps"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plot
import numpy as np
from PIL import Image, ImageDraw
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

COLORS = ["#2878B5", "#E69F00", "#009E73", "#CC79A7"]
OBJECTIVES = ["raw_kl", "standardized_full", "conditional", "matched_conditional"]
OBJECTIVE_LABELS = ["Raw KL", "Std. full", "Conditional", "Matched\nconditional"]
plot.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                      "axes.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                      "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                      "axes.grid": True, "grid.alpha": 0.18, "savefig.dpi": 180})
SUMMARIES = {}
OUTPUTS = []


def read(name):
    return json.loads((ARTIFACTS / name).read_text())


def save(figure, stem, footer, bottom=0.065):
    figure.text(0.5, 0.014, footer, ha="center", va="bottom", fontsize=8, color="#52616B")
    figure.tight_layout(rect=(0, bottom, 1, 0.94))
    for extension in ("pdf", "svg", "png"):
        figure.savefig(HERE / f"{stem}.{extension}", bbox_inches="tight", facecolor="white")
    plot.close(figure)
    OUTPUTS.append(stem)


def summary(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75))}


def historical_oracle():
    source = ROOT / "reproductions/head_story_visuals_20261027/artifacts/relations.json"
    report = json.loads(source.read_text())
    figure, axes = plot.subplots(1, 2, figsize=(9.5, 3.7), sharey=True)
    names = ["linear", "cosine", "sample_amplitude"]
    labels = ["Linear KD", "Cosine KD", "Amplitude\nrestored"]
    results = {}
    for axis, seed in zip(axes, (2051, 2052)):
        for position, (name, label, color) in enumerate(zip(names, labels, COLORS)):
            row = report["models"][f"{name}_{seed}"]["summary"]["all"]
            before, after = row["kd"], row["relation_residual"]
            axis.plot([position-.13, position+.13], [before, after], color=color, linewidth=2)
            axis.scatter(position-.13, before, facecolor="white", edgecolor=color, s=65, linewidth=1.5, zorder=3)
            axis.scatter(position+.13, after, color=color, s=60, zorder=3)
            axis.annotate(f"{after:.3f}", (position+.13, after), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9)
            results[f"{name}_{seed}"] = {"before": before, "after": after}
        axis.set_xticks(range(3), labels)
        axis.set_title(f"CIFAR-100 endpoints · seed {seed}")
        axis.set_ylim(0, 1.4)
    axes[0].set_ylabel(r"Recorded weighted KD loss ($8\,\mathrm{KL}$, $T=4$)")
    axes[1].scatter([], [], facecolor="white", edgecolor="#52616B", label="Before calibration")
    axes[1].scatter([], [], color="#52616B", label="After calibration")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, fontsize=9,
                  loc="lower center", bbox_to_anchor=(.5,.05), ncol=2)
    figure.suptitle("Oracle scalar calibration leaves endpoint residuals", fontsize=13)
    save(figure, "fig_oracle_calib", "Historical aggregate records · 10,000 test images per seed · scalar domain α ≥ 0 · no image-level error bars", bottom=.17)
    SUMMARIES["historical_oracle"] = results


def same_state_oracle():
    report = read("oracle_same_state.json")
    figure, axes = plot.subplots(1, 2, figsize=(9.5, 3.8))
    for label, key, color in (("Affine", "raw", COLORS[0]), ("Angular substitution", "angular", COLORS[2])):
        case = report["cases"][key]
        for field, style in (("before", "--"), ("after", "-")):
            values = np.sort(case[field])
            axes[0].plot(values, np.arange(1, len(values)+1)/len(values), color=color, linestyle=style,
                         label=f"{label}, {field}")
    delta = np.array(report["cases"]["angular"]["after"])-np.array(report["cases"]["raw"]["after"])
    axes[0].set_xlabel(r"$T^2\mathrm{KL}$ to fixed teacher ($T=4$)")
    axes[0].set_ylabel("Empirical cumulative fraction")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("(a) Per-image oracle calibration")
    axes[1].hist(delta, bins=24, color=COLORS[2], alpha=.85)
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_xlabel("Post-oracle KL: angular − affine")
    axes[1].set_ylabel("Images")
    axes[1].set_title(f"(b) Paired residual difference · mean {delta.mean():+.3f}")
    figure.suptitle("Same-state calibration diagnostic on ImageNet-1K", fontsize=13)
    save(figure, "fig_oracle_same_state", "CE-trained MobileNetV2, epoch 100 · 256 training images · fixed teacher/features/directions · angular head is post-hoc")
    SUMMARIES["same_state_oracle"] = {key: {field:summary(case[field]) for field in ("before", "after", "alpha")} | {"boundary_zero_count":case["boundary_zero_count"]} for key,case in report["cases"].items()}
    SUMMARIES["same_state_oracle"]["paired_delta"] = summary(delta)


def tsne_figures():
    arrays = np.load(ARTIFACTS / "features.npz")
    manifest = read("samples.json")
    indices = manifest["embedding_indices"]
    labels = arrays["labels"][indices]
    classes = np.unique(labels)
    mapped = np.searchsorted(classes, labels)
    palette = plot.get_cmap("tab20")(np.linspace(0, 1, len(classes)))
    start = np.random.default_rng(7).normal(0, 1e-4, (len(labels), 2)).astype(np.float32)
    figure, axes = plot.subplots(2, 2, figsize=(9.0, 7.0))
    embeddings = {}
    metrics = {}
    for row, (case, title) in enumerate((("kd_pilot", "Linear + ordinary KD"), ("acd_pilot", "Angular + SICD reconstruction"))):
        features = arrays[case+"_features"][indices]
        unit = features/np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        centroids = np.stack([unit[labels==category].mean(0) for category in classes])
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        scatter = 1-(unit*centroids[mapped]).sum(1)
        similarities = centroids@centroids.T
        nearest = np.where(np.eye(len(classes), dtype=bool), -np.inf, similarities).max(1)
        metrics[case] = {"within_class_angular_dispersion": summary(scatter),
                         "nearest_other_centroid_cosine": summary(nearest),
                         "angular_silhouette": float(silhouette_score(unit, labels, metric="cosine")),
                         "feature_norm": summary(np.linalg.norm(features, axis=1))}
        for column, (values, kind) in enumerate(((features, "Raw pre-head features"), (unit, "L2-normalized features"))):
            embedding = TSNE(n_components=2, perplexity=30, init=start.copy(), learning_rate="auto",
                             max_iter=1000, random_state=7, n_jobs=1).fit_transform(values)
            embeddings[case+"_"+str(column)] = embedding
            axis = axes[row,column]
            axis.scatter(embedding[:,0], embedding[:,1], c=palette[mapped], s=15, alpha=.85, linewidths=0)
            axis.set_title(f"{title}\n{kind}", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            metrics[case][kind+"_trustworthiness"] = float(trustworthiness(values, embedding, n_neighbors=5))
    handles = [plot.Line2D([], [], color=palette[index], marker="o", linestyle="", markersize=4, label=str(category)) for index,category in enumerate(classes)]
    figure.legend(handles=handles, title="ImageNet class index", loc="lower center", bbox_to_anchor=(.5,.045), ncol=12, frameon=False, fontsize=7, title_fontsize=8)
    figure.suptitle("t-SNE of real short-run checkpoints · 210 updates only", fontsize=13)
    save(figure, "fig_tsne_pilot", "12 preselected classes × 20 training images · independent embeddings, matched settings · head and objective both differ", bottom=.13)
    np.savez_compressed(ARTIFACTS / "tsne_embeddings.npz", labels=labels, **embeddings)
    (ARTIFACTS / "representation_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    SUMMARIES["pilot_representation"] = metrics
    figure, axes = plot.subplots(1, 3, figsize=(10, 3.7))
    for position, case in enumerate(("kd_pilot", "acd_pilot")):
        record = metrics[case]
        for axis, name in zip(axes[:2], ("within_class_angular_dispersion", "nearest_other_centroid_cosine")):
            values = record[name]
            axis.errorbar(position, values["median"], yerr=[[values["median"]-values["q25"]],[values["q75"]-values["median"]]],
                          fmt="o", color=COLORS[position*2], capsize=5, markersize=7)
        axes[2].scatter(position, record["angular_silhouette"], color=COLORS[position*2], s=60)
    for axis in axes:
        axis.set_xlim(-.5,1.5)
        axis.set_xticks((0,1), ("Linear+KD", "Angular+SICD\nreconstruction"), fontsize=8)
    axes[0].set_title("(a) Within-class angular dispersion")
    axes[0].set_ylabel("1 − cosine to class centroid")
    axes[1].set_title("(b) Nearest class centroid")
    axes[1].set_ylabel("Cosine similarity")
    axes[2].set_title("(c) Original-space silhouette")
    axes[2].set_ylabel("Cosine-distance silhouette")
    figure.suptitle("Quantitative representation checks alongside t-SNE", fontsize=13)
    save(figure, "fig_representation_pilot", "Same 240 training images · 210-update pilots · medians/IQR describe images or classes, not seed uncertainty")
    figure, axes = plot.subplots(2, 3, figsize=(10.5, 6.5))
    for row, case in enumerate(("kd_pilot", "acd_pilot")):
        features = arrays[case+"_features"][indices]
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        for column, (perplexity, seed) in enumerate(((15,7),(30,17),(50,7))):
            initial = np.random.default_rng(seed).normal(0, 1e-4, (len(labels),2)).astype(np.float32)
            embedding = TSNE(perplexity=perplexity, init=initial, random_state=seed,
                             learning_rate="auto", max_iter=1000, n_jobs=1).fit_transform(features)
            axes[row,column].scatter(embedding[:,0], embedding[:,1], c=palette[mapped], s=12, alpha=.8, linewidths=0)
            label = "Linear+KD" if case=="kd_pilot" else "Angular+SICD (recon.)"
            axes[row,column].set_title(f"{label} · perplexity {perplexity}, seed {seed}", fontsize=9)
            axes[row,column].set_xticks([])
            axes[row,column].set_yticks([])
            embeddings[f"sensitivity_{case}_{perplexity}_{seed}"] = embedding
    figure.suptitle("t-SNE sensitivity · same L2-normalized pilot features", fontsize=13)
    save(figure, "fig_tsne_sensitivity", "All settings reported · separate t-SNE embeddings have arbitrary orientation and scale · qualitative short-run evidence")
    np.savez_compressed(ARTIFACTS / "tsne_embeddings.npz", labels=labels, **embeddings)


def gradient_plot(states, titles, stem):
    report = read("backbone_gradients.json")
    figure, axes = plot.subplots(2, 3, figsize=(11, 6.7))
    stats = {}
    for row, (state, title) in enumerate(zip(states,titles)):
        stats[state] = {}
        for position, (objective,color) in enumerate(zip(OBJECTIVES,COLORS)):
            values = [entry for entry in report["rows"] if entry["state"]==state and entry["objective"]==objective and not entry["excluded_zero_gradient"]]
            cosine = np.array([entry["cosine"] for entry in values])
            ratio = np.array([entry["norm_ratio"] for entry in values])
            jitter = np.linspace(-.13,.13,len(values))
            axes[row,0].scatter(position+jitter,cosine,color=color,s=23,zorder=3)
            axes[row,0].plot([position-.18,position+.18],[cosine.mean()]*2,color=color,linewidth=2)
            negative = int((cosine<0).sum())
            axes[row,1].bar(position,negative/len(values),color=color,width=.6)
            axes[row,1].text(position,negative/len(values)+.03,f"{negative}/{len(values)}",ha="center",fontsize=9)
            axes[row,2].scatter(position+jitter,ratio,color=color,s=23,zorder=3)
            stats[state][objective] = {"cosine":summary(cosine),"negative_batches":negative,"valid_batches":len(values),"norm_ratio":summary(ratio)}
        axes[row,0].axhline(0,color="#333333",linewidth=.8)
        axes[row,0].set_ylim(-1,1)
        axes[row,1].set_ylim(0,1.12)
        axes[row,2].set_yscale("log")
        axes[row,2].axhline(1,color="#333333",linewidth=.8)
        for axis in axes[row]:
            axis.set_xticks(range(4),OBJECTIVE_LABELS,fontsize=8)
        axes[row,0].set_ylabel(f"{title}\nCosine with CE",fontsize=9)
    axes[0,0].set_title("(a) Same-batch gradient alignment")
    axes[0,1].set_title("(b) Negative-cosine fraction")
    axes[0,2].set_title("(c) KD/CE gradient-norm ratio")
    figure.suptitle("Cross-objective backbone gradients at fixed student states",fontsize=13)
    save(figure,stem,"8 fixed batches × 8 training images · eval mode · backbone parameters excluding head · positive matching preserves direction")
    SUMMARIES[stem] = stats


def parameterization():
    report = read("parameterization.json")
    figure, axes = plot.subplots(3,3,figsize=(10.5,8.0),sharex="col")
    metrics = ("conditional_tv","top1_alternative_changed","deviation")
    for column,(factor,title) in enumerate((("common_scale","Common logit scale"),("class_norm","Class-norm heterogeneity"),("class_bias","Centered class bias"))):
        rows = [entry for entry in report["rows"] if entry["factor"]==factor]
        strengths = [entry["strength"] for entry in rows]
        for row,metric in enumerate(metrics):
            axis=axes[row,column]
            mean=[np.mean(entry[metric]) for entry in rows]
            axis.plot(strengths,mean,marker="o",color=COLORS[column],markersize=4,linewidth=1.8)
            if metric!="top1_alternative_changed":
                axis.fill_between(strengths,[np.quantile(entry[metric],.25) for entry in rows],
                                  [np.quantile(entry[metric],.75) for entry in rows],color=COLORS[column],alpha=.13)
            axis.axvline(1,color="#666666",linestyle=":",linewidth=.8)
            axis.set_ylim(bottom=0)
        axes[0,column].set_title(title)
        axes[2,column].set_xlabel("Multiplier" if column==0 else "Strength ρ (original = 1)")
    axes[0,0].set_ylabel("Conditional TV to original")
    axes[1,0].set_ylabel("Top alternative change fraction")
    axes[2,0].set_ylabel("1 − cosine to original KD gradient")
    figure.suptitle("Separate score-parameterization interventions",fontsize=13)
    save(figure,"fig_parameterization","CE epoch100 · 256 ImageNet training images · fixed features/directions/teacher · lines: means; bands: image IQR")
    SUMMARIES["parameterization"] = [{"factor":row["factor"],"strength":row["strength"],**{metric:summary(row[metric]) for metric in metrics}} for row in report["rows"]]


def gradient_sources():
    report=read("gradient_sources.json")
    matrices={"deviation":np.zeros((2,2)),"norm_ratio":np.zeros((2,2))}
    for row in report["rows"]:
        position=(int(row["jacobian"]=="angular"),int(row["probability_error"]=="angular"))
        for metric in matrices:
            matrices[metric][position]=np.mean(row[metric])
    figure,axes=plot.subplots(1,2,figsize=(7.6,3.1),sharey=True)
    for axis,(metric,matrix) in zip(axes,matrices.items()):
        image=axis.imshow(matrix,cmap="Blues",vmin=0,vmax=max(float(matrix.max()),1e-8))
        for row in range(2):
            for column in range(2):
                displayed = 0.0 if abs(matrix[row,column]) < 5e-4 else matrix[row,column]
                axis.text(column,row,f"{displayed:.3f}",ha="center",va="center",fontsize=15,
                          color="white" if matrix[row,column]>.55*matrix.max() else "#172B3A")
        axis.set_xticks((0,1),("Affine error","Angular error"))
        axis.set_yticks((0,1),("Affine Jacobian","Angular Jacobian"))
        if metric=="norm_ratio":
            axis.tick_params(axis="y",left=False,labelleft=False)
        axis.grid(False)
        axis.set_title("Mean direction deviation" if metric=="deviation" else "Mean norm ratio to affine")
        figure.colorbar(image,ax=axis,fraction=.05,pad=.04)
    figure.tight_layout(pad=.25,w_pad=1.0)
    for extension in ("pdf","svg","png"):
        figure.savefig(HERE/f"fig_gradient_sources.{extension}",bbox_inches="tight",
                       pad_inches=.025,facecolor="white")
    plot.close(figure)
    OUTPUTS.append("fig_gradient_sources")
    SUMMARIES["gradient_sources"]={key:value.tolist() for key,value in matrices.items()}


def conditional_dependency():
    report=read("conditional_dependency.json")
    figure,axes=plot.subplots(2,3,figsize=(10.5,6.5),sharex=True)
    stats=[]
    for row,side in enumerate(("student","teacher")):
        entries=[entry for entry in report["rows"] if entry["side"]==side]
        strengths=[entry["shift_std_units"] for entry in entries]
        for metric,color,style,label in (("raw_conditional_tv",COLORS[0],"--","Without standardization"),
                                         ("standardized_conditional_tv",COLORS[2],"-","All-class standardization")):
            axes[row,0].plot(strengths,[np.mean(entry[metric]) for entry in entries],color=color,linestyle=style,marker="o",markersize=3,label=label)
        for axis,metric,color in ((axes[row,1],"teacher_non_target_mass",COLORS[1]),(axes[row,2],"weighted_relation",COLORS[3])):
            axis.plot(strengths,[np.mean(entry[metric]) for entry in entries],color=color,marker="o",markersize=3)
        axes[row,0].set_ylabel(f"Change {side} target logit\nConditional TV to original",fontsize=9)
        for axis in axes[row]:
            axis.axvline(0,color="#777777",linestyle=":",linewidth=.8)
        stats.extend({"side":side,"shift":entry["shift_std_units"],**{key:float(np.mean(entry[key])) for key in ("raw_conditional_tv","standardized_conditional_tv","teacher_non_target_mass","weighted_relation")}} for entry in entries)
    axes[0,0].set_title("(a) Non-target distribution")
    axes[0,0].legend(frameon=False,fontsize=7)
    axes[0,1].set_title(r"(b) Teacher weight $1-p_y$")
    axes[0,2].set_title("(c) Weighted conditional loss")
    for axis in axes[1]:
        axis.set_xlabel("Target-logit shift / original logit std")
    figure.suptitle("Conditional supervision retains target-dependent effects",fontsize=13)
    save(figure,"fig_conditional_dependency","Only the indicated target logit changes · all non-target logits fixed · CE epoch100 · before backbone norm matching")
    SUMMARIES["conditional_dependency"]={"rows":stats,"numerical_checks":report["numerical_checks"]}


def overview():
    width=1440
    cell_width=width//2
    thumbnails=[]
    for stem in OUTPUTS:
        with Image.open(HERE/f"{stem}.png") as source:
            thumbnail=source.copy()
        thumbnail.thumbnail((cell_width-24,470))
        thumbnails.append(thumbnail)
    row_heights=[max(image.height for image in thumbnails[index:index+2])+48
                 for index in range(0,len(thumbnails),2)]
    height=sum(row_heights)
    canvas=Image.new("RGB",(width,height),"#edf1f4")
    draw=ImageDraw.Draw(canvas)
    for index,(stem,thumbnail) in enumerate(zip(OUTPUTS,thumbnails)):
        column,row=index%2,index//2
        row_top=sum(row_heights[:row])
        left=column*cell_width+(cell_width-thumbnail.width)//2
        canvas.paste(thumbnail,(left,row_top+30))
        draw.text((column*cell_width+14,row_top+8),stem,fill="#172B3A")
    canvas.save(HERE/"overview.png")


def evidence_manifest():
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    inputs=["status.json","samples.json","sources.json","features.npz",
            "oracle_same_state.json","parameterization.json","gradient_sources.json",
            "conditional_dependency.json","backbone_gradients.json"]
    sources=read("sources.json")
    sources={key:value for key,value in sources.items() if key!="pilot_config"}
    historical=ROOT/"reproductions/head_story_visuals_20261027/artifacts/relations.json"
    outputs=[HERE/f"{stem}.{extension}" for stem in OUTPUTS for extension in ("pdf","svg","png")]
    outputs += [HERE/"overview.png",HERE/"results_summary.json"]
    record={"probe_run":read("status.json"),"sources":sources,
            "historical_summary":{"path":str(historical.relative_to(ROOT)),"sha256":digest(historical),
                                  "replotted_saved_aggregates_only":True},
            "artifact_sha256":{name:digest(ARTIFACTS/name) for name in inputs},
            "script_sha256":{name:digest(HERE/name) for name in ("probe.py","render.py")},
            "output_sha256":{path.name:digest(path) for path in outputs},
            "plot_requirements_sha256":digest(HERE/"plot_requirements.lock.txt"),
            "numerical_checks":SUMMARIES["conditional_dependency"]["numerical_checks"],
            "no_training_performed":True}
    (HERE/"evidence_manifest.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")


def main():
    if read("status.json")["status"]!="completed":
        raise ValueError("CPU probes must finish before plotting")
    with threadpool_limits(limits=4):
        historical_oracle()
        same_state_oracle()
        tsne_figures()
        gradient_plot(("ce_final","ce_final_angular"),("CE endpoint / affine","Same endpoint / angular"),"fig_gradient_conflict")
        gradient_plot(("kd_pilot","acd_pilot"),("KD pilot / 210 updates","SICD pilot / 210 updates"),"fig_gradient_conflict_pilot")
        parameterization()
        gradient_sources()
        conditional_dependency()
    (ARTIFACTS/"figure_summary.json").write_text(json.dumps(SUMMARIES,indent=2,allow_nan=False)+"\n")
    (HERE/"results_summary.json").write_text(json.dumps(SUMMARIES,indent=2,allow_nan=False)+"\n")
    packages=sorted(f"{entry.metadata['Name']}=={entry.version}" for entry in importlib.metadata.distributions(path=[str(ARTIFACTS/"plot_deps")]))
    (HERE/"plot_requirements.lock.txt").write_text("\n".join(packages)+"\n")
    (ARTIFACTS/"render_metadata.json").write_text(json.dumps({"figures":OUTPUTS,"tsne":{"perplexity":30,"seed":7,"iterations":1000,"init":"identical fixed random arrays across matched panels","normalization":"per-panel explicitly labeled"}},indent=2)+"\n")
    overview()
    evidence_manifest()
    print(json.dumps({"status":"figures_complete","count":len(OUTPUTS),"output":str(HERE)}),flush=True)


if __name__=="__main__":
    main()
