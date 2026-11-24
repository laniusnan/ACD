"""Render the measured CIFAR-100 calibration and four-curve same-state probes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACTS = HERE / "artifacts"
SOURCE = ARTIFACTS / "cifar100_same_state_seed2051"
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / "mplconfig"))
sys.path.insert(0, str(ARTIFACTS / "plot_deps"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plot
import numpy as np

plot.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ecdf(axis, values, label, color, style):
    axis.step(np.sort(values), np.arange(1, len(values) + 1) / len(values), where="post",
              label=label, color=color, linestyle=style, linewidth=1.6)
    axis.set_xscale("symlog", linthresh=.01)
    axis.set_xlim(left=0)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel(r"Per-image $8\,\mathrm{KL}$ to teacher ($T=4$; symlog)")
    axis.set_ylabel("Empirical cumulative fraction")


def save(figure, stem, report, design, footer=None):
    if footer:
        figure.text(.5, .012, footer, ha="center", va="bottom", fontsize=7, color="#52616B")
    figure.tight_layout(rect=(0, .13, 1, .95) if footer else (0, 0, 1, 1))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    out_of_view_labels = set()
    for axis in figure.axes:
        for coordinate, limits in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
            for tick in coordinate.get_major_ticks() + coordinate.get_minor_ticks():
                if not min(limits) <= tick.get_loc() <= max(limits):
                    out_of_view_labels.update((id(tick.label1), id(tick.label2)))
    clipped = []
    for text in figure.findobj(matplotlib.text.Text):
        if id(text) in out_of_view_labels or not text.get_visible() or not text.get_text().strip():
            continue
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > figure.bbox.width + 1 or bounds.y1 > figure.bbox.height + 1:
            clipped.append(text.get_text())
    if clipped:
        raise ValueError(f"Figure text exceeds canvas: {clipped}")
    outputs = {}
    for extension in ("pdf", "svg", "png"):
        destination = HERE / f"{stem}.{extension}"
        figure.savefig(destination, bbox_inches="tight", facecolor="white", dpi=300)
        outputs[destination.name] = digest(destination)
    plot.close(figure)
    record = {"seed": 2051, "design": design, "probe": report,
        "source": str((SOURCE / "report.json").relative_to(ROOT)),
        "source_sha256": digest(SOURCE / "report.json"),
        "script_sha256": digest(__file__), "output_sha256": outputs,
        "plot_checks": {"text_inside_canvas": True, "all_samples_plotted": True},
        "new_training_performed": False, "new_evaluation_performed": False}
    (HERE / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote {stem}.pdf/.svg/.png/.json")


def render_seed(seed, calib_only=False):
    if seed != 2051:
        raise ValueError("Only the explicitly selected seed 2051 was evaluated")
    report = json.loads((SOURCE / "report.json").read_text())
    if digest(SOURCE / "oracle_arrays.npz") != report["oracle_arrays_sha256"]:
        raise ValueError("Oracle array checksum mismatch")
    with np.load(SOURCE / "oracle_arrays.npz") as stored:
        arrays = {key: stored[key] for key in stored.files}
    if any(len(values) != 10000 or not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Expected finite per-image arrays for all 10,000 test images")
    affine = report["cases"]["affine"]
    figure, axis = plot.subplots(figsize=(4.5, 3.5))
    ecdf(axis, arrays["affine_before"], "Before oracle calibration", "#2878B5", "--")
    ecdf(axis, arrays["affine_after"], "After oracle calibration", "#E69F00", "-")
    axis.set_xlabel("Per-image 8 KL to teacher (T=4)")
    axis.legend(frameon=False, loc="upper left", fontsize=8)
    axis.text(.98, .08,
              f"Mean: {affine['before']['mean']:.3f} → {affine['after']['mean']:.3f}\n"
              f"Residual: {100 * affine['after']['mean'] / affine['before']['mean']:.1f}% of original mean",
              transform=axis.transAxes, ha="right", multialignment="right", va="bottom", fontsize=8,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 3})
    save(figure, "fig_oracle_calib_seed2051", report,
         "Two ECDF curves: one linear checkpoint before/after per-image oracle scaling")
    if calib_only:
        return

    figure, axes = plot.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.12, 1]})
    for name, label, color in (("affine", "Affine", "#2878B5"),
                                ("angular", "Angular substitution", "#009E73")):
        for field, style in (("before", "--"), ("after", "-")):
            ecdf(axes[0], arrays[f"{name}_{field}"], f"{label}, {field}", color, style)
    axes[0].set_title("(a) Same state, four measured distributions")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7)
    axes[1].hist(arrays["delta"], bins=50, color="#009E73", alpha=.85)
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_xlabel(r"Post-oracle $8\,\mathrm{KL}$: angular − affine")
    axes[1].set_ylabel("Test image count")
    axes[1].set_title("(b) Paired residual difference")
    paired = report["paired_delta"]
    axes[1].text(.97, .97,
                 f"Mean Δ = {paired['mean']:+.4f}\n"
                 f"Angular better: {100 * paired['angular_better_fraction']:.2f}%\n"
                 "Δ < 0: angular better",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=8)
    save(figure, "fig_oracle_same_state_cifar100_seed2051", report,
         "Four ECDFs and paired post-oracle differences from the same frozen linear checkpoint",
         "CIFAR-100 · 10,000 test images · seed 2051 · same backbone, classifier directions and teacher\n"
         "Affine vs. post-hoc 30 cos(θ); independent per-image scalar α ≥ 0; no retraining")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(2051,), default=2051)
    parser.add_argument("--calib-only", action="store_true", help="Render only the standalone calibration figure")
    args = parser.parse_args()
    render_seed(args.seed, calib_only=args.calib_only)
