"""Plot the frozen linear-KD mismatch from verified, previously measured arrays."""

import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "artifacts/cifar100_same_state_seed2051"
STEM = "fig_output_mismatch_cifar100_seed2051"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "artifacts/mplconfig"))
sys.path.insert(0, str(HERE / "artifacts/plot_deps"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plot
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log_softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def load_inputs():
    report = json.loads((SOURCE / "report.json").read_text())
    inference_metadata = json.loads((SOURCE / "inference.json").read_text())
    if report["identity"] != inference_metadata["identity"]:
        raise ValueError("Inference and oracle identities differ")
    for name, expected in (("inference.npz", inference_metadata["arrays_sha256"]),
                           ("oracle_arrays.npz", report["oracle_arrays_sha256"])):
        if digest(SOURCE / name) != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
    with np.load(SOURCE / "inference.npz") as inference:
        teacher = inference["teacher_logits"].astype(np.float64)
        student = inference["affine_logits"].astype(np.float64)
    with np.load(SOURCE / "oracle_arrays.npz") as oracle:
        before, after, alpha = (oracle[name] for name in ("affine_before", "affine_after", "affine_alpha"))
    if teacher.shape != (10000, 100) or student.shape != teacher.shape:
        raise ValueError("Expected all 10,000 test images and all 100 classes")
    if any(values.shape != (10000,) for values in (before, after, alpha)):
        raise ValueError("Expected one scalar and two losses per test image")
    if not all(np.isfinite(values).all() for values in (teacher, student, before, after, alpha)):
        raise ValueError("Nonfinite source data")
    if (alpha < 0).any() or (after > before + 1e-8).any():
        raise ValueError("Invalid scalar oracle")
    teacher_centered = (teacher - teacher.mean(axis=1, keepdims=True)) / report["temperature"]
    student_centered = (student - student.mean(axis=1, keepdims=True)) / report["temperature"]
    calibrated_centered = alpha[:, None] * student_centered
    target_log = log_softmax(teacher_centered)
    reconstruction_errors = {}
    for name, scores, expected in (("before", student_centered, before),
                                    ("after", calibrated_centered, after)):
        reconstructed = report["loss_factor"] * (
            np.exp(target_log) * (target_log - log_softmax(scores))).sum(axis=1)
        error = float(np.max(np.abs(reconstructed - expected)))
        if error > 1e-10:
            raise ValueError(f"Centering changed the recorded {name} KL: {error}")
        reconstruction_errors[name] = error
    return report, before, after, teacher_centered, calibrated_centered, reconstruction_errors


def check_text_bounds(figure):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    out_of_view_labels = set()
    for axis in figure.axes:
        for coordinate, limits in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
            for tick in coordinate.get_major_ticks() + coordinate.get_minor_ticks():
                if not min(limits) <= tick.get_loc() <= max(limits):
                    out_of_view_labels.update((id(tick.label1), id(tick.label2)))
    for text in figure.findobj(matplotlib.text.Text):
        if id(text) in out_of_view_labels or not text.get_visible() or not text.get_text().strip():
            continue
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > figure.bbox.width + 1 or bounds.y1 > figure.bbox.height + 1:
            raise ValueError(f"Text exceeds the canvas: {text.get_text()}")


def render():
    report, before, after, teacher_centered, calibrated_centered, errors = load_inputs()
    plot.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .16})
    figure, axes = plot.subplots(1, 3, figsize=(8.1, 3.8), layout="constrained",
                                gridspec_kw={"width_ratios": [1, 1, .045]})
    distribution, agreement, color_axis = axes
    cumulative = np.arange(1, len(before) + 1) / len(before)
    for values, label, color, style in ((before, "Before calibration", "#2878B5", "--"),
                                        (after, "After oracle calibration", "#E69F00", "-")):
        distribution.step(np.sort(values), cumulative, where="post", color=color,
                          linestyle=style, linewidth=1.7, label=label)
    distribution.set_xscale("symlog", linthresh=.01)
    distribution.set_xlim(left=0)
    distribution.set_ylim(0, 1.01)
    distribution.set_box_aspect(1)
    distribution.set_title("(a) Scalar calibration leaves a residual", pad=10)
    distribution.set_xlabel(r"Per-image $8\,\mathrm{KL}$ ($T=4$; symlog)")
    distribution.set_ylabel("Empirical cumulative fraction")
    distribution.legend(frameon=False, loc="upper left", fontsize=8)
    remaining = float(100 * after.mean() / before.mean())
    distribution.text(.97, .08,
        f"Mean: {before.mean():.4f} → {after.mean():.4f}\n"
        f"Residual: {remaining:.1f}% of original mean\n"
        f"Post-oracle median: {np.median(after):.4f}",
        transform=distribution.transAxes, ha="right", va="bottom", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 3})
    lower = math.floor(min(teacher_centered.min(), calibrated_centered.min()))
    upper = math.ceil(max(teacher_centered.max(), calibrated_centered.max()))
    density = agreement.hexbin(teacher_centered.ravel(), calibrated_centered.ravel(),
        gridsize=90, extent=(lower, upper, lower, upper), mincnt=1,
        bins="log", cmap="viridis", linewidths=0, zorder=2)
    pairs_binned = int(density.get_array().sum())
    if pairs_binned != teacher_centered.size:
        raise ValueError("Density plot dropped image-class pairs")
    agreement.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.1,
                   color="#C44E52", label="Exact agreement", zorder=3)
    agreement.set_xlim(lower, upper)
    agreement.set_ylim(lower, upper)
    agreement.set_aspect("equal", adjustable="box")
    agreement.set_title("(b) Residual class-score mismatch", pad=10)
    agreement.set_xlabel(r"Teacher centered score: $(z^t_c-\bar{z}^t)/T$")
    agreement.set_ylabel(r"Calibrated student: $\alpha^*(z^s_c-\bar{z}^s)/T$")
    agreement.legend(frameon=False, loc="upper left", fontsize=8)
    colorbar = figure.colorbar(density, cax=color_axis)
    colorbar.set_label("Image–class pairs per bin (log scale)", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    figure.supxlabel(
        "CIFAR-100 · ResNet32×4 → ResNet8×4 · Linear KD · seed 2051 · 10,000 test images\n"
        "Per-image oracle scalar α ≥ 0; scores centered only, not standardized; all 1,000,000 image–class pairs included",
        fontsize=7.5, color="#52616B")
    check_text_bounds(figure)
    outputs = {}
    for extension in ("png", "pdf", "svg"):
        destination = HERE / f"{STEM}.{extension}"
        figure.savefig(destination, dpi=300, facecolor="white", bbox_inches="tight")
        outputs[destination.name] = digest(destination)
    plot.close(figure)
    metadata = {"design": "Two-panel mismatch diagnostic using only the frozen linear-KD student",
        "identity": report["identity"], "temperature": report["temperature"],
        "loss_factor": report["loss_factor"], "scalar_domain": report["scalar_domain"],
        "measurements": {"before": report["cases"]["affine"]["before"],
                         "after": report["cases"]["affine"]["after"], "remaining_percent": remaining},
        "density_plot": {"x": "(teacher_logits - mean_over_classes) / T",
                         "y": "alpha * (linear_student_logits - mean_over_classes) / T",
                         "color": "image-class count per hexagonal bin; logarithmic color normalization",
                         "gridsize": 90, "shared_axis_limits": [lower, upper],
                         "pairs_binned": pairs_binned, "standard_deviation_normalization": False,
                         "sampling_or_quantile_clipping": False},
        "checks": {"reconstructed_kl_max_absolute_error": errors,
                   "text_inside_canvas": True, "all_samples_and_classes_plotted": True},
        "sources_sha256": {str((SOURCE / name).relative_to(ROOT)): digest(SOURCE / name)
                           for name in ("report.json", "inference.json", "inference.npz", "oracle_arrays.npz")},
        "script_sha256": digest(__file__), "matplotlib_version": matplotlib.__version__,
        "output_sha256": outputs, "new_training_performed": False, "new_inference_performed": False,
        "limitations": "Single historical seed; excludes a purely positive-scalar explanation, not other causes of mismatch."}
    (HERE / f"{STEM}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"outputs": list(outputs), "pairs_binned": pairs_binned,
                      "checks": metadata["checks"]}, indent=2))


if __name__ == "__main__":
    render()
