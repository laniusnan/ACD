"""Export the standalone class-score panel from the verified oracle arrays."""

import json
import math

from render_output_mismatch import (
    HERE, ROOT, SOURCE, check_text_bounds, digest, load_inputs,
)
import matplotlib
import matplotlib.pyplot as plot

STEM = "fig_output_mismatch_right_cifar100_seed2051"


def render():
    report, _, _, teacher, calibrated, errors = load_inputs()
    plot.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "mathtext.fontset": "dejavusans", "axes.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .16,
    })
    figure, axis = plot.subplots(figsize=(4.6, 3.8), layout="constrained")
    lower = math.floor(min(teacher.min(), calibrated.min()))
    upper = math.ceil(max(teacher.max(), calibrated.max()))
    density = axis.hexbin(
        teacher.ravel(), calibrated.ravel(), gridsize=90,
        extent=(lower, upper, lower, upper), mincnt=1,
        bins="log", cmap="viridis", linewidths=0, zorder=2,
    )
    pairs_binned = int(density.get_array().sum())
    if pairs_binned != teacher.size:
        raise ValueError("Density plot dropped image-class pairs")
    axis.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.1,
              color="#C44E52", label="Exact agreement", zorder=3)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"Teacher centered score: $(z^t_c-\bar{z}^t)/T$")
    axis.set_ylabel(r"Calibrated student: $\alpha^*(z^s_c-\bar{z}^s)/T$")
    axis.legend(frameon=False, loc="upper left", fontsize=8)
    colorbar = figure.colorbar(density, ax=axis, fraction=.046, pad=.035)
    colorbar.set_label("Image–class pairs per bin (log scale)", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    check_text_bounds(figure)
    outputs = {}
    for extension in ("png", "pdf", "svg"):
        destination = HERE / f"{STEM}.{extension}"
        figure.savefig(destination, dpi=300, facecolor="white", bbox_inches="tight")
        outputs[destination.name] = digest(destination)
    plot.close(figure)
    metadata = {
        "design": "Standalone right panel: calibrated teacher-student class-score mismatch",
        "identity": report["identity"], "temperature": report["temperature"],
        "scalar_domain": report["scalar_domain"],
        "density_plot": {
            "x": "(teacher_logits - mean_over_classes) / T",
            "y": "alpha * (linear_student_logits - mean_over_classes) / T",
            "color": "image-class count per hexagonal bin; logarithmic color normalization",
            "gridsize": 90, "shared_axis_limits": [lower, upper],
            "pairs_binned": pairs_binned, "standard_deviation_normalization": False,
            "sampling_or_quantile_clipping": False,
        },
        "checks": {"reconstructed_kl_max_absolute_error": errors,
                   "text_inside_canvas": True, "all_samples_and_classes_plotted": True},
        "sources_sha256": {
            str((SOURCE / name).relative_to(ROOT)): digest(SOURCE / name)
            for name in ("report.json", "inference.json", "inference.npz", "oracle_arrays.npz")
        },
        "script_sha256": digest(__file__),
        "source_loader_sha256": digest(HERE / "render_output_mismatch.py"),
        "matplotlib_version": matplotlib.__version__, "output_sha256": outputs,
        "new_training_performed": False, "new_inference_performed": False,
    }
    (HERE / f"{STEM}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"outputs": list(outputs), "pairs_binned": pairs_binned,
                      "checks": metadata["checks"]}, indent=2))


if __name__ == "__main__":
    render()
