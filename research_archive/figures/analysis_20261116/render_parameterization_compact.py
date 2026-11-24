"""Render existing fixed-checkpoint measurements as three metric panels."""
from render import COLORS, HERE, read, plot, np
from matplotlib.ticker import PercentFormatter


def main():
    report = read("parameterization.json")
    fig, axes = plot.subplots(1, 3, figsize=(13.8, 3.25))
    factors = [("common_scale", "Common logit scale", "o"),
               ("class_norm", "Class-norm heterogeneity", "s"),
               ("class_bias", "Centered class bias", "^")]
    metrics = [("conditional_tv", "(a) Distribution change", "Conditional TV to original",
                "Same angles,\nshifted distributions"),
               ("top1_alternative_changed", "(b) Ranking change", "Top non-target class changed (%)",
                "Rank changes\nbeyond scaling"),
               ("deviation", "(c) Gradient change", "Gradient deviation (1 − cosine)",
                "KD gradients\nrotate")]
    for ax, (metric, title, ylabel, takeaway) in zip(axes, metrics):
        for color, (factor, label, marker) in zip(COLORS, factors):
            rows = sorted((r for r in report["rows"] if r["factor"] == factor),
                          key=lambda r: r["strength"])
            x = [r["strength"] for r in rows]
            ax.plot(x, [np.mean(r[metric]) for r in rows], color=color,
                    marker=marker, markersize=4, linewidth=1.8, label=label)
            if metric != "top1_alternative_changed":
                ax.fill_between(x, [np.quantile(r[metric], .25) for r in rows],
                                [np.quantile(r[metric], .75) for r in rows],
                                color=color, alpha=.10)
        ax.axvline(1, color="#666666", linestyle=":", linewidth=.8)
        ax.set_title(title)
        ax.set_xlabel("Intervention parameter (original = 1)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-.08, 4.08)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_ylim(bottom=0)
        if metric == "top1_alternative_changed":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            for factor, color, offset in [("class_norm", COLORS[1], (12, -3)),
                                          ("class_bias", COLORS[2], (12, 14))]:
                value = next(np.mean(r[metric]) for r in report["rows"]
                             if r["factor"] == factor and r["strength"] == 0)
                ax.annotate(f"{value:.2%}", (0, value), xytext=offset,
                            textcoords="offset points", color=color, fontsize=9)
        else:
            ax.set_yscale("symlog", linthresh=1e-4)
            ax.set_yticks([0, 1e-4, 1e-3, 1e-2, 1e-1, 1])
            ax.set_yticklabels(["0", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1"])
        ax.text(.70, .48 if metric == "top1_alternative_changed" else .33,
                takeaway, transform=ax.transAxes, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#263D52", linespacing=1.35,
                bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EEF3F8",
                      "edgecolor": "#9AAFC1", "linewidth": 1.0}, zorder=5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.02),
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .94), pad=.8, w_pad=5.5)
    for ax in axes:
        ax.set_xlabel("")
    for extension in ("pdf", "svg", "png"):
        fig.savefig(HERE / f"fig_parameterization_compact.{extension}",
                    bbox_inches="tight", pad_inches=.05, facecolor="white")
    plot.close(fig)


if __name__ == "__main__":
    main()
