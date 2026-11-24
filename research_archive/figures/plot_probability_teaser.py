"""Synthetic illustration, NOT measured KD/ACD predictions.

Run: python plot_probability_teaser.py
Requires: numpy, matplotlib, seaborn; Chinese labels require a CJK font.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "simulated_probability_teaser")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cat is the target class. All values are deliberately hand-constructed.
    teacher = np.array([0.55, 0.25, 0.05, 0.15])
    conventional = np.array([0.64, 0.04, 0.25, 0.07])
    conditional = teacher[1:] / teacher[1:].sum()  # dog:car:bird = 5:1:3
    acd = np.r_[0.73, (1.0 - 0.73) * conditional]
    distributions = [teacher, conventional, acd]
    for p in distributions:
        assert np.all(p >= 0) and np.isclose(p.sum(), 1)
    np.testing.assert_allclose(acd[1:] / acd[1:].sum(), conditional)

    sns.set_theme(context="paper", style="white", font_scale=1.1)
    available = {font.name for font in font_manager.fontManager.ttflist}
    cjk = next((name for name in ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                "Microsoft YaHei", "SimHei", "Droid Sans Fallback"]
                if name in available), None)
    if cjk is None:
        raise RuntimeError("Install a CJK font (e.g. Noto Sans CJK SC) to render 猫、狗、车、鸟.")
    plt.rcParams.update({"font.family": ["DejaVu Sans", cjk],
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "path", "axes.linewidth": 0.7})
    colors = sns.color_palette(["#D39A42", "#4C78A8", "#59A38A", "#9A7BAF"])
    titles = ["Teacher", "Student (Conventional KD)", "Student (Our ACD)"]
    notes = ["Non-target ratio  5 : 1 : 3",
             "Non-target ratio  4 : 25 : 7",
             "Non-target ratio  5 : 1 : 3"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), sharey=True)
    for ax, p, title, note in zip(axes, distributions, titles, notes):
        bars = ax.bar(np.arange(4), p, width=0.58, color=colors, edgecolor="none")
        ax.set_xticks(np.arange(4), ["猫", "狗", "车", "鸟"])
        ax.set_ylim(0, 0.82)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, fontsize=12, pad=15)
        ax.tick_params(axis="both", length=0, labelsize=11)
        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in p], padding=4, fontsize=10)
        ax.text(0.5, -0.20, note, transform=ax.transAxes, ha="center", fontsize=10)
    axes[0].set_ylabel("Probability", fontsize=12)
    fig.text(0.5, 0.03,
             "Synthetic illustration (not experimental results)  |  Target: 猫  |  Ratios: 狗 : 车 : 鸟",
             ha="center", fontsize=9, color="#64748B")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.26, wspace=0.23)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"probability_teaser.{extension}",
                    dpi=300, bbox_inches="tight", facecolor="white")
        if extension == "svg":
            svg_path = args.output_dir / "probability_teaser.svg"
            svg_path.write_text(
                "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)
    metadata = {"synthetic": True, "target": "cat", "classes": ["cat", "dog", "car", "bird"],
                "teacher": teacher.tolist(), "conventional": conventional.tolist(), "acd": acd.tolist(),
                "note": "Illustrative distributions, not model inference or causal evidence."}
    (args.output_dir / "values.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved synthetic PNG/PDF/SVG and values to {args.output_dir}")


if __name__ == "__main__":
    main()
