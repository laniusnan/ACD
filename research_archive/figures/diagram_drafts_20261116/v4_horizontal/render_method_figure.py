"""Draw taller ACI and SICD panels with readable flow arrows and bottom-row losses."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from render_aci_sicd import (  # noqa: E402
    PALETTE, arrow, configure_style, draw_aci_axes, draw_heatmap,
    load_cifar_examples, plt, routed_arrow, save_figure, synthetic_maps, text,
)
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402


LEFT_WIDTH = 9.50
RIGHT_WIDTH = 12.20
HEIGHT = 7.80
GAP = 0.25
WIDTH = LEFT_WIDTH + GAP + RIGHT_WIDTH
INCHES_PER_UNIT = 0.78
NOTE_COLOR = "#939BA3"


class Panel:
    """Keep the same local coordinates and physical type sizes in all exports."""

    def __init__(self, figure, canvas_size, bounds):
        self.figure = figure
        self.canvas_width, self.canvas_height = canvas_size
        self.left, self.bottom, self.width, self.height = bounds
        self.axis = self.child((0, 0, self.width, self.height))
        self.axis.set(xlim=(0, self.width), ylim=(0, self.height))
        self.axis.axis("off")

    def child(self, bounds, **options):
        x, y, width, height = bounds
        return self.figure.add_axes(
            ((self.left + x) / self.canvas_width,
             (self.bottom + y) / self.canvas_height,
             width / self.canvas_width, height / self.canvas_height), **options,
        )


def frame(panel, title, color):
    axis = panel.axis
    axis.add_patch(FancyBboxPatch(
        (0.10, 0.10), panel.width - 0.20, panel.height - 0.40,
        boxstyle="round,pad=0,rounding_size=0.045",
        facecolor="white", edgecolor=color, linewidth=1.15,
        linestyle=(0, (4, 3)), zorder=0,
    ))
    text(axis, (panel.width / 2, panel.height - 0.30), title,
         21, color, fontweight="bold",
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})


def block(axis, bounds, title, formula, color, *, dark=False, fill="white"):
    left, bottom, width, height = bounds
    axis.add_patch(FancyBboxPatch(
        (left, bottom), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.07",
        facecolor=PALETTE["ink"] if dark else fill,
        edgecolor=color, linewidth=1.15, zorder=3,
    ))
    foreground = "white" if dark else color
    text(axis, (left + width / 2, bottom + height * 0.74),
         title, 15, foreground)
    text(axis, (left + width / 2, bottom + height * 0.29),
         formula, 17, foreground)


def draw_aci(panel):
    axis = panel.axis
    frame(panel, "Angular Coordinate Interface (ACI)", PALETTE["student_edge"])
    text(axis, (2.30, 6.95), "(a) Cosine readout", 17, PALETTE["student_edge"])
    text(axis, (7.20, 6.95), "(b) Optional ETF initialization", 17, PALETTE["student_edge"])
    geometry_axes = (panel.child((0.30, 2.05, 4.00, 4.55)),
                     panel.child((5.20, 2.05, 4.00, 4.55)))
    draw_aci_axes(geometry_axes)
    for geometry_axis in geometry_axes:
        geometry_axis.set_title("")
        for label in geometry_axis.texts:
            value = label.get_text()
            label.set_fontsize(14)
            if value.startswith("Anchor "):
                label.set_text(r"$\hat{\mathbf{w}}_{s,k}$")
                label.set_fontsize(18)
            elif value == r"$\theta$":
                label.set_text(r"$\theta_{s,k}$")
                label.set_position((0.31, 0.23))
                label.set_fontsize(16)
                label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.15})
            elif value == "Equiangular at initialization only":
                label.set_text("Trainable after initialization")
                label.set_fontsize(13.5)
    formulas = (
        (2.30, r"$z_{s,k}=s\,\hat{\mathbf{w}}_{s,k}^{\top}\hat{\mathbf{h}}_s"
         r"=s\cos\theta_{s,k}$", r"$s>0$ fixed, $b_s=0$", 19),
        (7.20, r"$\hat{\mathbf{w}}_{s,i}(0)^{\top}\hat{\mathbf{w}}_{s,j}(0)"
         r"=-\frac{1}{C-1}$", r"$i\ne j$; initialization only", 20),
    )
    for center, formula, annotation, size in formulas:
        main = text(axis, (center, 1.25), formula, size)
        note = text(axis, (center, 0.65), annotation, 12, NOTE_COLOR, ha="right")
        # Resolve against actual typeset formula bounds, not the subplot edge.
        panel.figure.formula_notes.append((axis, main, note))


def draw_sicd(panel, example, heatmaps):
    axis = panel.axis
    frame(panel, "Scale-Invariant Conditional Distillation (SICD)", PALETTE["green"])
    image_axis = panel.child((0.30, 4.50, 1.10, 1.10), zorder=2)
    image_axis.imshow(example["pixels"], interpolation="bicubic")
    image_axis.axis("off")
    axis.add_patch(Rectangle((0.30, 4.50), 1.10, 1.10, fill=False,
                             edgecolor=PALETTE["line"], linewidth=1.0))
    text(axis, (0.85, 5.88), r"Input $x$", 17)
    arrow(axis, (1.45, 5.05), (1.71, 5.05), style="-")
    routed_arrow(axis, [(1.71, 5.05), (1.71, 6.10), (2.07, 6.10)])
    routed_arrow(axis, [(1.71, 5.05), (1.71, 4.00), (2.07, 4.00)])
    rows = (
        (6.10, "Teacher (frozen)", PALETTE["teacher"], "Original head",
         r"$z_t=W_t h_t+b_t$", r"$p=\mathrm{softmax}(\bar z_t/\tau)$"),
        (4.00, "Student (trainable)", PALETTE["student_edge"], "Cosine head (ACI)",
         r"$z_{s,k}=s\cos\theta_{s,k}$", r"$q=\mathrm{softmax}(\bar z_s/\tau)$"),
    )
    for index, (center, role, color, head, scores, probabilities) in enumerate(rows):
        draw_heatmap(axis, heatmaps[index], (2.12, center - 0.59, 1.18, 1.18))
        text(axis, (3.00, center + 0.88), role, 17, color)
        arrow(axis, (3.36, center), (3.94, center))
        text(axis, (3.65, center - 0.38), "Pool", 13, PALETTE["muted"])
        block(axis, (4.00, center - 0.45, 2.30, 0.90), head, scores, color, dark=True)
        arrow(axis, (6.37, center), (6.94, center))
        block(axis, (7.00, center - 0.45, 3.10, 0.90),
              "Standardize + softmax", probabilities, PALETTE["green"])
    text(axis, (2.71, 3.03), "Feature grids", 13, PALETTE["muted"])
    text(axis, (8.55, 5.05), r"$\bar z=(z-\mu(z))/\sigma(z)$", 18)
    text(axis, (10.85, 7.00), r"$k\ne y$", 16, PALETTE["green"])
    # Both objectives occupy the lower row, leaving the upper flow unobstructed.
    axis.add_patch(FancyBboxPatch(
        (0.35, 0.75), 6.35, 1.40,
        boxstyle="round,pad=0.025,rounding_size=0.07",
        facecolor=PALETTE["green_bg"], edgecolor=PALETTE["green"], linewidth=1.15, zorder=3))
    text(axis, (3.525, 1.77), "Conditional distillation", 17, PALETTE["green"])
    text(axis, (3.525, 1.18),
         r"$\mathcal{L}_{\mathrm{SICD}}=\tau^2(1-p_y)\,D_{\mathrm{KL}}(p^{-y}\,\Vert\,q^{-y})$",
         19, PALETTE["green"])
    axis.add_patch(FancyBboxPatch(
        (7.30, 0.75), 4.55, 1.40,
        boxstyle="round,pad=0.025,rounding_size=0.07",
        facecolor="white", edgecolor=PALETTE["green"], linewidth=1.15, zorder=3))
    text(axis, (9.575, 1.77), "Training objective", 17, PALETTE["green"])
    text(axis, (9.575, 1.18),
         r"$\mathcal{L}_{\mathrm{ACD}}=\mathcal{L}_{\mathrm{CE}}"
         r"+\lambda\cdot r\cdot\mathcal{L}_{\mathrm{SICD}}$", 18)
    arrow(axis, (6.78, 1.45), (7.22, 1.45), color=PALETTE["green"])
    routed_arrow(axis, [(10.17, 6.10), (11.50, 6.10), (11.50, 2.70),
                        (4.40, 2.70), (4.40, 2.22)], color=PALETTE["teacher"])
    text(axis, (10.85, 6.55), r"$p^{-y},\;1-p_y$", 16, PALETTE["teacher"])
    routed_arrow(axis, [(10.17, 4.00), (10.75, 4.00), (10.75, 2.40),
                        (5.50, 2.40), (5.50, 2.22)], color=PALETTE["green"])
    text(axis, (10.85, 4.45), r"$q^{-y}$", 16, PALETTE["green"])


def new_figure(width, height):
    figure = plt.figure(figsize=(width * INCHES_PER_UNIT, height * INCHES_PER_UNIT))
    figure.formula_notes = []
    return figure


def align_notes(figure):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis, main, note in figure.formula_notes:
        right = main.get_window_extent(renderer).x1
        x, _ = axis.transData.inverted().transform((right, 0))
        note.set_x(x)


def render(example, heatmaps):
    figures = {}
    for name, width, draw in (
        ("aci", LEFT_WIDTH, draw_aci),
        ("sicd", RIGHT_WIDTH, lambda panel: draw_sicd(panel, example, heatmaps)),
    ):
        figure = new_figure(width, HEIGHT)
        draw(Panel(figure, (width, HEIGHT), (0, 0, width, HEIGHT)))
        figures[name] = figure
    combined = new_figure(WIDTH, HEIGHT)
    draw_aci(Panel(combined, (WIDTH, HEIGHT), (0, 0, LEFT_WIDTH, HEIGHT)))
    draw_sicd(Panel(combined, (WIDTH, HEIGHT), (LEFT_WIDTH + GAP, 0, RIGHT_WIDTH, HEIGHT)),
              example, heatmaps)
    figures["method_horizontal"] = combined
    for figure in figures.values():
        align_notes(figure)
    return figures


def visible_text(figure):
    return [label.get_text() for label in figure.findobj(matplotlib.text.Text)
            if label.get_visible() and label.get_text()]


def validate(figures):
    assert visible_text(figures["method_horizontal"]) == visible_text(figures["aci"]) + visible_text(figures["sicd"])
    assert figures["aci"].get_figheight() == figures["sicd"].get_figheight()
    for label in visible_text(figures["method_horizontal"]):
        assert all(value not in label.lower() for value in
                   ("selecting the distillation", "standardized full kd", "cifar", "apple",
                    "conceptual", "gradient-strength", "group", r"\mathbf{h}_s/"))
    assert sum(len(axis.images) for axis in figures["method_horizontal"].axes) == 1
    for name, figure in figures.items():
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for _, main, note in figure.formula_notes:
            assert abs(main.get_window_extent(renderer).x1 - note.get_window_extent(renderer).x1) < 0.1
            assert note.get_fontsize() < main.get_fontsize()
        for axis in figure.axes:
            if axis.axison:
                continue
            canvas = axis.get_window_extent(renderer)
            label_bounds = []
            for label in axis.texts:
                extent = label.get_window_extent(renderer)
                if not (extent.x0 >= canvas.x0 - 1 and extent.x1 <= canvas.x1 + 1
                        and extent.y0 >= canvas.y0 - 1 and extent.y1 <= canvas.y1 + 1):
                    raise RuntimeError(f"Text outside its panel in {name}: {label.get_text()!r}")
                for previous, bounds in label_bounds:
                    if extent.overlaps(bounds):
                        raise RuntimeError(f"Overlapping text in {name}: {label.get_text()!r}, {previous!r}")
                label_bounds.append((label.get_text(), extent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--example", choices=("apple", "tiger", "pickup_truck"), default="apple")
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    font = configure_style()
    example = next(item for item in load_cifar_examples(args.data_root, args.download)
                   if item["class"] == args.example)
    heatmaps = synthetic_maps(args.seed)
    figures = render(example, heatmaps)
    validate(figures)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        save_figure(figure, args.output_dir, name)
        svg = args.output_dir / f"{name}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    manifest = {
        "revision": "V4 revision: taller ACI, smaller flow blocks with visible arrow shafts, both losses in the bottom row.",
        "outputs": [f"{name}.{suffix}" for name in figures for suffix in ("pdf", "svg", "png")],
        "layout_units": {"width": WIDTH, "height": HEIGHT, "left_width": LEFT_WIDTH,
                         "right_width": RIGHT_WIDTH, "gap": GAP, "inches_per_unit": INCHES_PER_UNIT},
        "assembly": "Same height, physical scale, and font sizes; no additional text or stretch transforms.",
        "aci_notes": {"fontsize_pt": 12, "color": NOTE_COLOR,
                      "alignment": "Right edges match their main formula's rendered bounds."},
        "font": font, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "dataset": "torchvision.datasets.CIFAR100", "split": "test",
        "input": {key: value for key, value in example.items() if key != "pixels"},
        "input_interpolation": "bicubic",
        "feature_maps": {"synthetic": True, "seed": args.seed,
                         "arrays": [values.tolist() for values in heatmaps]},
        "embedded_rasters": "One input photograph in sicd and assembly; all other elements vector.",
        "aci": "Unit-normalized feature and class-weight directions, positive fixed scale, no student bias. Normalization definitions omitted from graphic.",
        "etf": "Optional initialization only; t=0 pairwise condition; three-class example with 120-degree angles. Trainable directions thereafter.",
        "etf_scope": "Current CIFAR rerun does not enable ETF initialization.",
        "sicd": "Frozen original teacher head; trainable angular student; classwise standardization, softmax, weighted non-target conditional KL.",
        "standardization": "Per-image all-class mean and sample std (correction=1); numerical floor omitted.",
        "conditioning": "p^-_k = p_k/(1-p_y), q^-_k = q_k/(1-q_y), k != y.",
        "gradient_matching": "r is the detached standardized-full/relation gradient-norm ratio on non-head student parameters, same state and batch.",
        "omissions": "Batch averaging, denominator floors, computation of r; CE uses unstandardized student logits.",
        "patch_groups": "None; feature grids are schematic, not grouped patches.",
        "training_or_inference_run": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved 9 figure files and manifest in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
