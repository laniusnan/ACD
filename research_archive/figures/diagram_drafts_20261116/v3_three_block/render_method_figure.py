"""Render three reusable method blocks, following LIFT and PLACE Fig. 4's layout.

Only the layout is borrowed. The objectives and computation depict ACI/SICD.
Run from the repository root; see README.md for the existing plotting environment.
"""

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


LEFT_WIDTH = 8.50
RIGHT_WIDTH = 11.15
HEIGHT = 8.80
OBJECTIVE_HEIGHT = 1.90
GAP = 0.30
ACI_HEIGHT = HEIGHT - OBJECTIVE_HEIGHT - GAP
WIDTH = LEFT_WIDTH + GAP + RIGHT_WIDTH
INCHES_PER_UNIT = 0.78


class Panel:
    """Use identical local coordinates and physical type sizes in every export."""

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


def frame(panel, title, color, title_size=20):
    axis = panel.axis
    axis.add_patch(FancyBboxPatch(
        (0.10, 0.10), panel.width - 0.20, panel.height - 0.40,
        boxstyle="round,pad=0,rounding_size=0.045",
        facecolor="white", edgecolor=color, linewidth=1.15,
        linestyle=(0, (4, 3)), zorder=0,
    ))
    text(axis, (panel.width / 2, panel.height - 0.30), title,
         title_size, color, fontweight="bold",
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
         title, 15.5, foreground)
    text(axis, (left + width / 2, bottom + height * 0.29),
         formula, 19, foreground)


def draw_objective(panel):
    """A text/math-only block; p and q are standardized outputs defined at right."""
    axis = panel.axis
    frame(panel, "Selecting the distillation objective", PALETTE["teacher"])
    text(axis, (1.90, 1.00), r"$T^2\,\mathrm{KL}(p\Vert q)$", 23)
    text(axis, (3.57, 1.00), r"$\Longrightarrow$", 27, PALETTE["muted"])
    text(axis, (6.13, 1.00),
         r"$a\,T^2(1-p_y)\,\mathrm{KL}(p^{-}\Vert q^{-})$", 22,
         PALETTE["green"])
    text(axis, (1.90, 0.42), "Standardized full KD", 16)
    text(axis, (6.13, 0.42), "SICD: conditional relations", 16, PALETTE["green"])


def draw_aci(panel):
    axis = panel.axis
    frame(panel, "Angular Coordinate Interface (ACI)", PALETTE["student_edge"])
    text(axis, (LEFT_WIDTH / 2, 5.77),
         r"$\hat{\mathbf{h}}_s=\mathbf{h}_s/\|\mathbf{h}_s\|_2,"
         r"\qquad\hat{\mathbf{w}}_{s,k}=\mathbf{w}_{s,k}/\|\mathbf{w}_{s,k}\|_2$",
         20)
    text(axis, (2.20, 5.23), "(a) Cosine readout", 17, PALETTE["student_edge"])
    text(axis, (6.30, 5.23), "(b) Optional ETF initialization", 17, PALETTE["student_edge"])
    geometry_axes = (panel.child((0.30, 1.56, 3.80, 3.48)),
                     panel.child((4.40, 1.56, 3.80, 3.48)))
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
    text(axis, (2.20, 1.02),
         r"$z_{s,k}=s\,\hat{\mathbf{w}}_{s,k}^{\top}\hat{\mathbf{h}}_s"
         r"=s\cos\theta_{s,k}$", 19)
    text(axis, (2.20, 0.47), r"$s>0$ fixed, $b_s=0$", 16)
    text(axis, (6.30, 1.02),
         r"$\hat{\mathbf{w}}_{s,i}(0)^{\top}\hat{\mathbf{w}}_{s,j}(0)"
         r"=-\frac{1}{C-1}$", 20)
    text(axis, (6.30, 0.47), r"$i\ne j$; initialization only", 16)


def draw_sicd(panel, example, heatmaps):
    axis = panel.axis
    frame(panel, "Scale-Invariant Conditional Distillation (SICD)", PALETTE["green"])
    image_axis = panel.child((0.34, 4.77, 1.25, 1.25), zorder=2)
    image_axis.imshow(example["pixels"], interpolation="bicubic")
    image_axis.axis("off")
    axis.add_patch(Rectangle((0.34, 4.77), 1.25, 1.25, fill=False,
                             edgecolor=PALETTE["line"], linewidth=1.0))
    text(axis, (0.965, 6.31), r"Input $x$", 17)
    arrow(axis, (1.64, 5.395), (1.99, 5.395), style="-")
    routed_arrow(axis, [(1.99, 5.395), (1.99, 6.55), (2.25, 6.55)])
    routed_arrow(axis, [(1.99, 5.395), (1.99, 4.25), (2.25, 4.25)])
    rows = (
        (6.55, "Teacher (frozen)", PALETTE["teacher"], "Original head",
         r"$z_t=W_t h_t+b_t$", r"$p=\mathrm{softmax}(\bar z_t/T)$"),
        (4.25, "Student (trainable)", PALETTE["student_edge"], "Cosine head (ACI)",
         r"$z_{s,k}=s\cos\theta_{s,k}$", r"$q=\mathrm{softmax}(\bar z_s/T)$"),
    )
    for index, (center, role, color, head, scores, probabilities) in enumerate(rows):
        # Individual heatmap cells are visual texture, not spatial patch groups.
        draw_heatmap(axis, heatmaps[index], (2.30, center - 0.64, 1.28, 1.28))
        text(axis, (3.30, center + 1.00), role, 16, color)
        arrow(axis, (3.63, center), (4.20, center))
        text(axis, (3.94, center + 0.30), "Pool", 14, PALETTE["muted"])
        block(axis, (4.27, center - 0.56, 2.32, 1.12), head, scores, color, dark=True)
        arrow(axis, (6.66, center), (7.26, center))
        block(axis, (7.33, center - 0.56, 2.98, 1.12),
              "Standardize + softmax", probabilities, PALETTE["green"])
    text(axis, (2.94, 3.30), "Feature grids", 14, PALETTE["muted"])
    text(axis, (8.82, 5.40), r"$\bar z=(z-\mu(z))/\sigma(z)$", 20)
    block(axis, (3.40, 1.50, 6.91, 1.15), "Non-target conditional distillation",
          r"$L_{\mathrm{SICD}}=a\,T^2(1-p_y)\,\mathrm{KL}(p^{-}\Vert q^{-})$",
          PALETTE["green"], fill=PALETTE["green_bg"])
    arrow(axis, (8.82, 3.64), (8.82, 2.71), color=PALETTE["green"])
    text(axis, (7.02, 3.12), r"Condition on $k\ne y$", 16, PALETTE["green"])
    text(axis, (9.17, 3.12), r"$q^{-}$", 18, PALETTE["green"])
    routed_arrow(axis, [(10.36, 6.55), (10.79, 6.55),
                        (10.79, 2.08), (10.37, 2.08)], color=PALETTE["teacher"])
    text(axis, (10.80, 4.64), r"$p^{-},\;1-p_y$", 18,
         PALETTE["teacher"], rotation=90, backgroundcolor="white")
    text(axis, (RIGHT_WIDTH / 2, 0.66),
         r"$L=\lambda_{\mathrm{CE}}\,\mathrm{CE}(z_s,y)"
         r"+\lambda_{\mathrm{KD}}\,L_{\mathrm{SICD}}$", 22)


def render(example, heatmaps):
    """The assembly adds no title, caption, panel letters, or other text."""
    specifications = {
        "objective": (LEFT_WIDTH, OBJECTIVE_HEIGHT, draw_objective),
        "aci": (LEFT_WIDTH, ACI_HEIGHT, draw_aci),
        "sicd": (RIGHT_WIDTH, HEIGHT, lambda panel: draw_sicd(panel, example, heatmaps)),
    }
    figures = {}
    for name, (width, height, draw) in specifications.items():
        figure = plt.figure(figsize=(width * INCHES_PER_UNIT, height * INCHES_PER_UNIT))
        draw(Panel(figure, (width, height), (0, 0, width, height)))
        figures[name] = figure
    combined = plt.figure(figsize=(WIDTH * INCHES_PER_UNIT, HEIGHT * INCHES_PER_UNIT))
    draw_objective(Panel(combined, (WIDTH, HEIGHT),
                         (0, ACI_HEIGHT + GAP, LEFT_WIDTH, OBJECTIVE_HEIGHT)))
    draw_aci(Panel(combined, (WIDTH, HEIGHT), (0, 0, LEFT_WIDTH, ACI_HEIGHT)))
    draw_sicd(Panel(combined, (WIDTH, HEIGHT), (LEFT_WIDTH + GAP, 0, RIGHT_WIDTH, HEIGHT)),
              example, heatmaps)
    figures["method_three_block"] = combined
    return figures


def visible_text(figure):
    return [label.get_text() for label in figure.findobj(matplotlib.text.Text)
            if label.get_visible() and label.get_text()]


def validate(figures):
    # Assert the requested composition and omissions, not a new training claim.
    standalone = sum((visible_text(figures[name]) for name in ("objective", "aci", "sicd")), [])
    assert visible_text(figures["method_three_block"]) == standalone
    for label in visible_text(figures["method_three_block"]):
        assert all(value not in label.lower()
                   for value in ("cifar", "apple", "conceptual", "gradient-strength", "group"))
    assert not figures["objective"].images
    assert all(not axis.images for axis in figures["objective"].axes)
    assert sum(len(axis.images) for axis in figures["method_three_block"].axes) == 1
    for name, figure in figures.items():
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in figure.axes:
            # Check panel-local boundaries as well as save_figure's full-canvas check.
            if axis.axison:
                continue  # circular geometry axes include their own legend text
            canvas = axis.get_window_extent(renderer)
            for label in axis.texts:
                extent = label.get_window_extent(renderer)
                if not (extent.x0 >= canvas.x0 - 1 and extent.x1 <= canvas.x1 + 1
                        and extent.y0 >= canvas.y0 - 1 and extent.y1 <= canvas.y1 + 1):
                    raise RuntimeError(f"Text outside its panel in {name}: {label.get_text()!r}")


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
    manifest = {
        "layout_reference": {
            "title": "LIFT and PLACE: A Simple, Stable, and Effective Knowledge Distillation Framework for Lightweight Diffusion Models",
            "url": "https://arxiv.org/abs/2605.19729v3", "figure": 4, "page": 4,
            "scope": "Three-block layout only; no regression or PLACE grouping in ACI/SICD.",
        },
        "outputs": [f"{name}.{suffix}" for name in figures for suffix in ("pdf", "svg", "png")],
        "layout_units": {"width": WIDTH, "height": HEIGHT, "left_width": LEFT_WIDTH,
                         "right_width": RIGHT_WIDTH, "objective_height": OBJECTIVE_HEIGHT,
                         "aci_height": ACI_HEIGHT, "gap": GAP, "inches_per_unit": INCHES_PER_UNIT},
        "assembly": "Identical panel renderers and physical font sizes; no additional text.",
        "font": font, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "dataset": "torchvision.datasets.CIFAR100", "split": "test",
        "input": {key: value for key, value in example.items() if key != "pixels"},
        "input_interpolation": "bicubic",
        "feature_maps": {"synthetic": True, "seed": args.seed,
                         "arrays": [values.tolist() for values in heatmaps]},
        "embedded_rasters": "One input photograph in sicd and the assembly; all other elements vector.",
        "aci": "Bias-free cosine student head with unit feature/weight directions and fixed scale s.",
        "etf": "Separate OPTIONAL initial pairwise constraint at t=0; a three-class example is illustrated. Directions train thereafter.",
        "etf_scope": "The current CIFAR rerun does not enable ETF initialization; this is an optional variant illustration.",
        "objective": "From standardized full KD to teacher-mass-weighted conditional KD; the arrow is a design choice, not equality.",
        "sicd": "Frozen original teacher head; trainable angular student; per-image all-class standardization; non-target conditioning.",
        "standardization": "Classwise mean and sample std (correction=1); denominator floor omitted in drawing.",
        "conditioning": "p^-_k = p_k/(1-p_y), q^-_k = q_k/(1-q_y), k != y.",
        "gradient_matching": "a is the detached ratio of standardized-full/relation gradient norms over trainable non-head student parameters, at the same state and batch.",
        "omissions": "Batch mean, numerical floors and computation of a; a remains in loss formula. CE uses unstandardized student logits.",
        "patch_groups": "None. Feature-grid cells are schematic visual texture, not grouped patches or per-patch classifiers.",
        "training_or_inference_run": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved 12 figure files and manifest in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
