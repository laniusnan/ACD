"""Compose vector ACI geometry and a photo-containing SICD schematic."""

import argparse
import json
from pathlib import Path

from render_aci_sicd import (
    PALETTE,
    arrow,
    configure_style,
    draw_aci_axes,
    draw_heatmap,
    load_cifar_examples,
    plt,
    routed_arrow,
    save_figure,
    synthetic_maps,
    text,
)
from matplotlib.patches import FancyBboxPatch, Rectangle


CANVAS_WIDTH = 18.0
CANVAS_HEIGHT = 8.4


def add_panel_axis(figure, bounds, **options):
    left, bottom, width, height = bounds
    return figure.add_axes(
        (left / CANVAS_WIDTH, bottom / CANVAS_HEIGHT,
         width / CANVAS_WIDTH, height / CANVAS_HEIGHT), **options
    )


def frame(axis, bounds, color):
    left, bottom, width, height = bounds
    axis.add_patch(FancyBboxPatch(
        (left, bottom), width, height,
        boxstyle="round,pad=0,rounding_size=0.13",
        facecolor="white", edgecolor=color, linewidth=1.35,
        linestyle=(0, (5, 4)), zorder=0,
    ))


def block(axis, bounds, title, formula, color, dark=False, fill="white"):
    left, bottom, width, height = bounds
    axis.add_patch(FancyBboxPatch(
        (left, bottom), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.075",
        facecolor=PALETTE["ink"] if dark else fill,
        edgecolor=color, linewidth=1.2, zorder=3,
    ))
    foreground = "white" if dark else color
    text(axis, (left + width / 2, bottom + height * 0.73),
         title, 14, foreground)
    text(axis, (left + width / 2, bottom + height * 0.29),
         formula, 16, foreground)


def draw_aci_panel(figure, axis):
    frame(axis, (0.25, 0.25, 6.65, 7.9), PALETTE["student_edge"])
    text(axis, (3.575, 7.70), "Angular Coordinate Interface (ACI)",
         18, PALETTE["student_edge"], fontweight="bold")
    text(axis, (3.575, 7.20), "Remove magnitude and bias from the student readout", 13)
    text(axis, (3.575, 6.65),
         r"$z_{s,k}=s\cos\theta_{s,k}"
         r"=s\,\hat{\mathbf{w}}_{s,k}^{\top}\hat{\mathbf{h}}_s$", 22)
    axes = (
        add_panel_axis(figure, (0.52, 2.82, 3.0, 3.0)),
        add_panel_axis(figure, (3.69, 2.82, 3.0, 3.0)),
    )
    draw_aci_axes(axes)
    axes[0].set_title("(a) Angular coordinates", fontsize=14, pad=12)
    axes[1].set_title("(b) Optional ETF init.", fontsize=14, pad=12)
    for geometry_axis in axes:
        for label in geometry_axis.texts:
            label.set_fontsize(max(13, label.get_fontsize() * 1.12))
            if label.get_text() == "Equiangular at initialization only":
                label.set_text("Equiangular init. only")
            if label.get_text().startswith("Anchor "):
                label.set_text(r"$\hat{\mathbf{w}}_k$")
    block(axis, (0.78, 1.75, 5.59, 0.95), "Normalize features and class weights",
          r"$\hat{\mathbf{h}}_s=\mathbf{h}_s/\|\mathbf{h}_s\|_2,"
          r"\quad\hat{\mathbf{w}}_{s,k}=\mathbf{w}_{s,k}/\|\mathbf{w}_{s,k}\|_2$",
          PALETTE["student_edge"], fill="#F3F8FB")
    text(axis, (3.575, 1.25), "Trainable directions; ETF is optional initialization.", 13)
    text(axis, (3.575, 0.78), "Conceptual arrows, not measured gradients.",
         12.5, PALETTE["muted"])


def draw_sicd_panel(figure, axis, example, heatmaps):
    frame(axis, (7.2, 0.25, 10.55, 7.9), PALETTE["green"])
    text(axis, (12.475, 7.70), "SICD Training Pipeline", 20,
         PALETTE["green"], fontweight="bold")
    text(axis, (12.475, 7.20), "Real CIFAR-100 input  |  synthetic feature grids", 14)
    image_axis = add_panel_axis(figure, (7.55, 4.15, 1.28, 1.28), zorder=2)
    image_axis.imshow(example["pixels"], interpolation="bicubic")
    image_axis.axis("off")
    axis.add_patch(Rectangle((7.55, 4.15), 1.28, 1.28, fill=False,
                             edgecolor=PALETTE["line"], linewidth=1.1))
    text(axis, (8.19, 5.75), r"Input $x$", 15)
    text(axis, (8.19, 3.86), "CIFAR-100", 12.5)
    text(axis, (8.19, 3.55), example["class"].replace("_", " "),
         13, PALETTE["muted"])
    arrow(axis, (8.85, 4.79), (9.13, 4.79), style="-")
    routed_arrow(axis, [(9.13, 4.79), (9.13, 5.85), (9.40, 5.85)])
    routed_arrow(axis, [(9.13, 4.79), (9.13, 3.75), (9.40, 3.75)])
    row_specs = (
        (5.85, "Teacher (frozen)", PALETTE["teacher"], "Original head",
         r"$z_t=W_t h_t+b_t$", r"$p=\mathrm{softmax}(\bar z_t/T)$"),
        (3.75, "Student (trainable)", PALETTE["student_edge"], "Cosine head (ACI)",
         r"$z_{s,k}=s\cos\theta_{s,k}$", r"$q=\mathrm{softmax}(\bar z_s/T)$"),
    )
    for row_index, (center, label, color, head, scores, probabilities) in enumerate(row_specs):
        draw_heatmap(axis, heatmaps[row_index], (9.45, center - 0.675, 1.35, 1.35))
        for boundary in (1, 2):
            offset = boundary * 1.35 / 3
            axis.add_patch(Rectangle(
                (9.45 + offset, center - 0.675), 0, 1.35,
                edgecolor="white", linewidth=0.75, alpha=0.8, zorder=3,
            ))
            axis.add_patch(Rectangle(
                (9.45, center - 0.675 + offset), 1.35, 0,
                edgecolor="white", linewidth=0.75, alpha=0.8, zorder=3,
            ))
        text(axis, (10.13, center + 0.91), label, 13, color)
        arrow(axis, (10.85, center), (11.48, center))
        text(axis, (11.16, center + 0.29), "Pool", 12, PALETTE["muted"])
        block(axis, (11.55, center - 0.55, 2.15, 1.1),
              head, scores, color, dark=True)
        arrow(axis, (13.77, center), (14.37, center))
        block(axis, (14.45, center - 0.55, 2.65, 1.1),
              "Standardize + softmax", probabilities, PALETTE["green"])
    text(axis, (10.13, 2.79), "Feature grids", 12.5, PALETTE["muted"])
    text(axis, (15.77, 4.80), r"$\bar z=(z-\mu(z))/\sigma(z)$", 16)
    block(axis, (10.60, 1.65, 6.50, 1.04), "Non-target conditional distillation",
          r"$L_{\mathrm{SICD}}=a\,T^2(1-p_y)\,"
          r"\mathrm{KL}(p^{-}\Vert q^{-})$",
          PALETTE["green"], fill=PALETTE["green_bg"])
    arrow(axis, (15.77, 3.16), (15.77, 2.73), color=PALETTE["green"])
    text(axis, (14.18, 2.96), r"Condition on $k\ne y$", 13, PALETTE["green"])
    text(axis, (16.15, 2.96), r"$q^{-}$", 14, PALETTE["green"])
    routed_arrow(axis, [(17.15, 5.85), (17.46, 5.85),
                        (17.46, 2.17), (17.15, 2.17)], color=PALETTE["teacher"])
    text(axis, (17.48, 4.35), r"$p^{-},\;1-p_y$", 14,
         PALETTE["teacher"], rotation=90, backgroundcolor="white")
    text(axis, (13.85, 1.25), r"$a$: detached gradient-strength matching", 13)
    text(axis, (12.475, 0.70),
         r"$L=\lambda_{\mathrm{CE}}\,\mathrm{CE}(z_s,y)"
         r"+\lambda_{\mathrm{KD}}\,L_{\mathrm{SICD}}$"
         "   (student updates only)", 16)


def compose_figure(example, heatmaps):
    figure = plt.figure(figsize=(14.0, 14.0 * CANVAS_HEIGHT / CANVAS_WIDTH))
    axis = add_panel_axis(figure, (0, 0, CANVAS_WIDTH, CANVAS_HEIGHT))
    axis.set(xlim=(0, CANVAS_WIDTH), ylim=(0, CANVAS_HEIGHT))
    axis.axis("off")
    draw_aci_panel(figure, axis)
    draw_sicd_panel(figure, axis, example, heatmaps)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--example", choices=("apple", "tiger", "pickup_truck"), default="apple")
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    font = configure_style()
    examples = load_cifar_examples(args.data_root, args.download)
    example = next(item for item in examples if item["class"] == args.example)
    heatmaps = synthetic_maps(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = compose_figure(example, heatmaps)
    save_figure(figure, args.output_dir, "final_method_figure")
    manifest = {
        "figure": "final_method_figure.pdf",
        "font": font,
        "dataset": "torchvision.datasets.CIFAR100",
        "split": "test",
        "input": {key: value for key, value in example.items() if key != "pixels"},
        "interpolation": "bicubic",
        "feature_maps": "Synthetic NumPy noise; not model activations.",
        "feature_map_seed": args.seed,
        "feature_map_arrays": [values.tolist() for values in heatmaps],
        "grid_lines": "Visual cells only, not spatial patch groups or independent heads.",
        "aci": "Angular Coordinate Interface; optional ETF initialization only.",
        "sicd": "Original teacher head, angular student head; all-class standardization, "
                "non-target conditioning, teacher non-target mass and gradient-strength matching.",
        "gradient_strength_matching": "Detached ratio of standardized-full and relation-loss "
                                      "gradient norms on non-head student parameters, same state/batch.",
        "simplifications": "Batch reduction, denominator floors and gradient-norm computation "
                           "are omitted from the drawing; CE uses unstandardized student scores.",
        "embedded_rasters": "One real CIFAR-100 photo; all other elements are vector.",
    }
    manifest_path = args.output_dir / "final_method_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output_dir.resolve() / 'final_method_figure.pdf'}")


if __name__ == "__main__":
    main()
