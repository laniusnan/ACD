"""Compose vector ACI geometry and a photo-containing SICD schematic."""

import argparse
import json
from pathlib import Path

import numpy as np

from render_aci_sicd import (
    PALETTE,
    arrow,
    configure_style,
    dot,
    draw_heatmap,
    load_cifar_examples,
    plt,
    polar,
    routed_arrow,
    save_figure,
    synthetic_maps,
    text,
)
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle


CANVAS_WIDTH = 18.0
CANVAS_HEIGHT = 8.4
ACI_CANVAS_WIDTH = 7.15
ACI_ILLUSTRATION = {
    "feature_angle_degrees": 56.0,
    "feature_norms": [0.52, 1.12],
    "weight_angles_degrees": [10.0, 150.0, 245.0],
    "weight_norms": [1.12, 0.73, 1.27],
    "shared_scale": 30.0,
}
ACI_SOURCES = [
    "CLASSIFIER_DISCUSSION_20261026.md, sections 1.2 and 5",
    "docs/CLASSIFIER_FIGURE_GUIDE_20261027.md, sections B and C",
    "docs/CLASSIFIER_FIGURE_REUSE_20261116.md, sections 2 and 3",
    "reports/ACD_WRITING_ARGUMENT_20261110.md, Method and Figure implications",
    "reproductions/cifar100_acd_rerun_20261116/methods.py, AngularHead",
    "reproductions/cifar100_acd_rerun_20261116/protocol.json, angular_scale",
]


def add_panel_axis(figure, bounds, canvas_width=CANVAS_WIDTH, **options):
    left, bottom, width, height = bounds
    return figure.add_axes(
        (left / canvas_width, bottom / CANVAS_HEIGHT,
         width / canvas_width, height / CANVAS_HEIGHT), **options
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


def draw_readout_geometry(axis, normalized):
    axis.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal")
    axis.set_xticks(np.linspace(-1, 1, 5))
    axis.set_yticks(np.linspace(-1, 1, 5))
    axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axis.set_axisbelow(True)
    axis.grid(color=PALETTE["grid"], linewidth=0.55)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.add_patch(Circle(
        (0, 0), 1, fill=False, edgecolor="#9AA5AE", linewidth=1.1,
        linestyle="-" if normalized else (0, (3, 3)),
    ))
    arrow(axis, (-1.35, 0), (1.35, 0), color="#BAC2C9", style="-", width=0.6)
    arrow(axis, (0, -1.35), (0, 1.35), color="#BAC2C9", style="-", width=0.6)
    for class_index, (direction, magnitude) in enumerate(zip(
            ACI_ILLUSTRATION["weight_angles_degrees"], ACI_ILLUSTRATION["weight_norms"])):
        endpoint = polar(1 if normalized else magnitude, direction)
        arrow(axis, (0, 0), endpoint, color=PALETTE["teacher"], style="-", width=1.15)
        dot(axis, endpoint, 0.065, PALETTE["teacher"])
        label = (rf"$\hat{{\mathbf{{w}}}}_{class_index + 1}$" if normalized
                 else rf"$\mathbf{{w}}_{class_index + 1}$")
        offset = ((0.15, -0.22), (-0.19, 0.23), (-0.28, 0.01))[class_index]
        text(axis, endpoint + offset, label, 14, PALETTE["teacher"])
    feature_direction = ACI_ILLUSTRATION["feature_angle_degrees"]
    if normalized:
        endpoint = polar(1, feature_direction)
        arrow(axis, (0, 0), endpoint, color=PALETTE["student_edge"], width=1.5)
        dot(axis, endpoint, 0.065, PALETTE["student"], PALETTE["student_edge"])
        text(axis, (0.82, 1.13), r"$\hat{\mathbf{h}}=\widehat{\alpha\mathbf{h}}$",
             15, PALETTE["student_edge"])
    else:
        short_endpoint, long_endpoint = [polar(magnitude, feature_direction)
                                         for magnitude in ACI_ILLUSTRATION["feature_norms"]]
        arrow(axis, (0, 0), short_endpoint, color=PALETTE["student_edge"], width=1.5)
        arrow(axis, short_endpoint, long_endpoint, color=PALETTE["student_edge"], dashed=True)
        dot(axis, short_endpoint, 0.055, PALETTE["student"], PALETTE["student_edge"])
        dot(axis, long_endpoint, 0.055, PALETTE["student"], PALETTE["student_edge"])
        text(axis, short_endpoint + (-0.20, 0.20), r"$\mathbf{h}$", 15, PALETTE["student_edge"])
        text(axis, long_endpoint + (0.28, 0.14), r"$\alpha\mathbf{h}$", 15, PALETTE["student_edge"])
    axis.add_patch(Arc((0, 0), 0.63, 0.63, theta1=ACI_ILLUSTRATION["weight_angles_degrees"][0],
                       theta2=feature_direction, color=PALETTE["muted"], linewidth=1.0))
    text(axis, polar(0.48, 30), r"$\theta_1$", 14)
    dot(axis, (0, 0), 0.017, PALETTE["muted"])


def draw_aci_panel(figure, axis, canvas_width=CANVAS_WIDTH):
    frame(axis, (0.25, 0.25, 6.65, 7.9), PALETTE["student_edge"])
    text(axis, (3.575, 7.70), "Angular Coordinate Interface (ACI)",
         18, PALETTE["student_edge"], fontweight="bold")
    text(axis, (3.575, 7.20), "Replace the student readout; keep learnable directions", 13)
    text(axis, (3.575, 6.65),
         r"$z_{s,k}^{\mathrm{lin}}=\|\mathbf{h}_s\|_2\,"
         r"\|\mathbf{w}_{s,k}\|_2\cos\theta_{s,k}+b_{s,k}$", 20)
    axes = (
        add_panel_axis(figure, (0.52, 3.00, 3.0, 3.0), canvas_width=canvas_width),
        add_panel_axis(figure, (3.69, 3.00, 3.0, 3.0), canvas_width=canvas_width),
    )
    draw_readout_geometry(axes[0], normalized=False)
    draw_readout_geometry(axes[1], normalized=True)
    axes[0].set_title("(a) Affine readout", fontsize=14, pad=5)
    axes[1].set_title("(b) Angular readout", fontsize=14, pad=5)
    text(axis, (2.02, 2.91), "Norms affect scores", 13, PALETTE["muted"])
    text(axis, (5.19, 2.91), r"Same angles; unit norms", 13, PALETTE["student_edge"])
    block(axis, (0.67, 1.85, 2.64, 0.90), "L2-normalize both",
          r"$\hat{\mathbf{h}}_s,\;\hat{\mathbf{w}}_{s,k}$", PALETTE["student_edge"], fill="#F3F8FB")
    arrow(axis, (3.39, 2.32), (3.76, 2.32), color=PALETTE["student_edge"])
    block(axis, (3.84, 1.85, 2.64, 0.90), "Dot product; no bias",
          r"$z_{s,k}=s\,\hat{\mathbf{w}}_{s,k}^{\top}\hat{\mathbf{h}}_s$",
          PALETTE["student_edge"], fill="#F3F8FB")
    text(axis, (3.575, 1.51), r"Fixed shared $s=30$; class directions remain trainable", 13)
    text(axis, (3.575, 1.07), r"Blue: student features   |   Red: class weights   ($\alpha>0$)", 12.5)
    text(axis, (3.575, 0.62), "Student head only; no ETF constraint or extra cosine loss.",
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


def compose_aci_figure():
    figure = plt.figure(figsize=(14.0 * ACI_CANVAS_WIDTH / CANVAS_WIDTH,
                                  14.0 * CANVAS_HEIGHT / CANVAS_WIDTH))
    axis = add_panel_axis(figure, (0, 0, ACI_CANVAS_WIDTH, CANVAS_HEIGHT),
                          canvas_width=ACI_CANVAS_WIDTH)
    axis.set(xlim=(0, ACI_CANVAS_WIDTH), ylim=(0, CANVAS_HEIGHT))
    axis.axis("off")
    draw_aci_panel(figure, axis, canvas_width=ACI_CANVAS_WIDTH)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--aci-only", action="store_true", help="Render ACI without loading CIFAR-100.")
    parser.add_argument("--example", choices=("apple", "tiger", "pickup_truck"), default="apple")
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    font = configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(compose_aci_figure(), args.output_dir, "aci_cosine_head")
    aci_manifest = {
        "figure": "aci_cosine_head.pdf",
        "font": font,
        "illustration": ACI_ILLUSTRATION,
        "sources": ACI_SOURCES,
        "scope": "Same illustrative student directions before/after normalization, "
                 "not trained checkpoints or an empirical accuracy comparison. "
                 "Bias is an additive score term, not a feature-space displacement. "
                 "ETF and cosine attraction loss are not part of this head.",
        "feature_scaling": "Both positive scalar multiples map to one unit direction. "
                           "Raw feature norms may still change during training.",
        "embedded_rasters": 0,
    }
    (args.output_dir / "aci_cosine_head_manifest.json").write_text(
        json.dumps(aci_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output_dir.resolve() / 'aci_cosine_head.pdf'}")
    if args.aci_only:
        return
    examples = load_cifar_examples(args.data_root, args.download)
    example = next(item for item in examples if item["class"] == args.example)
    heatmaps = synthetic_maps(args.seed)
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
        "aci": "Angular Coordinate Interface; student feature and per-class weight normalization, "
               "no bias, fixed shared scale 30; freely trainable class directions, no ETF constraint.",
        "aci_illustration": ACI_ILLUSTRATION,
        "aci_sources": ACI_SOURCES,
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
