"""Draw corrected ACI geometry and SICD diagrams with real CIFAR-100 inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, font_manager
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as DrawingPath


PALETTE = {
    "student": "#92BDD7", "student_edge": "#326884", "teacher": "#953F50",
    "ink": "#30353B", "muted": "#69737C", "grid": "#E8ECF0",
    "green": "#397560", "green_bg": "#F1F7F3", "line": "#D5DCE1",
}
GROUP_COLORS = ["#B55D67", "#487FA5", "#69946C", "#B28A42"]


def configure_style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next(name for name in ("Times New Roman", "STIXGeneral", "DejaVu Serif")
                    if name in available)
    plt.rcParams.update({
        "font.family": selected, "font.size": 12, "mathtext.fontset": "stix",
        "text.usetex": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.unicode_minus": False,
    })
    return selected


def text(axis, position, content, size=12, color=None, **options):
    defaults = {"ha": "center", "va": "center", "zorder": 9}
    defaults.update(options)
    return axis.text(*position, content, fontsize=size,
                     color=color or PALETTE["ink"], **defaults)


def arrow(axis, start, end, color=None, dashed=False, style="-|>", width=1.2):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11,
                            linewidth=width, color=color or PALETTE["muted"],
                            linestyle=(0, (3, 3)) if dashed else "-",
                            shrinkA=1.5, shrinkB=2, zorder=5)
    axis.add_patch(patch)
    return patch


def routed_arrow(axis, points, color=None, dashed=False):
    path = DrawingPath(points, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=11,
                            color=color or PALETTE["muted"], linewidth=1.1,
                            linestyle=(0, (3, 3)) if dashed else "-", zorder=5)
    axis.add_patch(patch)


def dot(axis, position, radius, color, edge="white"):
    axis.add_patch(Circle(position, radius, facecolor=color, edgecolor=edge,
                          linewidth=0.9, zorder=8))


def box(axis, bounds, title, detail="", color=None, fill="white", dark=False):
    left, bottom, width, height = bounds
    stroke = color or PALETTE["line"]
    axis.add_patch(FancyBboxPatch((left, bottom), width, height,
                                 boxstyle="round,pad=0.025,rounding_size=0.07",
                                 facecolor=PALETTE["ink"] if dark else fill,
                                 edgecolor=stroke, linewidth=1.1, zorder=3))
    foreground = "white" if dark else color or PALETTE["ink"]
    text(axis, (left + width / 2, bottom + height * (0.66 if detail else 0.5)),
         title, 12, foreground)
    if detail:
        text(axis, (left + width / 2, bottom + height * 0.27), detail, 10.5, foreground)


def polar(radius, degrees):
    angle = np.deg2rad(degrees)
    return radius * np.array([np.cos(angle), np.sin(angle)])


def angular_arrow(axis, radius, start, stop):
    angle_start, angle_stop = np.deg2rad([start, stop])
    first, last = polar(radius, start), polar(radius, stop)
    handle = radius * 4 / 3 * np.tan((angle_stop - angle_start) / 4)
    first_control = first + handle * np.array([-np.sin(angle_start), np.cos(angle_start)])
    last_control = last - handle * np.array([-np.sin(angle_stop), np.cos(angle_stop)])
    path = DrawingPath([first, first_control, last_control, last],
                       [DrawingPath.MOVETO] + [DrawingPath.CURVE4] * 3)
    axis.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=12,
                                   color=PALETTE["muted"], linewidth=1.3, zorder=5))


def save_figure(figure, directory, stem):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    for label in figure.findobj(matplotlib.text.Text):
        if label.get_visible() and label.get_text():
            extent = label.get_window_extent(renderer)
            if not (extent.x0 >= canvas.x0 - 1 and extent.y0 >= canvas.y0 - 1
                    and extent.x1 <= canvas.x1 + 1 and extent.y1 <= canvas.y1 + 1):
                raise RuntimeError(f"Text outside canvas in {stem}: {label.get_text()!r}")
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(directory / f"{stem}.{suffix}", dpi=220, facecolor="white")
    plt.close(figure)


def draw_aci_axes(axes):
    for axis, title in zip(axes, ("(a) Angular coordinates", "(b) Optional ETF initialization")):
        axis.set(xlim=(-1.28, 1.28), ylim=(-1.25, 1.27), aspect="equal")
        axis.set_xticks(np.linspace(-1, 1, 9))
        axis.set_yticks(np.linspace(-1, 1, 9))
        axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axis.set_axisbelow(True)
        axis.grid(color=PALETTE["grid"], linewidth=0.55)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.add_patch(Circle((0, 0), 1, fill=False, edgecolor="#8A939B", linewidth=1.1))
        arrow(axis, (-1.13, 0), (1.14, 0), color="#ADB5BD", width=0.65)
        arrow(axis, (0, -1.12), (0, 1.13), color="#ADB5BD", width=0.65)
        dot(axis, (0, 0), 0.017, PALETTE["muted"])
        axis.set_title(title, fontsize=13, pad=9)
    left, right = axes
    anchor = polar(1, 30)
    arrow(left, (0, 0), anchor)
    features = [(0.60, 78), (0.43, 62), (0.72, 104), (0.60, -12), (0.42, 8), (0.74, -30)]
    for radius, direction in features:
        dot(left, polar(radius, direction), 0.035, PALETTE["student"], PALETTE["student_edge"])
    for feature_index in (0, 2, 5):
        radius, direction = features[feature_index]
        arrow(left, polar(radius + 0.06, direction), polar(1, direction), dashed=True)
    arrow(left, (0, 0), polar(0.6, 78), dashed=True, style="-")
    angular_arrow(left, 0.60, 72, 34)
    angular_arrow(left, 0.74, -24, 27)
    dot(left, anchor, 0.066, PALETTE["teacher"])
    left.add_patch(Arc((0, 0), 0.61, 0.61, theta1=30, theta2=78,
                       color=PALETTE["muted"], linewidth=1.1))
    text(left, polar(0.41, 54), r"$\theta$", 17)
    text(left, (0.79, 0.68), r"Anchor $\hat{\mathbf{w}}_k$", 11, PALETTE["teacher"])
    dot(left, (-0.97, -1.15), 0.035, PALETTE["student"], PALETTE["student_edge"])
    text(left, (-0.86, -1.15), "Student", 10, ha="left")
    dot(left, (0.18, -1.15), 0.045, PALETTE["teacher"])
    text(left, (0.29, -1.15), "Class anchor", 10, ha="left")
    angles = (90, 210, 330)
    positions = [polar(1, angle) for angle in angles]
    right.add_patch(Polygon(positions, closed=True, fill=False,
                            edgecolor=PALETTE["muted"], linestyle=(0, (4, 3)), linewidth=1.1))
    for angle, position in zip(angles, positions):
        arrow(right, (0, 0), position, dashed=True, style="-")
        dot(right, position, 0.066, PALETTE["teacher"])
        right.add_patch(Arc((0, 0), 0.57, 0.57, theta1=angle, theta2=angle + 120,
                            color=PALETTE["muted"], linewidth=1))
        text(right, polar(0.43, angle + 60), r"$120^{\circ}$", 10.5)
    text(right, (0, -1.15), "Equiangular at initialization only", 10, PALETTE["teacher"])


def draw_aci(directory):
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 4.7))
    figure.subplots_adjust(left=0.05, right=0.96, bottom=0.10, top=0.74, wspace=0.13)
    figure.suptitle("Angular Coordinate Interface (ACI)", y=0.97, fontsize=18)
    figure.text(0.5, 0.855,
                r"$z_{s,k}=s\cos\theta_{s,k}=s\,\hat{\mathbf{w}}_{s,k}^{\top}\hat{\mathbf{h}}_s$",
                ha="center", va="center", fontsize=18)
    draw_aci_axes(axes)
    figure.text(0.5, 0.035,
                "Conceptual geometry: trainable directions; no separate cosine loss or ETF penalty is assumed.",
                ha="center", fontsize=9, color=PALETTE["muted"])
    save_figure(figure, directory, "aci_geometry_corrected")


def load_cifar_examples(data_root, download):
    from torchvision.datasets import CIFAR100

    try:
        dataset = CIFAR100(root=str(data_root), train=False, download=download)
    except RuntimeError as error:
        raise RuntimeError("CIFAR-100 is missing or invalid. Pass --download, or use "
                           "--data-root pointing to the parent of cifar-100-python.") from error
    targets = np.asarray(dataset.targets)
    examples = []
    for class_name in ("tiger", "apple", "pickup_truck"):
        class_index = dataset.class_to_idx[class_name]
        image_index = int(np.flatnonzero(targets == class_index)[0])
        pixels = np.asarray(dataset[image_index][0]).copy()
        examples.append({"class": class_name, "index": image_index, "pixels": pixels,
                         "sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
    return examples


def canvas(title, subtitle):
    figure, axis = plt.subplots(figsize=(12.8, 6.3))
    figure.subplots_adjust(left=0.015, right=0.99, bottom=0.025, top=0.86)
    axis.set(xlim=(0, 18), ylim=(0, 8.0))
    axis.axis("off")
    figure.suptitle(title, y=0.975, fontsize=21)
    figure.text(0.5, 0.905, subtitle, ha="center", va="center", fontsize=11,
                color=PALETTE["muted"])
    return figure, axis


def draw_inputs(axis, examples):
    centers = [6.25, 4.35, 2.45]
    for example, center in zip(examples, centers):
        image_axis = axis.inset_axes([0.24, center - 0.59, 1.18, 1.18], transform=axis.transData)
        image_axis.imshow(example["pixels"], interpolation="bicubic")
        image_axis.set(xticks=[], yticks=[])
        for spine in image_axis.spines.values():
            spine.set(edgecolor=PALETTE["line"], linewidth=1)
        text(axis, (0.83, center - 0.82), example["class"].replace("_", " "), 10.5)
        arrow(axis, (1.48, center), (1.93, center), width=0.9)
    text(axis, (0.88, 7.5), "Real CIFAR-100", 12)
    text(axis, (0.88, 7.17), "example batch", 10.5, PALETTE["muted"])
    arrow(axis, (1.99, 2.45), (1.99, 6.25), style="-", width=0.9)


def synthetic_maps(seed):
    generator = np.random.default_rng(seed)
    return [generator.normal(0, 0.85, (12, 12)) for _ in range(2)]


def draw_heatmap(axis, values, bounds, grouped=False):
    left, bottom, width, height = bounds
    normalizer = colors.Normalize(vmin=-2.2, vmax=2.2, clip=True)
    color_map = plt.get_cmap("coolwarm")
    row_count, column_count = values.shape
    for row_index in range(row_count):
        for column_index in range(column_count):
            axis.add_patch(Rectangle(
                (left + column_index * width / column_count,
                 bottom + (row_count - row_index - 1) * height / row_count),
                width / column_count, height / row_count,
                facecolor=color_map(normalizer(values[row_index, column_index])),
                edgecolor="none", linewidth=0, zorder=2,
            ))
    axis.add_patch(Rectangle((left, bottom), width, height, facecolor="none",
                             edgecolor=PALETTE["line"], linewidth=1.0, zorder=3))
    if grouped:
        for group_index, (column_index, row_index) in enumerate(((0, 1), (1, 1), (0, 0), (1, 0))):
            axis.add_patch(Rectangle(
                (left + column_index * width / 2, bottom + row_index * height / 2),
                width / 2, height / 2, facecolor="none",
                edgecolor=GROUP_COLORS[group_index], linewidth=2.0, zorder=4,
            ))
            text(axis, (left + (column_index + 0.5) * width / 2,
                        bottom + (row_index + 0.5) * height / 2),
                 f"Group {group_index + 1}", 10, GROUP_COLORS[group_index],
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.7})


def draw_sicd(directory, examples, heatmaps):
    figure, axis = canvas("SICD Training Pipeline",
                          "Scale-Invariant Conditional Distillation | original teacher head; angular student head")
    draw_inputs(axis, examples)
    rows = [(5.75, "Teacher Feature Map", PALETTE["teacher"]),
            (2.65, "Student Feature Map", PALETTE["student_edge"])]
    for row_index, (center, label, color) in enumerate(rows):
        arrow(axis, (1.99, center), (2.85, center))
        text(axis, (2.43, center + 0.38), "Backbone", 9.5, PALETTE["muted"])
        draw_heatmap(axis, heatmaps[row_index], (2.9, center - 0.85, 1.75, 1.7))
        text(axis, (3.77, center + 1.15), label, 12, color)
        text(axis, (3.77, center - 1.08), "synthetic illustration", 9.5, PALETTE["muted"])
        arrow(axis, (4.72, center), (5.45, center))
        text(axis, (5.07, center + 0.32), "Pool", 9.5, PALETTE["muted"])
        box(axis, (5.5, center - 0.57, 2.15, 1.14),
            "Original head" if row_index == 0 else "Cosine Head (ACI)",
            r"$z_t=W_t h_t+b_t$" if row_index == 0 else r"$z_{s,k}=s\cos\theta_{s,k}$",
            color=color, dark=True)
        text(axis, (6.57, center - 0.85), "frozen" if row_index == 0 else "trainable; no bias",
             10, color)
        arrow(axis, (7.7, center), (8.25, center))
        box(axis, (8.3, center - 0.57, 2.15, 1.14), "Standardize", r"$(z-\mu)/\sigma$")
        arrow(axis, (10.5, center), (11.05, center))
        box(axis, (11.1, center - 0.57, 2.55, 1.14), "Non-target condition",
            r"$p^{-},\;1-p_y$" if row_index == 0 else r"$q^{-}$", color=PALETTE["green"])
        text(axis, (10.75, center + 0.92), r"softmax$(\cdot/T)$; label $y$", 10)
    box(axis, (14.4, 3.75, 3.2, 1.35), "Conditional relation loss",
        r"$L_r=T^2(1-p_y)\,\mathrm{KL}(p^{-}\Vert q^{-})$",
        color=PALETTE["green"], fill=PALETTE["green_bg"])
    routed_arrow(axis, [(13.72, 5.75), (14.05, 5.75), (14.05, 4.77), (14.33, 4.77)])
    routed_arrow(axis, [(13.72, 2.65), (14.05, 2.65), (14.05, 4.05), (14.33, 4.05)])
    text(axis, (15.98, 5.56), r"$L_{\mathrm{SICD}}=a\,L_r$", 20, PALETTE["green"])
    box(axis, (14.4, 1.75, 3.2, 1.25), "Gradient-strength matching",
        r"$a=\mathrm{stopgrad}(\|\nabla L_{\mathrm{full}}\|/\|\nabla L_r\|)$")
    arrow(axis, (16, 3.7), (16, 3.06))
    text(axis, (16, 1.37), "Same-state standardized-full reference", 9.2, PALETTE["muted"])
    box(axis, (8.45, 0.25, 5.7, 0.82), "Student objective",
        r"$L=\lambda_{\mathrm{CE}}\,\mathrm{CE}(z_s,y)+\lambda_{\mathrm{KD}}\,aL_r$",
        color=PALETTE["green"], fill=PALETTE["green_bg"])
    routed_arrow(axis, [(16, 1.7), (16, 0.66), (14.22, 0.66)])
    routed_arrow(axis, [(7.95, 2.65), (7.95, 0.66), (8.38, 0.66)], color=PALETTE["student_edge"])
    text(axis, (2.7, 0.64), "No spatial patch groups. Only the student is updated.",
         9.5, PALETTE["muted"], ha="left")
    save_figure(figure, directory, "sicd_training_pipeline")


def draw_patch_demo(directory, examples, heatmaps):
    figure, axis = canvas("SICD Training Pipeline — Patch-Group Concept",
                          "Hypothetical layout only: patch groups and multiple cosine heads are NOT the implemented SICD")
    draw_inputs(axis, examples)
    for row_index, center in enumerate((5.75, 2.65)):
        role = "Teacher" if row_index == 0 else "Student"
        arrow(axis, (1.99, center), (2.95, center))
        text(axis, (4.1, center + 1.25), f"{role} Feature Map", 13)
        draw_heatmap(axis, heatmaps[row_index], (3.0, center - 1.0, 2.2, 2.0), grouped=True)
        text(axis, (4.1, center - 1.24), "synthetic illustration; 2 × 2 groups", 10, PALETTE["muted"])
        for group_index in range(4):
            head_center = center + 1.02 - 0.68 * group_index
            group_column = group_index % 2
            group_row = 1 - group_index // 2
            patch_center = (3.0 + (group_column + 0.5) * 1.1,
                            center - 1.0 + (group_row + 0.5) * 1.0)
            arrow(axis, patch_center, (6.95, head_center), color=GROUP_COLORS[group_index], width=1.0)
            box(axis, (7.0, head_center - 0.255, 2.3, 0.51), f"Cosine Head {group_index + 1}",
                dark=True, color=GROUP_COLORS[group_index])
            routed_arrow(axis, [(9.38, head_center), (10.2, head_center),
                                (10.2, center), (11.05, center)], color=GROUP_COLORS[group_index])
        box(axis, (11.1, center - 0.55, 2.6, 1.10), "Group predictions",
            "illustrative aggregation")
    box(axis, (14.55, 3.65, 3.0, 1.45), "Illustrative loss stage",
        "no implemented objective asserted", color=PALETTE["green"], fill=PALETTE["green_bg"])
    routed_arrow(axis, [(13.76, 5.75), (14.1, 5.75), (14.1, 4.7), (14.48, 4.7)])
    routed_arrow(axis, [(13.76, 2.65), (14.1, 2.65), (14.1, 4.05), (14.48, 4.05)])
    text(axis, (16.05, 5.6), r"$L_{\mathrm{SICD}}$" + " (placeholder)", 18, PALETTE["green"])
    text(axis, (9, 0.5), "Real input photographs; simulated feature maps. This is a layout mock-up, not experimental evidence.",
         11, PALETTE["muted"])
    save_figure(figure, directory, "sicd_patch_group_concept_only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-100 if missing.")
    parser.add_argument("--patch-demo", action="store_true", help="Also render the explicitly hypothetical patch-group design.")
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = configure_style()
    examples = load_cifar_examples(args.data_root, args.download)
    heatmaps = synthetic_maps(args.seed)
    draw_aci(args.output_dir)
    draw_sicd(args.output_dir, examples, heatmaps)
    if args.patch_demo:
        draw_patch_demo(args.output_dir, examples, heatmaps)
    manifest = {
        "font": font, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "dataset": "CIFAR-100 official test split", "input_interpolation": "bicubic",
        "selection": "First test-set occurrence of each requested class; no output-based selection.",
        "examples": [{key: value for key, value in example.items() if key != "pixels"} for example in examples],
        "heatmaps": {"synthetic": True, "seed": args.seed, "arrays": [values.tolist() for values in heatmaps]},
        "aci": "Angular Coordinate Interface; optional 3-class ETF initialization is not an added loss.",
        "sicd": "Original teacher head; student ACI; all-class standardization; weighted non-target conditional KL; detached norm matching.",
        "patch_demo_rendered": args.patch_demo,
        "limitations": ["No model inference, training, or measured activations.",
                        "Photographs are raster images embedded in otherwise vector PDF/SVG diagrams.",
                        "Patch-group diagram is hypothetical and is not the implemented SICD.",
                        "Positive scale invariance is subject to the standard-deviation numerical floor.",
                        "All-class standardization and teacher weighting retain target dependence."],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Figures and provenance saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
