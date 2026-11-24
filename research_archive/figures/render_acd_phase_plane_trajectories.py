#!/usr/bin/env python3
"""Render the three read-only ACD endpoint phase-plane comparisons."""
import ctypes as C
import ctypes.util
import json
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
OUT = HERE / "mechanism_figures"
W, H = 1200, 690
INK, MUTED, GRID = "#172B3A", "#627381", "#DCE5EA"
PALETTES = {
    "component": ("#667788", "#4F86B5", "#D38B45", "#2F8A70"),
    "external": ("#4F86B5", "#8A71AB", "#B96878", "#A68D69", "#2F8A70"),
    "plugin": ("#4F86B5", "#2F8A70", "#8A71AB", "#B96878"),
}


def endpoint(row):
    points = [point for point in row["rows"] if point["epoch"] == 240]
    if len(points) != 1:
        raise ValueError(f"Expected one epoch-240 point for {row['label']}")
    return points[0]


class Svg:
    def __init__(self, title, description):
        self.a = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">',
                  f'<title>{escape(title)}</title><desc>{escape(description)}</desc>',
                  f'<rect width="{W}" height="{H}" fill="white"/>']
    def line(self, x1, y1, x2, y2, color=GRID, width=1, dash=""):
        self.a.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    def circle(self, x, y, r, fill, stroke="white", width=1.6):
        self.a.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')
    def text(self, x, y, text, size=15, color=INK, weight=400, anchor="start"):
        self.a.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(str(text))}</text>')
    def save(self, stem):
        path = OUT / f"{stem}.svg"
        path.write_text("\n".join(self.a + ["</svg>"]) + "\n")
        ET.parse(path)
        return path


def limits(data):
    # One completed arm is one point: the final (epoch-240) checkpoint.
    xs = [endpoint(row)["conditional_tv_standardized"] for row in data["trajectories"]]
    ys = [endpoint(row)["raw_kl"] for row in data["trajectories"]]
    def padded(values, floor=0):
        lo, hi = min(values), max(values)
        margin = max(.005, (hi - lo) * .10)
        return max(floor, lo - margin), hi + margin
    return padded(xs), padded(ys)


def render(filename, output, palette, title, subtitle):
    data = json.loads((OUT / filename).read_text())
    (xmin, xmax), (ymin, ymax) = limits(data)
    left, top, width, height = 150, 80, 760, 490
    s = Svg(title, "Each colored point is one completed arm's epoch-240 location on two fixed-probe discrepancy coordinates.")
    s.text(left, 38, title, 25, INK, 700)
    s.text(left, 62, subtitle, 14, MUTED)
    def xy(x, y):
        return left + (x - xmin) / (xmax - xmin) * width, top + height - (y - ymin) / (ymax - ymin) * height
    for i in range(6):
        xval = xmin + (xmax - xmin) * i / 5
        yval = ymin + (ymax - ymin) * i / 5
        px, py = xy(xval, ymin)[0], xy(xmin, yval)[1]
        s.line(px, top, px, top + height)
        s.line(left, py, left + width, py)
        s.text(px, top + height + 23, f"{xval:.3f}", 12, MUTED, 400, "middle")
        s.text(left - 12, py + 4, f"{yval:.2f}", 12, MUTED, 400, "end")
    s.line(left, top, left, top + height, "#9EABB5", 1.2)
    s.line(left, top + height, left + width, top + height, "#9EABB5", 1.2)
    s.text(left + width / 2, top + height + 53, "Standardized non-target relation discrepancy ↓", 16, INK, 400, "middle")
    s.a.append(f'<text x="44" y="{top + height / 2:.2f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="16" text-anchor="middle" fill="{INK}" transform="rotate(-90 44 {top + height / 2:.2f})">Raw-logit KL discrepancy ↓</text>')
    legend_x, legend_y = 950, 125
    for index, row in enumerate(data["trajectories"]):
        color = palette[index]
        point = endpoint(row)
        x, y = xy(point["conditional_tv_standardized"], point["raw_kl"])
        s.circle(x, y, 9, color)
        ly = legend_y + index * 33
        s.circle(legend_x + 11, ly, 5, color)
        s.text(legend_x + 33, ly + 5, row["label"], 14, INK)
    s.text(150, 672, "Each point is the final epoch-240 checkpoint. Coordinates are fixed-probe diagnostics, not Top-1.", 13, MUTED)
    return s.save(output)


def export(svg):
    rsvg = C.CDLL(ctypes.util.find_library("rsvg-2"))
    cairo = C.CDLL(ctypes.util.find_library("cairo"))
    gobj = C.CDLL(ctypes.util.find_library("gobject-2.0"))
    ptr, integer, double = C.c_void_p, C.c_int, C.c_double
    def bind(lib, name, args, result):
        fn = getattr(lib, name); fn.argtypes = args; fn.restype = result; return fn
    load = bind(rsvg, "rsvg_handle_new_from_file", [C.c_char_p, C.POINTER(ptr)], ptr)
    render = bind(rsvg, "rsvg_handle_render_cairo", [ptr, ptr], integer)
    image = bind(cairo, "cairo_image_surface_create", [integer, integer, integer], ptr)
    pdf = bind(cairo, "cairo_pdf_surface_create", [C.c_char_p, double, double], ptr)
    context = bind(cairo, "cairo_create", [ptr], ptr); scale = bind(cairo, "cairo_scale", [ptr, double, double], None)
    write = bind(cairo, "cairo_surface_write_to_png", [ptr, C.c_char_p], integer); destroy = bind(cairo, "cairo_destroy", [ptr], None)
    finish = bind(cairo, "cairo_surface_finish", [ptr], None); free = bind(cairo, "cairo_surface_destroy", [ptr], None)
    status = bind(cairo, "cairo_surface_status", [ptr], integer); unref = bind(gobj, "g_object_unref", [ptr], None)
    error = ptr(); handle = load(str(svg).encode(), C.byref(error))
    for ext, factor in (("png", 2.0), ("pdf", .42)):
        result = svg.with_suffix("." + ext)
        surface = image(0, W * 2, H * 2) if ext == "png" else pdf(str(result).encode(), W * factor, H * factor)
        ctx = context(surface); scale(ctx, factor, factor); assert render(handle, ctx)
        if ext == "png": assert not write(surface, str(result).encode())
        destroy(ctx); finish(surface); assert not status(surface); free(surface)
    unref(handle)


def main():
    outputs = (
        ("acd_component_ablation_phase_plane.json", "acd_component_ablation", "component", "ACD component comparison", "VGG13 → MobileNetV1 · seed 2102 · same initialization and data order"),
        ("acd_external_kd_phase_plane.json", "acd_external_kd_provisional", "external", "Output-objective comparison", "VGG13 → VGG8 · protocol-mixed descriptive snapshot; not a fair ranking"),
        ("acd_plugin_replacement_phase_plane.json", "acd_plugin_replacement_pilots", "plugin", "Output-loss replacement comparison", "One paired seed per feature pipeline · AT and ReviewKD have distinct teacher–student pairs"),
    )
    for filename, output, palette, title, subtitle in outputs:
        export(render(filename, output, PALETTES[palette], title, subtitle))


if __name__ == "__main__":
    main()
