#!/usr/bin/env python3
"""Render three same-input color-depth alternatives for ACD mechanism review."""
import base64
import ctypes as C
import ctypes.util
import json
import math
from pathlib import Path
import struct
import zlib
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
OUT = HERE / "mechanism_figures"
D = json.loads((OUT / "same_input_pixel_force_probe.json").read_text())
# Display the first three recorded inputs; keep all source measurements intact.
EXAMPLES = D['examples'][:3]
# Leave a deliberate visual margin under the compact color key.
W, H = 792, 110 + 256 * len(EXAMPLES)
INK, MUTED, GRID = "#172B3A", "#60727F", "#DDE5EA"
METHODS = (("angular", "Angular"), ("scale_norm", "+ Scale / Norm"), ("raw", "Raw affine"))
PALETTE = ((225, 234, 236), (156, 181, 179), (184, 169, 132), (151, 107, 118))


def rgb_hex(rgb):
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(v))) for v in rgb)


def png_uri(rgb):
    # Explicit nearest-neighbor enlargement avoids renderer-dependent smoothing.
    rgb = [[pixel for pixel in row for _ in range(8)] for row in rgb for _ in range(8)]
    rows = b"".join(b"\0" + bytes(c for pixel in row for c in pixel) for row in rgb)
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    value = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(value).decode()


def force_values(tensor):
    # Tensor is C×H×W. Whole-map L2 normalization makes force maps comparable
    # without encoding the (different) absolute KD gradient norm.
    norm = math.sqrt(sum(value * value for channel in tensor for row in channel for value in row))
    return [[[value / norm for value in row] for row in channel] for channel in tensor]


FORCES = [{name: force_values(example["force_maps"][name]) for name, _ in METHODS} for example in EXAMPLES]


class Svg:
    def __init__(self, title, subtitle):
        stops = ''.join(f'<stop offset="{i/3*100:.2f}%" stop-color="{rgb_hex(rgb)}"/>' for i, rgb in enumerate(PALETTE))
        self.a = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">', f'<title>{title}</title><desc>{subtitle}</desc>', f'<defs><linearGradient id="morandi" x1="0%" x2="100%">{stops}</linearGradient></defs>', f'<rect width="{W}" height="{H}" fill="white"/>']
    def text(self, x, y, value, size=16, color=INK, weight=400, anchor="start"):
        self.a.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{value}</text>')
    def rect(self, x, y, w, h, fill, opacity=1, stroke="none", sw=0, r=0):
        self.a.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{r}" fill="{fill}" fill-opacity="{opacity:.3f}" stroke="{stroke}" stroke-width="{sw}"/>')
    def line(self, x1, y1, x2, y2, color=GRID, width=1):
        self.a.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"/>')
    def image(self, x, y, side):
        self.a.append(f'<image x="{x}" y="{y}" width="{side}" height="{side}" preserveAspectRatio="none" style="image-rendering:pixelated" xlink:href="{png_uri(self.rgb)}"/>')
    def save(self, stem):
        path = OUT / f"{stem}.svg"; path.write_text("\n".join(self.a + ["</svg>"]) + "\n"); ET.parse(path); return path


def disagreement():
    s = Svg("Same-input directional disagreement with angular KD", "Fixed CIFAR-100 examples from one release checkpoint. Each row repeats one input across angular, scale/norm, and raw parameterizations. Shared muted multihue scale encodes local 1-cosine deviation from 0 to 2; overlay opacity is constant.")
    # Same continuous low-saturation scale across all non-reference panels.
    def morandi(v):
        v = max(0, min(1, v / 2)) * 3
        index = min(2, int(v)); t = v - index
        pair = (PALETTE[index], PALETTE[index+1])
        return rgb_hex(tuple(pair[0][k] * (1 - t) + pair[1][k] * t for k in range(3)))
    # Adjacent panels deliberately share their inner edges: comparison should
    # read as one continuous triptych rather than three isolated images.
    lefts = (12, 268, 524); side = 256
    top = tuple(12 + side * i for i in range(len(EXAMPLES)))
    bottom = 12 + side * len(EXAMPLES)
    # One outer frame encloses all displayed rows and three method columns.
    s.rect(lefts[0] - 1, top[0] - 1, side * 3 + 2, side * len(EXAMPLES) + 2,
           "#FAFAF9", 1, GRID, .8, 3)
    for row, example in enumerate(EXAMPLES):
        s.rgb = example["rgb"]
        reference = FORCES[row]["angular"]
        for col, (name, _) in enumerate(METHODS):
            x, y = lefts[col], top[row]
            s.image(x, y, side)
            force = FORCES[row][name]; cell = side / 32
            for i in range(32):
                for j in range(32):
                    if name == "angular": value = 0
                    else:
                        dot = sum(reference[c][i][j] * force[c][i][j] for c in range(3)); a = math.sqrt(sum(reference[c][i][j] ** 2 for c in range(3))); b = math.sqrt(sum(force[c][i][j] ** 2 for c in range(3)))
                        value = 1 - dot / (a * b + 1e-12)
                    # Keep small disagreements pale rather than invisible so the
                    # common Morandi scale reads as a literal angle magnitude.
                    if value > .015: s.rect(x + j * cell, y + i * cell, cell + .12, cell + .12, morandi(value), .30, "none", 0)
    # The compact color key is retained; shared-row inputs and selected examples
    # belong in the paper caption rather than in the artwork.
    for col, (_, label) in enumerate(METHODS): s.text(lefts[col] + side / 2, bottom+32, label, 17, INK, 700, "middle")
    s.text(156, bottom+65, "local force-angle difference", 13, MUTED)
    s.a.append(f'<rect x="350" y="{bottom+52}" width="206" height="12" rx="6" fill="url(#morandi)"/>')
    s.text(350, bottom+83, "0", 13, MUTED, 400, "middle"); s.text(453, bottom+83, "1", 13, MUTED, 400, "middle"); s.text(556, bottom+83, "2", 13, MUTED, 400, "middle")
    return s.save("same_input_force_directional_disagreement")


def export(svg):
    rsvg = C.CDLL(ctypes.util.find_library("rsvg-2")); cairo = C.CDLL(ctypes.util.find_library("cairo")); gobj = C.CDLL(ctypes.util.find_library("gobject-2.0")); P, I, Dbl = C.c_void_p, C.c_int, C.c_double
    def bind(lib, name, args, ret): fn = getattr(lib, name); fn.argtypes = args; fn.restype = ret; return fn
    load = bind(rsvg, "rsvg_handle_new_from_file", [C.c_char_p, C.POINTER(P)], P); render = bind(rsvg, "rsvg_handle_render_cairo", [P, P], I); png = bind(cairo, "cairo_image_surface_create", [I, I, I], P); pdf = bind(cairo, "cairo_pdf_surface_create", [C.c_char_p, Dbl, Dbl], P); context = bind(cairo, "cairo_create", [P], P); scale = bind(cairo, "cairo_scale", [P, Dbl, Dbl], None); write = bind(cairo, "cairo_surface_write_to_png", [P, C.c_char_p], I); destroy = bind(cairo, "cairo_destroy", [P], None); finish = bind(cairo, "cairo_surface_finish", [P], None); free = bind(cairo, "cairo_surface_destroy", [P], None); status = bind(cairo, "cairo_surface_status", [P], I); unref = bind(gobj, "g_object_unref", [P], None)
    error = P(); handle = load(str(svg).encode(), C.byref(error))
    for ext, factor in (("png", 2.0), ("pdf", .42)):
        output = svg.with_suffix("." + ext); surface = png(0, W * 2, H * 2) if ext == "png" else pdf(str(output).encode(), W * factor, H * factor); cr = context(surface); scale(cr, factor, factor); assert render(handle, cr)
        if ext == "png": assert not write(surface, str(output).encode())
        destroy(cr); finish(surface); assert not status(surface); free(surface)
    unref(handle)


if __name__ == "__main__":
    export(disagreement())
