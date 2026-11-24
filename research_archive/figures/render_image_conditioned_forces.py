#!/usr/bin/env python3
"""Render an editable image-conditioned, unit-force comparison panel."""
import base64
import ctypes as C
import ctypes.util
import json
from pathlib import Path
import struct
import zlib
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
OUT = HERE / "mechanism_figures"
DATA = json.loads((OUT / "image_conditioned_force_probe.json").read_text())
W, H = 1500, 610
INK, MUTED, GRID = "#172B3A", "#61727E", "#DCE3E8"
ANGULAR, SCALE, RAW = "#8CA0AA", "#E2A23A", "#B94D58"


def png_data_uri(rgb):
    """Encode the 32×32 uint8 CIFAR RGB array without a plotting dependency."""
    rows = b"".join(b"\0" + bytes(channel for pixel in row for channel in pixel) for row in rgb)
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    content = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(content).decode()


class Svg:
    def __init__(self):
        self.a = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">', '<title>Image-conditioned normalized KD-force directions</title><desc>Three fixed-quantile CIFAR-100 examples. Per card, angular, scale/norm, and raw KD feature forces are normalized and drawn in a pair-local plane.</desc>', '<rect width="1500" height="610" fill="white"/>']
    def text(self, x, y, value, size=16, color=INK, weight=400, anchor="start"):
        self.a.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{value}</text>')
    def line(self, x1, y1, x2, y2, color=GRID, width=1.5, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ''
        self.a.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{extra}/>')
    def rect(self, x, y, width, height, fill="white", stroke="none", sw=1, radius=12):
        self.a.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def circle(self, x, y, r, fill, stroke="none", sw=0):
        self.a.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def image(self, x, y, side, rgb):
        uri = png_data_uri(rgb)
        self.a.append(f'<image x="{x}" y="{y}" width="{side}" height="{side}" image-rendering="pixelated" preserveAspectRatio="none" xlink:href="{uri}"/>')
    def arrow(self, ox, oy, unit, color, width, endpoint):
        x, y = ox + 112 * unit["x"], oy - 112 * unit["y"]
        self.line(ox, oy, x, y, color, width)
        self.circle(x, y, endpoint, color, "white", 1.5)
        return x, y
    def save(self):
        path = OUT / "image_conditioned_force_compass.svg"
        path.write_text("\n".join(self.a + ["</svg>"]) + "\n")
        ET.parse(path)
        return path


def render():
    s = Svg()
    s.text(54, 56, "Fixed input, three normalized KD forces", 25, INK, 700)
    s.text(54, 84, "Each card uses one real CIFAR-100 image; arrow length is fixed so only the force direction remains.", 15, MUTED)
    legend = (("angular", ANGULAR), ("+ scale/norm", SCALE), ("raw affine", RAW))
    for i, (label, color) in enumerate(legend):
        x = 940 + i * 175
        s.line(x, 55, x + 24, 55, color, 4)
        s.circle(x + 24, 55, 4.5, color, "white", 1)
        s.text(x + 35, 61, label, 15, MUTED)

    for card, example in enumerate(DATA["examples"]):
        left = 53 + card * 480
        s.rect(left, 120, 430, 425, "#FBFCFD", GRID, 1)
        s.text(left + 26, 155, f"q={example['quantile']:.2f}", 17, INK, 700)
        s.text(left + 26, 181, f"{example['class_name']}  ·  image {example['index']}", 15, MUTED)
        s.image(left + 26, 210, 142, example["rgb"])
        s.rect(left + 26, 210, 142, 142, "none", "white", 1, 0)
        s.text(left + 97, 380, "input", 14, MUTED, 400, "middle")

        ox, oy = left + 300, 350
        s.line(ox - 132, oy, ox + 132, oy, GRID, 1, "4 5")
        s.line(ox, oy - 132, ox, oy + 132, GRID, 1, "4 5")
        s.circle(ox, oy, 4, INK)
        coords = example["unit_force_coordinates"]
        # Baseline is always horizontal by construction, enabling direct visual comparison across cards.
        s.arrow(ox, oy, coords["angular"], ANGULAR, 3, 6)
        s.arrow(ox, oy, coords["scale_norm"], SCALE, 4, 6)
        rx, ry = s.arrow(ox, oy, coords["raw"], RAW, 5, 6.5)
        import math
        angle = math.degrees(math.atan2(coords["raw"]["y"], coords["raw"]["x"]))
        s.text(left + 300, 505, f"raw rotation  {angle:.0f}°", 16, RAW, 700, "middle")
        s.text(left + 300, 529, f"1−cos = {example['raw_vs_angular_deviation']:.3f}", 13, MUTED, 400, "middle")
    s.line(54, 576, 1445, 576, GRID, 1)
    s.text(54, 600, "Selection: nearest q25/q50/q75 of raw-versus-angular force deviation among the fixed 128-image probe; not selected by accuracy or visual effect.", 13, MUTED)
    return s.save()


def export(svg):
    rsvg = C.CDLL(ctypes.util.find_library("rsvg-2")); cairo = C.CDLL(ctypes.util.find_library("cairo")); gobj = C.CDLL(ctypes.util.find_library("gobject-2.0")); P, I, D = C.c_void_p, C.c_int, C.c_double
    def bind(lib, name, args, ret):
        fn = getattr(lib, name); fn.argtypes = args; fn.restype = ret; return fn
    load = bind(rsvg, "rsvg_handle_new_from_file", [C.c_char_p, C.POINTER(P)], P); draw = bind(rsvg, "rsvg_handle_render_cairo", [P, P], I); png = bind(cairo, "cairo_image_surface_create", [I, I, I], P); pdf = bind(cairo, "cairo_pdf_surface_create", [C.c_char_p, D, D], P); context = bind(cairo, "cairo_create", [P], P); scale = bind(cairo, "cairo_scale", [P, D, D], None); write = bind(cairo, "cairo_surface_write_to_png", [P, C.c_char_p], I); destroy = bind(cairo, "cairo_destroy", [P], None); finish = bind(cairo, "cairo_surface_finish", [P], None); free = bind(cairo, "cairo_surface_destroy", [P], None); status = bind(cairo, "cairo_surface_status", [P], I); unref = bind(gobj, "g_object_unref", [P], None)
    error = P(); handle = load(str(svg).encode(), C.byref(error))
    for ext, factor in (("png", 2.0), ("pdf", 0.42)):
        output = svg.with_suffix("." + ext)
        surface = png(0, int(W * 2), int(H * 2)) if ext == "png" else pdf(str(output).encode(), W * factor, H * factor)
        cr = context(surface); scale(cr, factor, factor); assert draw(handle, cr)
        if ext == "png": assert not write(surface, str(output).encode())
        destroy(cr); finish(surface); assert not status(surface); free(surface)
    unref(handle)


if __name__ == "__main__":
    export(render())
