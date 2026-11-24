#!/usr/bin/env python3
"""Render three point-based ACD evidence figures from completed status records.

All inputs are final-reloaded official-test Top-1 values.  No checkpoint is
loaded and no training/evaluation is initiated by this renderer.
Legacy outputs are isolated under legacy_top1 so they cannot overwrite the
current endpoint discrepancy scatter plots.
"""
import ctypes as C
import ctypes.util
import json
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "mechanism_figures" / "legacy_top1"
W, H = 1300, 650
INK, MUTED, GRID = "#172B3A", "#60727F", "#DCE5EA"
COLORS = {"linear": "#627381", "cosine": "#4F86B5", "relation": "#D38B45", "acd": "#2F8A70", "dkd": "#8A71AB", "nckd": "#B96878", "lskd": "#A68D69"}


def status(path):
    value = json.loads((ROOT / path).read_text())
    if value["status"] != "completed" or value["epoch"] != 240:
        raise RuntimeError(f"not a completed epoch-240 result: {path}")
    return float(value["test"]["top1"])


def records():
    interaction = "reproductions/acd_cnn_interaction_20261109/artifacts"
    ablation = {}
    for pair in ("vgg13_mv1", "vgg13_vgg8"):
        ablation[pair] = {}
        for arm in ("linear_kd", "cosine_kd", "cosine_relation_matched", "cosine_standardized_full"):
            ablation[pair][arm] = [status(f"{interaction}/{pair}__s{seed}__{arm}/status.json") for seed in (2102, 2103)]
    reviewer = "reproductions/cvpr_reviewer_minimal_ready_20261106/artifacts"
    external = {
        "Cosine KD": {"values": ablation["vgg13_vgg8"]["cosine_kd"], "kind": "cosine", "provenance": "interaction"},
        "DKD": {"values": [status(f"{reviewer}/vgg13_vgg8__s{seed}__cosine_dkd/status.json") for seed in (2102, 2103)], "kind": "dkd", "provenance": "reviewer"},
        "NCKD": {"values": [status(f"{reviewer}/vgg13_vgg8__s2101__cosine_nckd_matched/status.json")], "kind": "nckd", "provenance": "reviewer"},
        "LSKD": {"values": [status(f"{reviewer}/vgg13_vgg8__s2101__cosine_lskd_source_recipe/status.json")], "kind": "lskd", "provenance": "author-recipe"},
        "ACD": {"values": ablation["vgg13_vgg8"]["cosine_standardized_full"], "kind": "acd", "provenance": "interaction"},
    }
    plugin = {
        "AT": {"KL": status("reproductions/acd_at_res32x4_sv1_priority_20261111/artifacts/res32x4_sv1__s2101__at_cosine_kl/status.json"), "ACD": status("reproductions/acd_at_res32x4_sv1_priority_20261111/artifacts/res32x4_sv1__s2101__at_cosine_sicd/status.json")},
        "ReviewKD": {"KL": status("reproductions/acd_reviewkd_wrn_priority_20261110/artifacts/wrn402_wrn401__s2101__review_cosine_kl/status.json"), "ACD": status("reproductions/acd_reviewkd_wrn_priority_20261110/artifacts/wrn402_wrn401__s2101__review_cosine_sicd/status.json")},
    }
    return {"ablation": ablation, "external": external, "plugin": plugin}


class Svg:
    def __init__(self, title, desc):
        self.a = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">', f'<title>{escape(title)}</title><desc>{escape(desc)}</desc>', f'<rect width="{W}" height="{H}" fill="white"/>']
    def text(self, x, y, value, size=16, color=INK, weight=400, anchor="start"):
        self.a.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(str(value))}</text>')
    def line(self, x1, y1, x2, y2, color=GRID, width=1, dash=""):
        self.a.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    def circle(self, x, y, r, fill, stroke="white", sw=1.5, opacity=1):
        self.a.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"/>')
    def save(self, stem):
        p = OUT / f"{stem}.svg"; p.write_text("\n".join(self.a + ["</svg>"]) + "\n"); ET.parse(p); return p


def axis(s, left, top, width, height, lo, hi, ylabel):
    for i in range(6):
        value = lo + (hi - lo) * i / 5; y = top + height - (value - lo) / (hi - lo) * height
        s.line(left, y, left + width, y, GRID, .8); s.text(left - 12, y + 5, f"{value:.1f}", 12, MUTED, 400, "end")
    s.line(left, top, left, top + height, "#9EABB5", 1.1); s.line(left, top + height, left + width, top + height, "#9EABB5", 1.1)
    s.text(34, top + height / 2, ylabel, 15, INK, 400, "middle")
    def y(value): return top + height - (value - lo) / (hi - lo) * height
    return y


def ablation(data):
    s = Svg("ACD component ablation", "Official-test Top-1 from strictly paired two-seed interaction results. Every dot is one seed and larger dots are two-seed means.")
    s.text(60, 55, "ACD component ablation", 25, INK, 700); s.text(60, 81, "Strictly paired interaction study · points: seeds 2102/2103 · large marker: mean", 15, MUTED)
    labels = (("linear_kd", "Linear\nKD", "linear"), ("cosine_kd", "Cosine\nKD", "cosine"), ("cosine_relation_matched", "Cosine +\nrelation", "relation"), ("cosine_standardized_full", "Full\nACD", "acd"))
    for panel, (pair, title) in enumerate((("vgg13_mv1", "VGG13 → MobileNetV1"), ("vgg13_vgg8", "VGG13 → VGG8"))):
        left = 100 + panel * 620; top, pw, ph = 145, 420, 370; y = axis(s, left, top, pw, ph, 64, 77, "Top-1 (%)" if panel == 0 else "")
        s.text(left + pw / 2, 120, title, 18, INK, 700, "middle")
        for i, (arm, label, kind) in enumerate(labels):
            x = left + 58 + i * 104; values = data[pair][arm]
            for offset, value in zip((-7, 7), values): s.circle(x + offset, y(value), 5, COLORS[kind], "white", 1.2, .70)
            mean = sum(values) / len(values); s.circle(x, y(mean), 8, COLORS[kind]); s.text(x, y(mean) - 13, f"{mean:.2f}", 12, COLORS[kind], 700, "middle")
            first, second = label.split("\n"); s.text(x, top + ph + 31, first, 12, INK, 700, "middle"); s.text(x, top + ph + 48, second, 12, INK, 700, "middle")
    s.text(60, 600, "All arms share each pair's teacher, student, seed-specific initialization, data order, 240-epoch budget, and one final official-test evaluation.", 13, MUTED)
    return s.save("acd_component_ablation")


def external(data):
    s = Svg("Provisional logit-KD comparison", "Protocol-mixed VGG13 to VGG8 official-test snapshot. It is a descriptive inventory, not a fair benchmark.")
    s.text(60, 55, "Provisional logit-KD comparison", 25, INK, 700); s.text(60, 81, "VGG13 → VGG8 · protocol-mixed snapshot · do not use as a fair method ranking", 15, "#A45D5D", 700)
    left, top, pw, ph = 125, 140, 1050, 360; y = axis(s, left, top, pw, ph, 72, 77, "Official-test Top-1 (%)")
    for i, (method, row) in enumerate(data.items()):
        x = left + 110 + i * 225; vals = row["values"]; color = COLORS[row["kind"]]
        offsets = (0,) if len(vals) == 1 else (-8, 8)
        for offset, value in zip(offsets, vals): s.circle(x + offset, y(value), 5, color, "white", 1.2, .65)
        mean = sum(vals) / len(vals); s.circle(x, y(mean), 9, color); s.text(x, y(mean) - 15, f"{mean:.2f}", 13, color, 700, "middle")
        s.text(x, top + ph + 30, method, 15, INK, 700, "middle"); s.text(x, top + ph + 50, row["provenance"], 11, MUTED, 400, "middle")
    s.text(60, 585, "Filled large marker = available-seed mean; faint small markers = individual seeds. Provenance differs across methods (interaction / reviewer / author recipe).", 13, MUTED)
    s.text(60, 610, "This temporary figure is intentionally labelled provisional; a unified KD/DKD/LSKD/DIST/SKD/ACD protocol is required for a publishable comparison.", 13, "#A45D5D", 700)
    return s.save("acd_external_kd_provisional")


def plug_in(data):
    s = Svg("ACD as an output-loss replacement", "Paired one-seed pilot results. Lines retain the feature pipeline and replace cosine KL with cosine SICD.")
    s.text(60, 55, "ACD as an output-loss replacement", 25, INK, 700); s.text(60, 81, "One completed seed per pipeline · only the output loss changes: cosine KL → cosine SICD", 15, MUTED)
    left, top, pw, ph = 130, 145, 920, 350; y = axis(s, left, top, pw, ph, 74.5, 80, "Official-test Top-1 (%)")
    for i, (pipe, values) in enumerate(data.items()):
        x = left + 240 + i * 440; x1, x2 = x - 64, x + 64; a, b = values["KL"], values["ACD"]
        s.line(x1, y(a), x2, y(b), "#9CADB7", 2.2); s.circle(x1, y(a), 8, "#697A86"); s.circle(x2, y(b), 8, COLORS["acd"])
        s.text(x1, y(a) - 15, f"{a:.2f}", 13, "#53646F", 700, "middle"); s.text(x2, y(b) - 15, f"{b:.2f}", 13, COLORS["acd"], 700, "middle")
        s.text(x, top + ph + 36, pipe, 17, INK, 700, "middle"); s.text(x, top + ph + 57, f"Δ {b - a:+.2f} pp", 14, COLORS["acd"] if b >= a else "#B94D58", 700, "middle")
    s.circle(1090, 155, 6, "#697A86"); s.text(1102, 160, "cosine KL", 13, MUTED); s.circle(1090, 180, 6, COLORS["acd"]); s.text(1102, 185, "cosine SICD", 13, MUTED)
    s.text(60, 585, "AT uses ResNet32×4 → ShuffleNetV1; ReviewKD uses WRN40-2 → WRN40-1. These are paired pilots, not independent repeats or a universality claim.", 13, MUTED)
    return s.save("acd_plugin_replacement_pilots")


def export(svg):
    rsvg=C.CDLL(ctypes.util.find_library("rsvg-2")); cairo=C.CDLL(ctypes.util.find_library("cairo")); gobj=C.CDLL(ctypes.util.find_library("gobject-2.0")); P,I,D=C.c_void_p,C.c_int,C.c_double
    def bind(lib,n,args,ret): fn=getattr(lib,n);fn.argtypes=args;fn.restype=ret;return fn
    load=bind(rsvg,"rsvg_handle_new_from_file",[C.c_char_p,C.POINTER(P)],P); render=bind(rsvg,"rsvg_handle_render_cairo",[P,P],I); png=bind(cairo,"cairo_image_surface_create",[I,I,I],P); pdf=bind(cairo,"cairo_pdf_surface_create",[C.c_char_p,D,D],P); ctx=bind(cairo,"cairo_create",[P],P); scale=bind(cairo,"cairo_scale",[P,D,D],None); write=bind(cairo,"cairo_surface_write_to_png",[P,C.c_char_p],I); destroy=bind(cairo,"cairo_destroy",[P],None); finish=bind(cairo,"cairo_surface_finish",[P],None); free=bind(cairo,"cairo_surface_destroy",[P],None); status=bind(cairo,"cairo_surface_status",[P],I); unref=bind(gobj,"g_object_unref",[P],None)
    error=P(); handle=load(str(svg).encode(),C.byref(error))
    for ext,factor in (("png",2.0),("pdf",.42)):
        output=svg.with_suffix("."+ext); surface=png(0,W*2,H*2) if ext=="png" else pdf(str(output).encode(),W*factor,H*factor); cr=ctx(surface);scale(cr,factor,factor);assert render(handle,cr)
        if ext=="png":assert not write(surface,str(output).encode())
        destroy(cr);finish(surface);assert not status(surface);free(surface)
    unref(handle)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); data=records(); (OUT / "acd_evidence_figure_data.json").write_text(json.dumps(data, indent=2) + "\n")
    for fn in (ablation, external, plug_in): export(fn(data[fn.__name__] if fn.__name__ == "ablation" else data["external"] if fn.__name__ == "external" else data["plugin"]))
