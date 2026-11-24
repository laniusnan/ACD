#!/usr/bin/env python3
"""Render editable, data-backed SVG figures without training or model inference.

Default: render the committed evidence snapshot (Python standard library).
--collect: explicitly refresh selected completed results from local ledgers.
--raster: also export PNG/PDF using system librsvg and cairo, without a browser.
No training source, checkpoint, queue or original evidence artifact is changed.
"""
import argparse
import ctypes as C
import ctypes.util
import hashlib
from html import escape
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "evidence_figures"
INK, MUTED, LINE = "#172B3A", "#627381", "#DCE3E8"
ORANGE, BLUE, GREEN, RED = "#D88A2D", "#357DB2", "#23856D", "#B94D58"
PURPLE, GRAY = "#8B71AB", "#7B8793"


def collect():
    sources = {}

    def load(rel):
        p = ROOT / rel
        raw = p.read_bytes()
        sources[rel] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    probe = load("paper_assets/acd_figures/artifacts/teaser_probe.json")
    assert probe["status"] == "completed" and probe["probe"]["n"] == 256
    study = "cvpr_angular_interface_interaction_20261109"
    parent = "cvpr_vit_rkd_three_seed_extension_recovery_20261109"
    cache = {}

    def result(study_name, name, endpoint, samples):
        if study_name not in cache:
            cache[study_name] = load(f"reproductions/{study_name}/artifacts/state.json")
        job = cache[study_name]["jobs"][name]
        r = job["result"]
        assert job["status"] == "completed", name
        assert r["epoch"] == 240 and r["reload_verified"], name
        assert r[endpoint]["samples"] == samples, name
        assert r["official_test_evaluations"] == (1 if endpoint == "test" else 0)
        return {"job": name, "study": study_name, "epoch": r["epoch"],
                "reload_verified": r["reload_verified"], "endpoint": endpoint,
                "samples": samples, "top1": r[endpoint]["top1"],
                "checkpoint_sha256": r.get("checkpoint_sha256")}

    interaction = {}
    for seed in (2102, 2103):
        interaction[str(seed)] = {}
        for arm in ("linear_ce_only", "cosine_ce_only", "linear_kd", "cosine_kd"):
            src = parent if seed == 2102 and arm.endswith("_kd") else study
            interaction[str(seed)][arm] = result(src, f"vit_res8x4__s{seed}__{arm}", "test", 10000)
    interventions = {}
    for label, study_name, arm in [
        ("ordinary", "amplitude_decoupling_20261101", "cosine_kd"),
        ("full", "amplitude_decoupling_20261101", "cosine_standardized_kd"),
        ("binary", "signal_component_causality_loggingfix_20261102", "standardized_binary_matched"),
        ("relation", "signal_component_causality_loggingfix_20261102", "standardized_relation_matched"),
        ("strength", "standardization_mechanism_20261101", "standardized_matched_backbone"),
    ]:
        interventions[label] = result(study_name, f"vgg13_mv1__s2101__{arm}", "validation", 5000)
    for name in (study, "signal_component_causality_loggingfix_20261102", "standardization_mechanism_20261101"):
        load(f"reproductions/{name}/protocol.json")
    for name in ("study.py", "losses.py"):
        p = ROOT / f"reproductions/{study}/{name}"
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"snapshot_date": "2026-11-10", "synthetic": False,
            "scope": "Existing evidence only; no training, inference or new gradient diagnostic.",
            "probe": probe, "interaction": interaction, "interventions": interventions,
            "sources_sha256": sources,
            "limitations": ["Probe: one pair/seed, first 256 official-test images.",
                "Interaction: two seeds, ViT-T to ResNet8x4; negative KD-specific head interaction.",
                "Components: one seed, VGG13 to MobileNetV1, 45k/5k student split; teacher saw holdout.",
                "Component and full-strength controls use different gradient-matching references.",
                "No evidence here establishes gradient conflict or universal geometric compatibility."]}


class SVG:
    def __init__(self, title, subtitle, height=580):
        self.h = height
        self.items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height}" viewBox="0 0 1400 {height}" role="img">',
                      f'<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>',
                      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#627381"/></marker></defs>',
                      '<rect width="1400" height="100%" fill="white"/>']
        self.text(40, 43, title, 30, weight=700)
        self.text(40, 75, subtitle, 18, MUTED)

    def text(self, x, y, value, size=22, color=INK, weight=400, anchor="start"):
        self.items.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(str(value))}</text>')

    def lines(self, x, y, values, size=20, color=MUTED, step=29, anchor="start"):
        for i, v in enumerate(values):
            self.text(x, y + step*i, v, size, color, anchor=anchor)

    def rect(self, x, y, w, h, fill, stroke="none", radius=8):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def line(self, x1, y1, x2, y2, color=LINE, width=1.5, dash=False, arrow=False):
        extra = (' stroke-dasharray="6 5"' if dash else '') + (' marker-end="url(#arrow)"' if arrow else '')
        self.items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def circle(self, x, y, r, color):
        self.items.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    def path(self, pts, color=BLUE, width=2, arrow=False):
        d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts)
        extra = ' marker-end="url(#arrow)"' if arrow else ''
        self.items.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')

    def panel(self, x, title):
        self.text(x, 124, title, 23, weight=700)

    def footer(self, *lines):
        y = self.h - 22 - 24*(len(lines)-1)
        self.line(40, y-20, 1360, y-20)
        self.lines(40, y, lines, 17, MUTED, 24)

    def save(self, name):
        target = OUT / f"{name}.svg"
        content = "\n".join(self.items + ["</svg>"]) + "\n"
        ET.fromstring(content)
        target.write_text(content, encoding="utf-8")
        return target


def teaser(d):
    s = SVG("Fixed angular relations, different class supervision",
            "Checkpoint evidence  /  ViT-T → ResNet8×4  /  seed 2102  /  first 256 CIFAR-100 test images")
    s.panel(40, "(a) Linear-logit components")
    s.rect(40, 153, 382, 227, "#F5F7F9")
    s.text(65, 197, "zₖ = ‖h‖ ‖wₖ‖ cos θₖ + bₖ", 26, weight=600)
    for y, title, note, color in [(244, "Direction", "cos θₖ", BLUE),
                                  (291, "Magnitude", "‖h‖ ‖wₖ‖", ORANGE),
                                  (338, "Class bias", "bₖ", PURPLE)]:
        s.circle(69, y-6, 5, color)
        s.text(85, y, title, 21, color, 600)
        s.text(388, y, note, 22, anchor="end")
    s.lines(42, 414, ["Same checkpoint and input;", "introduce radial terms, then bias.", "No teacher–student feature alignment."], 18)
    s.panel(470, "(b) Leading alternative changes")
    s.text(470, 155, "Non-target top-1 change (%)", 18, MUTED)
    base, x0, scale = 407, 510, 6.8
    s.line(x0, 180, x0, base)
    s.line(x0, base, 885, base)
    for v in (0, 10, 20, 30):
        y = base-scale*v
        s.text(x0-12, y+6, v, 17, MUTED, anchor="end")
        s.line(x0-4, y, x0, y)
    order = d["probe"]["ordering"]
    for xc, prefix, label in [(602, "teacher", "Teacher"), (790, "linear_student", "Student")]:
        for offset, term, color in [(-32, "magnitude", ORANGE), (32, "bias", PURPLE)]:
            v = order[f"{prefix}_{term}"]["top1_changed_pct"]
            x = xc+offset
            s.rect(x-23, base-v*scale, 46, v*scale, color, radius=2)
            s.text(x, base-v*scale-12, f"{v:.1f}", 21, color, 700, "middle")
        s.text(xc, 438, label, 20, anchor="middle")
    s.circle(538, 474, 5, ORANGE); s.text(552, 480, "+ Magnitude", 17)
    s.circle(731, 474, 5, PURPLE); s.text(745, 480, "+ Bias", 17)
    s.panel(948, "(c) Top-5 set retention")
    s.text(948, 155, "Mean non-target top-5 overlap (%)", 18, MUTED)
    for i, (label, key, color) in enumerate([
        ("Teacher · magnitude", "teacher_magnitude", ORANGE),
        ("Teacher · bias", "teacher_bias", PURPLE),
        ("Student · magnitude", "linear_student_magnitude", ORANGE),
        ("Student · bias", "linear_student_bias", PURPLE)]):
        y = 211+i*65
        v = order[key]["mean_top5_overlap_pct"]
        s.text(948, y, label, 18)
        s.rect(948, y+10, 310, 10, "#EAEFF2", radius=3)
        s.rect(948, y+10, 310*v/100, 10, color, radius=3)
        s.text(1345, y+22, f"{v:.1f}", 21, color, 600, "end")
    s.text(948, 486, "Full track = 100% retained", 17, MUTED)
    s.footer("Magnitude: angular → dot-product scores. Bias: dot-product → full logits. Comparisons are sequential, not additive effects.",
             "Observed sensitivity does not establish that angular ordering is correct, or that KD gradients conflict.")
    return s.save("teaser_parameterization")


def interface(d):
    s = SVG("An angular interface is a structural choice—not a guaranteed KD gain",
            "Analysis  /  separate forward-coordinate properties from the head × supervision interaction", 650)
    s.panel(40, "(a) Effective student classifier coordinates")
    p = d["probe"]["classifier_parameters"]
    vals = p["linear_weight_norms"]
    x0, y0, w, h = 90, 370, 520, 170
    s.text(45, 164, "Linear weights vs. normalized cosine weights", 19, MUTED)
    s.text(90, 191, "Weight norm", 17, MUTED)
    for v in (1, 1.5, 2, 2.5):
        y = y0-(v-.8)/1.7*h
        s.text(x0-14, y+5, f"{v:g}", 17, MUTED, anchor="end")
        s.line(x0-4, y, x0, y)
    s.line(x0, y0-h, x0, y0)
    s.line(x0, y0, x0+w, y0)
    s.path([(x0+i*w/99, y0-(v-.8)/1.7*h) for i, v in enumerate(vals)], GRAY, 1.7)
    ya = y0-(1-.8)/1.7*h
    s.line(x0, ya, x0+w, ya, BLUE, 3)
    s.text(620, ya-9, "ACI: 1", 18, BLUE, anchor="end")
    for k in (0, 49, 99):
        s.text(x0+k*w/99, y0+27, k, 17, MUTED, anchor="middle")
    s.text(350, y0+55, "Class index (not sorted)", 18, MUTED, anchor="middle")
    s.rect(40, 450, 590, 90, "#F1F6FA")
    s.text(60, 480, "ACI: zₛ,ₖ = γ cos θₛ,ₖ", 24, BLUE, 600)
    s.text(60, 513, "Effective weight norm = 1; student bias = 0.", 19)
    s.text(44, 568, "γ = 30 is separate; this is not a teacher-head change.", 18, MUTED)
    s.panel(722, "(b) Is the head benefit specific to KD?")
    s.text(722, 164, "Cosine − linear accuracy (percentage points)", 19, MUTED)
    x0, y0, scale = 770, 405, 29
    s.line(x0, 199, x0, y0); s.line(x0, y0, 1340, y0)
    for v in (0, 2, 4, 6):
        s.text(x0-14, y0-scale*v+6, v, 17, MUTED, anchor="end")
    for xc, seed in [(919, "2102"), (1200, "2103")]:
        r = d["interaction"][seed]
        ce = r["cosine_ce_only"]["top1"]-r["linear_ce_only"]["top1"]
        kd = r["cosine_kd"]["top1"]-r["linear_kd"]["top1"]
        for x, value, color in [(xc-40, ce, GRAY), (xc+40, kd, BLUE)]:
            s.rect(x-26, y0-scale*value, 52, scale*value, color, radius=2)
            s.text(x, y0-scale*value-13, f"+{value:.2f}", 22, color, 600, "middle")
        s.text(xc, 437, f"Seed {seed}", 20, anchor="middle")
        s.text(xc, 507, f"Interaction: {kd-ce:+.2f} pp", 22, RED, 600, "middle")
    s.circle(836, 468, 5, GRAY); s.text(851, 474, "CE only", 18)
    s.circle(1035, 468, 5, BLUE); s.text(1050, 474, "Ordinary KD", 18)
    s.text(722, 558, "Head gain is larger under CE in both seeds.", 21, RED)
    s.footer("ViT-T → ResNet8×4; 240 epochs; final reloaded 10,000-image test endpoints. CE/KD coefficients retain the registered recipe.",
             "Interaction = head gain under KD − head gain under CE. These two seeds do not support a KD-specific head advantage.")
    return s.save("analysis_angular_interface")


def conditional(d):
    s = SVG("Which supervision should be retained?",
            "Analysis  /  exact loss decomposition plus existing component interventions", 675)
    s.panel(40, "(a) Separate confidence from alternatives")
    s.rect(40, 155, 605, 82, "#F5F7F9")
    s.text(63, 187, "p = softmax(N(zₜ) / T),  q = softmax(N(zₛ) / T)", 20)
    s.text(63, 219, "N(z) = (z − mean(z)) / std(z)", 22, weight=600)
    s.text(48, 279, "KL(p ‖ q) = B + (1 − pᵧ) R", 29, weight=600)
    s.rect(40, 305, 286, 111, "#F5F0F8")
    s.rect(344, 305, 301, 111, "#EDF6F2")
    s.text(61, 339, "B · Target confidence", 21, PURPLE, 700)
    s.lines(61, 370, ["Bernoulli KL", "[pᵧ, 1 − pᵧ] vs. [qᵧ, 1 − qᵧ]"], 17)
    s.text(363, 339, "R · Non-target relation", 21, GREEN, 700)
    s.lines(363, 370, ["Conditional KL(p⁻ ‖ q⁻)", "p⁻ₖ = pₖ / (1 − pᵧ),  k ≠ y"], 18)
    s.text(47, 457, "SICD retains the weighted conditional term.", 22, GREEN, 600)
    s.lines(47, 492, ["Training includes T² and detached gradient matching.",
                           "Target information still enters through normalization", "and the teacher non-target mass (1 − pᵧ)."], 18)
    s.panel(708, "(b) Component value is an empirical question")
    s.lines(708, 158, ["VGG13 → MobileNetV1 · seed 2101 · cosine head", "45k train / 5k student holdout · epoch 240"], 18, step=27)
    x0, x1 = 1025, 1335
    def xp(v): return x0+(v-60)/16*(x1-x0)
    specs = [("Ordinary KD", "ordinary", BLUE), ("Full standardized", "full", GRAY),
             ("Confidence only *", "binary", PURPLE), ("Conditional only *", "relation", GREEN),
             ("Full + norm control †", "strength", INK)]
    for i, (label, key, color) in enumerate(specs):
        y = 252+i*56
        v = d["interventions"][key]["top1"]
        s.text(708, y+6, label, 20, color)
        s.circle(xp(v), y, 7, color)
        s.text(xp(v)+12, y+6, f"{v:.2f}", 19, color, 600)
    s.line(x0, 503, x1, 503)
    for v in (60, 65, 70, 75):
        s.line(xp(v), 503, xp(v), 510)
        s.text(xp(v), 535, v, 18, MUTED, anchor="middle")
    s.text((x0+x1)/2, 566, "Holdout Top-1 (%)", 19, MUTED, anchor="middle")
    s.footer("* Component gradient norm matched to full standardized KD. † Full standardized gradient norm matched to ordinary KD (different reference).",
             "Relation: +0.80 pp vs. full; −0.66 pp vs. the separate norm control. Teacher saw the holdout images. One seed; no significance claim.",
             "The 5k holdout is not the official test set. Results support a bounded component comparison, not universally harmful confidence.")
    return s.save("analysis_conditional_supervision")


def method(d):
    s = SVG("ACD: construct the interface, then select the supervision",
            "Method  /  student-side ACI + standardized conditional distillation  /  computational graph, not an empirical result", 710)
    s.rect(200, 308, 283, 183, "#F1F6FA")
    s.text(219, 338, "ACI · student interface", 21, BLUE, 700)
    s.rect(520, 134, 835, 357, "#F0F7F3")
    s.text(542, 164, "SICD · select scale-invariant conditional relations", 23, GREEN, 700)

    def box(x, y, w, h, title, note, color=INK):
        s.rect(x, y, w, h, "white", LINE)
        s.text(x+w/2, y+33, title, 21, color, 600, "middle")
        if note: s.text(x+w/2, y+62, note, 17, MUTED, anchor="middle")

    box(40, 195, 135, 88, "Teacher", "frozen", ORANGE)
    box(222, 195, 240, 88, "Original classifier", "zₜ = Wₜhₜ + bₜ", ORANGE)
    box(550, 195, 210, 88, "Standardize", "N(zₜ)", ORANGE)
    box(814, 195, 220, 88, "Condition on k ≠ y", "p⁻; retain 1 − pᵧ", ORANGE)
    box(40, 356, 135, 88, "Student", "trainable", BLUE)
    box(222, 356, 240, 88, "Angular classifier", "zₛ,ₖ = γ cos θₛ,ₖ", BLUE)
    box(550, 356, 210, 88, "Standardize", "N(zₛ)", BLUE)
    box(814, 356, 220, 88, "Condition on k ≠ y", "q⁻", BLUE)
    for y in (239, 400):
        for a, b in [(175, 222), (462, 550), (760, 814)]:
            s.line(a, y, b-5, y, MUTED, 2, arrow=True)
    s.text(783, 310, "softmax(· / T)", 17, MUTED, anchor="middle")
    box(1090, 268, 240, 106, "Conditional loss", "Lᵣ = T²(1 − pᵧ) KL(p⁻ ‖ q⁻)", GREEN)
    s.path([(1034, 239), (1060, 239), (1060, 300), (1086, 300)], MUTED, 2, True)
    s.path([(1034, 400), (1060, 400), (1060, 344), (1086, 344)], MUTED, 2, True)
    s.text(219, 474, "Normalize hₛ, wₛ,ₖ; no bias", 17, BLUE)
    s.text(551, 470, "N(z): per-example centering and standard deviation over classes", 18, GREEN)
    s.rect(40, 526, 743, 105, "#F5F7F9")
    s.text(60, 556, "Detached gradient-strength matching", 21, weight=600)
    s.text(60, 590, "a = stopgrad( ‖∇ Lfull‖₂ / ‖∇ Lᵣ‖₂ )", 24)
    s.text(60, 618, "Same minibatch and state; non-head student parameters; extra backward work.", 17, MUTED)
    box(884, 535, 446, 89, "Training objective", "L = λCE CE(zₛ, y) + λKD a Lᵣ", GREEN)
    s.line(783, 578, 879, 578, MUTED, 2, arrow=True)
    s.path([(1210, 374), (1370, 374), (1370, 579), (1335, 579)], MUTED, 2, True)
    s.circle(500, 400, 3, BLUE)
    s.path([(500, 400), (500, 508), (866, 508), (866, 553), (879, 553)], BLUE, 2, True)
    s.text(802, 502, "CE", 17, BLUE)
    s.footer("Teacher classifier is unchanged. Standardization does not remove arbitrary class-dependent teacher bias or weight-norm effects.",
             "No shared teacher–student feature-space projection. Labels y define the conditional mask; only the student is optimized.")
    return s.save("method_acd")


def export(svg):
    """Use the installed SVG engine; no network, browser or Python graphics deps."""
    r = C.CDLL(ctypes.util.find_library("rsvg-2"))
    c = C.CDLL(ctypes.util.find_library("cairo"))
    g = C.CDLL(ctypes.util.find_library("gobject-2.0"))
    def bind(lib, name, args, restype):
        f = getattr(lib, name); f.argtypes = args; f.restype = restype
        return f
    ptr, integer, double = C.c_void_p, C.c_int, C.c_double
    load = bind(r, "rsvg_handle_new_from_file", [C.c_char_p, C.POINTER(ptr)], ptr)
    render = bind(r, "rsvg_handle_render_cairo", [ptr, ptr], integer)
    png_surface = bind(c, "cairo_image_surface_create", [integer, integer, integer], ptr)
    pdf_surface = bind(c, "cairo_pdf_surface_create", [C.c_char_p, double, double], ptr)
    context = bind(c, "cairo_create", [ptr], ptr)
    scale = bind(c, "cairo_scale", [ptr, double, double], None)
    write = bind(c, "cairo_surface_write_to_png", [ptr, C.c_char_p], integer)
    destroy = bind(c, "cairo_destroy", [ptr], None)
    finish = bind(c, "cairo_surface_finish", [ptr], None)
    free = bind(c, "cairo_surface_destroy", [ptr], None)
    status = bind(c, "cairo_surface_status", [ptr], integer)
    unref = bind(g, "g_object_unref", [ptr], None)
    error = ptr()
    handle = load(str(svg).encode(), C.byref(error))
    if not handle: raise RuntimeError(f"SVG load failed: {svg}")
    h = int(ET.parse(svg).getroot().attrib["height"])
    for ext, factor in [("png", 2.0), ("pdf", .36)]:
        target = svg.with_suffix("."+ext)
        surface = png_surface(0, 2800, h*2) if ext == "png" else pdf_surface(str(target).encode(), 504, h*factor)
        cr = context(surface); scale(cr, factor, factor)
        if not render(handle, cr): raise RuntimeError(f"SVG rendering failed: {svg}")
        if ext == "png" and write(surface, str(target).encode()): raise RuntimeError("PNG write failed")
        destroy(cr); finish(surface)
        if status(surface): raise RuntimeError("Cairo surface failed")
        free(surface)
    unref(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--raster", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    snapshot = OUT / "evidence.json"
    if args.collect:
        snapshot.write_text(json.dumps(collect(), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    d = json.loads(snapshot.read_text())
    assert not d["synthetic"]
    assert len(d["probe"]["classifier_parameters"]["linear_weight_norms"]) == 100
    for fn in (teaser, interface, conditional, method):
        svg = fn(d)
        if args.raster: export(svg)
        print(svg.relative_to(ROOT))


if __name__ == "__main__":
    main()
