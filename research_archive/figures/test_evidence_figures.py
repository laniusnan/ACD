"""Read-only integrity tests for the published figure snapshot and SVG assets."""
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
OUT = HERE / "evidence_figures"


class EvidenceFigures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = json.loads((OUT / "evidence.json").read_text())

    def test_real_endpoints(self):
        self.assertFalse(self.d["synthetic"])
        for row in list(self.d["interaction"].values()) + [self.d["interventions"]]:
            for r in row.values():
                self.assertTrue(r["reload_verified"])
                self.assertEqual(r["epoch"], 240)
                self.assertGreater(r["top1"], 0)
                self.assertLessEqual(r["top1"], 100)
                self.assertEqual(len(r["checkpoint_sha256"]), 64)

    def test_negative_interaction_is_retained(self):
        for seed, expected in [("2102", -4.24), ("2103", -4.05)]:
            r = self.d["interaction"][seed]
            value = (r["cosine_kd"]["top1"] - r["linear_kd"]["top1"]
                     - r["cosine_ce_only"]["top1"] + r["linear_ce_only"]["top1"])
            self.assertAlmostEqual(value, expected)

    def test_component_comparisons(self):
        r = self.d["interventions"]
        self.assertAlmostEqual(r["relation"]["top1"]-r["full"]["top1"], .8)
        self.assertAlmostEqual(r["relation"]["top1"]-r["strength"]["top1"], -.66)
        self.assertTrue(all(x["endpoint"] == "validation" and x["samples"] == 5000 for x in r.values()))

    def test_probe_and_sources(self):
        p = self.d["probe"]
        self.assertEqual(p["probe"]["n"], 256)
        self.assertEqual(len(p["classifier_parameters"]["linear_weight_norms"]), 100)
        for r in p["ordering"].values():
            self.assertTrue(0 <= r["top1_changed_pct"] <= 100)
            self.assertTrue(0 <= r["mean_top5_overlap_pct"] <= 100)
        for v in self.d["sources_sha256"].values():
            self.assertEqual(len(v), 64)

    def test_editable_svg_and_exports(self):
        for name in ("teaser_parameterization", "analysis_angular_interface",
                     "analysis_conditional_supervision", "method_acd"):
            p = OUT / (name + ".svg")
            root = ET.parse(p).getroot()
            ns = {"s": "http://www.w3.org/2000/svg"}
            self.assertTrue(root.findall("s:text", ns))
            self.assertIsNotNone(root.find("s:title", ns))
            self.assertIsNotNone(root.find("s:desc", ns))
            self.assertFalse(any(line.rstrip() != line for line in p.read_text().splitlines()))
            for ext, signature in [("png", b"\x89PNG"), ("pdf", b"%PDF")]:
                self.assertTrue(p.with_suffix("."+ext).read_bytes().startswith(signature))


if __name__ == "__main__":
    unittest.main()
