# Three-block ACI / SICD method figure

This revision follows the **layout** of Figure 4 (page 4) of
[LIFT and PLACE](https://arxiv.org/abs/2605.19729v3): a short objective block
at upper left, an ACI block beneath it, and a full-height SICD block at right.
It depicts our ACI/SICD method, without importing LIFT's regression objective
or PLACE's output grouping. V1 and V2 remain available in their original locations.

## Files

| File in `figures/` | Content |
| --- | --- |
| `objective.pdf` | Standalone upper-left block: standardized full KD → weighted conditional KD. Text and mathematics only. |
| `aci.pdf` | Standalone lower-left block: cosine readout and optional ETF initialization, each with its own formula. |
| `sicd.pdf` | Standalone right block: original teacher classifier, student ACI, standardized probabilities, conditional relation loss. |
| `method_three_block.pdf` | Three blocks assembled at the same physical font and drawing scales. No extra text or panel labels. |
| `manifest.json` | Reference, layout dimensions, dataset image index/hash, synthetic feature arrays, mathematical scope. |

All four figures also have editable SVG and PNG previews. PDF is recommended
for Overleaf. `overleaf.tex` includes both an assembled-PDF example and an
alternative using all three standalone PDFs.

The ACI and objective blocks are entirely vector. SICD and the assembled figure
contain one raster input photograph; all heatmap cells, geometry, arrows,
formulas, and text are vector. Times New Roman is used if installed, with
STIXGeneral / DejaVu Serif fallbacks and STIX mathematics.

## Requested changes

- Removed the original ACI bottom “Conceptual …” line.
- Moved the cosine equation below panel (a). Panel (b) now has the distinct ETF
  condition `w_hat_s,i(0)^T w_hat_s,j(0) = -1/(C-1)` for `i != j`.
  The `(0)` explicitly means initialization. Directions remain trainable.
- Removed the right block's dataset/class labels, “Real CIFAR-100 input …”
  subtitle, and the explanatory line for the gradient matching coefficient.
  The coefficient `a` remains in the objective.
- Did not add grouped patches. The feature grids have no group outlines,
  per-patch classifiers, or group losses.

## Mathematical and visual scope

The upper-left arrow is a **choice of supervision**, not an algebraic equality.
Both `p` and `q` refer to the all-class standardized probabilities defined in
the right block. ACI only changes the student classifier: features and class
weights are normalized, the shared positive scale is fixed, and there is no bias.
The teacher retains its original classifier.

The ETF drawing is a **three-class example**, not a projection of the actual
100-class anchors. For `C=3`, the displayed condition gives inner product `-1/2`
and angles of 120 degrees. It is an **optional initialization**, not a regularizer
or a constraint held throughout training. The current CIFAR rerun does not use
ETF initialization. Geometric arrows illustrate geometry, not measured gradients.

For every example, SICD standardizes the full class logit vector using its mean
and sample standard deviation (correction 1), then forms temperature-softened
probabilities. Non-target distributions are `p^-_k = p_k/(1-p_y)` and
`q^-_k = q_k/(1-q_y)` for `k != y`. The teacher non-target mass `(1-p_y)` weights
their KL divergence. `a` is the detached ratio of standardized-full and
conditional-loss gradient norms over trainable non-head student parameters,
computed at the same state and batch. The figure shows per-example notation;
the implementation averages over the batch before applying `a`. Numerical
denominator floors and the computation of `a` are omitted for space. CE uses
the student's unstandardized logits. Only the student is updated.

SICD operates on **class-output distributions after pooling**. Spatial patch
grouping is neither part of this implementation nor its claimed contribution.
PLACE's grouping serves that paper's diffusion-output objective; including a
similar grouping stage here would imply a different method.

The input is the same real image as V1 (default: official CIFAR-100 test index 9,
class apple), displayed with bicubic interpolation. Its provenance is recorded
here and in the manifest rather than printed inside the figure. Feature grids
are the same deterministic synthetic arrays as V1, **not model activations**.
No model inference or training is performed by this renderer.

## Reproduce

From the repository root, using the existing dataset and plotting environment:

```bash
PYTHONPATH=paper_assets/acd_figures/analysis_20261116/artifacts/plot_deps \
  .venv/bin/python -B \
  paper_assets/acd_figures/diagram_drafts_20261116/v3_three_block/render_method_figure.py \
  --data-root reproductions/cifar100_acd_rerun_20261116/data
```

On another machine with NumPy, Matplotlib, PyTorch and torchvision installed,
use `python3` and point `--data-root` to the parent of `cifar-100-python`.
`--download` is optional and only needed if the dataset is absent.
The renderer reuses the original sibling `../render_aci_sicd.py` drawing helpers.
It verifies that the assembled text exactly matches the three standalone blocks,
that removed labels are absent, and that text remains inside the panel/canvas.

Implementation references: `../../README.md` is the figure-assets overview;
the method's current executable definitions are in
`../../../../reproductions/cifar100_acd_rerun_20261116/methods.py`
(`AngularHead`, `standardize`, `relation`, and `objective`).
