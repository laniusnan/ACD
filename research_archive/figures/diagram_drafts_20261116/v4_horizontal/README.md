# Equal-height ACI and horizontal SICD

V4 contains two modules, ACI on the left and a wider, compact SICD pipeline on
the right. The repeated objective panel is omitted. Both modules have the same
height and drawing scale; circles, photographs, and type are not stretched.

## Outputs

| File in `figures/` | Content |
| --- | --- |
| `method_horizontal.pdf` | Complete two-module figure, with no added composition text. |
| `aci.pdf` | Cosine geometry and optional ETF initialization. |
| `sicd.pdf` | Teacher/student branches feeding a conditional loss at the right. |
| `manifest.json` | Layout, image provenance, synthetic arrays and scientific scope. |

Each figure also has an editable SVG and PNG preview. Use PDF for Overleaf.
[`overleaf.tex`](overleaf.tex) provides a complete-figure example and an
alternative assembling the two standalone PDFs.

## Changes from V3

- Removed the standalone objective block from this version.
- Removed the feature/weight normalization equation beneath the ACI title.
- Kept the two main equations beneath ACI (a) and (b). Their explanatory notes
  use 12 pt light gray text (`#939BA3`), down from 16 pt dark text. Each note's
  right edge is aligned to its own equation's measured right edge.
- Reduced ACI's height from 6.60 to 5.85 drawing units after removing the top
  equation. SICD is rearranged from 11.15 × 8.80 to 15.40 × 5.85 units, placing
  the conditional loss beside the teacher/student branches. The shared
  standardization formula stays between the branches and the total objective
  sits beneath the horizontal flow.
- Retained the previously requested removal of dataset/class labels, the
  coefficient explanation, and the conceptual footer. No patch grouping is
  introduced. V1–V3 remain archived in their existing locations.

## Meaning of the diagram

ACI uses unit-normalized student features and class weights, a shared fixed
positive scale, and no student-head bias. The hats in the main equation retain
this meaning even though the separate normalization definition is omitted.
The teacher retains its original affine classifier.

ETF is **optional initialization**, not a loss or a training-time constraint.
The pairwise equation is explicitly evaluated at time 0. The three-class
example has pairwise inner product −1/2 and angles of 120 degrees; it is not a
projection of a 100-class ETF. Directions remain trainable thereafter. The
current CIFAR rerun does not enable ETF initialization.

SICD standardizes the full class logit vector per image using its mean and
sample standard deviation (correction 1), then computes temperature-softened
probabilities. For `k != y`, `p^-_k = p_k/(1-p_y)` and
`q^-_k = q_k/(1-q_y)`. The teacher non-target mass weights the conditional KL.
The displayed product spans two lines in the loss box.

The coefficient `a` is the detached standardized-full/relation gradient-norm
ratio over non-head student parameters at the same state and batch. Batch
averaging, numerical floors, and calculation of `a` are omitted from the figure.
CE uses unstandardized student logits. No spatial patch groups or per-patch
classifiers are part of this computation.

The photograph and deterministic synthetic feature arrays are unchanged from
V3. Feature grids and geometric arrows are schematic, not measured activations
or gradients. Only the input photo is raster; all remaining figure elements
are vector. Rendering performs no training or inference.

## Reproduce

From the repository root:

```bash
PYTHONPATH=paper_assets/acd_figures/analysis_20261116/artifacts/plot_deps \
  .venv/bin/python -B \
  paper_assets/acd_figures/diagram_drafts_20261116/v4_horizontal/render_method_figure.py \
  --data-root reproductions/cifar100_acd_rerun_20261116/data
```

On another machine, use Python with Matplotlib, NumPy, PyTorch and torchvision,
and set `--data-root` to the parent of `cifar-100-python`. The renderer reuses
`../render_aci_sicd.py`. It verifies equal panel height, equation-note alignment,
panel text boundaries, text overlaps, requested label omissions and exact
agreement between assembled and standalone text. It strips only trailing
whitespace from generated SVG text.
