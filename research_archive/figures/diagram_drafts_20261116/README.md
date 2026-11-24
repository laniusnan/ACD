# ACI / SICD conceptual diagram drafts

These are conceptual diagrams, not empirical feature visualizations. Existing paper figures are unchanged.

## Separate versions

- **V4 / equal-height horizontal layout (latest requested revision)**: [`v4_horizontal/`](v4_horizontal/README.md) removes the repeated objective block and ACI's top normalization equation. Small gray notes are right-aligned to the two ACI equations. SICD is wider and shorter, with its conditional loss to the right of both branches. [Combined PDF](v4_horizontal/figures/method_horizontal.pdf); standalone [ACI](v4_horizontal/figures/aci.pdf) and [SICD](v4_horizontal/figures/sicd.pdf). Both modules share the same height; all figures also have SVG/PNG exports and [Overleaf code](v4_horizontal/overleaf.tex).

- **V3 / three-block layout (archived)**: [`v3_three_block/`](v3_three_block/README.md) follows the layout of *LIFT and PLACE*, Fig. 4. Upper-left objective, lower-left ACI with separate cosine/ETF formulas, full-height right SICD. [Combined PDF](v3_three_block/figures/method_three_block.pdf); standalone [objective](v3_three_block/figures/objective.pdf), [ACI](v3_three_block/figures/aci.pdf), and [SICD](v3_three_block/figures/sicd.pdf). All have SVG/PNG exports. The assembly adds no text; requested dataset/class and coefficient annotations are removed. No patch grouping is introduced.

- **V1 / original**: `render_final_method_figure.py` and `figures/final_method_figure.{pdf,png,svg}` are restored byte-for-byte from commit `ade8d26`. The original standalone ACI diagram remains at `figures/aci_geometry_corrected.pdf`.
- **V2 / revised cosine readout**: source, helper snapshot, documentation, `method_aci.tex`, and unchanged rendered outputs are stored independently in `v2_cosine_readout/`. Main figure: `v2_cosine_readout/figures/final_method_figure.pdf`; standalone ACI: `v2_cosine_readout/figures/aci_cosine_head.pdf`.
- Each script defaults to its own sibling `figures/` directory. Do not explicitly reuse the same `--output-dir` if you want to preserve both versions.
- V1–V3 remain available. The original V1 documentation follows; see the V4 README for the latest horizontal revision and its reproduction command.

## Outputs

- `figures/final_method_figure.pdf`: combined, side-by-side dashed ACI/SICD panels. Geometry is drawn directly on the same canvas with `fig.add_axes`, never imported as a bitmap. The default photo is the real CIFAR-100 apple (test index 9). The right-hand 3-by-3 grid overlay is visual scaffolding, **not** spatial patch grouping or independent heads. Its feature grids remain explicitly synthetic.
- `figures/aci_geometry_corrected.pdf`: **Angular Coordinate Interface** with `z_s,k = s cos(theta_s,k)`, not an invented additive cosine/ETF loss. The right panel explicitly illustrates **optional** three-class ETF initialization; it does not assert that every current experiment uses it or that directions remain equiangular during training.
- `figures/sicd_training_pipeline.pdf`: the actual documented classifier/output-loss flow. The teacher retains its original classifier; only the student uses ACI. Per-example all-class standardization precedes softmax and non-target conditioning. The teacher non-target mass weights the conditional KL; the current training recipe additionally uses a detached gradient-norm ratio with a same-state standardized-full reference over non-head student parameters. Numerical denominator protection and batch reduction are omitted from the drawing.
- `figures/sicd_patch_group_concept_only.pdf`: optional requested 2-by-2 patch/multi-head layout, conspicuously labeled **hypothetical**. It is not the implemented SICD and should not be used as its method figure.

All outputs also have SVG and PNG previews. ACI is fully vector. Pipeline PDFs/SVGs contain embedded raster photographs; the heatmaps (colored rectangles), text, arrows and boxes are vector. A figure containing photographs is not fully vector.

The three input examples are the first official CIFAR-100 test occurrences of `tiger`, `apple`, and `pickup_truck`. CIFAR-100 does not have the ordinary `cat` or `automobile` classes from CIFAR-10. The input display uses bicubic interpolation, not super-resolution. Heatmaps are reproducible NumPy-generated noise, prominently labeled synthetic; they are not extracted from any teacher or student. `figures/manifest.json` records image indices, raw RGB hashes and synthetic arrays. No inference or training is performed.

## Run

Dependencies: NumPy, Matplotlib, and mutually compatible PyTorch/torchvision packages. The serif font falls back to STIXGeneral/DejaVu Serif when Times New Roman is absent. An external LaTeX installation is not needed.

```bash
python3 render_aci_sicd.py --data-root ./data --download --patch-demo
```

`--download` explicitly permits torchvision to obtain missing CIFAR-100 data. For an existing dataset, point `--data-root` to the parent of `cifar-100-python` and omit `--download`.

From this repository root, using the existing read-only dataset and plotting environment:

```bash
PYTHONPATH=paper_assets/acd_figures/analysis_20261116/artifacts/plot_deps \
  .venv/bin/python -B paper_assets/acd_figures/diagram_drafts_20261116/render_aci_sicd.py \
  --data-root reproductions/cifar100_acd_rerun_20261116/data --patch-demo
```

### Combined method figure

Run from the repository root (the script reuses the sibling `render_aci_sicd.py`):

```bash
PYTHONPATH=paper_assets/acd_figures/analysis_20261116/artifacts/plot_deps \
  .venv/bin/python -B paper_assets/acd_figures/diagram_drafts_20261116/render_final_method_figure.py \
  --data-root reproductions/cifar100_acd_rerun_20261116/data
```

Use `--example tiger` or `--example pickup_truck` to change the photo. On another machine with the dependencies installed, use `python3`, a suitable `--data-root`, and optionally `--download`. `--output-dir .` puts `final_method_figure.pdf` in the current directory; by default it is written to the sibling `figures` directory with SVG/PNG previews and a separate `final_method_figure_manifest.json`. This PDF contains one raster photograph; the geometric panels, simulated heatmap cells, and all annotations remain vector. Use it at full two-column width, rather than as a single-column figure.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figs/final_method_figure.pdf}
  \caption{Method overview. Left: the angular coordinate interface normalizes
  student features and class weights; the ETF configuration illustrates optional
  initialization only. Right: SICD standardizes output scores and distills
  teacher-weighted non-target conditional relations. The teacher is frozen and
  only the student is updated. The CIFAR-100 input is real; feature grids and
  geometric arrows are schematic rather than measured model outputs.}
  \label{fig:method_overview}
\end{figure*}
```

## Scientific scope

ACI defines an angular student interface, not a feature-level cosine attraction loss. Arrows in the geometric illustration are conceptual alignment directions, not measured optimization gradients. If ETF initialization is not part of the described configuration, omit the optional right panel from the manuscript rather than implying it was used.

SICD does not introduce spatial patch groups or a classifier per patch. It also does not replace the teacher's classifier with a cosine head. Removing an explicit target-confidence KL term does not eliminate target dependence through standardization or the teacher non-target mass. The diagram does not claim that SICD eliminates CE/KD gradient conflict.

Method references within the repository: `../README.md` ("Method: construct the interface, then select supervision"), `../render_evidence_figures.py` (`method`), and the current `reproductions/cifar100_acd_rerun_20261116/methods.py`. These are read-only references; this draft does not modify training code.
