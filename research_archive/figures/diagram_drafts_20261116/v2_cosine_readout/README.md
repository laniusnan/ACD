# V2: cosine-readout ACI / SICD diagram draft

This is a separately stored candidate, not an accepted replacement. V1 remains in the parent directory. No figures were rerendered during separation and no side-by-side comparison image was created. Both versions remain available for user evaluation.

These are conceptual diagrams, not empirical feature visualizations. Historical analysis figures are unchanged. The local `render_aci_sicd.py` is an unchanged helper snapshot, so this version renders independently. The default output directory is this version's own `figures/`; do not point `--output-dir` to V1 when retaining both.

## Outputs

- `figures/final_method_figure.pdf`: combined, side-by-side dashed ACI/SICD panels. The revised ACI panel compares an affine readout with its angular counterpart at the **same illustrative directions**, then shows normalization, dot product, shared scale, and removal of bias. Geometry is drawn directly with `fig.add_axes`, never imported as a bitmap. The default photo is the real CIFAR-100 apple (test index 9). The right-hand 3-by-3 grid overlay is visual scaffolding, **not** spatial patch grouping or independent heads. Its feature grids remain explicitly synthetic; this revision leaves the SICD drawing unchanged.
- `figures/aci_cosine_head.pdf`: the revised ACI panel exported separately as a fully vector figure. Non-equally-spaced class directions intentionally avoid suggesting an ETF constraint. `aci_cosine_head_manifest.json` records its illustrative vectors and source documents.
- `method_aci.tex`: corresponding English Method paragraph and equations, ready to adapt in Overleaf with `amsmath`/`amssymb`. It distinguishes the student forward-score invariance from unsupported claims about training dynamics.
- `method_aci_body.tex`: separately saved manuscript-body revision from 2026-11-17, retaining the requested ACI subsection and Cosine Classifier subsubsection. It is an alternative to `method_aci.tex`, not an additional section to include alongside it. The earlier draft and all figures remain unchanged.
- `../figures/aci_geometry_corrected.pdf`: **V1 draft**, retained unchanged for evaluation. Its right panel illustrates optional three-class ETF initialization, which is not the core ACI method and is absent from the revised combined figure.
- `../figures/sicd_training_pipeline.pdf`: the actual documented classifier/output-loss flow. The teacher retains its original classifier; only the student uses ACI. Per-example all-class standardization precedes softmax and non-target conditioning. The teacher non-target mass weights the conditional KL; the current training recipe additionally uses a detached gradient-norm ratio with a same-state standardized-full reference over non-head student parameters. Numerical denominator protection and batch reduction are omitted from the drawing.
- `../figures/sicd_patch_group_concept_only.pdf`: optional requested 2-by-2 patch/multi-head layout, conspicuously labeled **hypothetical**. It is not the implemented SICD and should not be used as its method figure.

All outputs also have SVG and PNG previews. ACI is fully vector. Pipeline PDFs/SVGs contain embedded raster photographs; the heatmaps (colored rectangles), text, arrows and boxes are vector. A figure containing photographs is not fully vector.

The three input examples are the first official CIFAR-100 test occurrences of `tiger`, `apple`, and `pickup_truck`. CIFAR-100 does not have the ordinary `cat` or `automobile` classes from CIFAR-10. The input display uses bicubic interpolation, not super-resolution. Heatmaps are reproducible NumPy-generated noise, prominently labeled synthetic; they are not extracted from any teacher or student. `../figures/manifest.json` records image indices, raw RGB hashes and synthetic arrays. No inference or training is performed.

## Run

Dependencies: NumPy, Matplotlib, and mutually compatible PyTorch/torchvision packages. The serif font falls back to STIXGeneral/DejaVu Serif when Times New Roman is absent. An external LaTeX installation is not needed.

`--download` explicitly permits torchvision to obtain missing CIFAR-100 data. For an existing dataset, point `--data-root` to the parent of `cifar-100-python` and omit `--download`.

### Combined method figure

Run from the repository root (the script reuses the sibling `render_aci_sicd.py`):

```bash
PYTHONPATH=paper_assets/acd_figures/analysis_20261116/artifacts/plot_deps \
  .venv/bin/python -B paper_assets/acd_figures/diagram_drafts_20261116/v2_cosine_readout/render_final_method_figure.py \
  --data-root reproductions/cifar100_acd_rerun_20261116/data
```

Use `--example tiger` or `--example pickup_truck` to change the photo. On another machine with the dependencies installed, use `python3`, a suitable `--data-root`, and optionally `--download`. `--output-dir .` puts `final_method_figure.pdf` in the current directory; by default it is written to the sibling `figures` directory with SVG/PNG previews and a separate `final_method_figure_manifest.json`. This PDF contains one raster photograph; the geometric panels, simulated heatmap cells, and all annotations remain vector. Use it at full two-column width, rather than as a single-column figure.

The same command also exports `aci_cosine_head.{pdf,svg,png}`. To draw just the ACI panel without CIFAR-100 or torchvision, replace `--data-root ...` with `--aci-only`.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figs/v2_cosine_readout/final_method_figure.pdf}
  \caption{Method overview. Left: ACI replaces the student's affine readout
  with a bias-free cosine head. Feature and class-weight normalization remove
  their direct magnitude dependence from the forward scores while preserving
  learnable directions; a fixed scale is shared across samples and classes.
  The two geometric views use the same illustrative directions, not different
  trained checkpoints. Right: SICD standardizes output scores and distills
  teacher-weighted non-target conditional relations. The teacher is frozen and
  only the student is updated. The CIFAR-100 input is real; feature grids and
  geometric vectors are schematic rather than measured model outputs.}
  \label{fig:method_overview_v2}
\end{figure*}
```

## Scientific scope

### Why the ACI panel was revised

The classifier discussion (sections 1.2 and 5) and figure guide (sections B/C) support **feature normalization + per-class weight normalization + no bias + a fixed shared scale**, while preserving learnable directions. They do not establish that ETF initialization is essential, that features are explicitly attracted to class centers, or that removing radial gradients invariably helps. Consequently the method panel now explains the implemented readout rather than illustrating those hypotheses.

- Left geometry: blue `h` and `alpha h` have the same direction but different magnitudes; red class weights have unequal norms. The affine formula shows bias as a scalar score offset, **not** a displacement in feature space.
- Right geometry: those two features map to one unit direction, and each class weight is normalized independently. All original angles are preserved for this illustration. The class directions remain learnable and need not be equiangular.
- Bottom computation: normalize the sample feature and each weight row, compute their dot product, and multiply by the shared fixed scale (`s=30` in the cited implementation). This is not applying a cosine function to logits or matching teacher/student features directly.
- Invariance describes forward scores for positive rescaling outside the numerical protection regime. Raw norms can still evolve and affect optimization through normalization derivatives, the shared backbone, and regularization. The illustration does not demonstrate final-accuracy gains, universally harmful radial gradients, or eliminated CE/KD conflict.

The revised panel uses vector arrows, not optimization arrows, and contains no ETF construction or standalone cosine loss. The original optional-ETF draft is retained unchanged as a historical output.

SICD does not introduce spatial patch groups or a classifier per patch. It also does not replace the teacher's classifier with a cosine head. Removing an explicit target-confidence KL term does not eliminate target dependence through standardization or the teacher non-target mass. The diagram does not claim that SICD eliminates CE/KD gradient conflict.

Method references within the repository: `CLASSIFIER_DISCUSSION_20261026.md` (1.2, 5), `docs/CLASSIFIER_FIGURE_GUIDE_20261027.md` (B/C), `docs/CLASSIFIER_FIGURE_REUSE_20261116.md` (2/3), `reports/ACD_WRITING_ARGUMENT_20261110.md` (Method and Figure implications), `../../README.md` ("Method: construct the interface, then select supervision"), and the current `reproductions/cifar100_acd_rerun_20261116/methods.py` / `protocol.json`. These are read-only references; this draft does not modify training code.
