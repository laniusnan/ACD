# Supplementary-material draft

This directory is the source-of-truth draft for the ACD supplementary material.
It is independent of the author PDF (`../acd_v1.pdf`), whose LaTex source is
unavailable; it can be pasted into the paper project as an appendix or
supplement without altering that manuscript.

- `supplementary.tex` is the paper-facing source.
- `evidence_manifest.json` maps every result or implementation statement to
  its repository evidence.

The draft is restricted to reproducible facts: released assets, the completed
seven-arm ResNet32x4-to-ResNet8x4 comparison, and frozen CIFAR-100 reruns.
Historical paper-table rows, incomplete queue entries, ImageNet, and detection
are not presented as new verified results.

## Overleaf use

Copy `supplementary.tex` into Overleaf.  It compiles without figures.  To add
approved figures, upload their PDFs from `../figs/` or `../research_archive/`
and enable the commented figure blocks.  A standard LaTex installation with
`article`, `amsmath`, `booktabs`, and `hyperref` is sufficient; no bibliography
file is required.

## Evidence policy

"Verified" means an epoch-240 endpoint with fixed configuration and checkpoint
identity, evaluated on all 10,000 official CIFAR-100 test images.  A
single-seed result remains a single-seed result.  Do not replace this wording
with significance or generality claims before the registered replications are
complete.
