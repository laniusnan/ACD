# Supplementary figure and table assets

This list is the selection guide for `supplementary.tex`.  It records what an
asset measures, the wording that remains supported by its evidence, and the
file to upload to Overleaf.  Start with S1 and the two tables already embedded
in the supplement; add the local diagnostics only if the main paper needs the
extra mechanism detail.

| ID | Overleaf upload source | Intended role | Supported caption claim |
| --- | --- | --- | --- |
| S1 | `../research_archive/figures/analysis_20261116/fig_output_mismatch_cifar100_seed2051.pdf` | Oracle scalar-calibration diagnostic | Per-image nonnegative scalar calibration reduces mean `8 KL` at `T=4` from 1.2198 to 0.5575 for one frozen CIFAR-100 endpoint, but leaves residual class-relative disagreement. |
| S2 | `../research_archive/figures/mechanism_figures/mechanism_parameterization_force.pdf` | Fixed-state parameterization diagnostic | With directions and input fixed, score parameterization changes non-target distributions and the local KD feature gradient in the recorded checkpoint probe. |
| S3 | `../research_archive/figures/analysis_20261116/fig_parameterization_compact.pdf` | Compact parameterization sensitivity evidence | Controlled score interventions change the output distribution and local loss signal; this is not an end-to-end accuracy experiment. |
| S4 | `../research_archive/figures/analysis_20261116/fig_supervision_problem_ab.pdf` | Supervision decomposition diagnostic | Use only with its fixed-state scope and source checkpoint stated in the caption. |
| S5 | `../figs/method.pdf` | Method overview | ACI uses a bias-free cosine head and SICD uses standardized, teacher-mass-weighted conditional distillation. Check ETF wording before use because ETF is optional and is disabled for released reruns. |

The two tables in `supplementary.tex` require no external graphics:

- Table S1 is the released ResNet32x4-to-ResNet8x4 seven-arm comparison,
  evaluated on all 10,000 official test images. It is a seed-3301 endpoint
  comparison with one pair and one shared recipe.
- Table S2 lists seed-3201 frozen ACD rerun endpoints. It supports model
  availability and reproducibility; it is not a KD baseline table.

Do not include `imagenet_gradcam_20261117/fig_imagenet_baseline_gradcampp.*`
as an ACD visualization. It is a CE-versus-KD baseline diagnostic. Do not use
historical tables or user-transcribed ETF values as verified supplement results
unless a seed, configuration, and checkpoint provenance are recovered.

## Overleaf insertion

After uploading a selected PDF under `figures/`, enable `graphicx` in the
preamble and use the following pattern:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_output_mismatch_cifar100_seed2051.pdf}
  \caption{Output mismatch beyond scalar confidence. For a frozen CIFAR-100
  endpoint, independent per-image nonnegative logit scaling reduces mean
  $8D_{\mathrm{KL}}$ at $T=4$ from 1.2198 to 0.5575, while class-relative
  score disagreement remains. This oracle diagnostic uses teacher outputs and
  is not a training or deployment procedure.}
  \label{fig:supp-oracle}
\end{figure}
```

Use the original artifact label in the caption when referring to a frozen
checkpoint, and retain its dataset, seed, epoch, and fixed-state status.
