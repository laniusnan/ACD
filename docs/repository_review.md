# Repository review — 2026-11-18

This review focuses on using ACD to train, evaluate, inspect results and share weights. Existing experimental hyperparameters are preserved, including the ImageNet example learning rate of 0.1.

## Addressed gaps

| Finding | Change / evidence |
|---|---|
| Evaluation constructed the training split | Separate evaluation dataset path; tests assert no training builder is called |
| No convenient teacher evaluation | `--evaluate-teacher` selects the teacher's affine architecture |
| Only console Top-1 output | Top-1, Top-5, CE, sample count and optional metrics JSON with checkpoint identity |
| Loading always enabled Python unpickling | Default weights-only loading; explicit opt-in for trusted legacy files |
| Wrong checkpoint/config combinations were hard to identify | Model identity validation plus strict state loading |
| Smoke/export files failed resume with opaque key errors | Explicit resume format validation and actionable messages |
| Fresh runs could mix existing output logs | Reject existing run outputs unless resuming |
| No pretrained model delivery path | Twelve ACD-rerun/teacher weights plus twelve historical head-control/teacher weights, source/export checksums, configs, model guides and a download utility |
| Historical KD models were hard to distinguish from full ACD | Separate control release and guide, paired variants, recipe limits, missing VGG13 teacher and reused teacher dependencies |
| No reusable training figures | Workflow, method, results and archived training plots in `figs/`; plot tool for user runs |
| Gradient matching assumed a reusable parameter list | Materialize parameter iterators before both gradient calls; matched-norm regression test |
| ETF invalid dimensions/scales were hard to diagnose | Explicit validation, dtype/device preservation and geometric tests |

Resume behavior is checked against uninterrupted training using a simulated interruption. Release evaluation provides the real-data check separately from synthetic workflow tests.

## Content organization

The root README prioritizes install, download, train, evaluate and plot commands. Longer implementation notes live under `docs/`. Large checkpoints, generated user outputs and datasets stay outside Git history. Release weights omit optimizer state. Detailed manuscript revision drafts are not part of this user-facing cleanup.

The copied model definitions retain upstream structure and attribution; unused feature-return compatibility is retained because these definitions also support feature diagnostics. The historical augmentation sampler is preserved because changing it would change the recipe used by released models. No code was removed merely because it was not needed for a README example.

## Remaining scope

- Four additional CIFAR configs and two ImageNet examples have no complete released ACD run. They are documented as support/examples, not benchmark reproductions.
- The current trainer is a single-device float32 reference. AMP, DDP and accumulation need explicit treatment of the gradient-matching norm before implementation.
- The current task is classification; detection training is outside the package.
- Exact end-to-end equivalence between a new refactored training run and the archived trainer has not been established. Exported final weights have been evaluated independently.
- No new best-test selection policy was added. Released models are the final epoch 240 checkpoints; training logs should be used to inspect runs without representing test-set selection as validation.
