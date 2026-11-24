# Same-input directional disagreement

The PDF, PNG and SVG display three fixed official CIFAR-100
test examples (indices 0, 1, 2: mountain, forest, seal). Each row uses the same input under three
post-hoc score parameterizations on one affine-trained checkpoint.
The former fourth row (index 3, mushroom) is omitted from the figure; the
source probe retains all four examples and their original measurements.

The student is VGG13 → MobileNetV1, linear KD, seed 2051, epoch 240,
restored from `model-backup-20261027` / `head_transfer_20261021-001.tar.gz`.
The matching VGG13 teacher is restored from `teacher-models-000.tar.gz`.
Checkpoint and model-source hashes are verified by the extraction script.
This supersedes the earlier two-example seed-2102 rendering; all displayed rows
now share the recovered seed-2051 checkpoint.

For each input, compute the negative temperature-4 KL gradient with respect
to normalized input pixels. Angular scores use cosine similarity at scale 1;
Scale/Norm restores feature magnitude and class-weight norms; Raw affine
also restores bias. These are score interventions, not three trained models.

At each pixel, colors encode `1 - cosine` between its RGB-gradient vector
and the angular reference, using one shared 0–2 muted multihue color scale
and constant overlay opacity. Angular has zero disagreement by definition;
its uncolored image does not mean zero gradient or better semantic attention.
The arbitrary angular scale is retained from the earlier visualization and
does not isolate class-dependent effects from common-scale effects.

Reproduce from the restored weights:

```bash
.venv/bin/python -B paper_assets/acd_figures/extract_release_pixel_forces.py
.venv/bin/python -B paper_assets/acd_figures/render_same_input_force_variants.py
```

No optimizer updates or model-state mutations occur. This local input-gradient
diagnostic does not establish accuracy gains or training-time conflict rates.
