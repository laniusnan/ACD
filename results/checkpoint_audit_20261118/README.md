# Checkpoint audit — 2026-11-18

See [the full audit](../../docs/checkpoint_audit_20261118.md). These files describe 11 CIFAR-100 pairs × 7 arms. They are an inventory and unscheduled gap list, not a training queue.

`inventory.json` preserves per-run provenance and separates a file located in a current release from a result record without surviving weights. The 39 missing cells and 6 additional recipe gaps use different definitions; neither is a claim that the historical experiment never ran.
