# Artifacts

Generated benchmark artifacts belong here and are ignored by default.

Exception: `analysis/article/` is checked in because it contains the small
GitHub-facing result tables referenced by the findings note.

Expected subdirectories:

- `frozen/` for BFCL manifests and clean subset records.
- `generated/` for candidate noisy examples.
- `audits/` for automatic and manual audit logs.
- `accepted/` for verified noisy examples.
- `results/` for clean, noisy, and paired evaluation outputs.
- `analysis/` for metrics, tables, and figures.
