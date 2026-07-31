# Architecture and output contracts

## Runtime flow

```text
DAT / TIFF / CBVS / SEG-Y
          │
          ├─ unified reader → canonical [Z,Y,X] float32 + valid-trace mask
          ├─ optional standalone FEF preprocessing
          │
          ├─ inference profile → checkpoint + default threshold + sweep grid
          │
          ├─ patch normalization → 3D U-Net → weighted overlap blending
          │
          ├─ threshold selection → profile / fixed / Otsu / quantile / labeled best
          │
          └─ probability + uint8 mask + preview + threshold sweep + JSON
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `src/model.py` | Network definition only |
| `src/checkpoint.py` | Native and converted checkpoint loading |
| `src/data.py` | Raw volume I/O, normalization, paired datasets |
| `src/cbvs.py` | Memory-mapped post-stack OpendTect CBVS reader |
| `src/volumes.py` | Unified TIFF/CBVS/DAT/SEG-Y reader and SEG-Y writer |
| `src/filters.py` | F3 fault-enhancement similarity/median/diffusion filter |
| `src/inference.py` | Device selection, tiling, patch normalization, blending |
| `src/pipeline.py` | Software-facing one-volume prediction/evaluation API |
| `src/profiles.py` | Input-domain detection and portable profile configuration |
| `src/thresholds.py` | Threshold strategies and probability summaries |
| `src/visualization.py` | Orthogonal QC and threshold-sweep figures |
| `src/losses.py` | Standard and conservative false-positive-aware losses |
| `src/metrics.py` | Streaming labeled segmentation metrics |

CLI files under `script/` contain argument parsing and format-specific
orchestration only. Shared numerical logic belongs under `src/`.

## Axis and dtype contract

- Internal arrays: `[Z,Y,X]`, `float32`.
- Model tensors: `[N,1,Z,Y,X]`, normalized `float32`.
- Probability TIFF: `[Z,Y,X]`, `float32`, range `[0,1]`.
- Mask TIFF: `[Z,Y,X]`, `uint8`, values `{0,1}`.
- Missing CBVS traces remain invalid during normalization and are forced to zero
  in probability and mask outputs.
- SEG-Y geometry defaults to inline/crossline header bytes 189/193 and preserves
  the source trace order when an enhanced SEG-Y is written.
- Raw `.dat` output returns to the source storage order unless
  `--model-order-output` is specified.

## Threshold policy

Profile thresholds are versioned configuration, not network constants. Every
run records profile, checkpoint, threshold value, threshold method, positive
fraction and probability percentiles in JSON. A survey-specific supervised
calibration takes precedence over profile defaults; Otsu and quantile modes are
fallback exploratory tools.

Saved probability TIFF is the stable boundary between expensive neural-network
inference and interactive interpretation. `script/adjust_threshold.py` regenerates masks
and QC views without running the model again; a future UI should call the same
threshold and visualization functions when its slider changes.

## Fault-enhancement contract

The F3 OpendTect graph is reconstructed as normalized opposite-neighbour trace
similarity, lateral `3×3` median, median value at the minimum-similarity position,
then a `similarity < 0.85` gate. Standalone SEG-Y does not carry the separately
stored dip-steering cube, so `script/preprocess_sgy.py` labels its output
`non-steered-approximation` in JSON. It must not be presented as sample-identical
to OpendTect's dip-steered output.

## Migration checklist

1. Copy the repository, including `model/` checkpoint files.
2. Install with `python -m pip install -e .`.
3. Run `faultseg-verify` and `python -m unittest discover -s tests`.
4. Keep survey data outside Git or under ignored `data/`.
5. Put generated volumes under ignored `output/`.
6. Preserve only compact reports/QC figures under `results/` or `docs/`.
