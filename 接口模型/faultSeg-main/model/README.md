# Bundled model

`faultseg-best.pt` is the only deployment checkpoint included with this
repository. It contains 1,459,585 parameters and is approximately 5.6 MiB.

Validation performance on 20 synthetic `128³` volumes at the calibrated Dice
threshold of `0.518`:

| Precision | Recall | Dice | IoU |
|---:|---:|---:|---:|
| 0.8342 | 0.8685 | 0.8510 | 0.7406 |

