# FaultSeg3D PyTorch

A portable PyTorch project for 3D seismic fault segmentation. It supports
training, evaluation, threshold calibration, visualization, and tiled inference
for DAT, TIFF, OpendTect CBVS, and SEG-Y volumes. A trained checkpoint and four
small demo inputs are included, so inference works immediately after setup.

## Origin and citation

This project is adapted from Xinming Wu's official Keras implementation:

- Original repository: [xinwucwp/faultSeg](https://github.com/xinwucwp/faultSeg)
- Paper: [FaultSeg3D: using synthetic datasets to train an end-to-end convolutional neural network for 3D seismic fault segmentation](https://doi.org/10.1190/geo2018-064_faultseg3d.1)

Please cite the original work:

```bibtex
@article{wu2019faultseg3d,
  author  = {Xinming Wu and Luming Liang and Yunzhi Shi and Sergey Fomel},
  title   = {FaultSeg3D: Using synthetic datasets to train an end-to-end
             convolutional neural network for 3D seismic fault segmentation},
  journal = {Geophysics},
  volume  = {84},
  number  = {3},
  pages   = {IM35--IM45},
  year    = {2019},
  doi     = {10.1190/geo2018-064_faultseg3d.1}
}
```

The original project is distributed for non-commercial research under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Data

Only compact demo volumes are stored in this repository. Full datasets are
available from the original authors:

- [Synthetic training and validation datasets](https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8)
- [Training seismic volumes](https://drive.google.com/open?id=1I-kBAfc_ag68xQsYgAHbqWYdddk4XHHd)
- [F3 subset and reference prediction](https://drive.google.com/drive/folders/1aw_f29yXloAeLclOvIshfuBukaOVQAJ1)
- [Netherlands F3 public survey](https://terranubis.com/datainfo/Netherlands-Offshore-F3-Block-Complete)

The `demo/` directory contains the same `32³` seismic crop in four formats:
`input.sgy`, `input.tif`, `input.cbvs`, and `input.dat`. `label.tif` is included
for a quick labeled evaluation.

## Environment

Python 3.8+ and PyTorch 2.4+ are recommended. CUDA is selected automatically.

```bash
git clone https://github.com/acse-ym722/faultSeg.git
cd faultSeg
python -m pip install -e .
faultseg-verify
```

## Quick inference

Run any bundled format with the included model:

```bash
faultseg-predict demo/input.sgy output/sgy --profile synthetic --patch-size 32,32,32 --overlap 8,8,8
faultseg-predict demo/input.tif output/tif --profile synthetic --patch-size 32,32,32 --overlap 8,8,8
faultseg-predict demo/input.cbvs output/cbvs --profile synthetic --patch-size 32,32,32 --overlap 8,8,8
faultseg-predict demo/input.dat output/dat --shape 32,32,32 --profile synthetic --patch-size 32,32,32 --overlap 8,8,8
```

Each command writes `probability.tif`, `mask.tif`, QC figures, and
`result.json`. A saved probability can be thresholded again without rerunning
the model:

```bash
faultseg-threshold output/tif/probability.tif output/tif/mask_p070.tif \
  --threshold 0.70 --seismic demo/input.tif
```

## Evaluation and model performance

The bundled `model/faultseg-best.pt` contains 1,459,585 parameters. On 20
synthetic `128³` validation volumes, using the Dice-calibrated threshold `0.518`:

| Precision | Recall | Dice | IoU | Specificity |
|---:|---:|---:|---:|---:|
| 0.8342 | 0.8685 | **0.8510** | 0.7406 | 0.9865 |

Run a labeled demo evaluation:

```bash
faultseg-predict demo/input.tif output/evaluation \
  --label demo/label.tif --threshold best --profile synthetic \
  --patch-size 32,32,32 --overlap 8,8,8
```

For a full downloaded validation set:

```bash
faultseg-evaluate --checkpoint model/faultseg-best.pt \
  --seis data/validation/seis --fault data/validation/fault \
  --threshold auto --optimize dice --output-dir output/evaluation
```
