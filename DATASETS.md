# Datasets

Chain-of-Prompts is evaluated on **11 cell/nucleus instance-segmentation benchmarks**, each on
its official evaluation split, reorganized into a single unified layout. No dataset is authored
by us; every dataset is third-party and governed by its own license (tables below).

## Three groups

The eleven benchmarks split along two axes, staining (H&E vs. non-H&E) and cell-type annotation:

- **H&E, typed** (carry `mask_semantic`; one click *per cell type*): CoNIC, CoNSeP, PanNuke.
- **H&E, untyped** (single implicit type; one click *per image*): MoNuSeg, TNBC, CryoNuSeg, CPM-17.
- **non-H&E, untyped** (other microscopy modalities; one click *per image*): CellBinDB (DAPI),
  Cellpose, Kromp, LIVECell.

## Folder layout

The data is split into three top-level folders, one per group:

```
data_H@E_typed/{dataset}/{split}/{image, mask_instance, mask_semantic}   # CoNIC, CoNSeP, PanNuke
data_H@E_untyped/{dataset}/test/{image, mask_instance}                   # MoNuSeg, TNBC, CryoNuSeg, CPM-17
data_non-H@E_untyped/{dataset}/test/{image, mask_instance}               # CellBinDB, Cellpose, Kromp, LIVECell
```

- `image/` RGB input: an H&E crop, or a converted microscopy image (non-H&E).
- `mask_instance/` instance-id label map, RGB-encoded as **id = B·256 + G**, gap-free (the R
  channel is unused; black `(0,0,0)` = background). Decoding is by unique color, so the exact
  channel packing does not affect scoring: the evaluator treats each distinct color as one
  instance.
- `mask_semantic/` palette-indexed PNG, pixel value = cell-type id (0 = background, 1..T).
  **Typed datasets only** (CoNIC, CoNSeP, PanNuke).

The evaluation split is `test` for every dataset **except PanNuke**, which uses a non-standard
`Fold1 / Fold2 / Fold3` split and is evaluated on the last fold (`PanNuke/Fold3/`, not
`PanNuke/test/`). CellBinDB is multimodal (DAPI, mIF, ssDNA, H&E); this work uses **only the
DAPI subset (303 images)** and refers to it simply as CellBinDB.

## What is included in this repository

Only data we are licensed to redistribute is published here. Everything else is shipped as an
**empty folder structure** (each leaf carries a `PLACE_FILES_HERE.txt` placeholder so git keeps
the folders) together with the official download link, following the
[COIN](https://github.com/shjo-april/COIN) policy.

- **Full test split** (redistributable license, small enough to host): **MoNuSeg, TNBC,
  CryoNuSeg, Cellpose**.
- **Format sample, 1 image** (redistributable license, full split omitted only for repository
  size): **CoNIC, PanNuke, LIVECell**. The full splits are redistributable; download them from
  the source to reproduce the numbers.
- **Folder structure only, no data** (no clear redistribution license): **CoNSeP, CPM-17,
  CellBinDB, Kromp**. Download each from its source and drop the files into the prepared folders.

Reproducing the reported metrics requires the full evaluation split of every dataset, obtained
from its official source (tables below).

## Dataset statistics (evaluation split)

N/img and T/img are the mean per-image counts of per-instance ($\mathcal{P}_N$) and per-type
($\mathcal{P}_T$) prompting; click reduction is $1 - \sum_k T_k / \sum_k N_k$ over the split.

| Dataset | Group | #Types | Eval split | #Images | N/img | T/img | Click reduction |
|---|---|:--:|---|--:|--:|--:|--:|
| CoNIC | H&E typed | 6 | test | 4,980 | 114.4 | 3.88 | 96.6% |
| CoNSeP | H&E typed | 7 | test | 14 | 626.9 | 3.86 | 99.4% |
| PanNuke | H&E typed | 5 | Fold3 | 2,722 | 24.5 | 2.05 | 91.6% |
| MoNuSeg | H&E untyped | - | test | 14 | 478.4 | 1.00 | 99.8% |
| TNBC | H&E untyped | - | test | 10 | 67.2 | 1.00 | 98.5% |
| CryoNuSeg | H&E untyped | - | test | 10 | 242.4 | 1.00 | 99.6% |
| CPM-17 | H&E untyped | - | test | 32 | 118.9 | 1.00 | 99.2% |
| CellBinDB (DAPI) | non-H&E | - | test | 303 | 80.5 | 1.00 | 98.8% |
| Cellpose | non-H&E | - | test | 68 | 105.9 | 1.00 | 99.1% |
| Kromp | non-H&E | - | test | 37 | 137.5 | 1.00 | 99.3% |
| LIVECell | non-H&E | - | test | 1,512 | 297.6 | 1.00 | 99.7% |
| **All (11)** | - | - | - | **9,702** | 118.1 | 2.78 | **97.6%** |

7 H&E + 4 non-H&E; 3 typed + 8 untyped. Total 9,702 evaluation images, mean 118.1 cells/image.
Because untyped datasets take one click per image (T/img = 1.00), a single prompt replaces
hundreds of per-instance clicks (e.g. LIVECell: 297.6 → 1.00).

## Licenses, sources, and inclusion

Licenses were verified from each dataset's official page or publication where accessible. The
Kromp pipeline (`perlfloccri/NuclearSegmentationPipeline`) is MIT-licensed, but that covers the
code only; its image dataset (BioStudies S-BSST265) states no redistribution license, so we
treat Kromp as unlicensed for redistribution. "In repo" is what is physically shipped here, per
the inclusion policy above.

| Dataset | Source | License | In repo |
|---|---|---|---|
| CoNIC | https://conic-challenge.grand-challenge.org/ | CC BY-NC-SA 4.0 (built on Lizard) | sample (1 image) |
| CoNSeP | https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/ | none stated | structure only |
| PanNuke | https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/ | CC BY-NC-SA 4.0 | sample (1 image) |
| MoNuSeg | https://monuseg.grand-challenge.org/ | CC BY-NC-SA 4.0 | full |
| TNBC | https://zenodo.org/record/2579118 | CC BY 4.0 | full |
| CryoNuSeg | https://github.com/masih4/CryoNuSeg | CC BY-NC-SA 4.0 | full |
| CPM-17 | https://drive.google.com/drive/folders/1sJ4nmkif6j4s2FOGj8j6i_Ye7z9w0TfA | not specified | structure only |
| CellBinDB (DAPI) | https://doi.org/10.1093/gigascience/giaf069 | no clear redistribution license | structure only |
| Cellpose | https://www.cellpose.org/ | CC BY-NC | full |
| Kromp | https://github.com/perlfloccri/NuclearSegmentationPipeline | code MIT; dataset license not stated | structure only |
| LIVECell | https://sartorius-research.github.io/LIVECell/ | CC BY-NC 4.0 | sample (1 image) |

## Cell types (typed datasets)

- **CoNIC** (6 types; colon, Lizard-derived): Neutrophil, Epithelial, Lymphocyte, Plasma,
  Eosinophil, Connective.
- **CoNSeP** (7 types; colorectal adenocarcinoma, HoVer-Net original labels): Other,
  Inflammatory, Healthy epithelial, Dysplastic/malignant epithelial, Fibroblast, Muscle,
  Endothelial.
- **PanNuke** (5 types; 19 tissues): Neoplastic, Inflammatory, Connective/Soft, Dead, Epithelial.

## License & attribution

Each redistributed dataset is shared only under its original license and with attribution to
the source authors. Where a dataset has no clear or redistributable license (CoNSeP, CPM-17,
CellBinDB, Kromp), we do not redistribute it and ship only the empty folder structure with the
official download link. Review the original terms before use and cite the original works in any
publication. For licensing or ethical concerns, contact the original dataset creators.
