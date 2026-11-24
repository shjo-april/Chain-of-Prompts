<div align="center">

# One Click per Cell Type Suffices: Training-free Group Interaction for Cell Instance Segmentation

<a href="https://accv2026.org/conference-program/#oral-8"><img src="https://img.shields.io/badge/ACCV%202026-Oral-c8102e?style=for-the-badge" height="36" alt="ACCV 2026: Oral"></a>
<a href="https://asci.artificialintelligencepathology.org/#asci-spotlights"><img src="https://img.shields.io/badge/NeurIPS%202026%20Workshop%20ASCI-Spotlight-c8102e?style=for-the-badge" height="36" alt="NeurIPS 2026 Workshop ASCI: Spotlight"></a>

**ACCV 2026 Oral**: one of 36 oral papers, an estimated top 3% of submissions<br>
**NeurIPS 2026 Workshop ASCI Spotlight**: one of 10 spotlight talks, non-archival

**Method: Chain-of-Prompts (CoP)**

[![Project Page](https://img.shields.io/badge/Project-Page-gold?style=for-the-badge)](https://shjo-april.github.io/Chain-of-Prompts)
[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge)](https://arxiv.org/pdf/2605.29429)
[![arXiv](https://img.shields.io/badge/arXiv-2605.29429-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2605.29429)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-8c1b13?style=for-the-badge)](./LICENSE)

[Sanghyun Jo](https://shjo-april.github.io/)<sup>1,2</sup>, Seo Jin Lee<sup>1</sup>, Seohyung Hong<sup>1</sup>, Yoorim Gang<sup>1</sup>, [Hyeongsub Kim](https://sites.google.com/view/hyeongsub/)<sup>1,3</sup>, Hyungseok Seo<sup>1†</sup>, [Kyungsu Kim](https://aibl.snu.ac.kr/pi-information)<sup>1†</sup>

<sup>1</sup>Seoul National University &nbsp; <sup>2</sup>OGQ &nbsp; <sup>3</sup>LG CNS &nbsp; <sup>†</sup>Corresponding authors

</div>

---

> **ACCV 2026 (Oral).** Accepted at ACCV 2026, the Asian Conference on Computer Vision, and
> selected as an **Oral** presentation: one of **36 oral papers** among 378 accepted, presented
> in [Oral Session 8: Segmentation & Medical Imaging](https://accv2026.org/conference-program/#oral-8).
> The top 3% figure is an estimate: ACCV 2026 has not published its submission count, and paper
> IDs run up to 1417 (36 / 1417 ≈ 2.5%).
>
> **NeurIPS 2026 Workshop ASCI (Spotlight, non-archival).** The same paper was also selected as
> one of 10 **Spotlight** talks at NeurIPS 2026 Workshop ASCI, [AI at Scale for Clinical Impact: Cancer Pathology Foundation Models](https://asci.artificialintelligencepathology.org/#asci-spotlights).
> This workshop version is **non-archival**: it is not included in the workshop proceedings, so
> please cite the ACCV 2026 paper.

> **This is the demo release.** It contains the training-free inference core, the unified
> SAM interface, a demo notebook, and the benchmark test splits. Evaluation
> will be added incrementally. Please ⭐ star and 👁️ watch for updates.

<div align="center">
  <img src="./assets/hero.gif" width="100%">
  <br><em>The same clicks with and without CoP, on an H&E image (CoNSeP) and a label-free phase-contrast image (LIVECell). SAM 3 alone segments only the clicked cells (AJI 0.009 and 0.004), while CoP reaches AJI 0.451 and 0.699 from the same 3 clicks and 1 click. Clicking every cell takes 245 and 87 clicks to reach 0.499 and 0.748. <a href="#-reproducing-the-hero-gif">Reproduce this GIF.</a></em>
</div>

## 💡 TL;DR

Cell-specific models break on unseen cell types, and interactive foundation models such as
SAM 3 need one click per cell. **Chain-of-Prompts (CoP)** turns a **single click per cell
type** into every same-type instance, shifting interaction from per-instance **O(N)** to
per-type **O(T)**, fully training-free. Across **eleven benchmarks** it retains **over 90%**
of the per-instance upper bound while cutting annotation clicks by **97.6%** on average
(**42.5× fewer**). On the CoNSeP example below, **3 clicks replace 245** (81.7× fewer) at
**93.9%** of the upper bound.

<div align="center">
  <img src="./assets/teaser.png" width="100%">
  <br><em>Figure 1. Pretrained cell-specific models fail on unseen cell types (dashed boxes) and interactive models need one click per cell; CoP needs only one click per cell type.</em>
</div>

<div align="center">
  <img src="./assets/figure2.png" width="100%">
  <br><em>Figure 2. From 245 clicks to 3. Manual per-instance prompting O(N) vs. CoP group prompting O(T) on CoNSeP/test_2, with the AJI-vs-#prompts curve: CoP reaches <b>93.9%</b> of the per-instance upper bound with <b>81.7× fewer prompts</b>.</em>
</div>

## 📊 Results

CoP is training-free and adds no parameters to the frozen backbone. It is evaluated on
**eleven benchmarks** on their official test splits, in three groups. Numbers below are AJI
(see the tables for Dice and the full baseline set).

### One click per cell type: H&E, cell-type-annotated

A single click per cell type (𝒫ₜ) keeps **over 90%** of SAM 3's per-instance upper bound on
all three datasets (e.g., **0.731 vs. 0.801** AJI on CoNIC) and **surpasses every
fully-supervised and open-vocabulary baseline**, with no training. The table spans six
regimes: text (𝒯), visual (𝒱), unsupervised (✗), mask supervision (ℳ), per-instance points
(𝒫ₙ), and per-type points (𝒫ₜ) across **CoNIC, CoNSeP, and PanNuke**.

<div align="center">
  <img src="./assets/table_typed.png" width="100%">
</div>

### One click per image: H&E, untyped

When cells are morphologically homogeneous they act as a single type, so one click per image
propagates to the whole field: CoP retains **98–99%** of the per-instance upper bound for both
μSAM and SAM 3, still outperforming every fully-supervised baseline, on **MoNuSeg, TNBC,
CryoNuSeg, and CPM-17**.

<div align="center">
  <img src="./assets/table_he_untyped.png" width="100%">
</div>

### One click per image: non-H&E, untyped

Under a staining-modality shift, H&E-trained supervised models collapse (CellViT drops from
**0.676** AJI on MoNuSeg to **0.002** on CellBinDB), while CoP, reading only the frozen
encoder's features, preserves **over 95%** of the per-instance upper bound on all four
non-H&E benchmarks: **CellBinDB, Cellpose, Kromp, and LIVECell**.

<div align="center">
  <img src="./assets/table_non-he_untyped.png" width="100%">
</div>

### Qualitative Comparison

<div align="center">
  <img src="./assets/qualitative.png" width="100%">
  <br><em>Figure 4. Fully-supervised baselines miss cell populations outside their training distribution and collapse to near-empty masks under a modality change (dashed boxes); CoP recovers the missing instances from one click per type, or per image when types are unlabeled.</em>
</div>

## 🔍 How It Works

<div align="center">
  <img src="./assets/method_overview.png" width="100%">
</div>

A frozen SAM encoder is run **once** per image, producing a high-resolution feature map
𝐹ₕ and a low-resolution feature map 𝐹ₗ. For each user click, CoP applies two
training-free steps with no backprop:

- **HSG (Hierarchical Similarity Gating).** Gates the two scales via the element-wise product
  𝑆ₕ ⊙ 𝑆ₗ (𝐹ₕ localizes densely while 𝐹ₗ isolates the cell type), thresholds the
  gated map non-parametrically at **τ = μ + σ**, and runs connected-component labeling to
  extract a set of reliable same-type points (**precision > 96%** at every iteration).
- **FPR (Farthest Prompt Recursion).** Re-prompts the reliable point farthest (in image space)
  from all previous clicks, feeds it back into HSG, and repeats until no new points appear.

The converged point set is decoded into instance masks by the frozen mask decoder
(overlap guard + NMS at IoU > 0.5).

<div align="center">
  <img src="./assets/umap.png" width="100%">
  <br><em>Figure 5. UMAP of the frozen SAM encoder's features at GT cell centroids (CoNSeP/test_2). 𝐹ₕ mixes cell types, while 𝐹ₗ already groups same-type cells, before any prompt and without training.</em>
</div>

The frozen SAM encoder already clusters same-type cells in feature space before any prompt is
given, so grouping needs the right propagation rule, not fine-tuning. This structure comes
from SAM's prompt–mask pretraining, not encoder scale: swapping in DINOv3, EVA-CLIP, or SDXL
features drops CoP to ≤ 0.435 AJI on CoNIC (vs. 0.731 with SAM 3).

## 📦 What's in This Release

- **Training-free core (`core/cop.py`).** HSG + FPR + decoding: the entire method in one
  compact file that maps one to one to the paper. It carries **no evaluation, metric, or
  plotting code**, so the method itself is easy to read and reuse.
- **Unified SAM interface (`core/sam`).** A single `ImageSAM(checkpoint, backend)` (and
  `VideoSAM`) API across the SAM family used in the paper: **SAM1-H**, **μSAM (micro-SAM)**,
  **SAM 2.1**, and **SAM 3**, plus HQ-SAM, ZIM, and EdgeTAM. Frozen multi-scale encoder
  features are exposed through `encode_image()`, which is exactly what CoP consumes, so
  swapping a backbone is a one-line change.
- **Runnable demo.** [`demo.ipynb`](./demo.ipynb) is an image-only minimal run of CoP on a
  single image.
- **Reproducible hero GIF (`hero_gif/`).** The two scripts and the recorded runs behind the GIF
  at the top (see [Reproducing the Hero GIF](#-reproducing-the-hero-gif)).
- **Benchmark test splits** ship in a unified layout (see [Datasets](#-datasets)).

## 🗂️ Repository Structure

```
Chain-of-Prompts/
├── core/
│   ├── cop.py                 # Chain-of-Prompts: HSG + FPR + decoding (the public core)
│   └── sam/                   # Unified SAM interface (SAM1-H, μSAM, SAM 2.1, SAM 3, HQ-SAM, ZIM, EdgeTAM)
│       ├── __init__.py        #   ImageSAM / VideoSAM wrappers (encode_image, predict)
│       ├── build_sam.py       #   backend builders
│       ├── predictor.py       #   per-backend predictors
│       ├── configs/           #   model configs
│       └── modeling/          #   sam1 / sam2 / sam3 / zim model code
├── examples/                  # demo image + Figure 2 reference data
├── assets/                    # paper figures + hero GIF
├── hero_gif/                  # scripts + recorded runs that reproduce assets/hero.gif
├── demo.ipynb                 # image-only minimal demo
├── data_H@E_typed/            # H&E, cell-type-annotated (CoNIC, CoNSeP, PanNuke)
├── data_H@E_untyped/          # H&E, untyped (MoNuSeg, TNBC, CryoNuSeg, CPM-17)
├── data_non-H@E_untyped/      # non-H&E microscopy (CellBinDB, Cellpose, Kromp, LIVECell)
├── requirements.txt
└── DATASETS.md                # dataset licenses and attributions
```

Where to look first: `core/cop.py` is the entire method; `core/sam/__init__.py` is the one
interface used for every backbone.

## 🛠️ Installation

```bash
python3 -m venv venv && source ./venv/bin/activate     # Windows: .\venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.1, CUDA ≥ 11.8, a single GPU (the demo runs on a
12 GB card).

### SAM 3 checkpoint (gated)

SAM 3 is released by Meta under the [SAM License](https://ai.meta.com/sam) and is **gated**
on Hugging Face. We do **not** redistribute the weights. To run the demos:

1. Accept the license and request access at the official model page (`facebook/sam3`).
2. Download the checkpoint into a local folder, for example `./checkpoints/sam3/`.
3. Point the notebooks to it via `MODEL_DIR`.

```python
from core.sam import ImageSAM
model = ImageSAM("./checkpoints/sam3", "sam3")     # backend in {sam1, sam2, sam3, zim}
```

## 🚀 Quick Start

Minimal, image only:

```python
import cv2
from core.sam import ImageSAM
from core.cop import ChainOfPrompts

model = ImageSAM("./checkpoints/sam3", "sam3")
cop   = ChainOfPrompts(model)

image  = cv2.imread("./examples/CoNSeP_test_2.png")     # BGR
clicks = [(292, 395), (171, 539), (684, 958)]           # one click per cell type
label_map, overlay, info = cop.segment(image, clicks)
print(info["num_instances"], "cells from", info["num_clicks"], "clicks")
```

- [`demo.ipynb`](./demo.ipynb): one click per cell type, round by round, on a single image.

## 🎬 Reproducing the Hero GIF

The GIF at the top is drawn from real runs: every mask, click, and AJI value in it comes from the
recorded outputs in [`hero_gif/results`](./hero_gif/results). It is built in two steps.

```bash
python hero_gif/build_gif.py                               # CPU: recorded runs -> assets/hero.gif
python hero_gif/run_models.py --ckpt ./checkpoints/sam3    # GPU: rerun both scenes -> outputs/hero_gif
```

- [`build_gif.py`](./hero_gif/build_gif.py) composes the frames from
  [`scenes.json`](./hero_gif/scenes.json) (images, clicks, captions) and the recorded runs, with
  no model. With the Segoe UI of Windows 10 and Pillow 11.3.0 it reproduces `assets/hero.gif`
  byte for byte. Without Segoe UI it falls back to DejaVu Sans: same layout, different glyphs.
- [`run_models.py`](./hero_gif/run_models.py) reruns SAM 3 on both scenes and reports whether the
  recorded run is reproduced exactly. Per scene it runs CoP from the scene's clicks, the same
  clicks without CoP, and per-cell clicking (the same clicks first, then the centroid of every
  other ground-truth cell in id order), scoring the exact AJI as the clicks accumulate. Pass
  `--out hero_gif/results` to build the GIF from your own run.

| Scene | Cells | Clicks | Same clicks, w/o CoP | w/ CoP | Every cell clicked, w/o CoP | Clicks for w/o CoP to match |
|---|--:|--:|--:|--:|--:|--:|
| H&E: CoNSeP `test_2` | 245 | 3 | 0.009 | **0.451** | 0.499 (245 clicks) | 212 |
| Non-H&E: LIVECell `SKOV3_Phase_F4_1_01d00h00m_3` | 87 | 1 | 0.004 | **0.699** | 0.748 (87 clicks) | 82 |

Values are AJI against the ground truth for one recorded run per image. They are not benchmark
averages (see [Results](#-results)), and for CoNSeP `test_2` they are not the numbers of Figure 2,
which comes from the paper's own run of the same image and clicks.

- **Ground truth.** The LIVECell pair ships in `data_non-H@E_untyped/LIVECell/test/`. CoNSeP is
  not redistributable, so `run_models.py` needs its instance mask at the path given in
  `scenes.json` (`build_gif.py` does not).
- **Click order.** A mask that overlaps two or more existing cells is rejected, so per-cell
  clicking depends on the order of the clicks. The GIF fixes it to the ground-truth id order.
- **LIVECell click.** The click follows the paper's rule (a random reliable cell) at seed 3,
  which lands on a central cell. Seeds 0 to 9 give AJI 0.44 to 0.72 (mean 0.63); their clicks are
  the `seed_clicks` of `scenes.json`.
- **Environment.** The recorded run used PyTorch 2.8.0 (CUDA 12.8). Results on a single image
  vary with the numerical environment (PyTorch, CUDA, GPU), because the propagated points are
  not identical across machines. The same script on another GPU with PyTorch 2.7.0 (CUDA 11.8)
  gave AJI 0.4505 w/ CoP and 0.5058 with every cell clicked on CoNSeP (recorded: 0.4508 and
  0.4987), 0.6600 and 0.7508 on LIVECell (recorded: 0.6993 and 0.7477), and 0.47 to 0.72
  (mean 0.66) over the ten seed clicks. `run_models.py` lists what differs from the recorded
  run, and the committed GIF always shows the recorded one.

## 📁 Datasets

CoP is evaluated on **eleven benchmarks** on their official test splits, reorganized into a
single unified layout and grouped along two axes: **staining** (H&E vs. non-H&E) and
**type annotation** (typed vs. untyped):

```
data_H@E_typed/                                          # H&E, cell-type-annotated  (one click per type)
├── CoNIC/test/{image, mask_instance, mask_semantic}     #   4,980 images · 256×256      · 6 types
├── CoNSeP/test/{image, mask_instance, mask_semantic}    #      14 images · 1000×1000    · 7 types
└── PanNuke/Fold3/{image, mask_instance, mask_semantic}  #   2,722 images · 256×256      · 5 types  (eval split = Fold3)

data_H@E_untyped/                                        # H&E, untyped  (one click per image)
├── MoNuSeg/test/{image, mask_instance}                  #      14 images · 1000×1000
├── TNBC/test/{image, mask_instance}                     #      10 images · 512×512
├── CryoNuSeg/test/{image, mask_instance}                #      10 images · 512×512
└── CPM-17/test/{image, mask_instance}                   #      32 images · 500–600 px

data_non-H@E_untyped/                                    # non-H&E microscopy, untyped  (one click per image)
├── CellBinDB/test/{image, mask_instance}                #     303 images · 256×256      (DAPI subset)
├── Cellpose/test/{image, mask_instance}                 #      68 images · 135–576 px
├── Kromp/test/{image, mask_instance}                    #      37 images · 430–1024 px
└── LIVECell/test/{image, mask_instance}                 #   1,512 images · 520×704       (label-free phase-contrast)
```

Within a dataset the subfolders share one filename per example: `image` is the input crop,
`mask_instance` is the instance label map (instance id encoded as B·256 + G, gap-free, decoded
by unique RGB color), and `mask_semantic` (typed datasets only) is the per-cell-type label map
(0 = background, 1..T). Together the eleven test splits total **9,702 images**.

Only data we are licensed to redistribute is published here: **full** test splits of **MoNuSeg,
TNBC, CryoNuSeg, and Cellpose**, and a **1-image format sample** of **CoNIC, PanNuke, and
LIVECell** (redistributable but large; LIVECell also ships the image used in the hero GIF). The
datasets with no clear redistribution license
(**CoNSeP, CPM-17, CellBinDB, Kromp**) are **not shipped**; download them from their official
sources and recreate the layout. Full source links, licenses, and per-dataset statistics are in
[DATASETS.md](./DATASETS.md).

## 🙏 Acknowledgements

CoP is a successor to our annotation-free cell segmentation work
**[COIN (ICCV 2025)](https://github.com/shjo-april/COIN)**, and is powered by the
**Segment Anything Model** family from Meta. We thank the
authors of every dataset and foundation model we build on.

## 📖 Citation

```bibtex
@inproceedings{jo2026cop,
  title     = {One Click per Cell Type Suffices: Training-free Group Interaction for Cell Instance Segmentation},
  author    = {Jo, Sanghyun and Lee, Seo Jin and Hong, Seohyung and Gang, Yoorim and Kim, Hyeongsub and Seo, Hyungseok and Kim, Kyungsu},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## 📜 License

Our first-party code is released under [**CC BY-NC 4.0**](https://creativecommons.org/licenses/by-nc/4.0/)
(Creative Commons Attribution-NonCommercial 4.0): free for **non-commercial academic
research and education** with attribution. Any commercial use requires prior written
permission. For commercial licensing, please contact **both**:

- Sanghyun Jo: shjo.april@gmail.com
- Kyungsu Kim: kyskim@snu.ac.kr

The vendored Segment Anything model code under `core/sam` and the SAM 3 weights are
governed by Meta's [SAM License](https://ai.meta.com/sam). Datasets follow their original
licenses (see [DATASETS.md](./DATASETS.md)). The logos under `hero_gif/logos` belong to their
institutions and are not covered by this license.
