# Dataset Licenses and Attributions

This repository ships the **test split only** of the seven benchmarks used in the paper,
reorganized into a unified structure for reproducibility:

```
./data/{dataset}/test/{image, mask, mask_semantic}        # cell-type-annotated
./data_wo_type/{dataset}/test/{image, mask}               # morphologically homogeneous
```

Each dataset is redistributed under its original license, with attribution to the source
authors. Where a dataset has no clear license, we **do not redistribute the data** and
provide the official download link instead (following the policy of
[COIN](https://github.com/shjo-april/COIN)). Please review the original terms before use,
and cite the original works in any publication.

---

## Cell-type-annotated benchmarks (`./data`)

### 1. CoNIC
- Source: https://conic-challenge.grand-challenge.org/
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (built on the Lizard dataset)
- **Not included** here due to its size (about 700 MB). Please download it from the source. The license permits non-commercial redistribution with attribution and same-license sharing.

### 2. CoNSeP
- Source: https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/
- License: none stated; the original page is currently unavailable.
- **Not included** due to the missing license. Please obtain it from the source.

### 3. GlaS
- Source: https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/
- License: no explicit license; provided for non-commercial research under the original challenge terms.
- Included with attribution. If you require a formal license, please contact the original authors.

---

## Morphologically homogeneous benchmarks (`./data_wo_type`)

### 4. MoNuSeg
- Source: https://monuseg.grand-challenge.org/
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Included. Redistribution permitted for non-commercial use with attribution and same-license sharing.

### 5. TNBC
- Source: https://zenodo.org/record/2579118
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Included. Redistribution permitted with attribution.

### 6. CryoNuSeg
- Source: https://github.com/masih4/CryoNuSeg
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Included. Redistribution permitted for non-commercial use with attribution and same-license sharing.

### 7. CPM-17
- Source: https://drive.google.com/drive/folders/1sJ4nmkif6j4s2FOGj8j6i_Ye7z9w0TfA
- License: not specified.
- **Not included** due to the unclear license. Please download it manually from the source.

---

For licensing or ethical concerns, please contact the original dataset creators.
