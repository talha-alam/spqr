# Data Access Policy — SPQR Benchmark

> **ECCV 2026 Compliance Notice.** In accordance with the ECCV 2026 Dataset/Benchmark Release Policy, this document discloses which components of the SPQR benchmark are freely available and which require a lightweight access request. Access is **never categorically denied**; see below for how to request the restricted components.

---

## Summary of Availability

| Component | Status | How to Access |
|-----------|--------|---------------|
| Full evaluation code & scripts | ✅ Public | This repository |
| BFT training code (all 3 profiles) | ✅ Public | This repository |
| Benign fine-tuning datasets (COCO, multilingual, artistic, medical) | ✅ Public | Original sources (see links below) |
| SPQR metric computation library | ✅ Public | This repository |
| Visualization & analysis notebooks | ✅ Public | This repository |
| **Harmful evaluation prompts** (ViSU, I2P, RAB compilations in `data/prompts/`) | ⚠️ Restricted | [Access request form ↓](#requesting-access) |

---

## Freely Available Components

All code in this repository — including the SPQR metric implementations, BFT training scripts, evaluation pipelines, and visualization notebooks — is released under the **MIT License** and requires no access request.

### Benign Fine-Tuning Datasets

These are sourced from publicly available datasets and must be obtained from their original providers:

- **General (COCO)** — [cocodataset.org](https://cocodataset.org) (CC BY 4.0)
- **Multilingual COCO** — [github.com/tylin/coco-caption](https://github.com/tylin/coco-caption) and multilingual extensions
- **Artistic dataset** — Sourced from Stability AI's publicly released assets
- **Medical (NIH CXR)** — [kaggle.com/datasets/nih-chest-xrays](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **Medical (Brain MRI)** — [kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## Restricted Components: Harmful Evaluation Prompts

The `data/prompts/` directory contains curated and compiled prompt sets derived from **ViSU**, **I2P**, and **Ring-A-Bell (RAB)**, used to evaluate Safety (S) and Robustness (R) in the SPQR benchmark. Because these prompts describe sexually explicit, violent, or otherwise harmful scenarios, we gate access behind a lightweight acknowledgment form to:

1. Ensure use is limited to **safety research and benchmark evaluation**,
2. Discourage misuse for generating harmful content, and
3. Maintain a record of the research community using this benchmark, in line with responsible disclosure norms.

> **Note on original datasets.** ViSU, I2P, and RAB are independent works with their own terms. Requestors remain responsible for complying with those terms directly. Links: [ViSU / SafeCLIP](https://github.com/aimagelab/safe-clip), [I2P](https://huggingface.co/datasets/AIML-TUDA/i2p), [Ring-A-Bell](https://github.com/chiayi-hsu/Ring-A-Bell).

---

## Requesting Access

Access to the harmful prompt compilations in `data/prompts/` is granted via a **GitHub Issue form**. The process is:

1. Open a new Issue using the **[📋 Data Access Request](../../issues/new?template=data_access_request.yml)** template.
2. Complete all required fields (name, institution, research purpose, terms agreement).
3. A maintainer will review your request within **5 business days**.
4. Upon approval, you will receive access to the prompts either directly in this repository (unlocked path) or via a private release link.

Requests are not rejected without cause. Legitimate academic, industrial safety research, and red-teaming purposes are all acceptable.

---

## Terms of Use for Harmful Prompts

By requesting and receiving access to `data/prompts/`, you agree to the following:

1. **Research use only.** You will use these prompts solely for academic research, safety evaluation, or benchmark reproduction — not to generate, distribute, or solicit harmful content.
2. **No redistribution.** You will not publicly redistribute, re-host, or publish the raw prompt files beyond what is necessary for reproducibility in a peer-reviewed publication.
3. **Attribution.** Any work using these prompts must cite the SPQR paper (see below) and the original dataset papers (ViSU, I2P, RAB).
4. **Compliance with source terms.** You will independently comply with the terms of use of the original datasets (ViSU, I2P, RAB).
5. **Misuse reporting.** If you discover that these prompts are being used to generate harmful content, please notify us at `mohammed.alam@mbzuai.ac.ae`.

---

## Citation

If you use the SPQR benchmark or its harmful prompt compilations, please cite:

```bibtex
@inproceedings{alam2026spqr,
  title     = {SPQR: A Multi-Dimensional Benchmark for Safety Alignment under Benign Model Adaptation},
  author    = {Alam, Mohammed Talha and Saadi, Nada and Shamshad, Fahad and Lukas, Nils
               and Nandakumar, Karthik and Karray, Fakhri and Poppi, Samuele},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

---

## Contact

For questions about data access or the terms of use:

**Mohammed Talha Alam** — mohammed.alam@mbzuai.ac.ae  
Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
