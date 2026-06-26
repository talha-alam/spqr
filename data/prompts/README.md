# ⚠️ Harmful Evaluation Prompts — Access Required

This directory contains curated prompt compilations derived from **ViSU**, **I2P**, and **Ring-A-Bell (RAB)**, used to evaluate the **Safety (S)** and **Robustness (R)** axes of the SPQR benchmark.

Because these prompts describe sexually explicit, violent, or otherwise harmful scenarios, **access requires a brief acknowledgment form** in line with responsible disclosure norms and the ECCV 2026 Dataset Release Policy.

---

## How to Get Access

1. **Read** the full [Data Access Policy (`DATA_ACCESS.md`)](../../DATA_ACCESS.md).
2. **Submit** the [📋 Data Access Request Issue](../../issues/new?template=data_access_request.yml).
3. **Wait** for maintainer approval (~5 business days).
4. **Receive** the prompts or an access link in the Issue thread.

> Access is **never categorically denied**. All legitimate safety research and benchmark evaluation purposes are welcome.

---

## What's in This Directory (upon access)

| File | Source Dataset | Categories | # Prompts |
|------|---------------|------------|-----------|
| `visu_prompts.csv` | ViSU (SafeCLIP, ECCV 2024) | Nudity, Violence, Weapons, Brutality, Blood | ~1,000 |
| `i2p_prompts.csv` | I2P (Safe Latent Diffusion, CVPR 2023) | Multi-category NSFW | ~4,703 |
| `rab_prompts.csv` | Ring-A-Bell (arXiv 2023) | Jailbreak / adversarial | ~200 |

---

## Original Dataset Citations

If you use these prompts, please also cite the original works:

```bibtex
@inproceedings{poppi2024safeclip,
  title     = {Safe-CLIP: Removing NSFW Concepts from Vision-and-Language Models},
  author    = {Poppi, Samuele and Poppi, Tomaso and Cocchi, Federico and
               Cornia, Marcella and Baraldi, Lorenzo and Cucchiara, Rita},
  booktitle = {ECCV},
  year      = {2024}
}

@inproceedings{schramowski2023safe,
  title     = {Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models},
  author    = {Schramowski, Patrick and Brack, Manuel and Deiseroth, Björn and Kersting, Kristian},
  booktitle = {CVPR},
  year      = {2023}
}

@article{tsai2023ring,
  title   = {Ring-A-Bell! How Reliable are Concept Removal Methods for Diffusion Models?},
  author  = {Tsai, Yu-Lin and Hsu, Chia-Yi and Xie, Chulin and Lin, Chih-Hsun and
             Chen, Jia-You and Li, Bo and Chen, Pin-Yu and Yu, Chia-Mu and Huang, Chun-Ying},
  journal = {arXiv preprint arXiv:2310.10012},
  year    = {2023}
}
```

---

## Contact

Questions about access: **mohammed.alam@mbzuai.ac.ae**
