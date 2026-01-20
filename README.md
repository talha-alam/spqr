# SPQR: Standardized Benchmark for Safety Alignment in Text-to-Image Models

[![arXiv](https://img.shields.io/badge/arXiv-2025.19558-b31b1b.svg)](https://arxiv.org/abs/2511.19558)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **"SPQR: A Standardized Benchmark for Modern Safety Alignment Methods in Text-to-Image Diffusion Models"**

## 📋 Overview

SPQR evaluates safety-aligned text-to-image diffusion models across **four critical dimensions**:

- **S**afety (↑): Suppression of unsafe content using LLaVA-Guard + NudeNet
- **P**rompt adherence (↑): Text-image alignment via CLIP score  
- **Q**uality (↑): Visual fidelity measured by FID
- **R**obustness (↑): Stability under benign fine-tuning (BFT)

**Final Score**: Harmonic mean of S, P, Q, R → single leaderboard metric

```
SPQR = 4 / (1/S + 1/P + 1/Q + 1/R)
```

## 🎯 Key Contributions

1. **Unintentional Attacker Threat Model**: First formalization of safety degradation through benign fine-tuning
2. **Unified Benchmark**: Single metric combining safety, utility, and robustness
3. **Comprehensive Evaluation**: Multilingual, domain-specific, and OOD analyses
4. **Key Findings**: 
   - LoRA BFT offers superior stability over full fine-tuning
   - Top methods (RECE, UCE, MACE) succeed via distribution-aware alignment

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/SPQR.git
cd SPQR

# Install dependencies
pip install -r requirements.txt

# Run evaluation on a method
python scripts/run_benchmark.py \
    --method rece \
    --model_path path/to/model \
    --bft_profile standard \
    --scenario general
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install NudeNet for safety evaluation
pip install nudenet

# Install additional evaluation dependencies
pip install timm transformers diffusers accelerate
```

## 📊 Datasets

### Harmful Prompts (for Safety evaluation)
- **VISU**: Category-wise harmful prompts
- **I2P**: Inappropriate prompts dataset  
- **RAB (Ring-A-Bell)**: Jailbreak prompts

### Benign Data (for BFT)
- **General**: COCO subset (~5K samples)
- **Multilingual**: Arabic, Spanish, French, Hindi
- **Domain-Specific**: Artistic styles, Medical imaging

Download instructions: See `docs/datasets.md`

## 🔬 Evaluated Methods

Currently supported safety alignment methods:

| Method | Type | Paper |
|--------|------|-------|
| **RECE** | Conditioning | [ECCV 2024](link) |
| **UCE** | Unified Editing | [WACV 2024](link) |
| **MACE** | Multi-Concept | [CVPR 2024](link) |
| **ESD** | Concept Erasure | [ICCV 2023](link) |
| **SalUn** | Gradient-based | [arXiv 2023](link) |
| **STEREO** | Adversarial | [arXiv 2024](link) |
| **SPM** | Prompt Steering | [CVPR 2024](link) |
| **AdvUnlearn** | Defensive | [NeurIPS 2024](link) |
| **FMN** | Forgetting | [CVPRW 2024](link) |
| **EraseDiff** | Influence | [CVPR 2025](link) |
| **Scissorhands** | Connection | [ECCV 2024](link) |

## 📈 Leaderboard (General Scenario)

| Method | Safety ↑ | Prompt ↑ | Quality ↑ | Robust ↑ | **SPQR ↑** |
|--------|----------|----------|-----------|----------|------------|
| **RECE** | 0.938 | 0.292 | 0.934 | **0.980** | **0.608** |
| **UCE** | 0.926 | 0.293 | 0.919 | 0.942 | **0.602** |
| **MACE** | **0.996** | 0.267 | 0.907 | 0.657 | **0.542** |
| SPM | 0.920 | **0.294** | 0.946 | 0.684 | 0.571 |
| ESD | 0.936 | 0.289 | **0.950** | 0.684 | 0.568 |
| SalUn | **0.998** | 0.253 | 0.724 | 0.726 | 0.518 |

Full results across all scenarios: `results/leaderboard.md`

## 🏃 Running Evaluations

### 1. Full SPQR Benchmark

```bash
python scripts/run_benchmark.py \
    --method rece \
    --model_path checkpoints/rece_sd15 \
    --bft_profile standard \
    --scenario general \
    --output_dir results/rece
```

### 2. Individual Metrics

```python
from spqr.metrics import compute_safety_score, compute_fid_score

# Safety evaluation
safety = compute_safety_score(dataset_path="path/to/generated_images")

# Quality evaluation  
fid = compute_fid_score(real_path="path/to/real", gen_path="path/to/generated")
```

### 3. Benign Fine-Tuning (BFT)

```bash
python spqr/attacks/bft_trainer.py \
    --models_dir checkpoints/aligned_models \
    --train_data_dir data/bft_datasets/coco \
    --output_dir outputs/after_bft \
    --params full \
    --curriculum 1000,3000,5000 \
    --num_train_epochs 10
```

## 📝 BFT Profiles

Three fine-tuning profiles to test robustness:

1. **Lite (LoRA)**: Low-rank adaptation, minimal parameter update
2. **Moderate**: Cross-attention layers only
3. **Standard**: Full UNet fine-tuning

See `configs/bft_profiles.yaml` for details.

## 🔍 Analysis & Visualization

```bash
# Category-wise analysis
python scripts/analyze_categories.py --results results/

# Generate radar plots
jupyter notebook notebooks/visualization.ipynb
```

## 🤝 Contributing

We welcome contributions! See `CONTRIBUTING.md` for guidelines.

To add a new safety method:
1. Implement in `methods/your_method/`
2. Add config to `configs/methods.yaml`
3. Submit PR with evaluation results

## 📖 Citation

```bibtex
@article{alam2025spqr,
  title={SPQR: A Standardized Benchmark for Modern Safety Alignment Methods in Text-to-Image Diffusion Models},
  author={Alam, Mohammed Talha and Saadi, Nada and Shamshad, Fahad and Lukas, Nils and Nandakumar, Karthik and Karray, Fakhri and Poppi, Samuele},
  journal={arXiv preprint arXiv:2511.19558},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact:
- Mohammed Talha Alam: mohammed.alam@mbzuai.ac.ae
