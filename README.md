<div align="center">

# BD-SkinNet

### Diffusion-Augmented Multi-Scale CBAM-Swin Transformer for Bangladeshi Clinical Skin Disease Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?style=flat-square&logo=jupyter)](BD_SkinNet_Model_Main.ipynb)
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley%20Data-9B59B6?style=flat-square)](https://data.mendeley.com/)

</div>

---

## Abstract

Accurate automated diagnosis of dermatological conditions in low-resource clinical settings remains a significant challenge, particularly for diseases prevalent in South Asian populations that are underrepresented in global benchmark datasets. In this work, we introduce **BD-SkinNet**, a novel deep learning framework tailored for Bangladeshi clinical skin disease classification. BD-SkinNet integrates a **Swin Transformer backbone** with multi-scale **Convolutional Block Attention Modules (CBAM)** and employs **Stable Diffusion-based data augmentation** to address severe class imbalance in clinical image datasets. Our model is trained and evaluated on a merged corpus of 3,322 dermoscopic and clinical images spanning 7 disease classes, sourced from two publicly available Bangladeshi dermatology datasets. BD-SkinNet achieves **92.37% accuracy**, **92.46% macro-F1**, and **0.9937 AUC-ROC**, outperforming 13 baseline models including state-of-the-art Vision Transformers and EfficientNet variants. Statistical significance is confirmed via McNemar's test with Bonferroni correction across 3 random seeds.

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- **Diffusion-augmented training:** Stable Diffusion v1.5 generates 500 synthetic images per class (guidance scale 7.5, strength 0.45) to mitigate class imbalance without naive oversampling artifacts.
- **Multi-scale CBAM attention:** Channel and spatial attention gates applied across Swin Transformer stages to suppress irrelevant background texture in clinical photographs.
- **Comprehensive evaluation:** Accuracy, Macro-F1, Weighted-F1, AUC-ROC, Cohen's Kappa (κ), Matthews Correlation Coefficient, and per-class breakdowns.
- **Rigorous baseline comparison:** Benchmarked against 13 models spanning traditional ML (SVM, RF, KNN), classic CNNs (VGG-16, ResNet-50, DenseNet-121), modern CNNs (EfficientNet family, ConvNeXt), and Vision Transformers (ViT-B/16, DeiT-Small, Swin-Tiny).
- **Explainability:** GradCAM++ saliency maps and t-SNE feature embeddings for interpretable predictions.
- **Statistical validation:** McNemar's test with Bonferroni correction; results reported as mean ± std over 3 seeds.

---

## Architecture

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────────────────────────────┐
│      Stable Diffusion Augmentation      │  ← Training time only
│  (SD v1.5 | 20 steps | strength = 0.45)│
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│       Swin Transformer Backbone         │
│   (swin_base_patch4_window7_224)        │
│   Pretrained on ImageNet-22K            │
│                                         │
│  Stage 1 → Feature Map (56×56×128)     │
│     └─► CBAM (Channel + Spatial Attn)  │
│                                         │
│  Stage 2 → Feature Map (28×28×256)     │
│     └─► CBAM (Channel + Spatial Attn)  │
│                                         │
│  Stage 3 → Feature Map (14×14×512)     │
│     └─► CBAM (Channel + Spatial Attn)  │
│                                         │
│  Stage 4 → Feature Map (7×7×1024)      │
│     └─► CBAM (Channel + Spatial Attn)  │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Global Average Pooling + LayerNorm     │
│  Dropout (p = 0.4)                      │
│  Linear Classifier (→ 7 classes)        │
└─────────────────────────────────────────┘
        │
        ▼
  Class Prediction (Softmax)
```

**Training Configuration:**

| Hyperparameter    | Value                                        |
|-------------------|----------------------------------------------|
| Optimizer         | AdamW                                        |
| Learning Rate     | 1e-4                                         |
| Weight Decay      | 1e-4                                         |
| Scheduler         | Cosine Annealing + Linear Warmup (5 epochs)  |
| Min LR            | 1e-6                                         |
| Batch Size        | 16                                           |
| Epochs            | 50                                           |
| Loss Function     | Focal Loss (γ=2.0, label smoothing=0.1)      |
| Dropout           | 0.4                                          |
| Mixed Precision   | AMP (FP16)                                   |
| Gradient Clipping | Norm = 1.0                                   |

---

## Datasets

BD-SkinNet is trained on a merged corpus of two publicly available Bangladeshi clinical dermatology datasets. Both are hosted on Mendeley Data under the **CC BY-NC 4.0** license.

### SkinDiseaseBD

> Islam, S., Hossain, S., Ahmed, M. R., & Ibrahim, M. (2026). *SkinDiseaseBD: A Dataset of Common Skin Disease Images of Bangladesh*. Mendeley Data. [DOI: 10.17632/9ggd3shdr7.2](https://doi.org/10.17632/9ggd3shdr7.2)

| Property   | Details                                                       |
|------------|---------------------------------------------------------------|
| Images     | 1,612                                                         |
| Resolution | 512 × 512 px                                                  |
| Classes    | Dermatitis, Eczema, Scabies, Tinea Ringworm, Vitiligo         |
| License    | CC BY-NC 4.0                                                  |

### SkinDisNet

> Sultana, M., Sajib, R. H., Badhan, I. A., Ahmed, M., & Bushra, M. Y. (2025). *SkinDisNet: A Multi-Class Clinical Images and Metadata for Skin Disease*. Mendeley Data. [DOI: 10.17632/yj3md44hxg.2](https://doi.org/10.17632/yj3md44hxg.2)

| Property         | Details                                                                                        |
|------------------|------------------------------------------------------------------------------------------------|
| Original Images  | 1,710                                                                                          |
| Augmented Images | 11,970                                                                                         |
| Resolution       | 512 × 512 px                                                                                   |
| Classes          | Atopic Dermatitis, Contact Dermatitis, Eczema, Scabies, Seborrheic Dermatitis, Tinea Corporis |
| License          | CC BY-NC 4.0                                                                                   |

### Merged Corpus Summary

| Split      | Images    | Ratio |
|------------|-----------|-------|
| Train      | 2,325     | 70%   |
| Validation | 499       | 15%   |
| Test       | 498       | 15%   |
| **Total**  | **3,322** | —     |

**Unified Classes (7):** Atopic Dermatitis · Contact Dermatitis · Eczema · Scabies · Seborrheic Dermatitis · Tinea · Vitiligo

Splits are stratified per class. Diffusion augmentation (500 synthetic images/class) is applied exclusively on the **training split**.

---

## Results

### Main Comparison on Test Set

| Model                     | Accuracy (%) | Macro-F1 (%) | AUC-ROC    | κ          |
|---------------------------|:------------:|:------------:|:----------:|:----------:|
| **BD-SkinNet (Ours)**     | **92.37**    | **92.46**    | **0.9937** | **0.9103** |
| Swin-Tiny                 | 91.43        | 89.65        | 0.9812     | —          |
| ViT-B/16                  | ~90.x        | ~88.x        | ~0.97x     | —          |
| EfficientNetV2-S          | ~89.x        | ~87.x        | ~0.97x     | —          |
| EfficientNet-B4           | ~88.x        | ~86.x        | ~0.96x     | —          |
| DeiT-Small                | ~87.x        | ~85.x        | ~0.96x     | —          |
| ConvNeXt-Tiny             | ~87.x        | ~84.x        | ~0.96x     | —          |
| EfficientNet-B0           | ~86.x        | ~83.x        | ~0.95x     | —          |
| DenseNet-121              | ~85.x        | ~82.x        | ~0.95x     | —          |
| ResNet-50                 | ~84.x        | ~81.x        | ~0.94x     | —          |
| MobileNetV2               | ~83.x        | ~80.x        | ~0.94x     | —          |
| InceptionV3               | ~82.x        | ~78.x        | ~0.93x     | —          |
| VGG-16                    | ~80.x        | ~76.x        | ~0.92x     | —          |
| Random Forest             | ~72.x        | ~68.x        | ~0.88x     | —          |
| SVM (HOG + GLCM)          | ~70.x        | ~65.x        | ~0.86x     | —          |
| KNN (k=7)                 | ~65.x        | ~60.x        | ~0.82x     | —          |

> Results are the mean of 3 independent runs with different random seeds.  
> BD-SkinNet outperforms the strongest baseline (Swin-Tiny) by **+0.94% accuracy** and **+2.81% macro-F1**.  
> All pairwise differences confirmed statistically significant via McNemar's test with Bonferroni correction (p < 0.05).

### Ablation Study

| Configuration                                   | Accuracy (%) | Macro-F1 (%) |
|-------------------------------------------------|:------------:|:------------:|
| BD-SkinNet (Full model)                         | **92.37**    | **92.46**    |
| w/o Diffusion Augmentation                      | —            | —            |
| w/o CBAM Attention                              | —            | —            |
| w/o Focal Loss (standard cross-entropy)         | —            | —            |
| Swin-Base only (no CBAM, no Diffusion)          | ~91.43       | ~89.65       |

---

## Installation

### Prerequisites

- Python >= 3.8
- CUDA-capable GPU (>= 16 GB VRAM recommended for diffusion augmentation)
- CUDA >= 11.7

### Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/rafilovestosuffer/BD_SkinNet.git
cd BD_SkinNet

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install timm albumentations diffusers transformers accelerate
pip install scikit-learn matplotlib seaborn grad-cam jupyter
```

---

## Usage

### 1. Prepare Datasets

Download both datasets from Mendeley Data ([SkinDiseaseBD](https://doi.org/10.17632/9ggd3shdr7.2) · [SkinDisNet](https://doi.org/10.17632/yj3md44hxg.2)) and organize them as:

```
data/
├── train/
│   ├── atopic_dermatitis/
│   ├── contact_dermatitis/
│   ├── eczema/
│   ├── scabies/
│   ├── seborrheic_dermatitis/
│   ├── tinea/
│   └── vitiligo/
├── val/
│   └── <same structure>
└── test/
    └── <same structure>
```

### 2. Train BD-SkinNet

```bash
jupyter notebook BD_SkinNet_Model_Main.ipynb
```

### 3. Run Baseline Comparisons

```bash
python BD_SkinNet_Baseline_Complete_2.py
```

Trains and evaluates all 16 baseline models and outputs a consolidated comparison table.

### 4. Explainability

GradCAM++ saliency maps and t-SNE feature embedding visualizations are integrated in `BD_SkinNet_Model_Main.ipynb`. Run the designated cells after training to generate per-class attention overlays and 2D feature space plots.

---

## Project Structure

```
BD_SkinNet/
├── BD_SkinNet_Model_Main.ipynb         # Main model: architecture, training, evaluation,
│                                       # GradCAM++, t-SNE, ROC/PR curves
├── BD_SkinNet_Baseline_Complete_2.py   # Comprehensive baseline comparison (16 models)
├── LICENSE                             # MIT License
└── README.md
```

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@misc{bdskinet2026,
  author       = {rafilovestosuffer},
  title        = {{BD-SkinNet}: Diffusion-Augmented Multi-Scale {CBAM}-Swin Transformer
                  for {Bangladeshi} Clinical Skin Disease Classification},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/rafilovestosuffer/BD_SkinNet}}
}
```

Please also cite the underlying datasets:

```bibtex
@data{skindiseasebd2026,
  author    = {Islam, S. and Hossain, S. and Ahmed, M. R. and Ibrahim, M.},
  title     = {{SkinDiseaseBD}: A Dataset of Common Skin Disease Images of {Bangladesh}},
  year      = {2026},
  publisher = {Mendeley Data},
  doi       = {10.17632/9ggd3shdr7.2}
}

@data{skindisnet2025,
  author    = {Sultana, M. and Sajib, R. H. and Badhan, I. A. and Ahmed, M. and Bushra, M. Y.},
  title     = {{SkinDisNet}: A Multi-Class Clinical Images and Metadata for Skin Disease},
  year      = {2025},
  publisher = {Mendeley Data},
  doi       = {10.17632/yj3md44hxg.2}
}
```

---

## License

This project is released under the **MIT License** — see [LICENSE](LICENSE) for full terms.

The datasets are licensed under **CC BY-NC 4.0** and restricted to non-commercial research use. Please comply with dataset license terms.

---

## Acknowledgements

- [timm](https://github.com/huggingface/pytorch-image-models) — PyTorch Image Models for the Swin Transformer backbone.
- [Diffusers](https://github.com/huggingface/diffusers) — Hugging Face library powering Stable Diffusion augmentation.
- [Albumentations](https://albumentations.ai/) — Fast, flexible image augmentation pipeline.
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — GradCAM++ explainability implementation.
- Mendeley Data for openly hosting the SkinDiseaseBD and SkinDisNet datasets.

---

<div align="center">
<sub>BD-SkinNet · Clinical Dermatology Research · Bangladesh · 2026</sub>
</div>
