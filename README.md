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

Accurate automated diagnosis of dermatological conditions in low-resource clinical settings remains a significant challenge, particularly for diseases prevalent in South Asian populations that are underrepresented in global benchmark datasets. In this work, we introduce **BD-SkinNet**, a novel deep learning framework tailored for Bangladeshi clinical skin disease classification. BD-SkinNet integrates a **Swin Transformer backbone** with multi-scale **Convolutional Block Attention Modules (CBAM)** and employs **Stable Diffusion-based data augmentation** to address severe class imbalance in clinical image datasets. Our model is trained and evaluated on a merged corpus of 3,322 dermoscopic and clinical images spanning 7 disease classes, sourced from two publicly available Bangladeshi dermatology datasets. BD-SkinNet achieves **92.37% accuracy**, **92.46% macro-F1**, and **0.9937 AUC-ROC**, outperforming 15 baseline models including state-of-the-art Vision Transformers and EfficientNet variants. Statistical significance is confirmed via McNemar's test with Bonferroni correction across 3 random seeds.

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

- **Diffusion-augmented training:** Stable Diffusion v1.5 generates 1,039 synthetic images across underrepresented classes (guidance scale 7.5, strength 0.45) to mitigate class imbalance without naive oversampling artifacts, expanding the corpus from 3,322 to 4,361 images.
- **Multi-scale CBAM attention:** Channel and spatial attention gates applied across Swin Transformer stages to suppress irrelevant background texture in clinical photographs.
- **Comprehensive evaluation:** Accuracy, Macro-F1, Weighted-F1, AUC-ROC, Cohen's Kappa (κ), Matthews Correlation Coefficient, and per-class breakdowns.
- **Rigorous baseline comparison:** Benchmarked against 15 models spanning traditional ML (SVM+HOG/GLCM, Random Forest, KNN+HOG), classic CNNs (VGG-16, ResNet-50, InceptionV3, DenseNet-121, MobileNetV2), modern CNNs (EfficientNet-B0, EfficientNet-B4, EfficientNetV2-S, ConvNeXt-Tiny), and Vision Transformers (ViT-B/16, DeiT-Small, Swin-Tiny).
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

| Split                        | Images    | Ratio |
|------------------------------|-----------|-------|
| Train                        | 2,325     | 70%   |
| Validation                   | 499       | 15%   |
| Test                         | 498       | 15%   |
| **Total (pre-augmentation)** | **3,322** | —     |
| Diffusion-generated          | +1,039    | —     |
| **Total (post-augmentation)**| **4,361** | —     |

Splits are stratified per class. Diffusion augmentation is applied **exclusively on the training split**.

### Post-Augmentation Class Distribution (Training Set)

| Class                 | Images |
|-----------------------|-------:|
| Eczema                |    847 |
| Contact Dermatitis    |    779 |
| Scabies               |    644 |
| Tinea                 |    591 |
| Vitiligo              |    500 |
| Atopic Dermatitis     |    500 |
| Seborrheic Dermatitis |    500 |
| **Total**             |  **4,361** |

> Stable Diffusion v1.5 (20 steps, guidance scale 7.5, strength 0.45) synthesized 1,039 images targeted at the four minority classes (Vitiligo, Atopic Dermatitis, Seborrheic Dermatitis, Tinea) to reduce inter-class imbalance without duplicating real clinical samples.

---

## Results

### Main Comparison on Test Set

All results are mean ± std over 3 independent runs with different random seeds. Params = number of trainable parameters (millions).

| Category          | Model                      | Accuracy (%)      | Macro-F1 (%)      | AUC-ROC    | κ          | Params (M) |
|-------------------|----------------------------|:-----------------:|:-----------------:|:----------:|:----------:|:----------:|
| **Proposed**      | **BD-SkinNet (Ours)**      | **92.37 ±0.4**    | **92.46 ±0.4**    | **0.9937** | **0.9103** | **23.5**   |
| Vision Transformer| Swin-Tiny \[11\]           | 91.43 ±0.3        | 89.65 ±0.3        | 0.9812     | 0.9054     | 28.3       |
| Modern CNN        | ConvNeXt-Tiny \[9\]        | 90.87 ±0.5        | 89.12 ±0.5        | 0.9761     | 0.8981     | 28.6       |
| Modern CNN        | EfficientNetV2-S \[8\]     | 90.24 ±0.5        | 88.42 ±0.5        | 0.9724     | 0.8912     | 21.5       |
| Vision Transformer| ViT-B/16 \[10\]            | 89.14 ±0.6        | 87.26 ±0.6        | 0.9688     | 0.8812     | 86.6       |
| Modern CNN        | EfficientNet-B4 \[7\]      | 89.51 ±0.6        | 87.68 ±0.6        | 0.9681     | 0.8834     | 19.3       |
| Vision Transformer| DeiT-Small \[12\]          | 88.67 ±0.6        | 86.77 ±0.6        | 0.9652     | 0.8771     | 22.1       |
| Modern CNN        | EfficientNet-B0 \[7\]      | 87.73 ±0.7        | 85.84 ±0.7        | 0.9568     | 0.8612     | 5.3        |
| Classic CNN       | DenseNet-121 \[4\]         | 86.44 ±0.7        | 84.53 ±0.7        | 0.9451     | 0.8471     | 7.98       |
| Classic CNN       | ResNet-50 \[2\]            | 85.67 ±0.8        | 83.71 ±0.8        | 0.9387     | 0.8334     | 25.6       |
| Classic CNN       | InceptionV3 \[3\]          | 84.88 ±0.8        | 82.54 ±0.8        | 0.9311     | 0.8241     | 27.2       |
| Classic CNN       | MobileNetV2 \[5\]          | 83.12 ±0.9        | 81.18 ±0.9        | 0.9248     | 0.8128     | 3.41       |
| Classic CNN       | VGG-16 \[1\]               | 82.34 ±0.9        | 80.45 ±0.9        | 0.9124     | 0.8012     | 138.4      |
| Traditional ML    | Random Forest \[6\]        | 78.91             | 75.81             | 0.8731     | 0.7418     | —          |
| Traditional ML    | SVM + HOG/GLCM             | 76.42             | 73.60             | 0.8524     | 0.7103     | —          |
| Traditional ML    | KNN + HOG                  | 72.15             | 69.12             | 0.8301     | 0.6812     | —          |

> BD-SkinNet outperforms the strongest baseline (Swin-Tiny) by **+0.94% accuracy** and **+2.81% macro-F1**, while using **4.8M fewer parameters**.  
> DL results are mean ± std over 3 independent runs with different random seeds. Traditional ML models are deterministic (fixed seed).  
> All pairwise differences statistically significant via McNemar's test with Bonferroni correction (p < 0.05).

**References:**
\[1\] Simonyan & Zisserman, 2015 · \[2\] He et al., 2016 · \[3\] Szegedy et al., 2016 · \[4\] Huang et al., 2017  
\[5\] Sandler et al., 2018 · \[6\] Breiman, 2001 · \[7\] Tan & Le, 2019 · \[8\] Tan & Le, 2021  
\[9\] Liu et al., 2022 · \[10\] Dosovitskiy et al., 2021 · \[11\] Liu et al., 2021 · \[12\] Touvron et al., 2021

### Ablation Study

Each row removes one component from the full BD-SkinNet. ΔF1 = absolute drop in Macro-F1 from the full model, indicating each component's contribution.

| Configuration                   | Acc (%)    | Macro-F1 (%) | AUC-ROC    | ΔF1 (pp)     |
|---------------------------------|:----------:|:------------:|:----------:|:------------:|
| **Full model (BD-SkinNet)**     | **92.37**  | **92.46**    | **0.9937** | —            |
| w/o any augmentation            | 78.33      | 77.48        | 0.8914     | ↓ 14.98      |
| w/o ImageNet pretraining        | 81.44      | 80.87        | 0.9213     | ↓ 11.59      |
| w/o diffusion augmentation      | 86.14      | 85.73        | 0.9512     | ↓ 6.73       |
| w/o class-weighted loss         | 88.92      | 87.31        | 0.9621     | ↓ 5.15       |
| w/o attention module (CBAM)     | 89.84      | 89.21        | 0.9712     | ↓ 3.25       |

> **Key findings:**
> - Removing **all augmentation** causes the largest collapse (↓14.98 pp F1), confirming the data scarcity challenge in clinical Bangladeshi dermatology datasets.
> - **ImageNet pretraining** is the second most critical factor (↓11.59 pp), underscoring the value of transfer learning in low-data medical imaging.
> - **Diffusion augmentation alone** contributes ↓6.73 pp beyond standard augmentation, validating the generative synthesis strategy.
> - **Class-weighted loss** (↓5.15 pp) and **CBAM attention** (↓3.25 pp) provide complementary, additive gains.

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
python baseline_evaluation.py
```

Trains and evaluates all 15 baseline models and outputs a consolidated comparison table.

### 4. Explainability

GradCAM++ saliency maps and t-SNE feature embedding visualizations are integrated in `BD_SkinNet_Model_Main.ipynb`. Run the designated cells after training to generate per-class attention overlays and 2D feature space plots.

---

## Project Structure

```
BD_SkinNet/
├── BD_SkinNet_Model_Main.ipynb         # Main model: architecture, training, evaluation,
│                                       # GradCAM++, t-SNE, ROC/PR curves
├── baseline_evaluation.py              # Comprehensive baseline comparison (15 models)
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
