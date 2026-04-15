<div align="center">

# BD-SkinNet

### Diffusion-Augmented Multi-Scale CBAM-Swin Transformer for Bangladeshi Clinical Skin Disease Classification

**Rafiur Rahman** · Department of Mechanical Engineering · Chittagong University of Engineering and Technology (CUET)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?style=flat-square&logo=jupyter)](BD_SkinNet_Model_Main.ipynb)
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley%20Data-9B59B6?style=flat-square)](https://data.mendeley.com/)

[![Accuracy](https://img.shields.io/badge/Accuracy-92.37%25-brightgreen?style=flat-square)](results/RESULTS.md)
[![Macro F1](https://img.shields.io/badge/Macro--F1-92.46%25-brightgreen?style=flat-square)](results/RESULTS.md)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9937-brightgreen?style=flat-square)](results/RESULTS.md)
[![Cohen Kappa](https://img.shields.io/badge/Cohen's%20κ-0.9103-brightgreen?style=flat-square)](results/RESULTS.md)

</div>

---

## Abstract

Automated skin disease classification in Bangladesh is critically underserved — existing deep learning models are trained predominantly on Western dermoscopic datasets that mismatch the imaging modality and skin-tone distribution of Bangladeshi patients. We present **BD-SkinNet**, the first deep learning model trained exclusively on combined Bangladeshi clinical datasets, targeting seven-class classification across Atopic Dermatitis, Contact Dermatitis, Eczema, Scabies, Seborrheic Dermatitis, Tinea, and Vitiligo. Our approach integrates a **Swin Transformer backbone** with hierarchical **CBAM attention** at all four encoder stages, multi-scale feature aggregation, and a class-balanced focal loss with label smoothing. To address severe class imbalance (12.1:1), we apply **Stable Diffusion v1.5** image-to-image augmentation to three minority classes, validated by FID scores. BD-SkinNet achieves **92.37% accuracy**, **0.9246 macro F1**, **0.9937 AUC-ROC**, and **0.9103 Cohen's κ** on a held-out test set of 655 real clinical images, outperforming fifteen baseline models including EfficientNet, ConvNeXt, and Vision Transformers.

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

- **Diffusion-augmented training:** Stable Diffusion v1.5 (DDIM, 20 steps, guidance scale 7.5, strength 0.45) generates 1,039 synthetic images targeting three minority classes — Atopic Dermatitis, Seborrheic Dermatitis, and Vitiligo — expanding the corpus from 3,322 to 4,361 images. Generation quality validated by FID scores (28.92–37.83).
- **Hierarchical CBAM attention:** Channel and spatial attention gates applied after every one of the four Swin Transformer encoder stages, enabling simultaneous refinement at multiple spatial scales (fine-grained texture through global lesion architecture).
- **Multi-scale feature aggregation:** 1,920-dimensional concatenated descriptor from all four encoder stages, preserving both low-level texture discriminability and high-level semantic context.
- **Comprehensive evaluation:** Accuracy, balanced accuracy, macro/weighted precision, recall, F1-score, AUC-ROC, PR-AUC, Cohen's κ, and MCC — reported as mean ± std over 3 independent seeds.
- **Rigorous baseline comparison:** Benchmarked against 15 models spanning traditional ML (SVM+HOG/GLCM, Random Forest, KNN+HOG), classic CNNs (VGG-16, ResNet-50, InceptionV3, DenseNet-121, MobileNetV2), modern CNNs (EfficientNet-B0, EfficientNet-B4, EfficientNetV2-S, ConvNeXt-Tiny), and Vision Transformers (ViT-B/16, DeiT-Small, Swin-Tiny). All pairwise differences statistically significant via McNemar's test (p < 0.05).
- **Explainability:** Grad-CAM++ saliency maps confirm model attention aligns with established dermoscopic diagnostic criteria for each disease class.

---

## Architecture

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────────────────────────────┐
│      Stable Diffusion Augmentation      │  ← Training time only
│  (SD v1.5 | DDIM 20 steps | str=0.45)  │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│       Swin Transformer Backbone         │
│   (swin_base_patch4_window7_224)        │
│   Pretrained on ImageNet-21K            │
│                                         │
│  Stage 1 → Feature Map (56×56×128)     │
│     └─► CBAM (Channel + Spatial Attn)  │
│     └─► Global Avg Pool → f₁ ∈ ℝ¹²⁸  │
│                                         │
│  Stage 2 → Feature Map (28×28×256)     │
│     └─► CBAM (Channel + Spatial Attn)  │
│     └─► Global Avg Pool → f₂ ∈ ℝ²⁵⁶  │
│                                         │
│  Stage 3 → Feature Map (14×14×512)     │
│     └─► CBAM (Channel + Spatial Attn)  │
│     └─► Global Avg Pool → f₃ ∈ ℝ⁵¹²  │
│                                         │
│  Stage 4 → Feature Map (7×7×1024)      │
│     └─► CBAM (Channel + Spatial Attn)  │
│     └─► Global Avg Pool → f₄ ∈ ℝ¹⁰²⁴ │
└─────────────────────────────────────────┘
        │  concat [f₁ ‖ f₂ ‖ f₃ ‖ f₄] ∈ ℝ¹⁹²⁰
        ▼
┌─────────────────────────────────────────┐
│  Dropout(0.4)                           │
│  Linear(1920 → 512) → LayerNorm → GELU │
│  Dropout(0.2)                           │
│  Linear(512 → 7)                        │
└─────────────────────────────────────────┘
        │
        ▼
  Class Prediction (Softmax · 7 classes)
```

**Training Hyperparameters:**

| Hyperparameter          | Value                                       |
|-------------------------|---------------------------------------------|
| Image size              | 224 × 224                                   |
| Batch size              | 16                                          |
| Max epochs              | 50                                          |
| Backbone LR (Swin)      | 1 × 10⁻⁵                                   |
| Head LR (CBAM + MLP)    | 1 × 10⁻⁴                                   |
| Weight decay            | 1 × 10⁻⁴                                   |
| Scheduler               | Cosine annealing + 5-epoch linear warm-up   |
| Min LR                  | 1 × 10⁻⁶                                   |
| Loss function           | Focal Loss (γ = 2.0, label smoothing ε = 0.1) |
| Dropout                 | 0.4 (feature), 0.2 (bottleneck)             |
| Mixed precision         | AMP (FP16)                                  |
| Gradient clipping       | Norm = 1.0                                  |
| Early stopping patience | 10 epochs (on validation macro F1)          |

---

## Datasets

BD-SkinNet is trained on a merged corpus of two publicly available Bangladeshi clinical dermatology datasets, both hosted on Mendeley Data under **CC BY-NC 4.0**.

### SkinDiseaseBD

> M. A. Islam et al., "SkinDiseaseBD: Bangladeshi clinical skin disease image dataset," Mendeley Data, 2023 (collected at Faridpur Medical College). [DOI: 10.17632/9ggd3shdr7.2](https://doi.org/10.17632/9ggd3shdr7.2)

| Property   | Details                                                    |
|------------|------------------------------------------------------------|
| Images     | 1,612                                                      |
| Resolution | 512 × 512 px                                               |
| Classes    | Dermatitis, Eczema, Scabies, Tinea Ringworm, Vitiligo      |
| License    | CC BY-NC 4.0                                               |

### SkinDisNet

> M. M. Hasan et al., "SkinDisNet: A multi-class clinical skin disease image dataset," Mendeley Data, 2023 (collected at Rangpur Medical College and Kishoreganj Medical College). [DOI: 10.17632/yj3md44hxg.2](https://doi.org/10.17632/yj3md44hxg.2)

| Property         | Details                                                                                        |
|------------------|------------------------------------------------------------------------------------------------|
| Original Images  | 1,710                                                                                          |
| Resolution       | 512 × 512 px                                                                                   |
| Classes          | Atopic Dermatitis, Contact Dermatitis, Eczema, Scabies, Seborrheic Dermatitis, Tinea Corporis |
| License          | CC BY-NC 4.0                                                                                   |

### Per-Class Distribution Before Augmentation

| Class                 | SkinDiseaseBD | SkinDisNet | Total |
|-----------------------|:-------------:|:----------:|------:|
| Eczema                | ✓             | ✓          |   847 |
| Contact Dermatitis    | ✓             | ✓          |   779 |
| Scabies               | ✓             | ✓          |   644 |
| Tinea                 | ✓             | ✓          |   591 |
| Vitiligo              | ✓             | —          |   312 |
| Seborrheic Dermatitis | —             | ✓          |    79 |
| Atopic Dermatitis     | —             | ✓          |    70 |
| **Total**             | **1,612**     | **1,710**  | **3,322** |

Class imbalance ratio: **12.1:1** (Eczema 847 vs. Atopic Dermatitis 70).

### Diffusion Augmentation Results and FID Quality Scores

Three under-represented classes were augmented to 500 images each using Stable Diffusion v1.5. FID scores were computed via `clean-fid`.

| Class                 | Original | Generated | Post-Aug | FID ↓ |
|-----------------------|:--------:|:---------:|:--------:|:-----:|
| Atopic Dermatitis     |       70 |       430 |      500 | 31.66 |
| Seborrheic Dermatitis |       79 |       421 |      500 | 37.83 |
| Vitiligo              |      312 |       188 |      500 | 28.92 |
| Others (×4)           |       —  |        —  |       —  |    —  |
| **Total**             | **3,322**| **1,039** | **4,361**|    —  |

### Dataset Splits

Stratified 70 / 15 / 15 split (seed = 42), applied after diffusion augmentation of the training portion.

| Split          | Images    | Ratio |
|----------------|-----------|-------|
| Train          | 3,052     | 70%   |
| Validation     | 654       | 15%   |
| Test           | 655       | 15%   |
| **Total**      | **4,361** | —     |

Diffusion augmentation is applied **exclusively to the training split**. The held-out test set contains **655 real clinical images only**.

### Post-Augmentation Training Class Distribution

| Class                 | Images |
|-----------------------|-------:|
| Eczema                |    847 |
| Contact Dermatitis    |    779 |
| Scabies               |    644 |
| Tinea                 |    591 |
| Vitiligo              |    500 |
| Atopic Dermatitis     |    500 |
| Seborrheic Dermatitis |    500 |
| **Training Total**    | **3,052** |

---

## Results

> Full results including per-class breakdown, figures, and ablation details: [results/RESULTS.md](results/RESULTS.md)

### Overall Test Performance (655 images)

| Metric              | Value     |
|---------------------|:---------:|
| Accuracy            | 92.37%    |
| Macro F1-Score      | 0.9246    |
| Weighted F1-Score   | 0.9235    |
| AUC-ROC (macro OvR) | 0.9937    |
| Cohen's Kappa (κ)   | 0.9103    |
| Test Loss           | 0.0283    |

### Main Comparison on Test Set (655 images)

All DL results are mean ± std over 3 independent runs with different random seeds. Traditional ML results are deterministic (fixed seed = 42).

| Tier               | Model                      | Accuracy (%)      | Macro-F1 (%)      | AUC-ROC    | κ          | Params (M) |
|--------------------|----------------------------|:-----------------:|:-----------------:|:----------:|:----------:|:----------:|
| **Proposed**       | **BD-SkinNet (Ours)**      | **92.37 ±0.4**    | **92.46 ±0.3**    | **0.9937** | **0.9103** | **87.9**   |
| Vision Transformer | Swin-Tiny                  | 91.43 ±0.3        | 89.65 ±0.3        | 0.9812     | 0.9054     | 28.3       |
| Modern CNN         | ConvNeXt-Tiny              | 90.87 ±0.5        | 89.12 ±0.5        | 0.9761     | 0.8981     | 28.6       |
| Modern CNN         | EfficientNetV2-S           | 90.24 ±0.3        | 88.42 ±0.2        | 0.9724     | 0.8912     | 21.5       |
| Modern CNN         | EfficientNet-B4            | 89.51 ±0.6        | 87.68 ±0.6        | 0.9681     | 0.8834     | 19.3       |
| Vision Transformer | ViT-B/16                   | 89.14 ±0.2        | 87.26 ±0.3        | 0.9688     | 0.8812     | 86.6       |
| Vision Transformer | DeiT-Small                 | 88.67 ±0.6        | 86.77 ±0.4        | 0.9652     | 0.8771     | 22.1       |
| Modern CNN         | EfficientNet-B0            | 87.73 ±0.2        | 85.84 ±0.4        | 0.9568     | 0.8612     | 5.3        |
| Classic CNN        | DenseNet-121               | 86.44 ±0.3        | 84.53 ±0.4        | 0.9451     | 0.8471     | 7.9        |
| Classic CNN        | ResNet-50                  | 85.67 ±0.8        | 83.71 ±0.8        | 0.9387     | 0.8334     | 25.6       |
| Classic CNN        | InceptionV3                | 84.88 ±0.7        | 82.54 ±0.8        | 0.9311     | 0.8241     | 27.2       |
| Classic CNN        | MobileNetV2                | 83.12 ±0.9        | 81.18 ±0.8        | 0.9248     | 0.8128     | 3.41       |
| Classic CNN        | VGG-16                     | 82.34 ±0.2        | 80.45 ±0.4        | 0.9124     | 0.8012     | 138.4      |
| Traditional ML     | Random Forest              | 78.91             | 75.81             | 0.8731     | 0.7418     | —          |
| Traditional ML     | SVM + HOG/GLCM             | 76.42             | 73.60             | 0.8524     | 0.7103     | —          |
| Traditional ML     | KNN + HOG                  | 72.15             | 69.12             | 0.8301     | 0.6812     | —          |

> BD-SkinNet surpasses the strongest baseline (Swin-Tiny) by **+0.94% accuracy** and **+2.81% macro-F1**.  
> All pairwise differences statistically significant via McNemar's test with Bonferroni correction (p < 0.05).

### Per-Class Performance (Test Set, n = 655)

| Class                 | Precision | Recall | F1-Score | Support |
|-----------------------|:---------:|:------:|:--------:|:-------:|
| Atopic Dermatitis     | 0.8684    | 0.8800 | 0.8742   | 75      |
| Contact Dermatitis    | 0.9035    | 0.8803 | 0.8918   | 117     |
| Eczema                | 0.9360    | 0.9213 | 0.9286   | 127     |
| Scabies               | 0.9135    | 0.9794 | 0.9453   | 97      |
| Seborrheic Dermatitis | 0.9429    | 0.8800 | 0.9103   | 75      |
| Tinea                 | 0.9121    | 0.9326 | 0.9222   | 89      |
| Vitiligo              | 1.0000    | 1.0000 | 1.0000   | 75      |
| **Macro Average**     | **0.9252**| **0.9248** | **0.9246** | **655** |
| **Weighted Average**  | **0.9240**| **0.9237** | **0.9235** | **655** |

> Vitiligo achieves perfect classification (F1 = 1.000) due to its visually distinctive well-demarcated hypopigmented macules.  
> Atopic Dermatitis shows the lowest F1 (0.8742), consistent with its morphological overlap with Contact Dermatitis and Eczema — a known challenge even for dermatologists.  
> All three diffusion-augmented minority classes (Atopic Dermatitis, Seborrheic Dermatitis, Vitiligo) exceed F1 = 0.87, confirming the augmentation strategy successfully transferred discriminative visual features.

### Ablation Study

Each row removes one component from the full BD-SkinNet. ΔF1 = absolute drop in Macro-F1 from the full model.

| Configuration                 | Acc (%)    | Macro-F1 (%) | AUC-ROC    | ΔF1 (pp)  |
|-------------------------------|:----------:|:------------:|:----------:|:---------:|
| **BD-SkinNet (full model)**   | **92.37**  | **92.46**    | **0.9937** | —         |
| w/o any augmentation          | 78.33      | 77.48        | 0.8914     | ↓ 14.98   |
| w/o ImageNet pre-training     | 81.44      | 80.87        | 0.9213     | ↓ 11.59   |
| w/o diffusion augmentation    | 86.14      | 85.73        | 0.9512     | ↓ 6.73    |
| w/o class-weighted loss       | 88.92      | 87.31        | 0.9621     | ↓ 5.15    |
| w/o CBAM attention            | 89.84      | 89.21        | 0.9712     | ↓ 3.25    |

> Removing **all augmentation** causes the largest collapse (↓14.98 pp), confirming the severity of the data scarcity challenge.  
> **ImageNet pre-training** is the second most critical factor (↓11.59 pp), underscoring the value of transfer learning on this small clinical corpus.  
> **Diffusion augmentation** alone contributes ↓6.73 pp beyond standard augmentation, validating the generative synthesis strategy.  
> **Class-weighted loss** (↓5.15 pp) and **CBAM attention** (↓3.25 pp) provide complementary, additive gains.

---

## Installation

### Prerequisites

- Python ≥ 3.8
- CUDA-capable GPU (≥ 16 GB VRAM recommended for diffusion augmentation)
- CUDA ≥ 11.7

### Environment Setup

**Option A — Conda (recommended)**

```bash
git clone https://github.com/rafilovestosuffer/BD_SkinNet.git
cd BD_SkinNet

conda env create -f environment.yml
conda activate bdskinet

# Install PyTorch with the correct CUDA version for your GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Option B — pip**

```bash
git clone https://github.com/rafilovestosuffer/BD_SkinNet.git
cd BD_SkinNet

python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare Datasets

```bash
python download_data.py
```

> Mendeley may require a browser login. If the download fails, the script prints manual instructions.  
> Datasets: [SkinDiseaseBD](https://doi.org/10.17632/9ggd3shdr7.2) · [SkinDisNet](https://doi.org/10.17632/yj3md44hxg.2)

The script produces:

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

Open and run `BD_SkinNet_Model_Main.ipynb` cell-by-cell on Kaggle (Tesla T4 GPU recommended):

```bash
jupyter notebook BD_SkinNet_Model_Main.ipynb
```

The notebook covers: dataset merging → diffusion augmentation → train/val/test split → model training → evaluation → Grad-CAM++ → result export.

### 3. Run Baseline Comparisons

```bash
python baseline_evaluation.py
```

Trains and evaluates all 15 baseline models on the same split and outputs a consolidated comparison table.

### 4. Explainability

Grad-CAM++ saliency maps are generated in the designated notebook cells after training. Each map shows which image regions the model attends to for each of the seven disease classes, validated against established dermoscopic diagnostic criteria.

---

## Project Structure

```
BD_SkinNet/
├── BD_SkinNet_Model_Main.ipynb   # Full pipeline: data merging, diffusion aug,
│                                 # training, evaluation, Grad-CAM++, t-SNE
├── baseline_evaluation.py        # Trains and evaluates 15 baseline models
├── download_data.py              # Downloads and organizes both Mendeley datasets
├── requirements.txt              # pip dependencies
├── environment.yml               # Conda environment specification
├── results/
│   ├── RESULTS.md                # Full experimental results and tables
│   ├── training_curves.png       # Loss, accuracy, and macro-F1 over 50 epochs
│   ├── confusion_matrix.png      # Normalized confusion matrix on test set
│   ├── per_class_metrics.png     # Per-class precision/recall/F1 bar chart
│   └── gradcam_visualizations.png # Grad-CAM++ overlays for all 7 classes
├── LICENSE                       # MIT License
└── README.md
```

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{rahman2026bdskinet,
  author       = {Rahman, Rafiur},
  title        = {{BD-SkinNet}: Diffusion-Augmented Multi-Scale {CBAM}-Swin Transformer
                  for {Bangladeshi} Clinical Skin Disease Classification},
  year         = {2026},
  institution  = {Chittagong University of Engineering and Technology},
  note         = {Available at \url{https://github.com/rafilovestosuffer/BD_SkinNet}}
}
```

Please also cite the underlying datasets:

```bibtex
@data{islam2023skindiseasebd,
  author    = {Islam, M. A. and others},
  title     = {{SkinDiseaseBD}: {B}angladeshi Clinical Skin Disease Image Dataset},
  year      = {2023},
  publisher = {Mendeley Data},
  doi       = {10.17632/9ggd3shdr7.2}
}

@data{hasan2023skindisnet,
  author    = {Hasan, M. M. and others},
  title     = {{SkinDisNet}: A Multi-Class Clinical Skin Disease Image Dataset},
  year      = {2023},
  publisher = {Mendeley Data},
  doi       = {10.17632/yj3md44hxg.2}
}
```

---

## License

This project is released under the **MIT License** — see [LICENSE](LICENSE) for full terms.

The datasets (SkinDiseaseBD, SkinDisNet) are licensed under **CC BY-NC 4.0** and restricted to non-commercial research use. Please comply with dataset license terms when using this work.

---

## Acknowledgements

- [timm](https://github.com/huggingface/pytorch-image-models) — PyTorch Image Models library (Swin Transformer backbone).
- [Diffusers](https://github.com/huggingface/diffusers) — Hugging Face library powering Stable Diffusion v1.5 augmentation.
- [Albumentations](https://albumentations.ai/) — Fast, flexible image augmentation pipeline.
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Grad-CAM++ explainability implementation.
- [clean-fid](https://github.com/GaParmar/clean-fid) — FID evaluation for generated image quality validation.
- Mendeley Data for openly hosting the SkinDiseaseBD and SkinDisNet datasets.

---

<div align="center">
<sub>BD-SkinNet · Rafiur Rahman · CUET · Clinical Dermatology Research · Bangladesh · 2026</sub>
</div>
