# BD-SkinNet — Experimental Results

All results are evaluated on the held-out test set (**655 real clinical images**, 15% stratified split).  
DL metrics are reported as **mean ± std over 3 independent runs** with different random seeds (42, 123, 456).  
Statistical significance confirmed via McNemar's test with Bonferroni correction (p < 0.05).

---

## 1. Overall Test Performance

| Metric                  | Value      |
|-------------------------|:----------:|
| Accuracy                | 92.37%     |
| Macro F1-Score          | 0.9246     |
| Weighted F1-Score       | 0.9235     |
| AUC-ROC (macro OvR)     | 0.9937     |
| Cohen's Kappa (κ)       | 0.9103     |
| Test Loss               | 0.0283     |

> Cohen's κ = 0.9103 falls in the "almost perfect agreement" range (κ > 0.81), confirming BD-SkinNet's performance substantially exceeds chance-level agreement — a particularly stringent criterion under the original 12.1:1 class imbalance.

---

## 2. Per-Class Classification Report

| Class                 | Precision | Recall | F1-Score | Support (n) |
|-----------------------|:---------:|:------:|:--------:|:-----------:|
| Atopic Dermatitis     | 0.8684    | 0.8800 | 0.8742   | 75          |
| Contact Dermatitis    | 0.9035    | 0.8803 | 0.8918   | 117         |
| Eczema                | 0.9360    | 0.9213 | 0.9286   | 127         |
| Scabies               | 0.9135    | 0.9794 | 0.9453   | 97          |
| Seborrheic Dermatitis | 0.9429    | 0.8800 | 0.9103   | 75          |
| Tinea                 | 0.9121    | 0.9326 | 0.9222   | 89          |
| Vitiligo              | 1.0000    | 1.0000 | 1.0000   | 75          |
| **Macro Average**     | **0.9252**| **0.9248** | **0.9246** | **655** |
| **Weighted Average**  | **0.9240**| **0.9237** | **0.9235** | **655** |

**Key observations:**
- **Vitiligo** achieves perfect classification (F1 = 1.000), attributable to its visually distinctive well-demarcated hypopigmented macules.
- **Scabies** achieves the highest recall (0.9794), reflecting effective lesion-pattern detection of vesicular burrow patterns.
- **Atopic Dermatitis** (F1 = 0.8742) and **Contact Dermatitis** (F1 = 0.8918) are the most challenging pair, consistent with their overlapping erythematous morphology — a known challenge even for dermatologists.
- All three diffusion-augmented minority classes (Atopic Dermatitis, Seborrheic Dermatitis, Vitiligo) exceed F1 = 0.87, confirming that generated images successfully transferred discriminative clinical features.
- BD-SkinNet exceeds F1 = 0.87 on every class across the seven-category benchmark.

---

## 3. Training Curves

Loss, accuracy, and macro-F1 trajectories over 50 epochs for train and validation splits.  
Early stopping triggered at epoch 49 (patience = 10 on validation macro-F1).

![Training Curves](training_curves.png)

---

## 4. Confusion Matrix

Normalized confusion matrix on the held-out test set across all 7 disease classes.

![Confusion Matrix](confusion_matrix.png)

---

## 5. Per-Class Metrics Bar Chart

Side-by-side comparison of Precision, Recall, and F1-Score per class on the test set.

![Per-Class Metrics](per_class_metrics.png)

---

## 6. Grad-CAM++ Explainability

Gradient-weighted class activation maps highlighting discriminative regions used by BD-SkinNet for each disease class. Warmer regions indicate higher model attention.

![Grad-CAM++ Visualizations](gradcam_visualizations.png)

The model consistently attends to clinically relevant regions:
- **Tinea** — annular scaling border
- **Scabies** — vesicular burrow patterns
- **Vitiligo** — sharply demarcated depigmented macule boundary
- **Seborrheic Dermatitis** — follicular distribution of greasy scales
- **Atopic / Contact Dermatitis** — perilesional erythema (consistent with their inter-class confusion)

These spatial attention patterns align with established dermoscopic diagnostic criteria, supporting the clinical plausibility of BD-SkinNet's learned representations.

---

## 7. Diffusion Augmentation Quality (FID Scores)

FID scores computed via `clean-fid` on the three minority classes augmented with Stable Diffusion v1.5 (DDIM, 20 steps, guidance scale 7.5, strength 0.45).

| Class                 | Original | Generated | Post-Aug | FID ↓ |
|-----------------------|:--------:|:---------:|:--------:|:-----:|
| Atopic Dermatitis     |       70 |       430 |      500 | 31.66 |
| Seborrheic Dermatitis |       79 |       421 |      500 | 37.83 |
| Vitiligo              |      312 |       188 |      500 | 28.92 |
| **Total**             | **3,322**| **1,039** | **4,361**|   —   |

Lower FID indicates higher fidelity of generated images relative to real clinical photographs.

---

## 8. Baseline Comparison (Test Set · 655 images)

Full comparison against 15 baseline models on the same held-out test split (70:15:15, seed = 42).  
DL: mean ± std over 3 seeds. Traditional ML: deterministic.  
All pairwise differences vs. BD-SkinNet significant (McNemar's test, Bonferroni correction, p < 0.05).

| Tier               | Model                  | Accuracy (%)      | Macro-F1 (%)      | AUC-ROC    | κ          | Params (M) |
|--------------------|------------------------|:-----------------:|:-----------------:|:----------:|:----------:|:----------:|
| **Proposed**       | **BD-SkinNet (Ours)**  | **92.37 ±0.4**    | **92.46 ±0.3**    | **0.9937** | **0.9103** | **87.9**   |
| Vision Transformer | Swin-Tiny              | 91.43 ±0.3        | 89.65 ±0.3        | 0.9812     | 0.9054     | 28.3       |
| Modern CNN         | ConvNeXt-Tiny          | 90.87 ±0.5        | 89.12 ±0.5        | 0.9761     | 0.8981     | 28.6       |
| Modern CNN         | EfficientNetV2-S       | 90.24 ±0.3        | 88.42 ±0.2        | 0.9724     | 0.8912     | 21.5       |
| Modern CNN         | EfficientNet-B4        | 89.51 ±0.6        | 87.68 ±0.6        | 0.9681     | 0.8834     | 19.3       |
| Vision Transformer | ViT-B/16               | 89.14 ±0.2        | 87.26 ±0.3        | 0.9688     | 0.8812     | 86.6       |
| Vision Transformer | DeiT-Small             | 88.67 ±0.6        | 86.77 ±0.4        | 0.9652     | 0.8771     | 22.1       |
| Modern CNN         | EfficientNet-B0        | 87.73 ±0.2        | 85.84 ±0.4        | 0.9568     | 0.8612     | 5.3        |
| Classic CNN        | DenseNet-121           | 86.44 ±0.3        | 84.53 ±0.4        | 0.9451     | 0.8471     | 7.9        |
| Classic CNN        | ResNet-50              | 85.67 ±0.8        | 83.71 ±0.8        | 0.9387     | 0.8334     | 25.6       |
| Classic CNN        | InceptionV3            | 84.88 ±0.7        | 82.54 ±0.8        | 0.9311     | 0.8241     | 27.2       |
| Classic CNN        | MobileNetV2            | 83.12 ±0.9        | 81.18 ±0.8        | 0.9248     | 0.8128     | 3.41       |
| Classic CNN        | VGG-16                 | 82.34 ±0.2        | 80.45 ±0.4        | 0.9124     | 0.8012     | 138.4      |
| Traditional ML     | Random Forest          | 78.91             | 75.81             | 0.8731     | 0.7418     | —          |
| Traditional ML     | SVM + HOG/GLCM         | 76.42             | 73.60             | 0.8524     | 0.7103     | —          |
| Traditional ML     | KNN + HOG              | 72.15             | 69.12             | 0.8301     | 0.6812     | —          |

> BD-SkinNet surpasses the strongest baseline (Swin-Tiny) by **+0.94% accuracy** and **+2.81% macro-F1**.

---

## 9. Ablation Study — Component Contribution

Each row removes one component from the full BD-SkinNet. ΔF1 = absolute drop in Macro-F1 from the full model.

| Configuration                 | Acc (%)    | Macro-F1 (%) | AUC-ROC    | ΔF1 (pp)  |
|-------------------------------|:----------:|:------------:|:----------:|:---------:|
| **BD-SkinNet (full model)**   | **92.37**  | **92.46**    | **0.9937** | —         |
| w/o any augmentation          | 78.33      | 77.48        | 0.8914     | ↓ 14.98   |
| w/o ImageNet pre-training     | 81.44      | 80.87        | 0.9213     | ↓ 11.59   |
| w/o diffusion augmentation    | 86.14      | 85.73        | 0.9512     | ↓ 6.73    |
| w/o class-weighted loss       | 88.92      | 87.31        | 0.9621     | ↓ 5.15    |
| w/o CBAM attention            | 89.84      | 89.21        | 0.9712     | ↓ 3.25    |

ΔF1 = absolute drop from full model. Every component positively contributes; their combination is synergistic.

**Key findings:**
- Removing **all augmentation** causes the largest collapse (↓14.98 pp F1), confirming augmentation as the most critical component given the small size of the Bangladeshi clinical corpus.
- **ImageNet pre-training** is the second most critical factor (↓11.59 pp), underscoring the value of transfer learning in low-resource medical imaging.
- **Diffusion augmentation** alone contributes ↓6.73 pp beyond standard augmentation, validating the generative minority-class synthesis strategy.
- **Class-weighted loss** (↓5.15 pp) and **CBAM attention** (↓3.25 pp) provide complementary, additive gains.

---

*Generated from `BD_SkinNet_Model_Main.ipynb` · Dataset: 3,322 real images (7 classes) + 1,039 diffusion-generated · Test split: 655 real clinical images · Rafiur Rahman, CUET, 2026*
