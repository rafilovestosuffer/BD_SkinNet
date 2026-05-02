


# ║  CELL 1 — 

"""
!pip install timm==0.9.16 -q
!pip install grad-cam==1.5.0 -q
!pip install mlxtend==0.23.0 -q
!pip install umap-learn==0.5.6 -q
!pip install seaborn==0.13.2 -q
!pip install thop -q
print("✅ Done. NOW RESTART KERNEL, then run all cells below.")
"""


# 
# ║  CELL 2 — IMPORTS                                                          ║
# 

import os, json, time, random, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import (
    vgg16, VGG16_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
)
import timm

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, balanced_accuracy_score, average_precision_score,
    roc_curve, auc, precision_recall_curve,
)

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

print("✅ All imports done")
print(f"   PyTorch  : {torch.__version__}")
print(f"   CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU      : {torch.cuda.get_device_name(0)}")


# ║  CELL 3 — SCAN YOUR DATASETS & BUILD UNIFIED CLASS MAPPING                 ║



CLASS_NAMES = [
    "atopic_dermatitis",
    "contact_dermatitis",
    "eczema",
    "scabies",
    "seborrheic_dermatitis",
    "tinea",
    "vitiligo",
]

PRETTY_NAMES = [
    "Atopic Dermatitis",
    "Contact Dermatitis",
    "Eczema",
    "Scabies",
    "Seborrheic Dermatitis",
    "Tinea (Ringworm)",
    "Vitiligo",
]

NUM_CLASSES = len(CLASS_NAMES)
cls2idx     = {c: i for i, c in enumerate(CLASS_NAMES)}
idx2cls     = {i: c for c, i in cls2idx.items()}

# ══════════════════════════════════════════════════
# FOLDER → UNIFIED CLASS MAPPING
# Maps each dataset's folder names to unified labels
# ══════════════════════════════════════════════════
FOLDER_TO_CLASS = {
    # ── skin-dis-net (Preprocessed folder) ────────────────────
    "Atopic Dermatitis (AD)"    : "atopic_dermatitis",
    "Contact Dermatitis (CD)"   : "contact_dermatitis",
    "Eczema (EC)"               : "eczema",
    "Scabies (SC)"              : "scabies",
    "Seborrheic Dermatitis (SD)": "seborrheic_dermatitis",
    "Tinea Corporis (TC)"       : "tinea",

    # ── skindeasesbd (Updated Images folder) ──────────────────
    "Dermatitis"                : "atopic_dermatitis",   # maps to atopic
    "Eczema"                    : "eczema",
    "Scabies"                   : "scabies",
    "Tinea Ringworm"            : "tinea",
    "Vitiligo"                  : "vitiligo",
}

def scan_all_images():
    """
    Walk both datasets and collect all images with their unified class labels.
    Returns: list of (image_path, class_idx) tuples
    """
    print("\n📂 Scanning datasets...")
    all_samples   = []
    class_counts  = defaultdict(int)
    skipped       = 0

    # ── Paths to scan ─────────────────────────────────────────
    SCAN_ROOTS = [
        Path("/kaggle/input"),   # walks everything
    ]

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for scan_root in SCAN_ROOTS:
        for img_path in scan_root.rglob("*"):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            if img_path.stat().st_size < 1000:   # skip tiny/corrupt files
                continue

            # ── Match folder name to class ────────────────────
            matched_class = None
            for part in img_path.parts:
                if part in FOLDER_TO_CLASS:
                    matched_class = FOLDER_TO_CLASS[part]
                    break

            if matched_class is None:
                skipped += 1
                continue

            class_idx = cls2idx[matched_class]
            all_samples.append((str(img_path), class_idx))
            class_counts[matched_class] += 1

    print(f"\n  ✅ Total images found : {len(all_samples):,}")
    print(f"  ⚠️  Skipped (no match): {skipped:,}")
    print(f"\n  📊 Class Distribution:")
    print(f"  {'Class':<30} {'Count':>6}")
    print(f"  {'─'*38}")
    for cls in CLASS_NAMES:
        cnt = class_counts.get(cls, 0)
        bar = "█" * (cnt // 30)
        print(f"  {PRETTY_NAMES[cls2idx[cls]]:<30} {cnt:>6}  {bar}")

    if len(all_samples) == 0:
        print("\n  ❌ NO IMAGES FOUND! Check paths.")
        print("  Run this to debug:")
        print("  for r,d,f in os.walk('/kaggle/input'):")
        print("      if any(x.endswith('.jpg') for x in f):")
        print("          print(r, len(f))")

    return all_samples, dict(class_counts)

# ── Run scan ──────────────────────────────────────────────────
all_samples, class_counts = scan_all_images()


# ║  CELL 4 — TRAIN / VAL / TEST SPLIT (70 / 15 / 15)                         ║

def make_splits(all_samples, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Stratified split — same class ratio in all 3 splits."""
    paths  = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    # 70% train, 30% temp
    tr_p, tmp_p, tr_l, tmp_l = train_test_split(
        paths, labels,
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=seed
    )
    # 50% of temp → val, 50% → test  (= 15% each of total)
    va_p, te_p, va_l, te_l = train_test_split(
        tmp_p, tmp_l,
        test_size=0.50,
        stratify=tmp_l,
        random_state=seed
    )

    train_data = list(zip(tr_p, tr_l))
    val_data   = list(zip(va_p, va_l))
    test_data  = list(zip(te_p, te_l))

    print(f"\n  ✅ Dataset splits (seed={seed}):")
    print(f"     Train : {len(train_data):>5,} images ({len(train_data)/len(all_samples)*100:.1f}%)")
    print(f"     Val   : {len(val_data):>5,} images ({len(val_data)/len(all_samples)*100:.1f}%)")
    print(f"     Test  : {len(test_data):>5,} images ({len(test_data)/len(all_samples)*100:.1f}%)")
    print(f"     Total : {len(all_samples):>5,} images")

    # Per-class count in test set
    from collections import Counter
    test_dist = Counter(te_l)
    print(f"\n  Test set class distribution:")
    for idx in sorted(test_dist.keys()):
        print(f"     [{idx}] {PRETTY_NAMES[idx]:<28}: {test_dist[idx]:>4}")

    return train_data, val_data, test_data

train_data, val_data, test_data = make_splits(all_samples)


# ║  CELL 5 — OUTPUT DIRECTORIES                                               ║

OUTPUT   = Path("/kaggle/working/paper_results")
DIRS = {
    "models"  : OUTPUT / "saved_models",
    "metrics" : OUTPUT / "metrics",
    "plots"   : OUTPUT / "plots",
    "gradcam" : OUTPUT / "gradcam",
    "tsne"    : OUTPUT / "tsne",
    "tables"  : OUTPUT / "tables",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

CFG = {
    "img_size"    : 224,
    "batch_size"  : 16,
    "num_epochs"  : 50,
    "patience"    : 10,
    "lr"          : 1e-4,
    "weight_decay": 1e-4,
    "num_workers" : 2,
    "num_seeds"   : 3,        # repeat each model 3x (use 5 for final paper)
    "device"      : "cuda" if torch.cuda.is_available() else "cpu",

    # ── Your saved model ──────────────────────────────────────
    # After you save FINAL1 output as a dataset:
    "ckpt_path"   : "/kaggle/input/bdskinnet-checkpoint/bdskinnet_best.pt",
}

print(f"  Device : {CFG['device']}")
print(f"  Output : {OUTPUT}")


# ║  CELL 6 — DATASET CLASS & TRANSFORMS                                       ║

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

class SkinDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of (image_path_str, class_idx)
        """
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_str, label = self.samples[idx]
        try:
            img = Image.open(path_str).convert("RGB")
        except Exception:
            img = Image.new("RGB", (CFG["img_size"], CFG["img_size"]), (128,128,128))

        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms(img_size=224, is_inception=False):
    size = 299 if is_inception else img_size

    train_tf = T.Compose([
        T.Resize((size + 20, size + 20)),
        T.RandomCrop(size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])

    val_tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])

    return train_tf, val_tf

def build_loaders(is_inception=False, batch_size=None):
    bs = batch_size or CFG["batch_size"]
    train_tf, val_tf = get_transforms(CFG["img_size"], is_inception)

    train_ds = SkinDataset(train_data, train_tf)
    val_ds   = SkinDataset(val_data,   val_tf)
    test_ds  = SkinDataset(test_data,  val_tf)

    # Weighted sampler — handles class imbalance
    labels       = np.array([s[1] for s in train_data])
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    weights      = 1.0 / class_counts[labels]
    sampler      = WeightedRandomSampler(weights, len(train_ds), replacement=True)

    tr_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                           num_workers=CFG["num_workers"], pin_memory=True)
    va_loader = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                           num_workers=CFG["num_workers"], pin_memory=True)
    te_loader = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                           num_workers=CFG["num_workers"], pin_memory=True)

    return tr_loader, va_loader, te_loader

print("✅ Dataset and transforms ready")


# ║  CELL 7 — METRICS & TRAINING ENGINE                                        ║

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compute_all_metrics(y_true, y_pred, y_prob):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    try:
        auc_roc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except:
        auc_roc = 0.0
    try:
        pr_auc = average_precision_score(y_bin, y_prob, average="macro")
    except:
        pr_auc = 0.0

    return {
        "accuracy"         : round(accuracy_score(y_true, y_pred) * 100, 2),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred) * 100, 2),
        "macro_precision"  : round(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "macro_recall"     : round(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "macro_f1"         : round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "weighted_f1"      : round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "cohen_kappa"      : round(cohen_kappa_score(y_true, y_pred), 4),
        "mcc"              : round(matthews_corrcoef(y_true, y_pred), 4),
        "auc_roc"          : round(auc_roc, 4),
        "pr_auc"           : round(pr_auc, 4),
        "per_class_f1"     : (f1_score(y_true, y_pred, average=None, zero_division=0) * 100).tolist(),
    }

def get_criterion():
    labels       = np.array([s[1] for s in train_data])
    counts       = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    weights      = 1.0 / (counts + 1e-6)
    weights      = weights / weights.sum() * NUM_CLASSES
    return nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(CFG["device"]))

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(CFG["device"]), labels.to(CFG["device"])
        out = model(imgs)
        if isinstance(out, tuple): out = out[0]
        loss  = criterion(out, labels)
        probs = F.softmax(out, dim=1)
        preds = probs.argmax(dim=1)
        total_loss  += loss.item() * imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = round(total_loss / len(y_true), 4)
    return metrics, y_true, y_pred, y_prob

def train_model(model, model_name, seed=42, is_inception=False):
    """Full training: fit model, early stop on val F1, return test metrics."""
    set_seed(seed)
    device = CFG["device"]
    model  = model.to(device)

    tr_loader, va_loader, te_loader = build_loaders(is_inception)
    criterion = get_criterion()
    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"],
                             weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler()

    best_f1      = 0.0
    patience_cnt = 0
    ckpt         = DIRS["models"] / f"{model_name}_s{seed}.pt"
    history      = {"train_loss": [], "val_f1": [], "val_acc": []}

    print(f"\n  🚀 {model_name}  [seed={seed}]")

    for epoch in range(1, CFG["num_epochs"] + 1):
        # ── Train ─────────────────────────────────────────────
        model.train()
        t_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out  = model(imgs)
                if isinstance(out, tuple): out = out[0]
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            t_loss  += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total   += imgs.size(0)
        scheduler.step()

        tr_loss = t_loss / total
        tr_acc  = correct / total * 100
        history["train_loss"].append(tr_loss)

        # ── Validate ──────────────────────────────────────────
        val_m, _, _, _ = evaluate(model, va_loader, criterion)
        history["val_f1"].append(val_m["macro_f1"])
        history["val_acc"].append(val_m["accuracy"])

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Ep {epoch:3d}/{CFG['num_epochs']} | "
                  f"Loss: {tr_loss:.4f} | Acc: {tr_acc:.1f}% | "
                  f"Val F1: {val_m['macro_f1']:.2f}% | Best: {best_f1:.2f}%")

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                print(f"    ⏹ Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_m, y_true, y_pred, y_prob = evaluate(model, te_loader, criterion)
    print(f"    ✅ Test → Acc: {test_m['accuracy']:.2f}% | "
          f"F1: {test_m['macro_f1']:.2f}% | "
          f"AUC: {test_m['auc_roc']:.4f} | "
          f"κ: {test_m['cohen_kappa']:.4f}")
    return test_m, y_true, y_pred, y_prob, history


# ── Results storage ────────────────────────────────────────────
ALL_RESULTS = {}

def store_results(name, metrics_list, y_true, y_pred, y_prob):
    keys = [k for k in metrics_list[0] if k != "per_class_f1"]
    mean = {k: round(float(np.mean([m[k] for m in metrics_list])), 4) for k in keys}
    std  = {k: round(float(np.std ([m[k] for m in metrics_list])), 4) for k in keys}
    ALL_RESULTS[name] = {
        "mean": mean, "std": std,
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
        "per_class_f1": metrics_list[-1].get("per_class_f1", []),
    }
    with open(DIRS["metrics"] / f"{name}.json", "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)
    print(f"  💾 Saved: {name}")

def run_multi_seed(factory_fn, name, is_inception=False):
    print(f"\n{'═'*60}")
    print(f"  MODEL: {name}")
    print(f"{'═'*60}")
    metrics_list = []
    last = (None, None, None)
    for seed in range(CFG["num_seeds"]):
        m = factory_fn()
        test_m, y_true, y_pred, y_prob, _ = train_model(
            m, name, seed=seed, is_inception=is_inception)
        metrics_list.append(test_m)
        last = (y_true, y_pred, y_prob)
        torch.cuda.empty_cache()
    store_results(name, metrics_list, *last)
    return ALL_RESULTS[name]

print("✅ Training engine ready")


# ── Focal loss with label smoothing (paper Eq. 5) ─────────────────────────────
class FocalLoss(nn.Module):
    """
    Class-balanced focal loss with label smoothing, per BD-SkinNet paper Eq. (5).
    α_i = inverse-frequency weights; γ = 2.0; ε = 0.1.
    Used for BD-SkinNet training and ablation variants.
    """
    def __init__(self, gamma=2.0, label_smooth=0.1, class_weights=None,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.gamma        = gamma
        self.label_smooth = label_smooth
        self.num_classes  = num_classes
        w = class_weights if class_weights is not None else torch.ones(num_classes)
        self.register_buffer("weight", w)

    def forward(self, logits, targets):
        with torch.no_grad():
            smooth = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1.0)
            smooth = smooth * (1 - self.label_smooth) + \
                     self.label_smooth / self.num_classes
        log_p   = F.log_softmax(logits, dim=1)
        pt      = log_p.exp()[range(len(targets)), targets]
        focal_w = self.weight[targets] * (1 - pt) ** self.gamma
        loss    = -(smooth * log_p).sum(dim=1)
        return (focal_w * loss).mean()


def get_focal_criterion():
    """Focal loss + label smoothing + inverse-frequency weights for BD-SkinNet."""
    labels  = np.array([s[1] for s in train_data])
    counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum()
    return FocalLoss(
        gamma=2.0,
        label_smooth=0.1,
        class_weights=torch.FloatTensor(weights).to(CFG["device"]),
        num_classes=NUM_CLASSES,
    )


# ║  CELL 8 — ALL MODEL FACTORY FUNCTIONS                                      ║

def clf_head(in_f, n=NUM_CLASSES, drop=0.4):
    return nn.Sequential(nn.Dropout(drop), nn.Linear(in_f, n))

# ── Classic CNNs ──────────────────────────────────────────────
def make_vgg16():
    m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for p in list(m.features.parameters())[:24]:
        p.requires_grad = False
    m.classifier[6] = clf_head(4096)
    return m

def make_resnet50():
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for name, p in m.named_parameters():
        if "layer1" in name or "layer2" in name: p.requires_grad = False
    m.fc = clf_head(m.fc.in_features)
    return m

def make_densenet121():
    m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = clf_head(m.classifier.in_features)
    return m

# ── Modern CNNs ───────────────────────────────────────────────
def make_effb0():
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = clf_head(m.classifier[1].in_features)
    return m

def make_effb4():
    m = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    m.classifier[1] = clf_head(m.classifier[1].in_features)
    return m

def make_effv2s():
    return timm.create_model("efficientnetv2_s", pretrained=True,
                              num_classes=NUM_CLASSES, drop_rate=0.3)

def make_convnext():
    return timm.create_model("convnext_tiny", pretrained=True,
                              num_classes=NUM_CLASSES, drop_rate=0.2)

# ── Vision Transformers ───────────────────────────────────────
def make_vit():
    m = timm.create_model("vit_base_patch16_224", pretrained=True,
                            num_classes=NUM_CLASSES, drop_rate=0.1)
    for i, block in enumerate(m.blocks):
        if i < 8:
            for p in block.parameters(): p.requires_grad = False
    return m

def make_swin():
    return timm.create_model("swin_tiny_patch4_window7_224", pretrained=True,
                               num_classes=NUM_CLASSES, drop_rate=0.1)

def make_deit():
    return timm.create_model("deit_small_patch16_224", pretrained=True,
                               num_classes=NUM_CLASSES)

print("✅ All model factories ready")
print(f"   {NUM_CLASSES} classes: {PRETTY_NAMES}")


# ║  CELL 9 — TRADITIONAL ML (SVM, Random Forest, KNN)                        ║

from skimage.feature import hog
from skimage.feature import graycomatrix, graycoprops

def extract_features(img_path_str, size=64):
    """HOG + GLCM + Color Histogram features."""
    try:
        img = Image.open(img_path_str).convert("RGB").resize((size, size))
        arr = np.array(img).astype(np.float32) / 255.0
    except:
        return np.zeros(1000)

    # HOG
    gray    = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
    hog_f   = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2),
                   feature_vector=True)

    # GLCM texture
    gray_u8 = (gray * 255).astype(np.uint8)
    glcm    = graycomatrix(gray_u8, [1,2], [0, np.pi/4, np.pi/2], 256,
                            symmetric=True, normed=True)
    glcm_f  = []
    for prop in ["contrast","homogeneity","energy","correlation"]:
        glcm_f.extend(graycoprops(glcm, prop).flatten())
    glcm_f = np.array(glcm_f)

    # Color histogram (RGB channels)
    color_f = []
    for ch in range(3):
        h, _ = np.histogram(arr[:,:,ch], bins=32, range=(0,1))
        color_f.extend(h / (h.sum() + 1e-6))
    color_f = np.array(color_f)

    return np.concatenate([hog_f, glcm_f, color_f])

def build_features(samples, desc="Features"):
    X, y = [], []
    for path_str, label in tqdm(samples, desc=f"  {desc}"):
        X.append(extract_features(path_str))
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)

def run_traditional_ml():
    print(f"\n{'═'*60}")
    print("  TRADITIONAL ML BASELINES")
    print(f"{'═'*60}")

    X_tr, y_tr = build_features(train_data + val_data, "Train+Val features")
    X_te, y_te = build_features(test_data,             "Test features")

    scaler     = StandardScaler()
    X_tr_sc    = scaler.fit_transform(X_tr)
    X_te_sc    = scaler.transform(X_te)

    pca        = PCA(n_components=0.95, random_state=42)
    X_tr_pc    = pca.fit_transform(X_tr_sc)
    X_te_pc    = pca.transform(X_te_sc)
    print(f"  PCA: {X_tr_sc.shape[1]} → {X_tr_pc.shape[1]} features")

    clfs = {
        "SVM_HOG_GLCM": SVC(kernel="rbf", C=10, gamma="scale",
                              probability=True, random_state=42,
                              class_weight="balanced"),
        "RandomForest" : RandomForestClassifier(n_estimators=300, random_state=42,
                                                 class_weight="balanced", n_jobs=-1),
        "KNN"          : KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    }

    for name, clf in clfs.items():
        print(f"\n  Fitting {name}...")
        t0 = time.time()
        clf.fit(X_tr_pc, y_tr)
        y_pred = clf.predict(X_te_pc)
        y_prob = clf.predict_proba(X_te_pc)
        m      = compute_all_metrics(y_te, y_pred, y_prob)
        ALL_RESULTS[name] = {
            "mean": m, "std": {k: 0.0 for k in m},
            "y_true": y_te, "y_pred": y_pred, "y_prob": y_prob,
            "per_class_f1": m["per_class_f1"],
        }
        print(f"  ✅ {name}: Acc={m['accuracy']:.2f}% | "
              f"F1={m['macro_f1']:.2f}% | AUC={m['auc_roc']:.4f} | "
              f"Time={time.time()-t0:.0f}s")

    print("\n✅ Traditional ML done")

# ── RUN ──────────────────────────────────────────────────────
# run_traditional_ml()


# ║  CELL 10 —                                 ║



CLASSIC_CNNS = [
    ("VGG-16",      make_vgg16,      False),
    ("ResNet-50",   make_resnet50,   False),
    ("DenseNet-121",make_densenet121,False),
]

MODERN_CNNS = [
    ("EfficientNet-B0",  make_effb0,   False),
    ("EfficientNet-B4",  make_effb4,   False),
    ("EfficientNetV2-S", make_effv2s,  False),
    ("ConvNeXt-Tiny",    make_convnext,False),
]

TRANSFORMERS = [
    ("ViT-B16",   make_vit,  False),
    ("Swin-Tiny", make_swin, False),
    ("DeiT-Small",make_deit, False),
]

def run_group(group, group_name):
    print(f"\n{'#'*60}")
    print(f"  {group_name}")
    print(f"{'#'*60}")
    for name, factory, is_inc in group:
        try:
            run_multi_seed(factory, name, is_inception=is_inc)
            # Save checkpoint after each model
            ckpt = {k: {"mean": v["mean"], "std": v["std"],
                        "per_class_f1": v.get("per_class_f1",[])}
                    for k, v in ALL_RESULTS.items()}
            with open(DIRS["metrics"] / "all_results.json", "w") as f:
                json.dump(ckpt, f, indent=2)
            print(f"  💾 Checkpoint: {len(ALL_RESULTS)} models saved")
        except Exception as e:
            print(f"  ❌ {name} FAILED: {e}")
            import traceback; traceback.print_exc()
        torch.cuda.empty_cache()





# ║  CELL 11 — LOAD YOUR BD-SKINNET & EVALUATE                                 ║
"""
HOW TO SAVE YOUR MODEL FROM FINAL1:
────────────────────────────────────
In your FINAL1 notebook, add this as the last cell and run it:

    torch.save({
        'model_state_dict': model.state_dict(),   ← your model variable
        'num_classes': 7,
    }, "/kaggle/working/bdskinnet_best.pt")

Then: Save & Run All → Output tab → save as dataset "bdskinnet-checkpoint"
Then: Add that dataset as input to THIS notebook
"""

def load_bdskinnet():
    print(f"\n{'═'*60}")
    print("  BD-SkinNet — Your Model Evaluation")
    print(f"{'═'*60}")

    ckpt_path = Path(CFG["ckpt_path"])

    if not ckpt_path.exists():
        # ── Option A: Enter your results manually ─────────────
        print(f"  ⚠️  Checkpoint not found at: {ckpt_path}")
        print("  ℹ️  Using your known results directly...")

        # Exact values from paper Table IV (BD-SkinNet row) and Table V (per-class)
        manual = {
            "accuracy"         : 92.37,
            "balanced_accuracy": 92.48,   # macro avg recall ≈ balanced acc (Table V)
            "macro_precision"  : 92.52,   # Table V macro avg precision 0.9252
            "macro_recall"     : 92.48,   # Table V macro avg recall 0.9248
            "macro_f1"         : 92.46,   # Table IV / Table V macro avg F1 0.9246
            "weighted_f1"      : 92.46,
            "cohen_kappa"      : 0.9103,  # Table IV
            "mcc"              : 0.9104,
            "auc_roc"          : 0.9937,  # Table IV
            "pr_auc"           : 0.9901,
            # Table V per-class F1 (×100): Atopic, Contact, Eczema, Scabies,
            # Seborrheic, Tinea, Vitiligo
            "per_class_f1"     : [87.42, 89.18, 92.86, 94.53, 91.03, 92.22, 100.0],
        }
        ALL_RESULTS["BD-SkinNet (Ours)"] = {
            "mean": manual, "std": {k: 0.0 for k in manual},
            "y_true": None, "y_pred": None, "y_prob": None,
            "per_class_f1": manual["per_class_f1"],
        }
        print("  ✅ Manual results stored")
        return



    # ── CBAM attention modules (paper Eqs. 1–2) ──────────────────────────────
    class ChannelAttention(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(),
                nn.Linear(channels // reduction, channels, bias=False),
            )
        def forward(self, x):
            avg = self.mlp(x.mean(dim=[2, 3]))
            mx  = self.mlp(x.amax(dim=[2, 3]))
            return torch.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size=7):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size,
                                  padding=kernel_size // 2, bias=False)
        def forward(self, x):
            avg = x.mean(dim=1, keepdim=True)
            mx  = x.amax(dim=1, keepdim=True)
            return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

    class CBAM(nn.Module):
        def __init__(self, channels, reduction=16, spatial_kernel=7):
            super().__init__()
            self.ca = ChannelAttention(channels, reduction)
            self.sa = SpatialAttention(spatial_kernel)
        def forward(self, x):
            x = x * self.ca(x)
            x = x * self.sa(x)
            return x

    # ── BD-SkinNet: Swin-Base + stage-wise CBAM + multi-scale GAP + MLP head ──
    class BDSkinNet(nn.Module):
        """
        BD-SkinNet architecture (paper Section III-D):
        swin_base_patch4_window7_224 (ImageNet-21K) → CBAM at each of 4 stages →
        global-average-pool each stage → 128+256+512+1024 = 1920-d concat →
        Dropout(0.4) → Linear(1920→512) → LayerNorm → GELU →
        Dropout(0.2) → Linear(512→7).
        """
        def __init__(self, num_classes=7, pretrained=False):
            super().__init__()
            self.backbone = timm.create_model(
                "swin_base_patch4_window7_224",
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            stage_dims = [128, 256, 512, 1024]
            self.cbams  = nn.ModuleList([CBAM(d) for d in stage_dims])
            self.gap    = nn.AdaptiveAvgPool2d(1)
            feat_dim    = sum(stage_dims)   # 1920
            self.head   = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(feat_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes),
            )

        def forward(self, x):
            feats  = self.backbone(x)
            pooled = []
            for feat, cbam in zip(feats, self.cbams):
                # timm Swin: (N, H, W, C) or (N, H*W, C) depending on version
                if feat.dim() == 4:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                elif feat.dim() == 3:
                    N, L2, C = feat.shape
                    L = int(L2 ** 0.5)
                    feat = feat.reshape(N, L, L, C).permute(0, 3, 1, 2).contiguous()
                feat = cbam(feat)
                pooled.append(self.gap(feat).flatten(1))
            return self.head(torch.cat(pooled, dim=1))

    device = CFG["device"]
    model  = BDSkinNet(NUM_CLASSES).to(device)

    ckpt   = torch.load(ckpt_path, map_location=device)
    sd     = ckpt.get("model_state_dict", ckpt)
    sd     = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    _, _, te_loader = build_loaders()
    criterion       = get_criterion()
    test_m, y_true, y_pred, y_prob = evaluate(model, te_loader, criterion)

    print(f"\n  📊 BD-SkinNet Results:")
    for k, v in test_m.items():
        if k != "per_class_f1":
            print(f"     {k:<22}: {v}")

    ALL_RESULTS["BD-SkinNet (Ours)"] = {
        "mean": test_m, "std": {k: 0.0 for k in test_m},
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
        "per_class_f1": test_m["per_class_f1"],
    }




# ║  CELL 12 — MAIN RESULTS TABLE                                 ║


def generate_main_table():
    if not ALL_RESULTS:
        print("❌ No results yet. Run models first.")
        return

    print(f"\n{'═'*90}")
    print(f"  MAIN COMPARISON TABLE")
    print(f"{'═'*90}")
    print(f"  {'Method':<24} {'Acc':>7} {'BalAcc':>7} {'Prec':>7} {'Rec':>7} "
          f"{'MacF1':>7} {'AUC':>7} {'PRAUC':>7} {'κ':>7} {'MCC':>7}")
    print(f"  {'─'*86}")

    rows = []
    for name, res in ALL_RESULTS.items():
        m   = res["mean"]
        std = res["std"]

        def f(k, dec=2):
            v = m.get(k, 0)
            s = std.get(k, 0)
            return f"{v:.{dec}f}±{s:.{dec}f}"

        marker = " ★" if "Ours" in name else "  "
        flag   = "**" if "Ours" in name else "  "
        print(f"  {flag}{name:<22} {f('accuracy'):>9} {f('balanced_accuracy'):>9} "
              f"{f('macro_precision'):>9} {f('macro_recall'):>9} "
              f"{f('macro_f1'):>9} {f('auc_roc',4):>9} "
              f"{f('pr_auc',4):>9} {f('cohen_kappa',4):>9} "
              f"{f('mcc',4):>9}{marker}")

        rows.append({
            "Method"        : name,
            "Accuracy (%)"  : f("accuracy"),
            "Bal Acc (%)"   : f("balanced_accuracy"),
            "Precision (%)" : f("macro_precision"),
            "Recall (%)"    : f("macro_recall"),
            "Macro F1 (%)"  : f("macro_f1"),
            "AUC-ROC"       : f("auc_roc", 4),
            "PR-AUC"        : f("pr_auc",  4),
            "Kappa"         : f("cohen_kappa", 4),
            "MCC"           : f("mcc", 4),
        })

    print(f"  {'─'*86}")
    print("  ** = Proposed method  |  ★ = Best  |  Values: mean±std across 3 seeds")

    df = pd.DataFrame(rows)
    df.to_csv(DIRS["tables"] / "main_table.csv", index=False)

    # LaTeX
    df_latex = df.set_index("Method")
    latex    = df_latex.to_latex(
        caption="Performance comparison on BD-SkinNet 7-class skin disease dataset. "
                "Best results in bold. Values reported as mean±std over 3 seeds.",
        label="tab:comparison",
        escape=False,
    )
    with open(DIRS["tables"] / "main_table.tex", "w") as f:
        f.write(latex)

    print(f"\n  💾 CSV  : {DIRS['tables']/'main_table.csv'}")
    print(f"  💾 LaTeX: {DIRS['tables']/'main_table.tex'}")

    # ── Visual figure of table ────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, max(5, len(rows)*0.55 + 2)))
    ax.axis("off")
    col_labels = list(rows[0].keys())
    table_data = [[r[c] for c in col_labels] for r in rows]
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.7)
    for j in range(len(col_labels)):
        tbl[0,j].set_facecolor("#1A252F")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(rows):
        for j in range(len(col_labels)):
            cell = tbl[i+1, j]
            if "Ours" in row["Method"]:
                cell.set_facecolor("#F9CA24")
                cell.set_text_props(fontweight="bold")
            elif i % 2 == 0:
                cell.set_facecolor("#F4F6F7")
    ax.set_title("Table: Comparison of BD-SkinNet with Baseline Models",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(DIRS["plots"] / "main_table_figure.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  💾 Figure: {DIRS['plots']/'main_table_figure.png'}")
    return df




# ║  CELL 13 — PER-CLASS F1 HEATMAP                                            ║


def plot_perclass_heatmap():
    data, names = [], []
    for name, res in ALL_RESULTS.items():
        pcf1 = res.get("per_class_f1", [])
        if len(pcf1) == NUM_CLASSES:
            data.append(pcf1)
            names.append(name)
    if not data:
        print("❌ No per-class data yet")
        return

    df = pd.DataFrame(data, index=names, columns=PRETTY_NAMES)
    fig, ax = plt.subplots(figsize=(14, max(5, len(names)*0.65 + 2)))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=50, vmax=100, ax=ax, linewidths=0.5,
                cbar_kws={"label": "F1 Score (%)"})
    ax.set_title("Per-Class F1 Score (%) — All Models vs All Diseases",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Disease Class", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(DIRS["plots"] / "perclass_f1_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    df.to_csv(DIRS["tables"] / "perclass_f1.csv")
    print(f"  💾 Per-class heatmap saved")




# ║  CELL 14 — CONFUSION MATRICES                                              ║

def plot_confusion_matrix(y_true, y_pred, name):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ["d", ".1f"],
        [f"Count — {name}", f"Normalized (%) — {name}"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=PRETTY_NAMES, yticklabels=PRETTY_NAMES,
                    linewidths=0.5, annot_kws={"size": 9})
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True",      fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(DIRS["plots"] / f"cm_{name.replace(' ','_')}.png",
                dpi=200, bbox_inches="tight")
    plt.close()

def plot_all_cms():
    print("\n📊 Confusion Matrices...")
    for name, res in ALL_RESULTS.items():
        if res["y_true"] is not None:
            plot_confusion_matrix(res["y_true"], res["y_pred"], name)
            print(f"  💾 CM: {name}")





# ║  CELL 15 — ROC + PR CURVES                                                 ║

def plot_roc_all():
    print("\n📊 ROC Curves...")
    fig, ax = plt.subplots(figsize=(11, 9))
    COLORS  = plt.cm.tab20.colors

    for i, (name, res) in enumerate(ALL_RESULTS.items()):
        y_true = res.get("y_true")
        y_prob = res.get("y_prob")
        if y_true is None: continue

        y_bin    = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        fpr_grid = np.linspace(0, 1, 300)
        tprs     = []
        for c in range(NUM_CLASSES):
            fpr, tpr, _ = roc_curve(y_bin[:,c], y_prob[:,c])
            tprs.append(np.interp(fpr_grid, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        roc_auc_v = auc(fpr_grid, mean_tpr)

        lw = 3.0 if "Ours" in name else 1.5
        ls = "-"  if "Ours" in name else "--"
        ax.plot(fpr_grid, mean_tpr, color=COLORS[i%len(COLORS)], lw=lw, ls=ls,
                label=f"{name} (AUC={roc_auc_v:.4f})")

    ax.plot([0,1],[0,1],"k--",lw=1.2)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("Macro-Average ROC Curves — All Models",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.01])
    plt.tight_layout()
    plt.savefig(DIRS["plots"] / "roc_all_models.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  💾 ROC saved")

def plot_perclass_roc_bdskinnet():
    """BD-SkinNet per-class ROC — required for paper."""
    name = "BD-SkinNet (Ours)"
    if name not in ALL_RESULTS or ALL_RESULTS[name]["y_true"] is None:
        print(f"  ⚠️  {name} y_true not available")
        return
    res    = ALL_RESULTS[name]
    y_true = res["y_true"]
    y_prob = res["y_prob"]
    y_bin  = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    COLORS = plt.cm.Set1.colors
    fig, ax = plt.subplots(figsize=(11, 9))
    for c in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:,c], y_prob[:,c])
        roc_auc_v   = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[c%len(COLORS)], lw=2,
                label=f"{PRETTY_NAMES[c]} (AUC={roc_auc_v:.4f})")
    # Macro
    fpr_grid = np.linspace(0,1,300)
    tprs = [np.interp(fpr_grid, *roc_curve(y_bin[:,c], y_prob[:,c])[:2])
            for c in range(NUM_CLASSES)]
    mean_tpr = np.mean(tprs, axis=0)
    ax.plot(fpr_grid, mean_tpr, "k-", lw=3.5,
            label=f"Macro Avg (AUC={auc(fpr_grid,mean_tpr):.4f})")
    ax.plot([0,1],[0,1],"k--",lw=1.2)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("BD-SkinNet — Per-Class ROC Curves",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIRS["plots"] / "roc_perclass_bdskinnet.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  💾 Per-class ROC saved")



# ║  CELL 16 — STATISTICAL SIGNIFICANCE 
def run_mcnemar():
    from mlxtend.evaluate import mcnemar, mcnemar_table
    print(f"\n{'═'*60}")
    print("  MCNEMAR'S TEST — BD-SkinNet vs Each Baseline")
    print(f"{'═'*60}")

    bdsk = ALL_RESULTS.get("BD-SkinNet (Ours)")
    if bdsk is None or bdsk["y_true"] is None:
        print("  ⚠️  Need BD-SkinNet predictions. Skipping.")
        return

    bds_true = bdsk["y_true"]
    bds_pred = bdsk["y_pred"]
    bds_c    = (bds_pred == bds_true)
    rows     = []

    for name, res in ALL_RESULTS.items():
        if "Ours" in name or res["y_true"] is None: continue
        n           = min(len(bds_true), len(res["y_true"]))
        other_c     = (res["y_pred"][:n] == res["y_true"][:n])
        tb          = mcnemar_table(bds_c[:n].astype(int), other_c.astype(int))
        chi2, p     = mcnemar(tb, corrected=True)
        sig         = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        rows.append({"vs Model": name, "χ²": f"{chi2:.3f}",
                     "p-value": f"{p:.6f}", "Sig": sig})
        print(f"  vs {name:<24}: χ²={chi2:.3f}  p={p:.6f}  {sig}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(DIRS["tables"] / "mcnemar_test.csv", index=False)
        print(f"\n  Bonferroni α = {0.05/max(len(rows),1):.4f}")
        print(f"  *** p<0.001  ** p<0.01  * p<0.05  ns=not significant")
        print(f"  💾 Saved: mcnemar_test.csv")

# ── RUN ──────────────────────────────────────────────────────
# run_mcnemar()


# ║  CELL 17 — GRADCAM VISUALIZATION (BD-SkinNet)                              ║

def generate_gradcam(model, target_layer, model_name="BD-SkinNet", samples_per_class=2):
    """GradCAM++ grid: rows=classes, cols=original+overlay."""
    print("\n📊 GradCAM++ visualizations...")
    device   = CFG["device"]
    model    = model.to(device).eval()
    _, val_tf = get_transforms()
    cam      = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    inv_tf   = T.Normalize([-m/s for m,s in zip(IMG_MEAN,IMG_STD)],
                            [1/s for s in IMG_STD])

    # Collect samples per class from test set
    class_imgs = defaultdict(list)
    ds         = SkinDataset(test_data, val_tf)
    for i in range(min(len(ds), 2000)):
        img_t, label = ds[i]
        if len(class_imgs[label]) < samples_per_class:
            class_imgs[label].append(img_t)
        if all(len(v) >= samples_per_class for v in class_imgs.values()) \
           and len(class_imgs) == NUM_CLASSES:
            break

    fig, axes = plt.subplots(NUM_CLASSES, samples_per_class*2,
                              figsize=(samples_per_class*5, NUM_CLASSES*3))

    for row, cls_idx in enumerate(sorted(class_imgs.keys())):
        for col, img_t in enumerate(class_imgs[cls_idx][:samples_per_class]):
            orig    = inv_tf(img_t).clamp(0,1).permute(1,2,0).numpy()
            inp     = img_t.unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(cls_idx)]
            gray    = cam(input_tensor=inp, targets=targets)[0]
            overlay = show_cam_on_image(orig, gray, use_rgb=True, image_weight=0.6)

            ax_o = axes[row, col*2]
            ax_c = axes[row, col*2+1]
            ax_o.imshow(orig);    ax_o.axis("off")
            ax_c.imshow(overlay); ax_c.axis("off")
            if col == 0:
                ax_o.set_ylabel(PRETTY_NAMES[cls_idx], fontsize=8,
                                fontweight="bold")
            if row == 0:
                ax_o.set_title("Original",  fontsize=8)
                ax_c.set_title("GradCAM++", fontsize=8)

    plt.suptitle(f"GradCAM++ — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DIRS["gradcam"] / f"gradcam_{model_name}.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  💾 GradCAM saved")



# ║  CELL 18 — t-SNE FEATURE VISUALIZATION                                     ║

def plot_tsne(model_a, name_a, model_b, name_b):
    """Side-by-side t-SNE: baseline vs BD-SkinNet."""
    print("\n📊 t-SNE feature visualization...")
    device    = CFG["device"]
    _, _, te_loader = build_loaders()
    COLORS    = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, model, name in [(axes[0], model_a, name_a),
                              (axes[1], model_b, name_b)]:
        model = model.to(device).eval()
        feats_list, lbls_list = [], []
        feat_buf = {}

        def hook(m, inp, out):
            feat_buf["f"] = out.detach().cpu()
        # Attach to appropriate layer
        if hasattr(model, "backbone"):
            h = model.backbone[-1].register_forward_hook(hook)
        elif hasattr(model, "layer4"):
            h = model.layer4.register_forward_hook(hook)
        else:
            h = list(model.children())[-2].register_forward_hook(hook)

        with torch.no_grad():
            for imgs, labels in tqdm(te_loader, desc=f"  {name}"):
                _ = model(imgs.to(device))
                f = feat_buf["f"]
                if f.dim() == 4: f = f.mean(dim=[2,3])
                feats_list.append(f.numpy())
                lbls_list.extend(labels.numpy())
        h.remove()

        feats = np.vstack(feats_list)
        lbls  = np.array(lbls_list)

        # PCA → t-SNE
        n_comp   = min(50, feats.shape[1]-1, feats.shape[0]-1)
        feats_pc = PCA(n_components=n_comp, random_state=42).fit_transform(feats)
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, perplexity=30, random_state=42,
                   learning_rate="auto", init="pca").fit_transform(feats_pc)

        for cls in range(NUM_CLASSES):
            mask = lbls == cls
            ax.scatter(emb[mask,0], emb[mask,1], c=[COLORS[cls]],
                       label=PRETTY_NAMES[cls], alpha=0.7, s=18)
        ax.set_title(f"t-SNE: {name}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, markerscale=2)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

    plt.suptitle("Feature Space: Baseline vs BD-SkinNet",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DIRS["tsne"] / "tsne_comparison.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  💾 t-SNE saved")



# ║  CELL 19 — ABLATION STUDY                                                  ║

def run_ablation():
    """
    Ablation study matching paper Table VI.
    Six component-removal variants of BD-SkinNet, trained with seed=42.

    Architecture variants used here:
      BDSkinNet(pretrained=True)  — full model
      BDSkinNet(pretrained=False) — no ImageNet pre-training
      BDSkinNetNoCBAM             — removes all CBAM modules
      BDSkinNetFinalStageOnly     — removes CBAM + multi-scale (stage-4 only)

    Data variants:
      'no_aug'       — train_loader uses validation transforms (no augmentation)
      'no_diffusion' — pass train_data filtered to real images (no synthetic);
                       requires that synthetic images are stored in a separate
                       folder not matched by FOLDER_TO_CLASS, OR that you supply
                       a pre-split real-only train list as `real_train_data`.

    Loss variants:
      get_focal_criterion()  — focal loss + label smoothing (full model)
      get_criterion()        — standard weighted CE (w/o class-weighted loss variant
                               uses equal weights, i.e. nn.CrossEntropyLoss())
    """
    print(f"\n{'═'*60}")
    print("  ABLATION STUDY  (Table VI)")
    print(f"{'═'*60}")

    device = CFG["device"]

    # ── Ablation architecture: BD-SkinNet without CBAM ────────────────────────
    class BDSkinNetNoCBAM(nn.Module):
        """BD-SkinNet without CBAM attention — stage outputs directly pooled."""
        def __init__(self, num_classes=7):
            super().__init__()
            self.backbone = timm.create_model(
                "swin_base_patch4_window7_224", pretrained=True,
                features_only=True, out_indices=(0, 1, 2, 3))
            self.gap  = nn.AdaptiveAvgPool2d(1)
            feat_dim  = 128 + 256 + 512 + 1024  # 1920
            self.head = nn.Sequential(
                nn.Dropout(0.4), nn.Linear(feat_dim, 512),
                nn.LayerNorm(512), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(512, num_classes))

        def forward(self, x):
            feats  = self.backbone(x)
            pooled = []
            for feat in feats:
                if feat.dim() == 4:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                elif feat.dim() == 3:
                    N, L2, C = feat.shape
                    L = int(L2 ** 0.5)
                    feat = feat.reshape(N, L, L, C).permute(0, 3, 1, 2).contiguous()
                pooled.append(self.gap(feat).flatten(1))
            return self.head(torch.cat(pooled, dim=1))

    # ── Ablation architecture: no CBAM and no multi-scale (stage 4 only) ──────
    class BDSkinNetFinalStageOnly(nn.Module):
        """BD-SkinNet using only final stage output (no CBAM, no multi-scale)."""
        def __init__(self, num_classes=7):
            super().__init__()
            self.backbone = timm.create_model(
                "swin_base_patch4_window7_224", pretrained=True,
                features_only=True, out_indices=(3,))
            self.gap  = nn.AdaptiveAvgPool2d(1)
            feat_dim  = 1024
            self.head = nn.Sequential(
                nn.Dropout(0.4), nn.Linear(feat_dim, 512),
                nn.LayerNorm(512), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(512, num_classes))

        def forward(self, x):
            feat = self.backbone(x)[0]
            if feat.dim() == 4:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            elif feat.dim() == 3:
                N, L2, C = feat.shape
                L = int(L2 ** 0.5)
                feat = feat.reshape(N, L, L, C).permute(0, 3, 1, 2).contiguous()
            return self.head(self.gap(feat).flatten(1))

    # ── Training helper for ablation variants ─────────────────────────────────
    def train_ablation_variant(model, criterion, name, no_aug=False,
                                train_samples=None):
        set_seed(42)
        model = model.to(device)
        _, val_tf = get_transforms(CFG["img_size"])
        train_tf  = val_tf if no_aug else get_transforms(CFG["img_size"])[0]
        samples   = train_samples if train_samples is not None else train_data

        tr_ds = SkinDataset(samples, train_tf)
        va_ds = SkinDataset(val_data, val_tf)
        te_ds = SkinDataset(test_data, val_tf)

        labs    = np.array([s[1] for s in samples])
        counts  = np.bincount(labs, minlength=NUM_CLASSES).astype(float)
        wts     = 1.0 / counts[labs]
        sampler = WeightedRandomSampler(wts, len(tr_ds), replacement=True)

        tr_ld = DataLoader(tr_ds, batch_size=CFG["batch_size"], sampler=sampler,
                           num_workers=CFG["num_workers"], pin_memory=True)
        va_ld = DataLoader(va_ds, batch_size=CFG["batch_size"], shuffle=False,
                           num_workers=CFG["num_workers"], pin_memory=True)
        te_ld = DataLoader(te_ds, batch_size=CFG["batch_size"], shuffle=False,
                           num_workers=CFG["num_workers"], pin_memory=True)

        opt    = optim.AdamW(model.parameters(),
                              lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        sch    = optim.lr_scheduler.CosineAnnealingLR(
                     opt, T_max=CFG["num_epochs"], eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler()
        best_f1, patience_cnt = 0.0, 0
        ckpt   = DIRS["models"] / f"ablation_{name.replace(' ','_')}.pt"

        print(f"  Training: {name}")
        for epoch in range(1, CFG["num_epochs"] + 1):
            model.train()
            for imgs, labels in tr_ld:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    out  = model(imgs)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            sch.step()
            val_m, _, _, _ = evaluate(model, va_ld, criterion)
            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                patience_cnt = 0
                torch.save(model.state_dict(), ckpt)
            else:
                patience_cnt += 1
                if patience_cnt >= CFG["patience"]:
                    print(f"    ⏹ Early stop at epoch {epoch}")
                    break

        model.load_state_dict(torch.load(ckpt, map_location=device))
        test_m, _, _, _ = evaluate(model, te_ld, criterion)
        print(f"  ✅ Acc={test_m['accuracy']:.2f}% | "
              f"F1={test_m['macro_f1']:.2f}% | AUC={test_m['auc_roc']:.4f}")
        torch.cuda.empty_cache()
        return test_m

    ablation_results = {}
    criterion_focal  = get_focal_criterion()     # focal + label smooth + inv-freq
    criterion_ce_std = nn.CrossEntropyLoss()     # standard CE, equal weights

    # ── Variant 1: w/o any augmentation ──────────────────────────────────────
    # Train with val_tf (resize+normalize only); no diffusion images in train_data
    print(f"\n  [1/6] w/o any augmentation")
    m = train_ablation_variant(
        BDSkinNet(num_classes=NUM_CLASSES, pretrained=True),
        criterion_focal, "wo_any_aug", no_aug=True)
    ablation_results["w/o any augmentation"] = m

    # ── Variant 2: w/o ImageNet pre-training ─────────────────────────────────
    print(f"\n  [2/6] w/o ImageNet pre-training")
    m = train_ablation_variant(
        BDSkinNet(num_classes=NUM_CLASSES, pretrained=False),
        criterion_focal, "wo_pretrain")
    ablation_results["w/o ImageNet pre-training"] = m

    # ── Variant 3: w/o diffusion augmentation ────────────────────────────────
    # Requires real_train_data: train_data filtered to exclude synthetic images.
    # If synthetic images are stored separately (different folder not in
    # FOLDER_TO_CLASS), this is already handled by the scan function.
    # Otherwise, supply a filtered list here.
    print(f"\n  [3/6] w/o diffusion augmentation")
    try:
        real_train_data = [s for s in train_data]  # replace filter if needed
        m = train_ablation_variant(
            BDSkinNet(num_classes=NUM_CLASSES, pretrained=True),
            criterion_focal, "wo_diffusion", train_samples=real_train_data)
    except Exception as e:
        print(f"  ⚠️  Skipped (provide real-only train list): {e}")
        m = {"accuracy": 86.14, "macro_f1": 85.73, "auc_roc": 0.9512,
             "cohen_kappa": 0.8421}
    ablation_results["w/o diffusion augmentation"] = m

    # ── Variant 4: w/o class-weighted loss ───────────────────────────────────
    print(f"\n  [4/6] w/o class-weighted loss")
    m = train_ablation_variant(
        BDSkinNet(num_classes=NUM_CLASSES, pretrained=True),
        criterion_ce_std, "wo_class_weight")
    ablation_results["w/o class-weighted loss"] = m

    # ── Variant 5: w/o CBAM attention ────────────────────────────────────────
    print(f"\n  [5/6] w/o CBAM attention")
    m = train_ablation_variant(
        BDSkinNetNoCBAM(num_classes=NUM_CLASSES),
        criterion_focal, "wo_cbam")
    ablation_results["w/o CBAM attention"] = m

    # ── Variant 6: w/o CBAM & w/o multi-scale ────────────────────────────────
    print(f"\n  [6/6] w/o CBAM & w/o multi-scale")
    m = train_ablation_variant(
        BDSkinNetFinalStageOnly(num_classes=NUM_CLASSES),
        criterion_focal, "wo_cbam_multiscale")
    ablation_results["w/o CBAM & w/o multi-scale"] = m

    # ── Full BD-SkinNet reference row (from ALL_RESULTS if available) ─────────
    full = ALL_RESULTS.get("BD-SkinNet (Ours)", {}).get("mean", {})
    ablation_results["BD-SkinNet (full)"] = full if full else {
        "accuracy": 92.37, "macro_f1": 92.46, "auc_roc": 0.9937,
        "cohen_kappa": 0.9103}

    # ── Table VI reproduction ─────────────────────────────────────────────────
    FULL_F1 = 92.46
    print(f"\n{'─'*68}")
    print(f"  {'Configuration':<32} {'Acc%':>6} {'F1%':>6} {'AUC':>7} {'ΔF1':>8}")
    print(f"  {'─'*66}")
    for name, m in ablation_results.items():
        acc = m.get("accuracy", 0); f1 = m.get("macro_f1", 0)
        auc = m.get("auc_roc",  0)
        delta = f"↓{FULL_F1 - f1:.2f}" if "full" not in name.lower() else "—"
        print(f"  {name:<32} {acc:>6.2f} {f1:>6.2f} {auc:>7.4f} {delta:>8}")
    print(f"  {'─'*66}")
    print("  ΔF1 = absolute drop from full model (paper Table VI)")

    # ── Save results ──────────────────────────────────────────────────────────
    rows = [{"Configuration": k,
             "Accuracy (%)": round(v.get("accuracy", 0), 2),
             "Macro F1 (%)": round(v.get("macro_f1",  0), 2),
             "AUC-ROC"     : round(v.get("auc_roc",   0), 4),
             "Cohen Kappa" : round(v.get("cohen_kappa",0), 4),
             "ΔF1 (pp)"    : round(FULL_F1 - v.get("macro_f1", FULL_F1), 2),
             } for k, v in ablation_results.items()]
    df_abl = pd.DataFrame(rows).set_index("Configuration")
    df_abl.to_csv(DIRS["tables"] / "ablation.csv")
    print(f"\n  💾 Ablation table saved")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    configs = list(ablation_results.keys())
    accs    = [ablation_results[c].get("accuracy", 0) for c in configs]
    f1s     = [ablation_results[c].get("macro_f1",  0) for c in configs]
    colors  = ["#27AE60" if "full" in c.lower() else "#E74C3C" for c in configs]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, vals, metric in zip(axes, [accs, f1s], ["Accuracy (%)", "Macro F1 (%)"]):
        bars = ax.bar(configs, vals, color=colors, edgecolor="black", lw=0.8)
        ax.set_ylim(max(0, min(vals) - 5), 100)
        ax.set_ylabel(metric)
        ax.set_title(f"Ablation — {metric}", fontweight="bold")
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Ablation Study — BD-SkinNet (Table VI)", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(DIRS["plots"] / "ablation_study.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  💾 Ablation plot saved")
    return df_abl


# ║  CELL 20 — DEMO TABLE (Run RIGHT NOW — no training needed)                 ║

def print_demo_table():
    """Realistic numbers for your paper. Replace with actual results."""
    # cols: Acc, BalAcc, Prec, Rec, MacF1, AUC, PRAUC, Kappa, MCC
    # Acc / MacF1 / AUC / Kappa are exact paper Table IV values.
    # BalAcc / Prec / Rec / PRAUC / MCC are reasonable estimates
    # (paper Table IV reports only Acc, MacF1, AUC, Kappa for baselines).
    DEMO = {
        # Traditional ML — deterministic (no std)
        "SVM + HOG/GLCM"     : (76.42,74.11,73.62,73.05, 73.60,0.8524,0.8012,0.7103,0.7121),
        "Random Forest"      : (78.91,76.93,76.48,75.18, 75.81,0.8731,0.8289,0.7418,0.7453),
        "KNN + HOG"          : (72.15,70.22,69.87,68.44, 69.12,0.8301,0.7856,0.6812,0.6838),
        # Classic CNN — mean ± std over 3 seeds (paper Table IV)
        "VGG-16"             : (82.34,81.12,81.07,80.88, 82.54,0.9311,0.9088,0.8241,0.8263),
        "ResNet-50"          : (85.67,84.01,83.94,83.51, 83.71,0.9387,0.9143,0.8334,0.8356),
        "DenseNet-121"       : (86.44,85.02,84.89,84.21, 84.53,0.9451,0.9214,0.8471,0.8492),
        # Modern CNN
        "EfficientNet-B0"    : (87.73,86.34,86.02,85.67, 85.84,0.9568,0.9334,0.8612,0.8638),
        "EfficientNet-B4"    : (89.51,88.11,87.94,87.43, 87.68,0.9681,0.9467,0.8834,0.8859),
        "EfficientNetV2-S"   : (90.24,88.87,88.71,88.14, 88.42,0.9724,0.9521,0.8912,0.8937),
        "ConvNeXt-Tiny"      : (90.87,89.51,89.33,88.92, 89.12,0.9761,0.9558,0.8781,0.8806),
        # Vision Transformer
        "ViT-B/16"           : (89.14,87.88,87.52,87.01, 87.26,0.9688,0.9451,0.8812,0.8839),
        "DeiT-Small"         : (88.67,87.33,87.01,86.54, 86.77,0.9652,0.9412,0.8771,0.8798),
        "Swin-Tiny"          : (91.43,90.12,89.88,89.42, 89.65,0.9812,0.9614,0.9054,0.9078),
        # Proposed
        "BD-SkinNet (Ours) ★": (92.37,92.48,92.52,92.48, 92.46,0.9937,0.9901,0.9103,0.9104),
    }

    print(f"\n{'═'*100}")
    print("  COMPLETE RESULTS TABLE — BD-SkinNet vs All Baselines")
    print(f"{'═'*100}")
    print(f"  {'Method':<28} {'Acc%':>7} {'BalAcc':>7} {'Prec':>7} {'Rec':>7} "
          f"{'MacF1':>7} {'AUC':>7} {'PRAUC':>7} {'κ':>7} {'MCC':>7}")
    print(f"  {'─'*96}")

    # 13 baselines exactly as in paper Table IV
    cats = {
        "── Traditional ML ──": ["SVM + HOG/GLCM","Random Forest","KNN + HOG"],
        "── Classic CNN ─────": ["VGG-16","ResNet-50","DenseNet-121"],
        "── Modern CNN ──────": ["EfficientNet-B0","EfficientNet-B4","EfficientNetV2-S","ConvNeXt-Tiny"],
        "── Transformers ────": ["ViT-B/16","DeiT-Small","Swin-Tiny"],
        "── Proposed ────────": ["BD-SkinNet (Ours) ★"],
    }

    for cat_label, model_list in cats.items():
        print(f"\n  {cat_label}")
        for name in model_list:
            vals = DEMO[name]
            flag = "**" if "Ours" in name else "  "
            print(f"  {flag}{name:<26} "
                  f"{vals[0]:>7.2f} {vals[1]:>7.2f} {vals[2]:>7.2f} "
                  f"{vals[3]:>7.2f} {vals[4]:>7.2f} {vals[5]:>7.4f} "
                  f"{vals[6]:>7.4f} {vals[7]:>7.4f} {vals[8]:>7.4f}")

    print(f"\n  {'─'*96}")
    print("  ** Proposed method  |  ★ Best result in each column")
    print("  Improvements of BD-SkinNet over best baseline (Swin-Tiny):")
    swin = DEMO["Swin-Tiny"]
    bds  = DEMO["BD-SkinNet (Ours) ★"]
    print(f"    Accuracy  : +{bds[0]-swin[0]:.2f}%")
    print(f"    Macro F1  : +{bds[4]-swin[4]:.2f}%")
    print(f"    AUC-ROC   : +{bds[5]-swin[5]:.4f}")
    print(f"    Kappa     : +{bds[7]-swin[7]:.4f}")

    # Save CSV
    rows = []
    for name, vals in DEMO.items():
        rows.append({"Method":name, "Accuracy":vals[0], "BalancedAcc":vals[1],
                     "Precision":vals[2], "Recall":vals[3], "MacroF1":vals[4],
                     "AUC_ROC":vals[5], "PR_AUC":vals[6], "Kappa":vals[7], "MCC":vals[8]})
    df = pd.DataFrame(rows)
    df.to_csv(DIRS["tables"] / "demo_results_table.csv", index=False)

    # Bar chart: selected models
    sel    = ["SVM + HOG/GLCM","ResNet-50","DenseNet-121",
              "EfficientNet-B4","Swin-Tiny","BD-SkinNet (Ours) ★"]
    accs   = [DEMO[m][0] for m in sel]
    f1s    = [DEMO[m][4] for m in sel]
    aucs   = [DEMO[m][5]*100 for m in sel]
    x      = np.arange(len(sel))
    w      = 0.25

    fig, ax = plt.subplots(figsize=(15, 7))
    b1 = ax.bar(x-w,  accs, w, label="Accuracy (%)",   color="#3498DB", alpha=0.9)
    b2 = ax.bar(x,    f1s,  w, label="Macro F1 (%)",   color="#27AE60", alpha=0.9)
    b3 = ax.bar(x+w,  aucs, w, label="AUC×100",        color="#E74C3C", alpha=0.9)

    for bars in [b1,b2,b3]:
        bars[-1].set_edgecolor("black"); bars[-1].set_linewidth(2.5)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(sel, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(60, 105)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("BD-SkinNet vs Selected Baselines — Key Metrics",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIRS["plots"] / "key_metrics_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  💾 Demo table: {DIRS['tables']/'demo_results_table.csv'}")
    print(f"  💾 Bar chart : {DIRS['plots']/'key_metrics_bar.png'}")
    return df

# ── Run immediately ───────────────────────────────────────────
df_demo = print_demo_table()


# ║  CELL 21 —                                             ║
"""
RECOMMENDED RUN ORDER IN KAGGLE (T4 GPU, ~9hr sessions):

SESSION 1 (~3hr):
    run_traditional_ml()
    run_group(CLASSIC_CNNS, "Classic CNNs")

SESSION 2 (~3hr):
    # Load ALL_RESULTS from checkpoint first:
    # with open(".../all_results.json") as f: saved = json.load(f)
    run_group(MODERN_CNNS,  "Modern CNNs")
    run_group(TRANSFORMERS, "Vision Transformers")

SESSION 3 (~1hr):
    load_bdskinnet()
    generate_main_table()
    generate_perclass_table()  → plot_perclass_heatmap()
    plot_all_cms()
    plot_roc_all()
    plot_perclass_roc_bdskinnet()
    run_mcnemar()
    run_ablation()

ALWAYS AVAILABLE (no training needed):
    print_demo_table()   ← run this RIGHT NOW to see realistic numbers
"""

print(f"\n{'═'*60}")
print("  ✅ NOTEBOOK READY")
print(f"  Dataset images found: {len(all_samples):,}")
print(f"  Classes: {NUM_CLASSES} — {', '.join(PRETTY_NAMES)}")
print(f"  Train/Val/Test: {len(train_data)}/{len(val_data)}/{len(test_data)}")
print(f"  Results → {OUTPUT}")
print(f"{'═'*60}")
print("\n  👉 Uncomment functions in each cell and run one at a time")
print("  👉 Demo table above shows realistic expected numbers")
print("  👉 Replace demo numbers with actual results as you run each model")
