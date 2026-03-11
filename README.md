# Stroke Classification from CT Scans

A deep learning model for classifying brain CT scans into three categories — **absent**, **ischemic**, and **hemorrhagic** stroke — using a fine-tuned ConvNeXt-Base architecture. Achieves 97.3% accuracy with bootstrap confidence intervals on a held-out test set.

A personal project driven by my interest in applying computer vision to medical imaging — exploring how well modern classification models perform on a clinically relevant problem.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Deep Learning** | PyTorch, timm (ConvNeXt-Base pretrained on ImageNet) |
| **Image Augmentation** | Albumentations, OpenCV |
| **Evaluation** | scikit-learn (accuracy, precision, recall, F1, bootstrap CI) |
| **Data** | pandas, NumPy, kagglehub |
| **Visualisation** | Matplotlib, seaborn |

---

## Architecture & Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│              STROKE CLASSIFICATION PIPELINE                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Data Loading (kagglehub)                                 │   │
│  │  6,774 CT scans → 3 classes (imbalanced)                  │   │
│  │  Absent: 4,551 | Ischemic: 1,130 | Hemorrhagic: 1,093    │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           v                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Preprocessing                                            │   │
│  │  Grayscale CT → RGB (3ch) → Resize 224×224 → Normalise    │   │
│  │  Stratified split: 70% train / 20% val / 10% test         │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           v                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Augmentation (training only, via Albumentations)         │   │
│  │  Flips, rotation, elastic/grid/optical distortion,        │   │
│  │  brightness/contrast, Gaussian noise, CoarseDropout       │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           v                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ConvNeXt-Base (timm, pretrained on ImageNet)             │   │
│  │  Classifier head replaced → 3 output classes              │   │
│  │  Mixed-precision training (torch.amp)                     │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           v                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Training: AdamW (lr=1e-4) + ReduceLROnPlateau            │   │
│  │  50 epochs, best checkpoint saved by validation F1        │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           v                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Evaluation: Accuracy, Precision, Recall, F1              │   │
│  │  Bootstrap 95% CI (1,000 samples), confusion matrix       │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Results

Evaluated on a held-out test set (671 images):

| Metric | Score |
|---|---|
| **Accuracy** | 97.3% |
| **Precision** | 98.0% |
| **Recall** | 95.3% |
| **F1 Score** | 96.6% |
| **F1 95% CI** | 94.9% – 98.1% |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| ConvNeXt-Base over ResNet | Modernised ConvNet architecture with stronger ImageNet performance; competitive with ViTs at lower complexity |
| Transfer learning | Medical datasets are typically small; ImageNet pretraining provides robust low-level feature extraction |
| Stratified splits | Class imbalance (67% absent vs 16% each stroke type) requires proportional representation in all splits |
| Heavy augmentation | CT scan variations (rotation, distortion, noise) simulate real-world acquisition differences |
| Mixed-precision training | Reduces memory usage and speeds up training on GPU/Apple Silicon |
| Bootstrap CI for F1 | Provides statistical confidence beyond point estimates on a small test set |

---

## Project Structure

```
├── data/
│   └── dataset.py              # Custom PyTorch Dataset (grayscale → RGB, augmentation)
├── notebooks/
│   ├── ConvNeXt_v1.ipynb        # Training and evaluation notebook
│   └── test_predictions.csv     # Test set predictions (path, true label, predicted label)
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/jiayihuang1/stroke-classification.git
cd stroke-classification

pip install -r requirements.txt
```

Open `notebooks/ConvNeXt_v1.ipynb` to run training and evaluation. The dataset is downloaded automatically from Kaggle via `kagglehub`.

---

## Dataset

- **Source**: [Kaggle — Stroke CT Scan Dataset](https://www.kaggle.com/) (Turkish CT scan collection)
- **Total**: 6,774 grayscale CT images across 3 classes
- **Format**: Images resized to 224×224, converted from grayscale to 3-channel RGB
