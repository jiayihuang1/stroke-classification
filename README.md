# Stroke Classification from CT Scans

A deep learning model for classifying brain CT scans into three categories — **absent**, **ischemic**, and **hemorrhagic** stroke — using a fine-tuned ConvNeXt-Base architecture.

## Motivation

This was a personal project driven by my interest in applying computer vision to medical imaging. I wanted to explore how well modern image classification models could perform on a clinically relevant problem — distinguishing stroke types from CT scans — and learn about the challenges of working with medical data.

## Results

Evaluated on a held-out test set (671 images):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 97.3%  |
| Precision | 98.0%  |
| Recall    | 95.3%  |
| F1 Score  | 96.6%  |
| F1 95% CI | 94.9% – 98.1% |

## Approach

### Dataset

- **Source**: [Kaggle stroke CT scan dataset](https://www.kaggle.com/) (6,774 images)
- **Classes**: Absent (4,551), Ischemic (1,130), Hemorrhagic (1,093)
- **Split**: 70% train / 20% validation / 10% test (stratified)

### Model

- **Architecture**: ConvNeXt-Base (pretrained on ImageNet, via [timm](https://github.com/huggingface/pytorch-image-models))
- **Classifier head** replaced for 3-class output

### Training

- AdamW optimizer (lr=1e-4, weight_decay=1e-5)
- ReduceLROnPlateau scheduler (patience=5, factor=0.5)
- Mixed-precision training with `torch.amp`
- Best model checkpoint saved by validation F1 score
- 50 epochs

### Data Augmentation

Applied via [Albumentations](https://albumentations.ai/):
- Horizontal/vertical flips, rotation
- Elastic, grid, and optical distortion
- Brightness/contrast adjustment
- Gaussian noise, CoarseDropout

## Project Structure

```
├── data/
│   └── dataset.py          # Custom PyTorch Dataset class
├── notebooks/
│   ├── ConvNeXt_v1.ipynb    # Training and evaluation notebook
│   └── test_predictions.csv # Test set predictions
├── requirements.txt
└── README.md
```

## Tech Stack

Python, PyTorch, timm, Albumentations, OpenCV, scikit-learn, pandas, matplotlib, seaborn

## Getting Started

```bash
pip install -r requirements.txt
```

Open `notebooks/ConvNeXt_v1.ipynb` to run training and evaluation. The dataset is downloaded automatically from Kaggle via `kagglehub`.
