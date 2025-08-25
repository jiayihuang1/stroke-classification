# 🧠 ConvNeXt Stroke Classification  

Fine-tuning a ConvNeXt model for medical image classification (stroke detection with 3 categories: **absent, ischemic, hemorrhagic**).  

---

## 📖 Overview  
This project uses **ConvNeXt-Base** from the [timm](https://github.com/huggingface/pytorch-image-models) library, pretrained on ImageNet, and fine-tunes it for a 3-class stroke dataset.  

The dataset contains three categories:  
- 0 – Absent stroke  
- 1 – Ischemic stroke  
- 2 – Hemorrhagic stroke  

The model was trained with a stratified train/validation/test split and evaluated on accuracy, precision, recall, and F1 score.  

---

## 🏗️ Project Structure  

- `data/` — dataset (images & labels)  
  - `dataset.py` — custom `StrokeDataset` class  
- `ConvNeXt_v1.ipynb` — main notebook (training, evaluation, results)  
- `best_model.pth` — saved model (best validation F1 score)  
- `test_predictions.csv` — predictions on the test set  
- `README.md` — project documentation  


## 🔬 Methodology  

1. **Data Preparation**  
   - Train/val/test split (70/20/10).  
   - Data augmentation with Albumentations (resizing, normalization, flips, etc.).  

2. **Model Setup**  
   - Pretrained `convnext_base`.  
   - Final classifier adjusted to 3 output classes.

3. **Training**  
   - Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-5`).  
   - Scheduler: ReduceLROnPlateau (patience=5).  
   - Loss: CrossEntropyLoss.  
   - Mixed-precision training with `torch.amp`.  
   - Best model saved according to validation F1 score.  

4. **Evaluation Metrics**  
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score (macro)  

---

## 📊 Results  

| Metric        | Value |
|---------------|-------|
| Test Accuracy | 99.5% |
| Test Precision| 99.6% |
| Test Recall   | 99.4% |
| Test F1 Score | 99.5% |

Predictions are stored in `test_predictions.csv` with image paths, true labels, and predicted labels.  

---
