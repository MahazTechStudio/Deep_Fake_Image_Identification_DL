# Deep Fake Image Identification using Deep Learning

This project implements a robust deep learning pipeline for identifying AI-generated (Deep Fake) images using state-of-the-art Vision Transformers (ViT) and Convolutional Neural Networks (CNN).

## 🚀 Project Overview
The goal is to distinguish between real photographs and AI-generated images using the **Defactify Image Dataset**. We address common challenges such as large-scale data handling, class imbalance, and model generalization.

### Key Features
- **Cloud-to-Local Data Streaming**: Handles 7.6GB of data without full local download using Hugging Face `streaming=True`.
- **Balanced Data Extraction**: Custom logic to extract a perfectly balanced dataset (2,000 Real, 2,000 AI) to prevent majority-class bias.
- **Multi-Architecture Comparison**: Evaluates Vision Transformer (`vit_tiny`) against CNN (`ResNet18`).
- **Targeted Data Augmentation**: Implements ColorJitter and GaussianBlur to improve model specificity and reduce False Positives.
- **Comprehensive Evaluation**: Reports Accuracy, F1-Score, Precision, Recall, Specificity, and ROC/AUC.

## 📊 Results (Balanced 4,000-Sample Run)
| Experiment | Accuracy | F1-Score | Precision | Specificity |
| :--- | :--- | :--- | :--- | :--- |
| **ViT Binary (Task A)** | **87.20%** | **0.871** | **0.879** | **0.802** |
| **ViT Multiclass (Task B)**| **77.41%** | **0.774** | **0.784** | **N/A** |
| **ResNet18 Binary** | **81.70%** | **0.817** | **0.818** | **0.788** |

## 📂 Dataset Resource
- **Dataset Name**: [Defactify Image Veracity Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)
- **Size**: ~7.6 GB
- **Labels**: 
  - `Label_A`: Binary (0: Real, 1: AI-Generated)
  - `Label_B`: Multiclass (6 different AI generators)

## 🛠️ Setup and Installation

### Clone the Repository
```bash
git clone https://github.com/MahazTechStudio/Deep_Fake_Image_Identification_DL.git
cd Deep_Fake_Image_Identification_DL
```

### Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

### Install Dependencies
```bash
pip install torch torchvision timm datasets transformers pandas matplotlib seaborn scikit-learn tqdm openpyxl pillow
```

## 🚀 How to Run

### 1. Initial Data Check
Verify the dataset connection and label distribution:
```bash
python check_ratio.py
```

### 2. Main Training Pipeline
Run the balanced training and evaluation (this will generate Excel reports and figures in the `reports/` folder):
```bash
python train_and_evaluate.py
```

### 3. Generalization Proof
Generate raw text output to prove the model learned the data patterns:
```bash
python prove_generalization.py
```

## 📂 Project Structure
- `train_and_evaluate.py`: Core pipeline with balanced loading and weighted loss.
- `prove_generalization.py`: Mathematical proof of model generalization.
- `check_ratio.py`: Diagnostic script for class distribution.
- `reports/`: Sequential folders containing learning curves, confusion matrices, and ROC curves.
- `Summary_Results.xlsx`: Final performance comparison across all models.

## 🎓 Academic Summary
This project demonstrates that **Vision Transformers** outperform traditional CNNs in capturing the structural artifacts of AI-generated pixels. By implementing a **Balanced Data Extraction** strategy and **Weighted CrossEntropyLoss**, we successfully eliminated the bias toward the majority class, achieving a high **Specificity of 80.2%**.
