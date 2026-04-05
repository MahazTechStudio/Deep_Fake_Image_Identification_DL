import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
import timm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, classification_report
from tqdm import tqdm
from PIL import Image

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS = 1
LR = 2e-5
IMG_SIZE = 224
NUM_SAMPLES_TRAIN = 100 # Using small subset for demonstration
NUM_SAMPLES_VAL = 20
NUM_SAMPLES_TEST = 20

# Set style
sns.set_theme(style="whitegrid")

class ImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, label_col='Label_A'):
        self.dataset = hf_dataset
        self.transform = transform
        self.label_col = label_col
        self.data = list(hf_dataset) # Convert to list for indexing (since it's a small sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['Image'].convert('RGB')
        label = item[self.label_col]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_test_transforms

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle ViT output which is an object
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
            
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, acc, f1, precision, recall, all_labels, all_preds, all_probs

def plot_curves(history, name_prefix, save_dir):
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{name_prefix} Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{name_prefix} Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, '1_learning_curves.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, name_prefix, classes, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{name_prefix} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(save_dir, '2_confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_probs, name_prefix, save_dir):
    # For binary classification (Label_A)
    if y_probs.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(f'{name_prefix} ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, '3_roc_curve.png'))
        plt.close()

def collect_balanced_data(dataset, target_total, label_col, num_labels):
    # Dictionary to store data per class
    class_data = {i: [] for i in range(num_labels)}
    target_per_class = target_total // num_labels
    
    print(f"Collecting {target_total} balanced samples for {label_col} ({target_per_class} per class)...")
    pbar = tqdm(total=target_total)
    
    # Create a fresh iterator for each call to ensure we start from the beginning if needed
    stream_iter = iter(dataset['train'])
    
    while sum(len(v) for v in class_data.values()) < target_total:
        try:
            item = next(stream_iter)
            label = item[label_col]
            
            # Check if label is valid and if we still need samples for this class
            if label in class_data and len(class_data[label]) < target_per_class:
                class_data[label].append(item)
                pbar.update(1)
        except StopIteration:
            print(f"Warning: Reached end of stream. Collected {sum(len(v) for v in class_data.values())} samples.")
            break
            
    pbar.close()
    
    # Flatten dictionary to list and shuffle
    combined = []
    for samples in class_data.values():
        combined.extend(samples)
        
    np.random.shuffle(combined)
    return combined

def run_experiment(model_type, label_col, num_labels, classes):
    print(f"\n--- Starting Experiment: {model_type} for {label_col} ---")
    
    # Create experiment-specific report directory
    experiment_name = f"{model_type}_{label_col}"
    report_dir = os.path.join('reports', experiment_name)
    os.makedirs(report_dir, exist_ok=True)
    
    # Load and subset data
    print("Loading dataset...")
    dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", streaming=True)
    
    # Unified Balanced Data Extraction for both Binary and Multiclass
    train_data = collect_balanced_data(dataset, 4000, label_col, num_labels)
    val_data = collect_balanced_data(dataset, 1000, label_col, num_labels)
    test_data = collect_balanced_data(dataset, 1000, label_col, num_labels)
    
    # Safety check: if validation or test sets are empty, skip experiment
    if len(val_data) == 0 or len(test_data) == 0:
        print(f"Skipping {experiment_name} due to insufficient data.")
        return None
    
    train_trans, val_test_trans = get_transforms()
    
    train_ds = ImageDataset(train_data, transform=train_trans, label_col=label_col)
    val_ds = ImageDataset(val_data, transform=val_test_trans, label_col=label_col)
    test_ds = ImageDataset(test_data, transform=val_test_trans, label_col=label_col)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Initialize Model
    if model_type == 'ViT':
        # Using a much smaller ViT variant to avoid OOM
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_labels)
    else: # ResNet18 as a stable fallback for EfficientNet
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
    
    model.to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # Task 2: Implement Class Weights
    if num_labels == 2:
        class_weights = torch.tensor([1.0, 1.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using Weighted CrossEntropyLoss with weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1, val_prec, val_rec, _, _, _ = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
    # Final Evaluation
    print("Final Evaluation on Test Set...")
    test_loss, test_acc, test_f1, test_prec, test_rec, y_true, y_pred, y_probs = evaluate(model, test_loader, criterion)
    
    # Calculate Specificity for Binary Task
    if num_labels == 2:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"Test Specificity: {specificity:.4f}")
    else:
        specificity = None

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    
    report_text = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    print("\nClassification Report:")
    print(report_text)
    
    # Save classification report to text file
    with open(os.path.join(report_dir, '4_classification_report.txt'), 'w') as f:
        f.write(report_text)
    
    name_prefix = f"{model_type}_{label_col}"
    plot_curves(history, name_prefix, report_dir)
    plot_confusion_matrix(y_true, y_pred, name_prefix, classes, report_dir)
    plot_roc_curve(y_true, np.array(y_probs), name_prefix, report_dir)
    
    return {'Accuracy': test_acc, 'F1': test_f1, 'Precision': test_prec, 'Recall': test_rec, 'Specificity': specificity}

if __name__ == "__main__":
    os.makedirs('reports', exist_ok=True)
    
    results = {}
    
    # 1. ViT Binary
    results['ViT_Binary'] = run_experiment('ViT', 'Label_A', 2, ['Real', 'AI-Gen'])
    
    # 2. ViT Multiclass
    results['ViT_Multiclass'] = run_experiment('ViT', 'Label_B', 6, [f'Class_{i}' for i in range(6)])
    
    # 3. ResNet18 Binary
    results['ResNet18_Binary'] = run_experiment('ResNet18', 'Label_A', 2, ['Real', 'AI-Gen'])
    
    # Comparison Plot and Excel Report
    df_results = pd.DataFrame(results).T
    
    # Save to Excel
    try:
        df_results.to_excel('reports/Summary_Results.xlsx', index_label='Model_Config')
        print("\n[SUCCESS] Results saved to reports/Summary_Results.xlsx")
    except Exception as e:
        print(f"\n[ERROR] Could not save Excel: {e}. Saving to CSV instead.")
        df_results.to_csv('reports/Summary_Results.csv', index_label='Model_Config')
    
    plt.figure(figsize=(12, 7))
    df_results.plot(kind='bar', figsize=(12, 7))
    plt.title('Final Model Comparison: Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('reports/5_final_comparison_plot.png')
    plt.close()
    
    print("\n--- All Experiments Complete ---")
    print(df_results)
