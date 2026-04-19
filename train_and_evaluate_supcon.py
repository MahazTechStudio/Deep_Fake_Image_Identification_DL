import os
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import timm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5
IMG_SIZE = 224
TOTAL_SAMPLES = 4000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
LAMBDA_SUPCON = 0.2
TEMPERATURE = 0.07

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ImageDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["Image"].convert("RGB")
        label_a = int(item["Label_A"])
        label_b = int(item["Label_B"])

        if self.transform:
            image = self.transform(image)

        # Contrastive label:
        # real -> 0
        # fake generator classes -> label_b + 1
        contrastive_label = 0 if label_a == 0 else (label_b + 1)

        return image, label_a, contrastive_label

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_test_transforms

def collect_balanced_pool(dataset, total_samples):
    class_data = {0: [], 1: []}
    target_per_class = total_samples // 2

    print(f"Collecting balanced pool: {total_samples} samples ({target_per_class} per class)")
    pbar = tqdm(total=total_samples)

    for item in dataset["train"]:
        label = int(item["Label_A"])
        if len(class_data[label]) < target_per_class:
            class_data[label].append(item)
            pbar.update(1)

        if len(class_data[0]) >= target_per_class and len(class_data[1]) >= target_per_class:
            break

    pbar.close()

    combined = class_data[0] + class_data[1]
    random.shuffle(combined)
    return combined

def split_data(data):
    total = len(data)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data

class ViTWithProjection(nn.Module):
    def __init__(self, num_classes=2, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=0
        )
        feat_dim = int(self.backbone.num_features)

        self.classifier = nn.Linear(feat_dim, num_classes)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        proj = self.projection(features)
        proj = F.normalize(proj, dim=1)
        return logits, proj

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum
        loss = -mean_log_prob_pos.mean()
        return loss

def train_one_epoch(model, loader, optimizer, ce_criterion, supcon_criterion):
    model.train()
    total_loss = 0
    total_ce = 0
    total_supcon = 0
    all_preds = []
    all_labels = []

    for images, label_a, contrastive_label in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        label_a = label_a.to(DEVICE)
        contrastive_label = contrastive_label.to(DEVICE)

        optimizer.zero_grad()

        logits, proj = model(images)

        ce_loss = ce_criterion(logits, label_a)
        supcon_loss = supcon_criterion(proj, contrastive_label)
        loss = ce_loss + LAMBDA_SUPCON * supcon_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_supcon += supcon_loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label_a.cpu().numpy())

    return (
        total_loss / len(loader),
        total_ce / len(loader),
        total_supcon / len(loader),
        accuracy_score(all_labels, all_preds)
    )

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, label_a, _ in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(DEVICE)
            label_a = label_a.to(DEVICE)

            logits, _ = model(images)
            loss = criterion(logits, label_a)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_a.cpu().numpy())

    return (
        total_loss / len(loader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="weighted"),
        precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        all_labels,
        all_preds
    )

def main():
    set_seed()
    os.makedirs("reports_supcon", exist_ok=True)
    os.makedirs("reports_supcon/checkpoints", exist_ok=True)

    dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", streaming=True)
    full_data = collect_balanced_pool(dataset, TOTAL_SAMPLES)
    train_data, val_data, test_data = split_data(full_data)

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    train_tf, val_tf = get_transforms()

    train_ds = ImageDataset(train_data, train_tf)
    val_ds = ImageDataset(val_data, val_tf)
    test_ds = ImageDataset(test_data, val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ViTWithProjection(num_classes=2, proj_dim=128).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    ce_criterion = nn.CrossEntropyLoss()
    supcon_criterion = SupConLoss(temperature=TEMPERATURE)

    best_val_acc = -1
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_ce, train_supcon, train_acc = train_one_epoch(
            model, train_loader, optimizer, ce_criterion, supcon_criterion
        )

        val_loss, val_acc, val_f1, val_prec, val_rec, _, _ = evaluate(
            model, val_loader, ce_criterion
        )

        print(f"Train Total Loss: {train_loss:.4f} | CE: {train_ce:.4f} | SupCon: {train_supcon:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_total_loss": train_loss,
            "train_ce_loss": train_ce,
            "train_supcon_loss": train_supcon,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, "reports_supcon/checkpoints/best_model.pth")
            print("Best model updated.")

    model.load_state_dict(best_state)

    print("\nFinal Test Evaluation...")
    test_loss, test_acc, test_f1, test_prec, test_rec, y_true, y_pred = evaluate(
        model, test_loader, ce_criterion
    )

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Test Accuracy:    {test_acc:.4f}")
    print(f"Test F1 Score:    {test_f1:.4f}")
    print(f"Test Precision:   {test_prec:.4f}")
    print(f"Test Recall:      {test_rec:.4f}")
    print(f"Test Specificity: {specificity:.4f}")

    pd.DataFrame(history).to_csv("reports_supcon/epoch_metrics.csv", index=False)

    with open("reports_supcon/final_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Best Val Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write(f"Test Precision: {test_prec:.4f}\n")
        f.write(f"Test Recall: {test_rec:.4f}\n")
        f.write(f"Test Specificity: {specificity:.4f}\n")

if __name__ == "__main__":
    main()