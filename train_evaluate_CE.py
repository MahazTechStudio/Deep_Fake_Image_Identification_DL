import os
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
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

REPORT_DIR = "reports_ce_clean"
CHECKPOINT_DIR = os.path.join(REPORT_DIR, "checkpoints")


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
        label = int(item["Label_A"])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


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


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
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

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, f1, precision, recall, all_labels, all_preds


def main():
    set_seed()

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", streaming=True)
    full_data = collect_balanced_pool(dataset, TOTAL_SAMPLES)
    train_data, val_data, test_data = split_data(full_data)

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    train_tf, eval_tf = get_transforms()

    train_ds = ImageDataset(train_data, transform=train_tf)
    val_ds = ImageDataset(val_data, transform=eval_tf)
    test_ds = ImageDataset(test_data, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=2
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1, val_prec, val_rec, _, _ = evaluate(
            model, val_loader, criterion
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )
        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("Best model updated.")

    model.load_state_dict(best_state)

    print("\nFinal Test Evaluation...")
    test_loss, test_acc, test_f1, test_prec, test_rec, y_true, y_pred = evaluate(
        model, test_loader, criterion
    )

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy:    {test_acc:.4f}")
    print(f"Test F1 Score:    {test_f1:.4f}")
    print(f"Test Precision:   {test_prec:.4f}")
    print(f"Test Recall:      {test_rec:.4f}")
    print(f"Test Specificity: {specificity:.4f}")

    pd.DataFrame(history).to_csv(os.path.join(REPORT_DIR, "epoch_metrics.csv"), index=False)

    with open(os.path.join(REPORT_DIR, "final_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best Val Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write(f"Test Precision: {test_prec:.4f}\n")
        f.write(f"Test Recall: {test_rec:.4f}\n")
        f.write(f"Test Specificity: {specificity:.4f}\n")


if __name__ == "__main__":
    main()