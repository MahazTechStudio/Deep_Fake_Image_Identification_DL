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

TEMPERATURE = 0.07
PROJ_DIM = 128

# Paper ratio is alpha:beta:gamma = 100:90:10
# Kept in normalized form to preserve the same ratio without making gradients too large
ALPHA_CL = 1.0
BETA_SCL = 0.9
GAMMA_CE = 0.1

REPORT_DIR = "reports_bcl_paper"
CHECKPOINT_DIR = os.path.join(REPORT_DIR, "checkpoints")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrainDatasetBCL(Dataset):
    def __init__(self, data_list, transform_main=None, transform_aux=None):
        self.data = data_list
        self.transform_main = transform_main
        self.transform_aux = transform_aux

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["Image"].convert("RGB")
        label_a = int(item["Label_A"])

        img_main = self.transform_main(image) if self.transform_main else image
        img_aux = self.transform_aux(image) if self.transform_aux else image

        return img_main, img_aux, label_a


class EvalDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["Image"].convert("RGB")
        label_a = int(item["Label_A"])

        if self.transform:
            image = self.transform(image)

        return image, label_a


def get_transforms():
    train_transform_main = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.5, 2.0), value=0)
    ])

    train_transform_aux = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(1 / 1.3, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.5, 2.0), value=0)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform_main, train_transform_aux, eval_transform


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


class ViTBCL(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()

        self.encoder = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=0
        )

        feat_dim = int(self.encoder.num_features)

        self.proj_main = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim)
        )

        self.proj_aux = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim)
        )

        self.pred_head = nn.Linear(proj_dim * 2, 2)

    def forward_train(self, x_main, x_aux):
        f_main = self.encoder(x_main)
        f_aux = self.encoder(x_aux)

        v_main = self.proj_main(f_main)
        v_aux = self.proj_aux(f_aux)

        v_main_norm = F.normalize(v_main, dim=1)
        v_aux_norm = F.normalize(v_aux, dim=1)

        w = torch.cat([v_main, v_aux], dim=1)
        w_norm = F.normalize(w, dim=1)

        logits = self.pred_head(w)

        return logits, v_main_norm, v_aux_norm, w_norm

    def forward_eval(self, x):
        f = self.encoder(x)
        v_main = self.proj_main(f)
        v_aux = self.proj_aux(f)
        w = torch.cat([v_main, v_aux], dim=1)
        logits = self.pred_head(w)
        return logits


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        diag_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim.masked_fill_(diag_mask, -1e9)

        targets = torch.arange(batch_size, device=z.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)

        return F.cross_entropy(sim, targets)


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        similarity = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=features.device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum
        loss = -mean_log_prob_pos.mean()
        return loss


def train_one_epoch(model, loader, optimizer, ce_criterion, cl_criterion, scl_criterion):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_cl = 0.0
    total_scl = 0.0

    all_preds = []
    all_labels = []

    for img_main, img_aux, label_a in tqdm(loader, desc="Training", leave=False):
        img_main = img_main.to(DEVICE)
        img_aux = img_aux.to(DEVICE)
        label_a = label_a.to(DEVICE)

        optimizer.zero_grad()

        logits, v_main, v_aux, w_norm = model.forward_train(img_main, img_aux)

        l_ce = ce_criterion(logits, label_a)
        l_cl = cl_criterion(v_main, v_aux)
        l_scl = scl_criterion(w_norm, label_a)

        loss = ALPHA_CL * l_cl + BETA_SCL * l_scl + GAMMA_CE * l_ce
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += l_ce.item()
        total_cl += l_cl.item()
        total_scl += l_scl.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label_a.cpu().numpy())

    avg_loss = total_loss / len(loader)
    avg_ce = total_ce / len(loader)
    avg_cl = total_cl / len(loader)
    avg_scl = total_scl / len(loader)
    train_acc = accuracy_score(all_labels, all_preds)

    return avg_loss, avg_ce, avg_cl, avg_scl, train_acc


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model.forward_eval(images)
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

    train_tf_main, train_tf_aux, eval_tf = get_transforms()

    train_ds = TrainDatasetBCL(train_data, transform_main=train_tf_main, transform_aux=train_tf_aux)
    val_ds = EvalDataset(val_data, transform=eval_tf)
    test_ds = EvalDataset(test_data, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ViTBCL(proj_dim=PROJ_DIM).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    ce_criterion = nn.CrossEntropyLoss()
    cl_criterion = NTXentLoss(temperature=TEMPERATURE)
    scl_criterion = SupConLoss(temperature=TEMPERATURE)

    best_val_acc = -1.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_ce, train_cl, train_scl, train_acc = train_one_epoch(
            model, train_loader, optimizer, ce_criterion, cl_criterion, scl_criterion
        )

        val_loss, val_acc, val_f1, val_prec, val_rec, _, _ = evaluate(
            model, val_loader, ce_criterion
        )

        print(
            f"Train Total: {train_loss:.4f} | "
            f"CE: {train_ce:.4f} | "
            f"CL: {train_cl:.4f} | "
            f"SCL: {train_scl:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )
        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_total_loss": train_loss,
            "train_ce_loss": train_ce,
            "train_cl_loss": train_cl,
            "train_scl_loss": train_scl,
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
        model, test_loader, ce_criterion
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