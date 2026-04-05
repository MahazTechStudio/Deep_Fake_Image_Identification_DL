
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

# This script checks the exact ratio of Label 0 (Real) to Label 1 (AI) in the train_loader
def check_label_ratio():
    NUM_SAMPLES_TRAIN = 100
    label_col = 'Label_A'
    
    print("Fetching training samples from streaming dataset...")
    dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", streaming=True)
    train_iter = iter(dataset['train'])
    
    # Collect labels for the first 100 samples
    labels = []
    for i in range(NUM_SAMPLES_TRAIN):
        try:
            item = next(train_iter)
            labels.append(item[label_col])
        except StopIteration:
            break
            
    total = len(labels)
    label_0 = labels.count(0)
    label_1 = labels.count(1)
    
    print(f"\n--- Training Set Label Distribution (First {total} samples) ---")
    print(f"Label 0 (Real): {label_0} ({label_0/total*100:.2f}%)")
    print(f"Label 1 (AI):   {label_1} ({label_1/total*100:.2f}%)")
    print(f"Total Samples:  {total}")

if __name__ == "__main__":
    check_label_ratio()
