import torch
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_augmentation():
    # Load a small sample
    dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", streaming=True)
    train_split = dataset['train']
    
    # Get a sample image
    sample = next(iter(train_split))
    img = sample['Image']
    
    # Define augmentation pipeline
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        # Normalization usually comes here, but for visualization we'll skip it or denormalize
    ])
    
    # Generate 5 augmented versions of the same image
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 6, 1)
    plt.imshow(img.resize((224, 224)))
    plt.title("Original (Resized)")
    plt.axis('off')
    
    for i in range(5):
        aug_img = train_transforms(img)
        # Convert tensor back to image for display
        aug_img_np = aug_img.permute(1, 2, 0).numpy()
        
        plt.subplot(1, 6, i + 2)
        plt.imshow(aug_img_np)
        plt.title(f"Augmented {i+1}")
        plt.axis('off')
    
    plt.savefig('figures/4_augmentation_check.png')
    plt.close()
    print("Saved: figures/4_augmentation_check.png")

if __name__ == "__main__":
    visualize_augmentation()
