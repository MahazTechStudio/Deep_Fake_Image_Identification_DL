import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

# Set style for seaborn
sns.set_theme(style="whitegrid")

def load_and_analyze():
    print("Loading dataset in streaming mode...")
    # Load the dataset in streaming mode to save memory and time
    dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", streaming=True)
    train_split = dataset['train']

    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # --- Task 3.1: Class Imbalance Check ---
    print("Performing Class Imbalance Check (Sample of 2000)...")
    labels = []
    sample_size = 2000
    count = 0
    for item in train_split:
        labels.append(item['Label_A'])
        count += 1
        if count >= sample_size:
            break
    
    df_labels = pd.DataFrame(labels, columns=['Label_A'])
    df_labels['Type'] = df_labels['Label_A'].map({0: 'Real', 1: 'AI-Generated'})
    
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_labels, x='Type', palette='viridis')
    plt.title('Class Distribution (Sample of 2000)')
    plt.xlabel('Image Type')
    plt.ylabel('Count')
    plt.savefig('figures/1_class_distribution.png')
    plt.close()
    print("Saved: figures/1_class_distribution.png")

    # --- Task 3.2: Image Resolution Check ---
    print("Performing Image Resolution Check (First 500)...")
    resolutions = []
    count = 0
    # Resetting iterator by reloading or just continuing if possible
    # Since it's an iterable, we might need to reload or just use the first 500 from the same stream
    dataset_res = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", streaming=True)
    for item in dataset_res['train']:
        img = item['Image']
        resolutions.append({'Width': img.width, 'Height': img.height})
        count += 1
        if count >= 500:
            break
            
    df_res = pd.DataFrame(resolutions)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_res, x='Width', y='Height', alpha=0.5)
    plt.title('Image Resolution Check (First 500 Images)')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.savefig('figures/2_resolution_check.png')
    plt.close()
    print("Saved: figures/2_resolution_check.png")

    # --- Task 3.3: Visual Sanity Check ---
    print("Performing Visual Sanity Check (2x4 grid)...")
    real_imgs = []
    fake_imgs = []
    
    dataset_viz = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", streaming=True)
    for item in dataset_viz['train']:
        if item['Label_A'] == 0 and len(real_imgs) < 4:
            real_imgs.append(item)
        elif item['Label_A'] == 1 and len(fake_imgs) < 4:
            fake_imgs.append(item)
        
        if len(real_imgs) == 4 and len(fake_imgs) == 4:
            break
            
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Visual Sanity Check: Real (Top) vs AI-Generated (Bottom)', fontsize=20)
    
    for i in range(4):
        # Real images
        axes[0, i].imshow(real_imgs[i]['Image'])
        caption = real_imgs[i]['Caption']
        axes[0, i].set_title(f"Real\n{caption[:30]}...", fontsize=10)
        axes[0, i].axis('off')
        
        # AI-Generated images
        axes[1, i].imshow(fake_imgs[i]['Image'])
        caption = fake_imgs[i]['Caption']
        axes[1, i].set_title(f"AI-Gen\n{caption[:30]}...", fontsize=10)
        axes[1, i].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('figures/3_visual_sanity_check.png')
    plt.close()
    print("Saved: figures/3_visual_sanity_check.png")
    
    print("\nAll diagnostics complete. Check the 'figures/' folder.")

if __name__ == "__main__":
    load_and_analyze()
