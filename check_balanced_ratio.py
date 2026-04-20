import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def collect_balanced_pool(dataset, total_samples=4000):
    class_data = {0: [], 1: []}
    target_per_class = total_samples // 2

    print(f"Collecting balanced pool: {total_samples} total samples ({target_per_class} per class)")
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
    return combined


def plot_balanced_distribution(balanced_data, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Count labels
    labels = [item["Label_A"] for item in balanced_data]
    real_count = labels.count(0)
    ai_count = labels.count(1)
    total = real_count + ai_count
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Type': ['Real', 'AI-Generated'],
        'Count': [real_count, ai_count]
    })
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Create plot
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df, x='Type', y='Count', palette=['#2ecc71', '#e74c3c'])
    
    # Customize
    plt.title('Balanced Training Data Distribution (After Preprocessing)', fontsize=14, fontweight='bold')
    plt.xlabel('Image Type', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.ylim(0, total // 2 + 500)
    
    # Add value labels on top of bars
    for i, v in enumerate([real_count, ai_count]):
        ax.text(i, v + 50, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add percentage labels inside bars
    for i, v in enumerate([real_count, ai_count]):
        pct = (v / total) * 100
        ax.text(i, v / 2, f'{pct:.0f}%', ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    
    # Add a text box explaining the method
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.5, -0.15, f'Method: collect_balanced_pool()\nTotal samples: {total} (1:1 ratio)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=props, ha='center')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'balanced_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Real: {real_count} images (50%)")
    print(f"  AI-Generated: {ai_count} images (50%)")
    print(f"  Total: {total} images")
    
    return real_count, ai_count


def plot_comparison(raw_data_sample=None, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    # If raw data sample not provided, sample it now
    if raw_data_sample is None:
        print("\nSampling raw data distribution (first 4000 samples)...")
        dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", streaming=True)
        raw_labels = []
        count = 0
        for item in dataset["train"]:
            raw_labels.append(int(item["Label_A"]))
            count += 1
            if count >= 4000:
                break
        raw_real = raw_labels.count(0)
        raw_ai = raw_labels.count(1)
    else:
        raw_real = raw_data_sample[0]
        raw_ai = raw_data_sample[1]
    
    # Create comparison dataframe
    df_compare = pd.DataFrame({
        'Dataset': ['Raw (First 4000)', 'Balanced (4000)'],
        'Real': [raw_real, 2000],
        'AI-Generated': [raw_ai, 2000]
    })
    
    # Melt for seaborn
    df_melted = df_compare.melt(id_vars='Dataset', var_name='Image Type', value_name='Count')
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melted, x='Dataset', y='Count', hue='Image Type', palette=['#2ecc71', '#e74c3c'])
    
    plt.title('Comparison: Raw Dataset vs Balanced Training Data', fontsize=14, fontweight='bold')
    plt.xlabel('', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fontsize=10, fontweight='bold')
    
    plt.legend(title='Image Type', title_fontsize=11)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'raw_vs_balanced_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison: {output_path}")
    
    return raw_real, raw_ai


def main():
    print("=" * 60)
    print("Checking Balanced Distribution")
    print("=" * 60)
    
    # Load dataset in streaming mode
    print("\nLoading dataset...")
    dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset", streaming=True)
    
    # Apply the SAME balancing function used in your training scripts
    print("\nApplying collect_balanced_pool()...")
    balanced_data = collect_balanced_pool(dataset, total_samples=4000)
    
    # Plot the balanced distribution
    print("\nGenerating balanced distribution plot...")
    real_count, ai_count = plot_balanced_distribution(balanced_data)
    
    # Optional: Generate comparison plot (shows raw vs balanced side by side)
    print("\n" + "=" * 60)
    response = input("Generate comparison plot (raw vs balanced)? (y/n): ")
    if response.lower() == 'y':
        plot_comparison(output_dir="figures")
    
    print("\n" + "=" * 60)
    print("✓ DONE! Check the 'figures/' folder for:")
    print("   - balanced_distribution.png")
    print("   - raw_vs_balanced_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()