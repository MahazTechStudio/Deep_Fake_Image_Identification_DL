import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# File locations
# -----------------------------
FILES = {
    "CE Baseline": "reports_ce_clean/epoch_metrics.csv",
    "SupCon": "reports_supcon/epoch_metrics.csv",
    "Paper BCL": "reports_bcl_paper/epoch_metrics.csv",
}

OUTPUT_DIR = "analysis_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_csv_safe(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"[WARNING] File not found: {path}")
        return None


def plot_loss_curves():
    plt.figure(figsize=(12, 8))

    found_any = False

    for model_name, path in FILES.items():
        df = load_csv_safe(path)
        if df is None:
            continue

        found_any = True

        # Try to detect the right columns
        if "train_loss" in df.columns and "val_loss" in df.columns:
            plt.plot(df["epoch"], df["train_loss"], label=f"{model_name} Train Loss")
            plt.plot(df["epoch"], df["val_loss"], linestyle="--", label=f"{model_name} Val Loss")

        elif "train_total_loss" in df.columns and "val_loss" in df.columns:
            plt.plot(df["epoch"], df["train_total_loss"], label=f"{model_name} Train Loss")
            plt.plot(df["epoch"], df["val_loss"], linestyle="--", label=f"{model_name} Val Loss")

    if not found_any:
        print("[ERROR] No valid CSV files found for loss curves.")
        return

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curves_loss.png"))
    plt.close()
    print("[SAVED] analysis_figures/learning_curves_loss.png")


def plot_accuracy_curves():
    plt.figure(figsize=(12, 8))

    found_any = False

    for model_name, path in FILES.items():
        df = load_csv_safe(path)
        if df is None:
            continue

        found_any = True

        if "train_acc" in df.columns and "val_acc" in df.columns:
            plt.plot(df["epoch"], df["train_acc"], label=f"{model_name} Train Acc")
            plt.plot(df["epoch"], df["val_acc"], linestyle="--", label=f"{model_name} Val Acc")

    if not found_any:
        print("[ERROR] No valid CSV files found for accuracy curves.")
        return

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curves_accuracy.png"))
    plt.close()
    print("[SAVED] analysis_figures/learning_curves_accuracy.png")


def main():
    plot_loss_curves()
    plot_accuracy_curves()
    print("\nDone. Check the 'analysis_figures' folder.")


if __name__ == "__main__":
    main()