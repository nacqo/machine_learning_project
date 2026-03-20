import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULTS_DIR = "results/additional"
OUTPUT_DIR = "plots/additional"


def _load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_pca_sweep():
    df = _load(os.path.join(RESULTS_DIR, "additional_pca_sweep_results.csv"))
    plt.figure(figsize=(10, 5.5))
    for model in df["Model"].unique():
        sub = df[df["Model"] == model].copy()
        # Keep explicit ordering for the x-axis.
        order = ["No PCA", "PCA 70%", "PCA 80%", "PCA 90%", "PCA 95%", "PCA 99%"]
        sub["PCA Setting"] = pd.Categorical(sub["PCA Setting"], categories=order, ordered=True)
        sub = sub.sort_values("PCA Setting")
        plt.plot(sub["PCA Setting"], sub["F1-macro Mean"], marker="o", label=model)

    plt.title("Additional Experiment: PCA Sweep vs F1-macro")
    plt.xlabel("PCA Setting")
    plt.ylabel("F1-macro Mean")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "additional_pca_sweep_f1_macro.png"), dpi=220)
    plt.close()


def plot_significance_differences():
    df = _load(os.path.join(RESULTS_DIR, "additional_pca_significance_tests.csv"))
    metric_map = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "f1_weighted": "F1-weighted",
        "f1_macro": "F1-macro",
    }
    df["Metric Label"] = df["Metric"].map(metric_map).fillna(df["Metric"])

    plt.figure(figsize=(11.5, 5.5))
    sns.barplot(
        data=df,
        x="Metric Label",
        y="Mean Difference (PCA-NoPCA)",
        hue="Model",
        errorbar=None,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Additional Experiment: PCA95 - No PCA (Mean Difference)")
    plt.xlabel("Metric")
    plt.ylabel("Mean Difference")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "additional_pca_difference_bar.png"), dpi=220)
    plt.close()


def plot_confusion_heatmaps():
    df = _load(os.path.join(RESULTS_DIR, "additional_confusion_matrices.csv"))
    for model in df["Model"].unique():
        for pca_label in df["PCA"].unique():
            sub = df[(df["Model"] == model) & (df["PCA"] == pca_label)].copy()
            pivot = sub.pivot(index="Actual Class", columns="Predicted Class", values="Count")
            plt.figure(figsize=(5.2, 4.4))
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", cbar=True)
            plt.title(f"{model} - {pca_label} Confusion Matrix")
            plt.xlabel("Predicted Class")
            plt.ylabel("Actual Class")
            plt.tight_layout()
            file_name = f"additional_confusion_{model.lower().replace(' ', '_')}_{pca_label.lower().replace(' ', '_')}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=220)
            plt.close()


def plot_per_class_recall():
    df = _load(os.path.join(RESULTS_DIR, "additional_per_class_metrics.csv"))
    plt.figure(figsize=(11.5, 5.5))
    sns.barplot(data=df, x="Model", y="Recall", hue="PCA", errorbar=None)
    plt.ylim(0.7, 1.01)
    plt.title("Additional Experiment: Per-class Recall (Aggregated)")
    plt.ylabel("Recall")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "additional_per_class_recall.png"), dpi=220)
    plt.close()


def main():
    _ensure_output_dir()
    plot_pca_sweep()
    plot_significance_differences()
    plot_confusion_heatmaps()
    plot_per_class_recall()
    print(f"Saved additional visualizations in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
