import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from knN import (
    MushroomDataPreprocessor,
    StandardScalerNP,
    LogisticRegressionGD,
    train_val_test_split_np,
)


RESULTS_FILE = "results/initial/results_pca_comparison.csv"
BEST_FILE = "results/initial/results_pca_best_configs.csv"
GD_METRICS_FILE = "results/initial/gd_metrics.csv"
DATASET_FILE = "MushroomDataset/secondary_data.csv"
OUTPUT_DIR = "plots/initial"


def _load_csv(path_candidates):
    for path in path_candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find any of: {path_candidates}")


def _ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_class_distribution(dataset_path, output_dir):
    df = pd.read_csv(dataset_path, sep=";")
    if "class" not in df.columns:
        df = pd.read_csv(dataset_path)
    if "class" not in df.columns:
        raise ValueError("Dataset does not contain a 'class' column.")

    class_counts = df["class"].value_counts().sort_index()
    class_pct = class_counts / class_counts.sum() * 100

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(class_counts.index.astype(str), class_counts.values, color=["#4C78A8", "#F58518"])
    plt.title("Class Distribution (Secondary Mushroom)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.2)

    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{class_counts.values[i]}\n({class_pct.values[i]:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_best_metrics(best_df, output_dir):
    metric_cols = ["Accuracy Mean", "Balanced Accuracy Mean", "F1-macro Mean"]
    melted = best_df.melt(
        id_vars=["Model", "PCA"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score",
    )
    melted["Label"] = melted["Model"] + " | " + melted["PCA"]

    labels = list(melted["Label"].unique())
    metrics = metric_cols
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 5.5))
    for i, metric in enumerate(metrics):
        vals = []
        for lbl in labels:
            vals.append(melted[(melted["Label"] == lbl) & (melted["Metric"] == metric)]["Score"].iloc[0])
        plt.bar(x + (i - 1) * width, vals, width=width, label=metric)

    plt.title("Best Configuration Metrics by Model and PCA")
    plt.ylabel("Score")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0.75, 1.01)
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, "best_config_metrics.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_best_training_time(best_df, output_dir):
    plot_df = best_df.copy()
    plot_df["Label"] = plot_df["Model"] + " | " + plot_df["PCA"]
    plot_df = plot_df.sort_values("Training Time Mean", ascending=True)

    plt.figure(figsize=(10, 5))
    bars = plt.barh(plot_df["Label"], plot_df["Training Time Mean"], color="#54A24B")
    plt.title("Training Time (Best Configurations)")
    plt.xlabel("Mean Fit Time (seconds)")
    plt.grid(axis="x", alpha=0.2)

    for bar in bars:
        w = bar.get_width()
        plt.text(w, bar.get_y() + bar.get_height() / 2, f" {w:.4f}s", va="center", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "best_config_training_time.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def _extract_param_value(params_str, key):
    pattern = rf"{re.escape(key)}=([^,]+)"
    match = re.search(pattern, str(params_str))
    if not match:
        return None
    value = match.group(1).strip()
    if value == "None":
        return None
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def plot_decision_tree_depth_effect(results_df, output_dir):
    dt = results_df[results_df["Model"] == "Decision Tree"].copy()
    dt["max_depth"] = dt["Hyperparameters"].apply(lambda s: _extract_param_value(s, "max_depth"))
    dt = dt[dt["max_depth"].notna()].copy()
    if dt.empty:
        return

    depth_plot = (
        dt.groupby(["PCA", "max_depth"], as_index=False)["Accuracy Mean"]
        .mean()
        .sort_values(["PCA", "max_depth"])
    )

    plt.figure(figsize=(8, 5))
    for pca_label, grp in depth_plot.groupby("PCA"):
        grp = grp.sort_values("max_depth")
        plt.plot(grp["max_depth"], grp["Accuracy Mean"], marker="o", label=pca_label)

    plt.title("Decision Tree Depth vs Accuracy")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy Mean")
    plt.grid(alpha=0.25)
    plt.legend(title="PCA")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "decision_tree_depth_vs_accuracy.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gd_metrics(gd_df, output_dir):
    if gd_df is None or gd_df.empty:
        return

    cols = [
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
    ]
    available = [c for c in cols if c in gd_df.columns]
    if not available:
        return

    row = gd_df.iloc[0]
    labels = ["Train", "Validation", "Test"]
    values = [
        float(row.get("train_accuracy", np.nan)),
        float(row.get("val_accuracy", np.nan)),
        float(row.get("test_accuracy", np.nan)),
    ]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.title("Gradient Descent Logistic Regression (Accuracy)")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.01)
    plt.grid(axis="y", alpha=0.2)

    for bar, v in zip(bars, values):
        if not np.isnan(v):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "gd_logreg_accuracy.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gd_losses(gd_df, output_dir):
    if gd_df is None or gd_df.empty:
        return
    loss_cols = ["final_train_loss", "final_val_loss", "final_test_loss"]
    if not all(c in gd_df.columns for c in loss_cols):
        return

    row = gd_df.iloc[0]
    labels = ["Train", "Validation", "Test"]
    values = [float(row[c]) for c in loss_cols]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.title("Gradient Descent Logistic Regression (Final Loss)")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(axis="y", alpha=0.2)

    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "gd_logreg_final_loss.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gd_loss_surface(output_dir):
    """
    Build a loss surface over (learning_rate, l2) using validation loss.
    """
    preprocessor = MushroomDataPreprocessor("MushroomDataset/secondary_data.csv")
    preprocessor.preprocess()
    X, y = preprocessor.get_processed_data()

    X_train, X_val, _, y_train, y_val, _ = train_val_test_split_np(
        X, y, val_size=0.2, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScalerNP()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr_values = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2], dtype=float)
    l2_values = np.array([0.0, 1e-5, 1e-4, 1e-3, 1e-2], dtype=float)

    Z = np.zeros((len(l2_values), len(lr_values)), dtype=float)

    for i, l2 in enumerate(l2_values):
        for j, lr in enumerate(lr_values):
            clf = LogisticRegressionGD(
                lr=lr,
                n_epochs=60,
                batch_size=4096,
                l2=l2,
                fit_intercept=True,
                random_state=42,
            )
            clf.fit(X_train_scaled, y_train)
            Z[i, j] = clf.loss(X_val_scaled, y_val)

    LR, L2 = np.meshgrid(lr_values, l2_values)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(LR, L2, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_title("Gradient Descent Validation Loss Surface")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("L2 Regularization")
    ax.set_zlabel("Validation Loss")
    fig.colorbar(surf, shrink=0.6, aspect=12, label="Validation Loss")
    plt.tight_layout()
    out_surface = os.path.join(output_dir, "gd_loss_surface.png")
    plt.savefig(out_surface, dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    im = plt.imshow(Z, cmap="viridis", aspect="auto", origin="upper")
    plt.title("Gradient Descent Validation Loss Heatmap")
    plt.xlabel("Learning Rate")
    plt.ylabel("L2 Regularization")
    plt.xticks(np.arange(len(lr_values)), [f"{v:g}" for v in lr_values])
    plt.yticks(np.arange(len(l2_values)), [f"{v:g}" for v in l2_values])
    cbar = plt.colorbar(im)
    cbar.set_label("Validation Loss")
    plt.tight_layout()
    out_heatmap = os.path.join(output_dir, "gd_loss_surface_heatmap.png")
    plt.savefig(out_heatmap, dpi=220)
    plt.close()


def main():
    _ensure_output_dir(OUTPUT_DIR)

    results_df = _load_csv([RESULTS_FILE])
    best_df = _load_csv([BEST_FILE])
    gd_df = None
    if os.path.exists(GD_METRICS_FILE):
        gd_df = pd.read_csv(GD_METRICS_FILE)

    plot_class_distribution(DATASET_FILE, OUTPUT_DIR)
    plot_best_metrics(best_df, OUTPUT_DIR)
    plot_best_training_time(best_df, OUTPUT_DIR)
    plot_decision_tree_depth_effect(results_df, OUTPUT_DIR)
    plot_gd_metrics(gd_df, OUTPUT_DIR)
    plot_gd_losses(gd_df, OUTPUT_DIR)
    plot_gd_loss_surface(OUTPUT_DIR)

    print("Saved visualizations to:", OUTPUT_DIR)
    print("- class_distribution.png")
    print("- best_config_metrics.png")
    print("- best_config_training_time.png")
    print("- decision_tree_depth_vs_accuracy.png")
    if gd_df is not None:
        print("- gd_logreg_accuracy.png")
        print("- gd_logreg_final_loss.png")
    print("- gd_loss_surface.png")
    print("- gd_loss_surface_heatmap.png")


if __name__ == "__main__":
    main()
