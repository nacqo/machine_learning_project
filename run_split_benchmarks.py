import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from knN import (
    KNearestNeighbors,
    LogisticRegressionGD,
    MushroomDataPreprocessor,
    StandardScalerNP,
    f1_score_weighted,
    train_val_test_split_np,
)


DATASET_PATH = "MushroomDataset/secondary_data.csv"
OUT_PATH = "results/initial/split_benchmark_results.csv"


def _eval_metrics(y_true, y_pred):
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "f1_weighted": float(f1_score_weighted(y_true, y_pred)),
    }


def _tune_knn(X_train, y_train, X_val, y_val, k_values=(3, 5, 7, 11)):
    best_k = None
    best_val = -np.inf
    for k in k_values:
        clf = KNearestNeighbors(k=k, metric="euclidean").fit(X_train, y_train)
        pred = clf.predict(X_val)
        acc = float(np.mean(pred == y_val))
        if acc > best_val:
            best_val = acc
            best_k = k
    return best_k, best_val


def _tune_gd_logreg(
    X_train,
    y_train,
    X_val,
    y_val,
    lr_grid=(0.1, 0.05),
    l2_grid=(0.0, 1e-4),
    n_epochs=80,
    batch_size=4096,
):
    best = None
    best_val = -np.inf
    for lr in lr_grid:
        for l2 in l2_grid:
            clf = LogisticRegressionGD(
                lr=lr,
                n_epochs=n_epochs,
                batch_size=batch_size,
                l2=l2,
                fit_intercept=True,
                random_state=42,
            )
            start = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - start
            pred = clf.predict(X_val)
            acc = float(np.mean(pred == y_val))
            if acc > best_val:
                best_val = acc
                best = (lr, l2, train_time)
    lr, l2, train_time = best
    return lr, l2, best_val, train_time


def _tune_dt(X_train, y_train, X_val, y_val, depths=(5, 10, 20, None)):
    best_depth = None
    best_val = -np.inf
    for d in depths:
        clf = DecisionTreeClassifier(random_state=42, max_depth=d)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        acc = float(np.mean(pred == y_val))
        if acc > best_val:
            best_val = acc
            best_depth = d
    return best_depth, best_val


def _apply_pca95(X_train, X_val, X_test):
    pca = PCA(n_components=0.95, random_state=42)
    X_train_p = pca.fit_transform(X_train)
    X_val_p = pca.transform(X_val)
    X_test_p = pca.transform(X_test)
    return X_train_p, X_val_p, X_test_p


def main():
    pre = MushroomDataPreprocessor(DATASET_PATH).preprocess()
    X, y = pre.get_processed_data()

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_np(
        X, y, val_size=0.2, test_size=0.2, random_state=42, stratify=y
    )

    # Dataset stats (class distribution)
    counts = pd.Series(y).value_counts().sort_index()
    total = int(counts.sum())
    class0_count = int(counts.iloc[0])
    class1_count = int(counts.iloc[1])
    class0_pct = 100.0 * class0_count / total
    class1_pct = 100.0 * class1_count / total

    # Scale (for kNN and GD logreg)
    scaler = StandardScalerNP()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    rows = []

    # -----------------------
    # kNN (no PCA)
    # -----------------------
    best_k, val_acc = _tune_knn(X_train_s, y_train, X_val_s, y_val)
    X_trainval_s = np.vstack([X_train_s, X_val_s])
    y_trainval = np.concatenate([y_train, y_val])
    start = time.time()
    knn = KNearestNeighbors(k=best_k, metric="euclidean").fit(X_trainval_s, y_trainval)
    train_time = time.time() - start
    pred = knn.predict(X_test_s)
    m = _eval_metrics(y_test, pred)
    rows.append(
        {
            "Model": "kNN",
            "PCA": "None",
            "Val Acc": float(val_acc),
            "Test Acc": m["accuracy"],
            "F1 (weighted)": m["f1_weighted"],
            "Train time (s)": float(train_time),
            "Notes": f"k={best_k}",
        }
    )

    # -----------------------
    # kNN (PCA 95%)
    # -----------------------
    X_train_p, X_val_p, X_test_p = _apply_pca95(X_train_s, X_val_s, X_test_s)
    best_k, val_acc = _tune_knn(X_train_p, y_train, X_val_p, y_val)
    X_trainval_p = np.vstack([X_train_p, X_val_p])
    start = time.time()
    knn = KNearestNeighbors(k=best_k, metric="euclidean").fit(X_trainval_p, y_trainval)
    train_time = time.time() - start
    pred = knn.predict(X_test_p)
    m = _eval_metrics(y_test, pred)
    rows.append(
        {
            "Model": "kNN",
            "PCA": "95% var",
            "Val Acc": float(val_acc),
            "Test Acc": m["accuracy"],
            "F1 (weighted)": m["f1_weighted"],
            "Train time (s)": float(train_time),
            "Notes": f"k={best_k}",
        }
    )

    # -----------------------
    # GD Logistic Regression (no PCA)
    # -----------------------
    lr, l2, val_acc, train_time = _tune_gd_logreg(X_train_s, y_train, X_val_s, y_val)
    clf = LogisticRegressionGD(
        lr=lr, n_epochs=80, batch_size=4096, l2=l2, fit_intercept=True, random_state=42
    )
    start = time.time()
    clf.fit(X_trainval_s, y_trainval)
    train_time = time.time() - start
    pred = clf.predict(X_test_s)
    m = _eval_metrics(y_test, pred)
    rows.append(
        {
            "Model": "GD Logreg",
            "PCA": "None",
            "Val Acc": float(val_acc),
            "Test Acc": m["accuracy"],
            "F1 (weighted)": m["f1_weighted"],
            "Train time (s)": float(train_time),
            "Notes": f"lr={lr}, l2={l2}",
        }
    )

    # -----------------------
    # GD Logistic Regression (PCA 95%)
    # -----------------------
    X_train_p, X_val_p, X_test_p = _apply_pca95(X_train_s, X_val_s, X_test_s)
    lr, l2, val_acc, _ = _tune_gd_logreg(X_train_p, y_train, X_val_p, y_val)
    clf = LogisticRegressionGD(
        lr=lr, n_epochs=80, batch_size=4096, l2=l2, fit_intercept=True, random_state=42
    )
    start = time.time()
    clf.fit(np.vstack([X_train_p, X_val_p]), y_trainval)
    train_time = time.time() - start
    pred = clf.predict(X_test_p)
    m = _eval_metrics(y_test, pred)
    rows.append(
        {
            "Model": "GD Logreg",
            "PCA": "95% var",
            "Val Acc": float(val_acc),
            "Test Acc": m["accuracy"],
            "F1 (weighted)": m["f1_weighted"],
            "Train time (s)": float(train_time),
            "Notes": f"lr={lr}, l2={l2}",
        }
    )

    # -----------------------
    # Decision Tree (no PCA)
    # -----------------------
    best_depth, val_acc = _tune_dt(X_train, y_train, X_val, y_val)
    start = time.time()
    dt = DecisionTreeClassifier(random_state=42, max_depth=best_depth)
    dt.fit(np.vstack([X_train, X_val]), y_trainval)
    train_time = time.time() - start
    pred = dt.predict(X_test)
    m = _eval_metrics(y_test, pred)
    rows.append(
        {
            "Model": "Decision Tree",
            "PCA": "None",
            "Val Acc": float(val_acc),
            "Test Acc": m["accuracy"],
            "F1 (weighted)": m["f1_weighted"],
            "Train time (s)": float(train_time),
            "Notes": f"max_depth={best_depth}",
        }
    )

    # -----------------------
    # Decision Tree (PCA 95%)
    # -----------------------
    # Use scaled+PCA representation (consistent with other PCA runs)
    X_train_p, X_val_p, X_test_p = _apply_pca95(X_train_s, X_val_s, X_test_s)
    best_depth, val_acc = _tune_dt(X_train_p, y_train, X_val_p, y_val)
    start = time.time()
    dt = DecisionTreeClassifier(random_state=42, max_depth=best_depth)
    dt.fit(np.vstack([X_train_p, X_val_p]), y_trainval)
    train_time = time.time() - start
    pred = dt.predict(X_test_p)
    m = _eval_metrics(y_test, pred)
    rows.append(
        {
            "Model": "Decision Tree",
            "PCA": "95% var",
            "Val Acc": float(val_acc),
            "Test Acc": m["accuracy"],
            "F1 (weighted)": m["f1_weighted"],
            "Train time (s)": float(train_time),
            "Notes": f"max_depth={best_depth}",
        }
    )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)

    stats_path = "results/initial/dataset_class_stats.csv"
    pd.DataFrame(
        [
            {
                "class0_count": class0_count,
                "class0_pct": class0_pct,
                "class1_count": class1_count,
                "class1_pct": class1_pct,
                "n_total": total,
            }
        ]
    ).to_csv(stats_path, index=False)

    print(f"Saved {OUT_PATH}")
    print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()

