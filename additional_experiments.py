import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


DATASET_PATH = "MushroomDataset/secondary_data.csv"
RESULTS_DIR = "results/additional"


def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, sep=";")
    if "class" not in df.columns:
        df = pd.read_csv(path)
    if "class" not in df.columns:
        raise ValueError("Dataset must include 'class' column.")

    X = df.drop(columns=["class"]).copy()
    y = df["class"].copy()

    for col in X.columns:
        if X[col].dtype == "object":
            mode = X[col].mode()
            X[col] = X[col].fillna("missing" if mode.empty else mode.iloc[0])
        else:
            X[col] = X[col].fillna(X[col].median())

    if y.dtype == "object":
        mode = y.mode()
        y = y.fillna("missing" if mode.empty else mode.iloc[0])
        y = LabelEncoder().fit_transform(y)

    return X, np.asarray(y)


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )


def model_configs():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, max_depth=None, min_samples_leaf=1
        ),
        "kNN": KNeighborsClassifier(n_neighbors=3, weights="uniform"),
    }


def evaluate_split(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def run_pca_sweep(X, y, preprocessor, models):
    pca_settings = [
        ("No PCA", None),
        ("PCA 70%", 0.70),
        ("PCA 80%", 0.80),
        ("PCA 90%", 0.90),
        ("PCA 95%", 0.95),
        ("PCA 99%", 0.99),
    ]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    rows = []
    for model_name, model in models.items():
        for pca_label, pca_value in pca_settings:
            fold_metrics = []
            fit_times = []
            n_components_seen = []

            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                steps = [("preprocessing", clone(preprocessor))]
                if pca_value is not None:
                    steps.append(("pca", PCA(n_components=pca_value)))
                steps.append(("model", clone(model)))
                pipeline = Pipeline(steps)

                start = pd.Timestamp.now()
                pipeline.fit(X_train, y_train)
                fit_seconds = (pd.Timestamp.now() - start).total_seconds()
                y_pred = pipeline.predict(X_test)
                metrics = evaluate_split(y_test, y_pred)
                fold_metrics.append(metrics)
                fit_times.append(fit_seconds)

                if pca_value is not None:
                    n_components_seen.append(int(pipeline.named_steps["pca"].n_components_))

            fold_df = pd.DataFrame(fold_metrics)
            rows.append(
                {
                    "Model": model_name,
                    "PCA Setting": pca_label,
                    "Accuracy Mean": fold_df["accuracy"].mean(),
                    "Accuracy Std": fold_df["accuracy"].std(),
                    "Balanced Accuracy Mean": fold_df["balanced_accuracy"].mean(),
                    "F1-weighted Mean": fold_df["f1_weighted"].mean(),
                    "F1-macro Mean": fold_df["f1_macro"].mean(),
                    "Training Time Mean": float(np.mean(fit_times)),
                    "Training Time Std": float(np.std(fit_times)),
                    "Avg Components": np.mean(n_components_seen) if n_components_seen else np.nan,
                    "Num Folds": len(fold_metrics),
                }
            )

    return pd.DataFrame(rows)


def run_significance_tests(X, y, preprocessor, models):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    rows = []

    for model_name, model in models.items():
        no_pca_scores = defaultdict(list)
        pca_scores = defaultdict(list)

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe_no = Pipeline(
                [("preprocessing", clone(preprocessor)), ("model", clone(model))]
            )
            pipe_no.fit(X_train, y_train)
            pred_no = pipe_no.predict(X_test)
            m_no = evaluate_split(y_test, pred_no)
            for k, v in m_no.items():
                no_pca_scores[k].append(v)

            pipe_pca = Pipeline(
                [
                    ("preprocessing", clone(preprocessor)),
                    ("pca", PCA(n_components=0.95)),
                    ("model", clone(model)),
                ]
            )
            pipe_pca.fit(X_train, y_train)
            pred_pca = pipe_pca.predict(X_test)
            m_pca = evaluate_split(y_test, pred_pca)
            for k, v in m_pca.items():
                pca_scores[k].append(v)

        for metric in ["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro"]:
            baseline = np.asarray(no_pca_scores[metric], dtype=float)
            reduced = np.asarray(pca_scores[metric], dtype=float)
            diff = reduced - baseline

            if wilcoxon is not None:
                stat = wilcoxon(reduced, baseline, zero_method="wilcox", alternative="two-sided")
                p_value = float(stat.pvalue)
            else:  # fallback when scipy unavailable
                p_value = np.nan

            rows.append(
                {
                    "Model": model_name,
                    "Metric": metric,
                    "No PCA Mean": baseline.mean(),
                    "PCA95 Mean": reduced.mean(),
                    "Mean Difference (PCA-NoPCA)": diff.mean(),
                    "Wilcoxon p-value": p_value,
                    "Significant @ 0.05": bool(p_value < 0.05) if not np.isnan(p_value) else False,
                }
            )

    return pd.DataFrame(rows)


def run_confusion_and_class_metrics(X, y, preprocessor, models):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    labels = np.unique(y)

    conf_rows = []
    class_rows = []

    for model_name, model in models.items():
        for pca_label, pca_value in [("No PCA", None), ("PCA 95%", 0.95)]:
            y_true_all = []
            y_pred_all = []

            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                steps = [("preprocessing", clone(preprocessor))]
                if pca_value is not None:
                    steps.append(("pca", PCA(n_components=pca_value)))
                steps.append(("model", clone(model)))
                pipeline = Pipeline(steps)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                y_true_all.append(y_test)
                y_pred_all.append(y_pred)

            y_true_all = np.concatenate(y_true_all)
            y_pred_all = np.concatenate(y_pred_all)

            cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
            for i, actual_class in enumerate(labels):
                for j, predicted_class in enumerate(labels):
                    conf_rows.append(
                        {
                            "Model": model_name,
                            "PCA": pca_label,
                            "Actual Class": int(actual_class),
                            "Predicted Class": int(predicted_class),
                            "Count": int(cm[i, j]),
                        }
                    )

            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_all, y_pred_all, labels=labels, zero_division=0
            )
            for idx, cls in enumerate(labels):
                class_rows.append(
                    {
                        "Model": model_name,
                        "PCA": pca_label,
                        "Class": int(cls),
                        "Precision": float(precision[idx]),
                        "Recall": float(recall[idx]),
                        "F1": float(f1[idx]),
                        "Support": int(support[idx]),
                    }
                )

    return pd.DataFrame(conf_rows), pd.DataFrame(class_rows)


def rounded(df, digits=4):
    out = df.copy()
    num_cols = out.select_dtypes(include=["number"]).columns
    out[num_cols] = out[num_cols].round(digits)
    return out


def main():
    print("Running additional experiments...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X, y = load_dataset()
    preprocessor = build_preprocessor(X)
    models = model_configs()

    print("1) PCA sweep with repeated CV")
    pca_sweep_df = run_pca_sweep(X, y, preprocessor, models)
    pca_sweep_df = rounded(pca_sweep_df, digits=4)
    pca_sweep_path = os.path.join(RESULTS_DIR, "additional_pca_sweep_results.csv")
    pca_sweep_df.to_csv(pca_sweep_path, index=False)
    print(f"Saved {pca_sweep_path}")

    print("2) Paired significance tests (PCA vs No PCA)")
    significance_df = run_significance_tests(X, y, preprocessor, models)
    significance_df = rounded(significance_df, digits=6)
    significance_path = os.path.join(RESULTS_DIR, "additional_pca_significance_tests.csv")
    significance_df.to_csv(significance_path, index=False)
    print(f"Saved {significance_path}")

    print("3) Confusion matrices and per-class metrics")
    conf_df, per_class_df = run_confusion_and_class_metrics(X, y, preprocessor, models)
    confusion_path = os.path.join(RESULTS_DIR, "additional_confusion_matrices.csv")
    per_class_path = os.path.join(RESULTS_DIR, "additional_per_class_metrics.csv")
    conf_df.to_csv(confusion_path, index=False)
    rounded(per_class_df, digits=4).to_csv(per_class_path, index=False)
    print(f"Saved {confusion_path}")
    print(f"Saved {per_class_path}")
    print("Done.")


if __name__ == "__main__":
    main()
