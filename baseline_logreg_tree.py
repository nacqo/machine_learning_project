import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _find_csv_candidates():
    """Return CSV files from project root and MushroomDataset."""
    root_csv = [f for f in os.listdir() if f.lower().endswith(".csv")]

    dataset_dir = "MushroomDataset"
    dataset_csv = []
    if os.path.isdir(dataset_dir):
        dataset_csv = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.lower().endswith(".csv")
        ]

    candidates = root_csv + dataset_csv

    # Prefer the project's intended dataset for this study.
    preferred = "MushroomDataset/secondary_data.csv"
    if preferred in candidates:
        return [preferred] + [c for c in candidates if c != preferred]
    return candidates


def _load_dataset():
    """
    Load the first usable CSV.
    Tries ';' separator first (mushroom data format), then ','.
    """
    csv_files = _find_csv_candidates()
    print("CSV files found:", csv_files)

    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV file found in project root or MushroomDataset.")

    candidate_with_class = None
    fallback_candidate = None

    for file_name in csv_files:
        try:
            df = pd.read_csv(file_name, sep=";")
            if "class" not in df.columns:
                df = pd.read_csv(file_name)
            if "class" in df.columns:
                candidate_with_class = (file_name, df)
                break
            if len(df.columns) > 1 and fallback_candidate is None:
                fallback_candidate = (file_name, df)
        except Exception:
            continue

    if candidate_with_class is not None:
        file_name, df = candidate_with_class
        print("Using file:", file_name)
        return df

    if fallback_candidate is not None:
        file_name, df = fallback_candidate
        print("Using file:", file_name)
        return df

    raise ValueError("No usable CSV dataset found.")


def run_baseline_and_pca_comparison():
    df = _load_dataset()

    print(df.shape)
    print(df.columns.tolist())

    target = "class"
    if target not in df.columns:
        target = df.columns[-1]

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    for col in X.columns:
        if X[col].dtype == "object":
            if X[col].mode().empty:
                X[col] = X[col].fillna("missing")
            else:
                X[col] = X[col].fillna(X[col].mode()[0])
        else:
            X[col] = X[col].fillna(X[col].median())

    if y.dtype == "object":
        if y.mode().empty:
            y = y.fillna("missing")
        else:
            y = y.fillna(y.mode()[0])
        y = LabelEncoder().fit_transform(y)

    # Class-distribution visibility helps interpret imbalance-sensitive metrics.
    class_counts = pd.Series(y).value_counts().sort_index()
    class_pct = (class_counts / class_counts.sum() * 100).round(2)
    imbalance_df = pd.DataFrame({"count": class_counts, "percent": class_pct})
    print("\nClass distribution:")
    print(imbalance_df.to_string())

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    # Team note:
    # If your teammate implemented Logistic Regression only, Decision Tree must be implemented too.
    # If your teammate implemented Decision Tree only, Logistic Regression must be implemented too.
    model_hyperparams = {
        "Logistic Regression": [
            {"max_iter": 1000, "random_state": 42, "C": 0.5},
            {"max_iter": 1000, "random_state": 42, "C": 1.0, "class_weight": "balanced"},
            {"max_iter": 1000, "random_state": 42, "C": 2.0},
        ],
        "Decision Tree": [
            {"random_state": 42, "max_depth": 5, "min_samples_leaf": 1},
            {"random_state": 42, "max_depth": 10, "min_samples_leaf": 1},
            {"random_state": 42, "max_depth": 20, "min_samples_leaf": 2},
            {"random_state": 42, "max_depth": None, "min_samples_leaf": 1},
            {"random_state": 42, "max_depth": 20, "min_samples_leaf": 2, "class_weight": "balanced"},
        ],
        "kNN": [
            {"n_neighbors": 3, "weights": "uniform"},
            {"n_neighbors": 5, "weights": "uniform"},
            {"n_neighbors": 11, "weights": "distance"},
        ],
    }

    model_builders = {
        "Logistic Regression": LogisticRegression,
        "Decision Tree": DecisionTreeClassifier,
        "kNN": KNeighborsClassifier,
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_weighted": "f1_weighted",
        "f1_macro": "f1_macro",
    }

    for model_name, param_list in model_hyperparams.items():
        for params in param_list:
            model = model_builders[model_name](**params)
            params_label = ", ".join([f"{k}={v}" for k, v in params.items()])

            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model),
            ])
            cv_scores = cross_validate(
                pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
            )
            results.append(
                [
                    model_name,
                    params_label,
                    "No PCA",
                    cv_scores["test_accuracy"].mean(),
                    cv_scores["test_accuracy"].std(),
                    cv_scores["test_balanced_accuracy"].mean(),
                    cv_scores["test_f1_weighted"].mean(),
                    cv_scores["test_f1_weighted"].std(),
                    cv_scores["test_f1_macro"].mean(),
                    cv_scores["fit_time"].mean(),
                ]
            )

            pipeline_pca = Pipeline([
                ("preprocessing", preprocessor),
                ("pca", PCA(n_components=0.95)),
                ("model", model),
            ])
            cv_scores_pca = cross_validate(
                pipeline_pca, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
            )
            results.append(
                [
                    model_name,
                    params_label,
                    "With PCA",
                    cv_scores_pca["test_accuracy"].mean(),
                    cv_scores_pca["test_accuracy"].std(),
                    cv_scores_pca["test_balanced_accuracy"].mean(),
                    cv_scores_pca["test_f1_weighted"].mean(),
                    cv_scores_pca["test_f1_weighted"].std(),
                    cv_scores_pca["test_f1_macro"].mean(),
                    cv_scores_pca["fit_time"].mean(),
                ]
            )

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Hyperparameters",
            "PCA",
            "Accuracy Mean",
            "Accuracy Std",
            "Balanced Accuracy Mean",
            "F1-weighted Mean",
            "F1-weighted Std",
            "F1-macro Mean",
            "Training Time Mean",
        ],
    )

    # Rank by strongest balanced performance (F1-weighted, then Accuracy),
    # with training time as a tie-breaker.
    ranked_df = results_df.sort_values(
        by=["F1-weighted Mean", "Accuracy Mean", "Training Time Mean"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_configs_df = (
        ranked_df.groupby(["Model", "PCA"], as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )

    return results_df, best_configs_df


def _rounded_for_reporting(df, digits=4):
    """Return a copy with numeric columns rounded for readability."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(digits)
    return out


if __name__ == "__main__":
    results_df, best_configs_df = run_baseline_and_pca_comparison()
    results_report_df = _rounded_for_reporting(results_df, digits=4)
    best_report_df = _rounded_for_reporting(best_configs_df, digits=4)

    print(results_report_df)
    results_report_df.to_csv("results_pca_comparison.csv", index=False)
    print("Saved as results_pca_comparison.csv")

    best_report_df.to_csv("results_pca_best_configs.csv", index=False)
    print("Saved as results_pca_best_configs.csv")
    print("\nBest configuration per model and PCA condition:")
    print(best_report_df.to_string(index=False))

