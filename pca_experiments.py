import time
import os
import numpy as np
import pandas as pd

from knN import (
    MushroomDataPreprocessor,
    train_test_split_np,
    StandardScalerNP,
    KNearestNeighbors,
    f1_score_weighted,
)

# PCA from sklearn; install with:
#   pip install scikit-learn
from sklearn.decomposition import PCA


def run_pca_experiments(n_components_list):
    # Load and preprocess data
    preprocessor = MushroomDataPreprocessor("MushroomDataset/secondary_data.csv")
    preprocessor.preprocess()
    X, y = preprocessor.get_processed_data()

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split_np(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScalerNP()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    all_results = []

    print("\n" + "=" * 60)
    print("PCA EXPERIMENTS")
    print("=" * 60)

    for n_components in n_components_list:
        print(f"\n--- PCA with n_components = {n_components} ---")
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # Record explained variance ratio (for scree plot later if needed)
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"  Explained variance (cumulative): {explained_variance:.4f}")

        # Custom kNN (NumPy implementation) on PCA-reduced data
        for k in [3, 5, 7]:
            knn = KNearestNeighbors(k=k, metric="euclidean")
            start_time = time.time()
            knn.fit(X_train_pca, y_train)
            train_time = time.time() - start_time

            start_time = time.time()
            y_pred = knn.predict(X_test_pca)
            pred_time = time.time() - start_time

            accuracy = np.mean(y_pred == y_test)
            f1 = f1_score_weighted(y_test, y_pred)

            res_knn = {
                "model": f"kNN_PCA_k={k}",
                "accuracy": accuracy,
                "f1_score": f1,
                "train_time": train_time,
                "pred_time": pred_time,
                "n_features": X_train_pca.shape[1],
                "n_components": n_components,
            }
            all_results.append(res_knn)

    results_df = pd.DataFrame(all_results)
    return results_df


if __name__ == "__main__":
    # Choose a range of PCA dimensions to test
    n_components_list = [2, 5, 10, 20, 50]

    results_df = run_pca_experiments(n_components_list)

    print("\n" + "=" * 60)
    print("SUMMARY TABLE: PCA vs Models")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Optionally, save to CSV for use in your report
    os.makedirs("results/initial", exist_ok=True)
    results_df.to_csv("results/initial/pca_results.csv", index=False)

