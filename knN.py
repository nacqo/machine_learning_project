import numpy as np
import pandas as pd
from collections import Counter
import time
import os

class KNearestNeighbors:
    """
    k-Nearest Neighbors classifier implemented with NumPy.
    """
    
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data (lazy learning)."""
        self.X_train = X
        self.y_train = y
        return self
    
    def _compute_distances(self, x):
        """Compute distances from a single sample to all training samples."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def predict(self, X, batch_size=512):
        """
        Predict class labels for samples in X.

        Uses a vectorized distance computation for Euclidean metric to avoid
        extremely slow Python loops on larger datasets.
        """
        X = np.asarray(X, dtype=float)

        if self.metric != "euclidean":
            predictions = []
            for x in X:
                distances = self._compute_distances(x)
                k_indices = np.argsort(distances)[: self.k]
                k_labels = self.y_train[k_indices]
                most_common = Counter(k_labels).most_common(1)[0][0]
                predictions.append(most_common)
            return np.array(predictions)

        X_train = np.asarray(self.X_train, dtype=float)
        y_train = np.asarray(self.y_train)
        k = int(self.k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        # Precompute squared norms for fast (a-b)^2 expansion.
        train_sq = np.sum(X_train * X_train, axis=1)  # (n_train,)

        preds = np.empty(X.shape[0], dtype=y_train.dtype)
        n = X.shape[0]
        bs = int(batch_size) if batch_size is not None else n
        bs = max(1, bs)

        # For voting we assume labels are integer-like (as in this project).
        labels = np.unique(y_train)
        label_to_pos = {lbl: i for i, lbl in enumerate(labels)}

        for start in range(0, n, bs):
            end = min(n, start + bs)
            Xb = X[start:end]
            batch_sq = np.sum(Xb * Xb, axis=1)  # (batch,)

            # Squared Euclidean distances:
            # ||x - t||^2 = ||x||^2 + ||t||^2 - 2 x·t
            d2 = batch_sq[:, None] + train_sq[None, :] - 2.0 * (Xb @ X_train.T)

            # Get k nearest indices without full sort.
            nn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]  # (batch, k)
            nn_labels = y_train[nn_idx]  # (batch, k)

            # Majority vote (small k -> tiny inner loop OK).
            for i in range(nn_labels.shape[0]):
                counts = np.zeros(len(labels), dtype=int)
                for lbl in nn_labels[i]:
                    counts[label_to_pos[lbl]] += 1
                preds[start + i] = labels[int(np.argmax(counts))]

        return preds
    
    def score(self, X, y):
        """Calculate accuracy on test set."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


def train_test_split_np(X, y, test_size=0.2, random_state=None, stratify=None):
    """
    Simple NumPy implementation of train_test_split.
    Supports optional stratification on the label vector.
    """
    rng = np.random.RandomState(random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    if stratify is None:
        indices = np.arange(len(y))
        rng.shuffle(indices)
        n_test = int(len(y) * test_size)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
    else:
        stratify = np.asarray(stratify)
        unique_classes = np.unique(stratify)
        train_indices = []
        test_indices = []
        for cls in unique_classes:
            cls_idx = np.where(stratify == cls)[0]
            rng.shuffle(cls_idx)
            n_test_cls = int(len(cls_idx) * test_size)
            test_indices.extend(cls_idx[:n_test_cls])
            train_indices.extend(cls_idx[n_test_cls:])
        train_idx = np.array(train_indices)
        test_idx = np.array(test_indices)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_val_test_split_np(
    X,
    y,
    val_size=0.2,
    test_size=0.2,
    random_state=None,
    stratify=None,
):
    """
    Split arrays into train/val/test using NumPy only.

    - val_size and test_size are fractions of the full dataset.
    - Supports optional stratification on a label vector (typically y).
    """
    if val_size < 0 or test_size < 0 or (val_size + test_size) >= 1:
        raise ValueError("val_size and test_size must be >=0 and sum to < 1.")

    rng = np.random.RandomState(random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    n = len(y)
    n_val = int(round(n * val_size))
    n_test = int(round(n * test_size))

    if stratify is None:
        indices = np.arange(n)
        rng.shuffle(indices)
        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]
    else:
        stratify = np.asarray(stratify)
        unique_classes = np.unique(stratify)

        train_indices = []
        val_indices = []
        test_indices = []

        for cls in unique_classes:
            cls_idx = np.where(stratify == cls)[0]
            rng.shuffle(cls_idx)

            n_cls = len(cls_idx)
            n_test_cls = int(round(n_cls * test_size))
            n_val_cls = int(round(n_cls * val_size))

            test_indices.extend(cls_idx[:n_test_cls])
            val_indices.extend(cls_idx[n_test_cls : n_test_cls + n_val_cls])
            train_indices.extend(cls_idx[n_test_cls + n_val_cls :])

        train_idx = np.array(train_indices, dtype=int)
        val_idx = np.array(val_indices, dtype=int)
        test_idx = np.array(test_indices, dtype=int)

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

    return (
        X[train_idx],
        X[val_idx],
        X[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
    )


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    # Numerically stable sigmoid
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def _softmax(logits):
    logits = np.asarray(logits, dtype=float)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class LogisticRegressionGD:
    """
    Logistic regression trained with (mini-)batch gradient descent (NumPy only).

    Supports:
      - binary classification (sigmoid + log-loss)
      - multiclass classification (softmax + cross-entropy)
    """

    def __init__(
        self,
        lr=0.1,
        n_epochs=300,
        batch_size=None,
        l2=0.0,
        fit_intercept=True,
        random_state=None,
    ):
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.batch_size = batch_size  # None => full batch
        self.l2 = float(l2)
        self.fit_intercept = bool(fit_intercept)
        self.random_state = random_state

        self.classes_ = None
        self.W_ = None
        self.b_ = None
        self.history_ = None

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        return np.hstack([np.ones((X.shape[0], 1), dtype=float), X])

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        Xb = self._add_intercept(X)
        n_samples, n_features = Xb.shape

        bs = self.batch_size if self.batch_size is not None else n_samples
        bs = int(bs)
        if bs <= 0:
            raise ValueError("batch_size must be a positive int or None.")

        self.history_ = {"epoch": []}

        if n_classes == 2:
            # Map to {0,1} in class order.
            y01 = (y == self.classes_[1]).astype(float)
            self.W_ = np.zeros(n_features, dtype=float)

            for epoch in range(1, self.n_epochs + 1):
                indices = np.arange(n_samples)
                rng.shuffle(indices)

                for start in range(0, n_samples, bs):
                    batch_idx = indices[start : start + bs]
                    X_batch = Xb[batch_idx]
                    y_batch = y01[batch_idx]

                    logits = X_batch @ self.W_
                    p = _sigmoid(logits)
                    grad = (X_batch.T @ (p - y_batch)) / len(batch_idx)

                    # L2 regularization (skip intercept term)
                    if self.l2 > 0:
                        reg = self.W_.copy()
                        if self.fit_intercept:
                            reg[0] = 0.0
                        grad += self.l2 * reg

                    self.W_ -= self.lr * grad

                self.history_["epoch"].append(epoch)
        else:
            # Multiclass softmax: W shape (n_features, n_classes)
            class_to_index = {c: i for i, c in enumerate(self.classes_)}
            y_idx = np.vectorize(class_to_index.get)(y).astype(int)
            Y = np.eye(n_classes, dtype=float)[y_idx]  # (n_samples, n_classes)
            self.W_ = np.zeros((n_features, n_classes), dtype=float)

            for epoch in range(1, self.n_epochs + 1):
                indices = np.arange(n_samples)
                rng.shuffle(indices)

                for start in range(0, n_samples, bs):
                    batch_idx = indices[start : start + bs]
                    X_batch = Xb[batch_idx]
                    Y_batch = Y[batch_idx]

                    logits = X_batch @ self.W_
                    P = _softmax(logits)
                    grad = (X_batch.T @ (P - Y_batch)) / len(batch_idx)

                    if self.l2 > 0:
                        reg = self.W_.copy()
                        if self.fit_intercept:
                            reg[0, :] = 0.0
                        grad += self.l2 * reg

                    self.W_ -= self.lr * grad

                self.history_["epoch"].append(epoch)

        return self

    def loss(self, X, y):
        """
        Average cross-entropy loss (with optional L2 penalty).
        """
        if self.W_ is None or self.classes_ is None:
            raise ValueError("Model is not fitted.")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        Xb = self._add_intercept(X)
        n = Xb.shape[0]

        if self.W_.ndim == 1:
            y01 = (y == self.classes_[1]).astype(float)
            p = _sigmoid(Xb @ self.W_)
            eps = 1e-12
            p = np.clip(p, eps, 1 - eps)
            data_loss = -np.mean(y01 * np.log(p) + (1 - y01) * np.log(1 - p))
            reg = 0.0
            if self.l2 > 0:
                w = self.W_.copy()
                if self.fit_intercept:
                    w[0] = 0.0
                reg = 0.5 * self.l2 * np.sum(w * w)
            return float(data_loss + reg)

        # Multiclass
        class_to_index = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.vectorize(class_to_index.get)(y).astype(int)
        P = _softmax(Xb @ self.W_)
        eps = 1e-12
        P = np.clip(P, eps, 1 - eps)
        data_loss = -np.mean(np.log(P[np.arange(n), y_idx]))
        reg = 0.0
        if self.l2 > 0:
            W = self.W_.copy()
            if self.fit_intercept:
                W[0, :] = 0.0
            reg = 0.5 * self.l2 * np.sum(W * W)
        return float(data_loss + reg)

    def predict_proba(self, X):
        if self.W_ is None or self.classes_ is None:
            raise ValueError("Model is not fitted.")
        X = np.asarray(X, dtype=float)
        Xb = self._add_intercept(X)

        if self.W_.ndim == 1:
            p1 = _sigmoid(Xb @ self.W_)
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        P = _softmax(Xb @ self.W_)
        return P

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


def accuracy_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def train_logreg_gd(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    lr_grid=(0.1, 0.05),
    l2_grid=(0.0, 1e-4),
    n_epochs=80,
    batch_size=4096,
    random_state=42,
):
    """
    Tune logistic regression (GD) on validation set, then evaluate on test.
    Returns dict with best hyperparams + metrics.
    """
    best = None
    best_val = -np.inf

    for lr in lr_grid:
        for l2 in l2_grid:
            start = time.time()
            clf = LogisticRegressionGD(
                lr=lr,
                n_epochs=n_epochs,
                batch_size=batch_size,
                l2=l2,
                fit_intercept=True,
                random_state=random_state,
            )
            clf.fit(X_train, y_train)
            train_time = time.time() - start

            y_val_pred = clf.predict(X_val)
            val_acc = accuracy_np(y_val, y_val_pred)

            if val_acc > best_val:
                best_val = val_acc
                best = (clf, lr, l2, train_time)

    clf, lr, l2, train_time = best
    y_test_pred = clf.predict(X_test)

    return {
        "model": "LogReg_GD",
        "lr": lr,
        "l2": l2,
        "n_epochs": int(n_epochs),
        "batch_size": int(batch_size) if batch_size is not None else None,
        "val_accuracy": float(best_val),
        "test_accuracy": accuracy_np(y_test, y_test_pred),
        "test_f1_score": f1_score_weighted(y_test, y_test_pred),
        "train_time": float(train_time),
        "n_features": int(X_train.shape[1]),
    }


class StandardScalerNP:
    """
    Simple standardization using NumPy only.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def f1_score_weighted(y_true, y_pred):
    """
    Weighted F1-score implementation using NumPy only.
    Works for binary or multiclass labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(y_true)
    f1_scores = []
    weights = []

    for lbl in labels:
        tp = np.sum((y_true == lbl) & (y_pred == lbl))
        fp = np.sum((y_true != lbl) & (y_pred == lbl))
        fn = np.sum((y_true == lbl) & (y_pred != lbl))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1)
        weights.append(np.sum(y_true == lbl))

    f1_scores = np.array(f1_scores)
    weights = np.array(weights, dtype=float)
    return np.average(f1_scores, weights=weights)


class MushroomDataPreprocessor:
    """
    Load and preprocess the Secondary Mushroom dataset.
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.encoder_dicts = {}
    
    def load_data(self):
        """Load CSV with semicolon delimiter."""
        self.df = pd.read_csv(self.filepath, delimiter=';')
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        return self
    
    def handle_missing_values(self):
        """
        Handle missing values (empty strings represented as NaN).
        Strategy: Drop rows with missing target; fill features with 'unknown'.
        """
        # Target is 'class' column
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=['class'])
        print(f"Removed {initial_rows - len(self.df)} rows with missing target")
        
        # Fill missing feature values with 'unknown'
        self.df = self.df.fillna('unknown')
        print(f"Filled missing feature values with 'unknown'")
        return self
    
    def encode_categorical_features(self):
        """
        One-hot encode categorical features.
        Separate target (class) from features.
        """
        # Separate target and features
        self.y = self.df['class'].copy()
        X_temp = self.df.drop('class', axis=1)
        
        # One-hot encode all columns (they're all categorical)
        X_encoded = pd.get_dummies(X_temp, drop_first=False)
        
        self.feature_names = X_encoded.columns.tolist()
        self.X = X_encoded.values
        
        print(f"After one-hot encoding: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        return self
    
    def encode_target(self):
        """Encode target labels to integers."""
        unique_classes = self.y.unique()
        self.class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.y = self.y.map(self.class_mapping).values
        print(f"Target classes: {self.class_mapping}")
        return self
    
    def get_processed_data(self):
        """Return processed X and y."""
        return self.X, self.y
    
    def preprocess(self):
        """Run full preprocessing pipeline."""
        self.load_data()
        self.handle_missing_values()
        self.encode_categorical_features()
        self.encode_target()
        return self


def train_knn_baseline(X_train, X_test, y_train, y_test, k=5):
    """
    Train kNN on raw data and record metrics.
    
    Returns:
        dict: Contains accuracy, training time
    """
    start_time = time.time()
    knn = KNearestNeighbors(k=k, metric='euclidean')
    knn.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict and evaluate
    start_time = time.time()
    y_pred = knn.predict(X_test)
    pred_time = time.time() - start_time
    
    accuracy = np.mean(y_pred == y_test)
    
    # F1-score (weighted, handles binary or multiclass)
    f1 = f1_score_weighted(y_test, y_pred)
    
    return {
        'model': 'kNN',
        'k': k,
        'accuracy': accuracy,
        'f1_score': f1,
        'train_time': train_time,
        'pred_time': pred_time,
        'n_features': X_train.shape[1]
    }


if __name__ == '__main__':
    # Load and preprocess data
    preprocessor = MushroomDataPreprocessor('MushroomDataset/secondary_data.csv')
    preprocessor.preprocess()
    X, y = preprocessor.get_processed_data()

    # Train/val/test split (NumPy implementation)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_np(
        X, y, val_size=0.2, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (fit on train only; transform val/test)
    scaler = StandardScalerNP()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*60)
    print("kNN (tuned on validation, tested once)")
    print("="*60)
    
    # Test different k values
    k_values = [3, 5, 7, 11]
    best_k = None
    best_val_acc = -np.inf

    for k in k_values:
        knn = KNearestNeighbors(k=k, metric='euclidean')
        knn.fit(X_train_scaled, y_train)
        y_val_pred = knn.predict(X_val_scaled)
        val_acc = np.mean(y_val_pred == y_val)
        print(f"  k={k}: val_accuracy={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k

    # Retrain on train+val, evaluate on test
    X_trainval = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = np.concatenate([y_train, y_val])

    start_time = time.time()
    knn = KNearestNeighbors(k=best_k, metric='euclidean')
    knn.fit(X_trainval, y_trainval)
    train_time = time.time() - start_time

    start_time = time.time()
    y_test_pred = knn.predict(X_test_scaled)
    pred_time = time.time() - start_time

    test_acc = np.mean(y_test_pred == y_test)
    test_f1 = f1_score_weighted(y_test, y_test_pred)

    print(f"\nBest k={best_k} (val_accuracy={best_val_acc:.4f})")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print(f"Training time: {train_time:.4f}s")
    print(f"Prediction time: {pred_time:.4f}s")

    print("\n" + "="*60)
    print("Logistic Regression (Gradient Descent, tuned on validation)")
    print("="*60)

    gd_result = train_logreg_gd(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        lr_grid=(0.1, 0.05),
        l2_grid=(0.0, 1e-4),
        n_epochs=80,
        batch_size=4096,
        random_state=42,
    )
    print(pd.DataFrame([gd_result]).to_string(index=False))

    # Train best GD model again to capture learning curves for visualization
    best_lr = float(gd_result["lr"])
    best_l2 = float(gd_result["l2"])
    n_epochs = int(gd_result["n_epochs"])
    batch_size = int(gd_result["batch_size"]) if gd_result["batch_size"] is not None else None

    clf = LogisticRegressionGD(
        lr=best_lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        l2=best_l2,
        fit_intercept=True,
        random_state=42,
    )

    start_time = time.time()
    clf.fit(X_train_scaled, y_train)
    gd_curve_train_time = time.time() - start_time

    curve_rows = []
    # Compute loss/accuracy per epoch (uses final weights after each epoch)
    # We stored only epoch indices; recompute metrics after fit for each epoch isn't possible
    # without storing weights, so we record final train/val/test summary plus final loss values.
    # (Keeping it lightweight for this project.)
    curve_rows.append(
        {
            "lr": best_lr,
            "l2": best_l2,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "final_train_loss": clf.loss(X_train_scaled, y_train),
            "final_val_loss": clf.loss(X_val_scaled, y_val),
            "final_test_loss": clf.loss(X_test_scaled, y_test),
            "train_accuracy": accuracy_np(y_train, clf.predict(X_train_scaled)),
            "val_accuracy": accuracy_np(y_val, clf.predict(X_val_scaled)),
            "test_accuracy": accuracy_np(y_test, clf.predict(X_test_scaled)),
            "train_time_seconds": gd_curve_train_time,
        }
    )

    os.makedirs("results/initial", exist_ok=True)
    gd_metrics_path = os.path.join("results/initial", "gd_metrics.csv")
    pd.DataFrame(curve_rows).to_csv(gd_metrics_path, index=False)
    print(f"Saved {gd_metrics_path}")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    results_df = pd.DataFrame(
        [
            {
                "model": "kNN",
                "k": best_k,
                "val_accuracy": float(best_val_acc),
                "test_accuracy": float(test_acc),
                "test_f1_score": float(test_f1),
                "train_time": float(train_time),
                "pred_time": float(pred_time),
                "n_features": int(X_train.shape[1]),
            },
            gd_result,
        ]
    )
    print(results_df.to_string(index=False))