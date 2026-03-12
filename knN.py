import numpy as np
import pandas as pd
from collections import Counter
import time

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
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        predictions = []
        for x in X:
            distances = self._compute_distances(x)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
    
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
    
    # Train/test split (NumPy implementation)
    X_train, X_test, y_train, y_test = train_test_split_np(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (NumPy implementation)
    scaler = StandardScalerNP()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*60)
    print("kNN BASELINE (Raw Data)")
    print("="*60)
    
    # Test different k values
    k_values = [3, 5, 7, 11]
    results = []
    
    for k in k_values:
        result = train_knn_baseline(X_train_scaled, X_test_scaled, y_train, y_test, k=k)
        results.append(result)
        print(f"\nk={k}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  F1-score: {result['f1_score']:.4f}")
        print(f"  Training time: {result['train_time']:.4f}s")
        print(f"  Prediction time: {result['pred_time']:.4f}s")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))