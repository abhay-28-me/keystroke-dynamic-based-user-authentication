"""
Typing Trainer
---------------
Trains a model from the user's own live typing samples.
Uses synthetic impostors (very different timing patterns) when
no IKDD files are available.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)


def build_feature_vector(dwell_times: List[float], flight_times: List[float], size: int = 20) -> np.ndarray:
    """
    Build a fixed-size feature vector from raw dwell and flight times.
    Uses statistical features: mean, std, min, max + padded raw values.
    """
    def stats(vals):
        if len(vals) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        arr = np.array(vals, dtype=np.float32)
        return [float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))]

    def pad(vals, n):
        arr = list(vals)[:n]
        arr += [0.0] * (n - len(arr))
        return arr

    features = (
        stats(dwell_times) +
        stats(flight_times) +
        pad(dwell_times, size) +
        pad(flight_times, size)
    )
    return np.array(features, dtype=np.float32)


def generate_impostors(genuine_samples: List[np.ndarray], n: int = 30) -> List[np.ndarray]:
    """
    Generate synthetic impostor samples by creating timing patterns
    that are very different from the genuine user's patterns.
    """
    # Get the genuine user's mean dwell and flight times
    all_features = np.array(genuine_samples)
    genuine_mean = np.mean(all_features, axis=0)
    genuine_std  = np.std(all_features, axis=0) + 1e-6

    impostors = []
    for _ in range(n):
        # Impostors have very different timing — either much faster or much slower
        direction = np.random.choice([-1, 1], size=genuine_mean.shape)
        noise     = np.random.uniform(2.0, 4.0, size=genuine_mean.shape)
        impostor  = genuine_mean + direction * noise * genuine_std
        impostor  = np.clip(impostor, 0, None)
        impostors.append(impostor)

    return impostors


def train_from_typing(
    username : str,
    samples  : List[Tuple[List[float], List[float]]],  # list of (dwell_times, flight_times)
    data_dir : str = "data",
) -> Tuple[str, float, int]:
    """
    Train a model from the user's own typing samples.

    Args:
        username : user's username
        samples  : list of (dwell_times, flight_times) tuples

    Returns:
        (model_path, accuracy, n_samples)
    """
    if len(samples) < 5:
        raise ValueError("Please provide at least 5 typing samples.")

    # ── Build genuine feature vectors ──────────────────────────────────────
    genuine_vecs = []
    for dwell, flight in samples:
        vec = build_feature_vector(dwell, flight)
        genuine_vecs.append(vec)

    # ── Build impostor samples ─────────────────────────────────────────────
    # First try IKDD files if available
    impostor_vecs = []

    if os.path.exists(data_dir) and os.listdir(data_dir):
        try:
            from ml.parser import parse_ikdd_file
            from ml.features import extract_features

            ikdd_files = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".txt")
            ][:20]

            for fpath in ikdd_files:
                try:
                    dwell_d, flight_d = parse_ikdd_file(fpath)
                    all_dwell  = [v for vals in dwell_d.values()  for v in vals]
                    all_flight = [v for vals in flight_d.values() for v in vals]
                    if all_dwell:
                        vec = build_feature_vector(all_dwell[:50], all_flight[:50])
                        impostor_vecs.append(vec)
                except Exception:
                    continue
        except Exception:
            pass

    # Fall back to synthetic impostors if not enough IKDD data
    if len(impostor_vecs) < 10:
        impostor_vecs += generate_impostors(genuine_vecs, n=max(30, len(genuine_vecs) * 5))

    # ── Combine and train ──────────────────────────────────────────────────
    X = np.array(genuine_vecs + impostor_vecs)
    y = np.array([1] * len(genuine_vecs) + [0] * len(impostor_vecs))
    n_samples = len(X)

    if len(X) < 4:
        raise ValueError("Not enough data to train.")

    # Use stratified split only if both classes have enough samples
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )),
    ])
    pipeline.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, pipeline.predict(X_test))

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{username}.pkl")
    joblib.dump({
        "pipeline"    : pipeline,
        "model_type"  : "typing",   # flag: trained from live typing
    }, model_path)

    return model_path, round(accuracy * 100, 2), n_samples