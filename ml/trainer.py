"""
Model Trainer
--------------
Trains a Random Forest classifier for a specific user using their
IKDD dataset files as genuine samples and other users as impostors.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from typing import List, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from ml.parser import parse_ikdd_file, get_all_user_files
from ml.features import extract_features, select_top_keys

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_user_model(
    username       : str,
    user_files     : List[str],
    impostor_files : List[str],
    data_dir       : str,
) -> Tuple[str, float, int]:
    all_dwell  : Dict = {}
    all_flight : Dict = {}

    for fpath in user_files:
        dwell, flight = parse_ikdd_file(fpath)
        for k, v in dwell.items():
            all_dwell[k]  = all_dwell.get(k, [])  + v
        for k, v in flight.items():
            all_flight[k] = all_flight.get(k, []) + v

    dwell_keys, flight_keys = select_top_keys(all_dwell, all_flight)

    X, y = [], []

    for fpath in user_files:
        dwell, flight = parse_ikdd_file(fpath)
        vec = extract_features(dwell, flight, dwell_keys, flight_keys)
        X.append(vec)
        y.append(1)

    for fpath in impostor_files[:len(user_files) * 3]:
        try:
            dwell, flight = parse_ikdd_file(fpath)
            vec = extract_features(dwell, flight, dwell_keys, flight_keys)
            X.append(vec)
            y.append(0)
        except Exception:
            continue

    X = np.array(X)
    y = np.array(y)
    n_samples = len(X)

    if len(X) < 4:
        raise ValueError("Not enough samples to train. Provide more IKDD files.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    y_pred   = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    model_path = os.path.join(MODELS_DIR, f"{username}.pkl")
    joblib.dump({
        "pipeline"    : pipeline,
        "dwell_keys"  : dwell_keys,
        "flight_keys" : flight_keys,
    }, model_path)

    return model_path, round(accuracy * 100, 2), n_samples