"""
Predictor — handles both IKDD-trained and typing-trained models.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from typing import List, Tuple

from ml.features import extract_features, build_feature_vector

MODELS_DIR     = "saved_models"
AUTH_THRESHOLD = 0.25


def predict_user(
    username     : str,
    dwell_times  : List[float],
    flight_times : List[float],
) -> Tuple[bool, float]:

    model_path = os.path.join(MODELS_DIR, f"{username}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for '{username}'. Please enroll first.")

    saved      = joblib.load(model_path)
    pipeline   = saved["pipeline"]
    model_type = saved.get("model_type", "ikdd")

    if model_type == "typing":
        vec = build_feature_vector(dwell_times, flight_times)
    else:
        dwell_keys  = saved["dwell_keys"]
        flight_keys = saved["flight_keys"]
        dwell_dict  = _list_to_dict(dwell_times,  dwell_keys)
        flight_dict = _list_to_dict(flight_times, flight_keys)
        vec = extract_features(dwell_dict, flight_dict, dwell_keys, flight_keys)

    vec = vec.reshape(1, -1)
    proba        = pipeline.predict_proba(vec)[0]
    confidence   = float(proba[1])
    is_authentic = confidence >= AUTH_THRESHOLD

    return is_authentic, round(confidence, 4)


def _list_to_dict(values: List[float], keys: list) -> dict:
    result = {}
    for i, key in enumerate(keys):
        if i < len(values):
            result[key] = [values[i]]
    return result