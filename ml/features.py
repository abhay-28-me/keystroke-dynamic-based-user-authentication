"""
Feature Extraction
-------------------
Converts raw dwell/flight time dictionaries into a fixed-length
feature vector for ML training/prediction.

Features per entry:
  - mean
  - standard deviation
  - min
  - max

We use the TOP_N most common keys/digrams to keep feature size fixed.
"""
import numpy as np
from typing import Dict, List, Tuple

# Top N most common single keys and digrams to use as features
TOP_DWELL_KEYS  = 30
TOP_FLIGHT_KEYS = 50


def extract_features(
    dwell_times  : Dict[int, List[float]],
    flight_times : Dict[Tuple[int, int], List[float]],
    dwell_keys   : List[int],
    flight_keys  : List[Tuple[int, int]],
) -> np.ndarray:
    """
    Build a fixed-length feature vector from dwell and flight times.

    Args:
        dwell_times  : { key: [durations] }
        flight_times : { (k1,k2): [durations] }
        dwell_keys   : ordered list of key codes to use (defines feature positions)
        flight_keys  : ordered list of digrams to use (defines feature positions)

    Returns:
        1D numpy array of features
    """
    features = []

    # Dwell time features
    for key in dwell_keys:
        vals = dwell_times.get(key, [])
        features.extend(_stats(vals))

    # Flight time features
    for pair in flight_keys:
        vals = flight_times.get(pair, [])
        features.extend(_stats(vals))

    return np.array(features, dtype=np.float32)


def _stats(vals: List[float]) -> List[float]:
    """Return [mean, std, min, max] — zeros if no data."""
    if len(vals) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    arr = np.array(vals)
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
    ]


def select_top_keys(
    all_dwell  : Dict[int, List[float]],
    all_flight : Dict[Tuple[int, int], List[float]],
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Select the TOP_N most frequent keys and digrams from the dataset.
    This ensures consistent feature vectors across users.
    """
    # Sort by number of observations (most data = most reliable)
    sorted_dwell  = sorted(all_dwell.items(),  key=lambda x: len(x[1]), reverse=True)
    sorted_flight = sorted(all_flight.items(), key=lambda x: len(x[1]), reverse=True)

    dwell_keys  = [k    for k, _    in sorted_dwell[:TOP_DWELL_KEYS]]
    flight_keys = [pair for pair, _ in sorted_flight[:TOP_FLIGHT_KEYS]]

    return dwell_keys, flight_keys


def build_feature_vector(dwell_times: list, flight_times: list, size: int = 20):
    """Build a fixed-size feature vector from raw dwell and flight times."""
    import numpy as np

    def stats(vals):
        if len(vals) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        arr = np.array(vals, dtype=np.float32)
        return [float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))]

    def pad(vals, n):
        arr = list(vals)[:n]
        arr += [0.0] * (n - len(arr))
        return arr

    features = stats(dwell_times) + stats(flight_times) + pad(dwell_times, size) + pad(flight_times, size)
    return np.array(features, dtype=np.float32)