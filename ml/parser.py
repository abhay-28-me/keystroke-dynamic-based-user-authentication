"""
IKDD Dataset Parser
--------------------
Parses .txt files from the IKDD keystroke dataset.

File format:
  Line 1  : metadata (username, gender, age, ...)
  x-0, v1, v2, ... : dwell times for key x (how long key x was held, in ms)
  x-y, v1, v2, ... : flight times from key x to key y (time between presses, in ms)
  x-0 or x-y with no values : that key/digram was never recorded
"""
import os
from typing import Dict, List, Tuple


def parse_ikdd_file(filepath: str) -> Tuple[Dict[int, List[float]], Dict[Tuple[int,int], List[float]]]:
    """
    Parse an IKDD .txt file.

    Returns:
        dwell_times  : { key_code: [durations in ms] }
        flight_times : { (key1, key2): [durations in ms] }
    """
    dwell_times  : Dict[int, List[float]]            = {}
    flight_times : Dict[Tuple[int, int], List[float]] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if i == 0:
            # Skip metadata line
            continue

        parts = line.split(",")
        key_pair = parts[0].strip()   # e.g. "32-0" or "32-65"
        values   = [float(v.strip()) for v in parts[1:] if v.strip()]

        if not values:
            continue

        keys = key_pair.split("-")
        k1, k2 = int(keys[0]), int(keys[1])

        if k2 == 0:
            # Dwell time for key k1
            dwell_times[k1] = dwell_times.get(k1, []) + values
        else:
            # Flight time from key k1 to key k2
            pair = (k1, k2)
            flight_times[pair] = flight_times.get(pair, []) + values

    return dwell_times, flight_times


def get_all_user_files(data_dir: str, username_prefix: str) -> List[str]:
    """
    Find all IKDD files for a given user prefix in the data directory.
    e.g. username_prefix = "user164" matches user164_(1).txt, user164_(2).txt, etc.
    """
    matched = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt") and username_prefix in fname:
            matched.append(os.path.join(data_dir, fname))
    return matched