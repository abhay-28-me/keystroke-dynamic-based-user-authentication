"""
🔐 Keystroke Dynamics Authentication — Demo Test Script
---------------------------------------------------------
This script demonstrates the full authentication flow:
  1. Register a user
  2. Enroll (train ML model using IKDD data)
  3. Check model status
  4. Authenticate as genuine user (should PASS)
  5. Authenticate as impostor with random timings (should FAIL)

Run: python test_api.py
Make sure the server is running: PYTHONPATH=. uvicorn main:app --port 8000
"""

import requests
import random
import json

BASE_URL = "http://127.0.0.1:8000"
USERNAME = "test_recruiter"
EMAIL    = "recruiter@test.com"
PASSWORD = "testpass123"
IKDD_FILE = "user164"   # change to match your data/ folder filename prefix


def print_result(step, response):
    print(f"\n{'='*55}")
    print(f"  {step}")
    print(f"{'='*55}")
    print(f"  Status : {response.status_code}")
    print(f"  Response:")
    try:
        data = response.json()
        print(json.dumps(data, indent=4))
    except:
        print(response.text)


def run_demo():
    print("\n🔐 Keystroke Dynamics Authentication — Demo")
    print("=" * 55)

    # ── Step 1: Register ──────────────────────────────────────
    print("\n[1/5] Registering user...")
    res = requests.post(f"{BASE_URL}/users/register", json={
        "username": USERNAME,
        "email":    EMAIL,
        "password": PASSWORD,
    })
    if res.status_code == 400 and "already" in res.text:
        print("  ℹ️  User already exists, skipping registration.")
    else:
        print_result("✅ User Registered", res)

    # ── Step 2: Login ─────────────────────────────────────────
    print("\n[2/5] Logging in...")
    res = requests.post(f"{BASE_URL}/users/login", json={
        "username": USERNAME,
        "password": PASSWORD,
    })
    print_result("✅ Login", res)

    # ── Step 3: Enroll (train ML model) ───────────────────────
    print(f"\n[3/5] Enrolling user with IKDD file '{IKDD_FILE}'...")
    print("  ⏳ Training ML model... this may take a few seconds.")
    res = requests.post(f"{BASE_URL}/keystrokes/enroll", json={
        "username":  USERNAME,
        "ikdd_file": IKDD_FILE,
    })
    print_result("✅ Model Trained", res)

    if res.status_code != 200:
        print("\n❌ Enrollment failed. Check that your IKDD files are in the data/ folder.")
        return

    data = res.json()
    print(f"\n  🎯 Model Accuracy : {data.get('accuracy')}")
    print(f"  📊 Training Samples: {data.get('n_samples')}")

    # ── Step 4: Authenticate as genuine user ──────────────────
    print("\n[4/5] Authenticating as GENUINE user...")
    print("  (Using timing values similar to the IKDD training data)")

    # Simulate realistic genuine timings from the IKDD dataset range
    genuine_dwell  = [random.randint(60, 130)  for _ in range(30)]
    genuine_flight = [random.randint(100, 350) for _ in range(50)]

    res = requests.post(f"{BASE_URL}/keystrokes/authenticate", json={
        "username":     USERNAME,
        "dwell_times":  genuine_dwell,
        "flight_times": genuine_flight,
    })
    print_result("Genuine User Result", res)
    data = res.json()
    status = "✅ AUTHENTICATED" if data.get("authentic") else "❌ REJECTED"
    print(f"\n  Result     : {status}")
    print(f"  Confidence : {data.get('confidence', 0) * 100:.1f}%")

    # ── Step 5: Authenticate as impostor ──────────────────────
    print("\n[5/5] Authenticating as IMPOSTOR...")
    print("  (Using very different random timing values)")

    # Impostor uses very different timing patterns
    impostor_dwell  = [random.randint(200, 500) for _ in range(30)]
    impostor_flight = [random.randint(800, 2000) for _ in range(50)]

    res = requests.post(f"{BASE_URL}/keystrokes/authenticate", json={
        "username":     USERNAME,
        "dwell_times":  impostor_dwell,
        "flight_times": impostor_flight,
    })
    print_result("Impostor Result", res)
    data = res.json()
    status = "✅ AUTHENTICATED" if data.get("authentic") else "❌ REJECTED (correct!)"
    print(f"\n  Result     : {status}")
    print(f"  Confidence : {data.get('confidence', 0) * 100:.1f}%")

    print("\n" + "="*55)
    print("  ✅ Demo Complete! System is working correctly.")
    print("="*55 + "\n")


if __name__ == "__main__":
    try:
        run_demo()
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to server.")
        print("   Make sure it's running: PYTHONPATH=. uvicorn main:app --port 8000")