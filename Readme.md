# 🔐 Keystroke Dynamics Authentication System

A behavioral biometrics authentication system that identifies users based on their **unique typing patterns** using machine learning. Built with **FastAPI** and **scikit-learn**, trained on the [IKDD Keystroke Dataset](https://github.com/MachineLearningVisionRG/IKDD) or your own live typing data.

---

## 🧠 How It Works

Every person types differently — the time they hold each key (**dwell time**) and the time between keystrokes (**flight time**) is unique to them. This system:

1. **Enrolls** a user by learning their keystroke timing patterns
2. **Trains** a Random Forest classifier to distinguish genuine user vs impostor
3. **Authenticates** by analyzing a new typing session and predicting if it's the same person

---

## ✨ Features

- 🔑 **Two enrollment methods** — use IKDD dataset files OR type live in the browser
- 🤖 **ML-powered** — Random Forest classifier with StandardScaler pipeline
- 🌐 **Browser-based login** — real-time keystroke capture in a sleek dark UI
- 🔒 **JWT authentication** — secure token-based login
- 🗄️ **SQLite database** — zero setup required
- 📖 **Auto API docs** — Swagger UI at `/docs`

---

## 🚀 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.11 |
| ML | scikit-learn (Random Forest), numpy, joblib |
| Database | SQLite + SQLAlchemy ORM |
| Auth | JWT (python-jose) + bcrypt (passlib) |
| Frontend | HTML, CSS, Vanilla JS |
| Server | Uvicorn (ASGI) |

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.11 (required — scikit-learn doesn't support 3.14 yet)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/keystroke-auth.git
cd keystroke-auth
```

### 2. Create & Activate Virtual Environment
```bash
python3.11 -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Add IKDD Dataset Files
If you want to use IKDD-based enrollment, place your `.txt` files in the `data/` folder:
```bash
mkdir data
# Copy your IKDD .txt files here
```

### 5. Run the Server
```bash
PYTHONPATH=$(pwd) uvicorn main:app --port 8000
```

Server starts at **http://127.0.0.1:8000**

### 6. Open the Frontend
Open `login.html` in your browser:
```bash
open login.html    # Mac
# or double-click login.html in File Explorer
```

---

## 🧪 How to Test

### Option A — Live Enrollment (No dataset needed)
1. Open `login.html` in browser
2. **Register tab** → create an account
3. **Live Enroll tab** → type `the quick brown fox` 5 times, press Enter after each
4. Click **Train My Model**
5. **Login tab** → type anything, click **Analyze & Login**
6. ✅ You should be authenticated!

### Option B — IKDD Dataset Enrollment
1. Place IKDD `.txt` files in `data/` folder
2. **Register tab** → create an account
3. **IKDD Enroll tab** → enter username + file prefix (e.g. `user164`)
4. **Login tab** → type and authenticate

### Option C — Run the automated test script
```bash
# Make sure server is running first
python3 test_api.py
```
This automatically runs the full flow and shows genuine vs impostor results.

---

## 📖 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/users/register` | Register new user |
| POST | `/users/login` | Login and get JWT token |
| POST | `/keystrokes/enroll` | Train model using IKDD dataset |
| POST | `/keystrokes/enroll-typing` | Train model from live typing samples |
| POST | `/keystrokes/authenticate` | Authenticate via keystroke pattern |
| GET | `/keystrokes/status/{username}` | Check if model is trained |

Full interactive docs: **http://127.0.0.1:8000/docs**

---

## 📁 Project Structure

```
keystroke-auth/
├── main.py                   # FastAPI entry point
├── database.py               # SQLite DB setup
├── models.py                 # SQLAlchemy models
├── schemas.py                # Pydantic schemas
├── login.html                # Frontend UI
├── test_api.py               # Automated demo script
├── requirements.txt
├── .env.example
├── data/                     # Place IKDD .txt files here
├── saved_models/             # Trained .pkl models (auto-created)
├── routers/
│   ├── users.py              # Register & login
│   └── keystrokes.py        # Enroll & authenticate
└── ml/
    ├── parser.py             # IKDD dataset parser
    ├── features.py           # Feature extraction (dwell/flight stats)
    ├── trainer.py            # IKDD-based model trainer
    ├── typing_trainer.py     # Live typing model trainer
    └── predictor.py          # Prediction engine
```

---

## 🔬 ML Details

**Features extracted:**
- Dwell time (key hold duration) — mean, std, min, max
- Flight time (time between keystrokes) — mean, std, min, max
- Raw padded timing values (top 20 dwell + 20 flight)

**Model:** Random Forest (100 trees) with StandardScaler, `class_weight='balanced'`

**Training:** Genuine user samples vs impostor samples (from other IKDD users or synthetic)

**Authentication threshold:** 0.25 confidence score (tunable in `ml/predictor.py`)

---

## 🔑 Environment Variables

Create a `.env` file (copy from `.env.example`):

```
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///./keystroke_auth.db
```

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).