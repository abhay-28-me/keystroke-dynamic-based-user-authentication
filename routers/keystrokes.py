"""
Keystrokes Router — Enroll & Authenticate
"""
import sys, os

# Add project root to path so 'ml' package can be found
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from models import User, TrainedModel
from schemas import KeystrokeEnrollRequest, AuthenticateRequest, AuthResult, ModelStatus, TypingEnrollRequest
from ml.trainer   import train_user_model
from ml.predictor import predict_user

router   = APIRouter(prefix="/keystrokes", tags=["Keystrokes"])
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@router.post("/enroll", response_model=ModelStatus)
def enroll(payload: KeystrokeEnrollRequest, db: Session = Depends(get_db)):
    """
    Train a keystroke model for the user using their IKDD data file(s).
    The ikdd_file field should be the filename prefix, e.g. 'user164'.
    All matching files in data/ folder will be used for training.
    All other files will be used as impostors.
    """
    # Check user exists
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found. Please register first.")

    # Find genuine user files
    if not os.path.exists(DATA_DIR):
        raise HTTPException(status_code=500, detail=f"Data directory not found: {DATA_DIR}")

    all_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".txt")
    ]

    user_files = [f for f in all_files if payload.ikdd_file in os.path.basename(f)]
    if not user_files:
        raise HTTPException(
            status_code=404,
            detail=f"No IKDD files found matching '{payload.ikdd_file}' in data/ folder."
        )

    impostor_files = [f for f in all_files if f not in user_files]

    try:
        model_path, accuracy, n_samples = train_user_model(
            username       = payload.username,
            user_files     = user_files,
            impostor_files = impostor_files,
            data_dir       = DATA_DIR,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save or update trained model record in DB
    record = db.query(TrainedModel).filter(TrainedModel.username == payload.username).first()
    if record:
        record.model_path = model_path
        record.accuracy   = f"{accuracy}%"
        record.n_samples  = n_samples
    else:
        record = TrainedModel(
            username   = payload.username,
            model_path = model_path,
            accuracy   = f"{accuracy}%",
            n_samples  = n_samples,
        )
        db.add(record)
    db.commit()
    db.refresh(record)

    return ModelStatus(
        username   = payload.username,
        trained    = True,
        accuracy   = record.accuracy,
        n_samples  = record.n_samples,
        created_at = record.created_at,
    )


@router.post("/authenticate", response_model=AuthResult)
def authenticate(payload: AuthenticateRequest, db: Session = Depends(get_db)):
    """
    Authenticate a user based on their keystroke dynamics.
    Send raw dwell_times and flight_times from a typing session.
    """
    # Check model exists
    record = db.query(TrainedModel).filter(TrainedModel.username == payload.username).first()
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for '{payload.username}'. Please enroll first."
        )

    try:
        is_authentic, confidence = predict_user(
            username     = payload.username,
            dwell_times  = payload.dwell_times,
            flight_times = payload.flight_times,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return AuthResult(
        username   = payload.username,
        authentic  = is_authentic,
        confidence = confidence,
        message    = (
            f"✅ Authenticated! Confidence: {confidence:.0%}"
            if is_authentic else
            f"❌ Authentication failed. Confidence: {confidence:.0%}"
        )
    )


@router.get("/status/{username}", response_model=ModelStatus)
def model_status(username: str, db: Session = Depends(get_db)):
    """Check if a user has a trained keystroke model."""
    record = db.query(TrainedModel).filter(TrainedModel.username == username).first()
    if not record:
        return ModelStatus(username=username, trained=False, accuracy=None, n_samples=None, created_at=None)
    return ModelStatus(
        username   = username,
        trained    = True,
        accuracy   = record.accuracy,
        n_samples  = record.n_samples,
        created_at = record.created_at,
    )


# ── Enroll by Typing ──────────────────────────────────────────────────────────
@router.post("/enroll-typing", response_model=ModelStatus)
def enroll_by_typing(payload: TypingEnrollRequest, db: Session = Depends(get_db)):
    from ml.typing_trainer import train_from_typing

    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found. Please register first.")

    samples = [(s.dwell_times, s.flight_times) for s in payload.samples]

    try:
        model_path, accuracy, n_samples = train_from_typing(
            username = payload.username,
            samples  = samples,
            data_dir = DATA_DIR,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    record = db.query(TrainedModel).filter(TrainedModel.username == payload.username).first()
    if record:
        record.model_path = model_path
        record.accuracy   = f"{accuracy}%"
        record.n_samples  = n_samples
    else:
        record = TrainedModel(
            username   = payload.username,
            model_path = model_path,
            accuracy   = f"{accuracy}%",
            n_samples  = n_samples,
        )
        db.add(record)
    db.commit()
    db.refresh(record)

    return ModelStatus(
        username   = payload.username,
        trained    = True,
        accuracy   = record.accuracy,
        n_samples  = record.n_samples,
        created_at = record.created_at,
    )