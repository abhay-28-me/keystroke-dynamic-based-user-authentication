"""
Pydantic Schemas — Pydantic v1
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# ── User ──────────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    username : str      = Field(..., min_length=3, max_length=50, example="abhay28")
    email    : EmailStr = Field(..., example="abhay@email.com")
    password : str      = Field(..., min_length=6, example="secret123")

class UserOut(BaseModel):
    id         : int
    username   : str
    email      : str
    is_active  : bool
    created_at : datetime

    class Config:
        orm_mode = True

class UserLogin(BaseModel):
    username : str = Field(..., example="abhay28")
    password : str = Field(..., example="secret123")


# ── Token ─────────────────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token : str
    token_type   : str = "bearer"


# ── Keystroke ─────────────────────────────────────────────────────────────────
class KeystrokeEnrollRequest(BaseModel):
    username  : str        = Field(..., example="abhay28")
    ikdd_file : str        = Field(..., example="user164_(3)")
    """
    The filename (without .txt) of the IKDD dataset file to use for training.
    File must be present in the data/ folder.
    """

class AuthenticateRequest(BaseModel):
    username        : str        = Field(..., example="abhay28")
    dwell_times     : List[float] = Field(..., example=[80.0, 90.0, 70.0, 60.0, 110.0])
    flight_times    : List[float] = Field(..., example=[150.0, 200.0, 180.0, 160.0])
    """
    dwell_times:  list of key hold durations (ms) from the typing session
    flight_times: list of time between consecutive key presses (ms)
    """

class AuthResult(BaseModel):
    username    : str
    authentic   : bool
    confidence  : float   # probability score 0.0 - 1.0
    message     : str

class ModelStatus(BaseModel):
    username   : str
    trained    : bool
    accuracy   : Optional[str]
    n_samples  : Optional[int]
    created_at : Optional[datetime]


# ── Typing Enrollment ─────────────────────────────────────────────────────────
class TypingSample(BaseModel):
    dwell_times  : List[float] = Field(..., example=[80.0, 90.0, 70.0])
    flight_times : List[float] = Field(..., example=[150.0, 200.0, 180.0])

class TypingEnrollRequest(BaseModel):
    username : str                = Field(..., example="abhay28")
    samples  : List[TypingSample] = Field(..., min_items=5, example=[])
    """
    Provide at least 5 typing samples of the same phrase.
    Each sample contains dwell_times and flight_times captured while typing.
    """