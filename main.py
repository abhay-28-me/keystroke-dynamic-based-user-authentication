"""
Keystroke Dynamics Authentication System — FastAPI Backend
-----------------------------------------------------------
Start: PYTHONPATH=. uvicorn main:app --reload --port 8000
Docs:  http://localhost:8000/docs
"""
import sys, os

# Add project root to Python path — fixes 'ml' module not found
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from routers import users, keystrokes

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title       = "🔐 Keystroke Dynamics Authentication",
    description = "Authenticate users based on their unique typing patterns using the IKDD dataset and a Random Forest classifier.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.include_router(users.router)
app.include_router(keystrokes.router)


@app.get("/", tags=["Health"])
def root():
    return {
        "service" : "Keystroke Dynamics Authentication API",
        "status"  : "online",
        "version" : "1.0.0",
        "docs"    : "/docs",
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}