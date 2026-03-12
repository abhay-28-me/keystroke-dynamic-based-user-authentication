"""
Database Models
---------------
Tables: users, trained_models
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(50), unique=True, nullable=False, index=True)
    email      = Column(String(120), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(50), unique=True, nullable=False, index=True)
    model_path = Column(String(255), nullable=False)   # path to saved .pkl file
    accuracy   = Column(String(20), nullable=True)     # training accuracy
    n_samples  = Column(Integer, nullable=True)        # number of training samples
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())