"""
Database configuration.

This module initializes the SQLAlchemy database engine,
session factory, and declarative base used throughout
the Gate Surveillance backend.

Components:
    - engine: Connection to the SQLite database.
    - SessionLocal: Factory for creating database sessions.
    - Base: Base class for all ORM models.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------
# Database Configuration
# ---------------------------------------------------------------------

# SQLite database used by the backend.
DATABASE_URL = "sqlite:///gate_surveillance.db"

# ---------------------------------------------------------------------
# SQLAlchemy Engine
# ---------------------------------------------------------------------

# Creates the database engine.
# 'check_same_thread=False' allows database access from multiple threads,
# which is required because the video processing pipeline runs in a
# separate thread from the FastAPI server.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# ---------------------------------------------------------------------
# Session Factory
# ---------------------------------------------------------------------

# Creates independent database sessions for read/write operations.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ---------------------------------------------------------------------
# Declarative Base
# ---------------------------------------------------------------------

# Base class inherited by all SQLAlchemy ORM models.
Base = declarative_base()
