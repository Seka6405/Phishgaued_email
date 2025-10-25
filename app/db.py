# app/db.py
from sqlmodel import SQLModel, create_engine, Session
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent.parent / "phishguard.db"
ENGINE = create_engine(f"sqlite:///{DB_FILE}", echo=False, connect_args={"check_same_thread": False})

def create_db_and_tables():
    from app.models import User, EmailRecord, Feedback, ModelVersion
    SQLModel.metadata.create_all(ENGINE)

def get_session():
    with Session(ENGINE) as session:
        yield session
