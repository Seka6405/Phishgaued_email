# app/models.py
from sqlmodel import SQLModel, Field, Column, JSON
from typing import Optional
from datetime import datetime

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True)
    password_hash: str
    name: Optional[str] = None
    role: str = Field(default="user")  # user | admin
    created_at: datetime = Field(default_factory=datetime.utcnow)
    settings: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))

class EmailRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    subject: Optional[str] = None
    body: str = Field()
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    explanation: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email_id: int = Field(foreign_key="emailrecord.id")
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    label: str  # 'phish' | 'legit'
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ModelVersion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    path: str
    metrics: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
