from datetime import datetime
from app import db
import uuid

class Member(db.Model):
    __tablename__ = 'members'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = db.Column(db.DateTime, nullable=True, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow)
    email = db.Column(db.String(255), unique=True, nullable=True)
    name = db.Column(db.String(255), nullable=True)
    password = db.Column(db.String(255), nullable=True)
    role = db.Column(db.Enum('USER', 'ADMIN'), nullable=True)
    
    # Relationships
    cats = db.relationship('Cat', backref='member', lazy=True) 