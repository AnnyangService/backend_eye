from datetime import datetime
from app import db
import uuid

class Cat(db.Model):
    __tablename__ = 'cat'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = db.Column(db.DateTime, nullable=True, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow)
    birth_date = db.Column(db.Date, nullable=True)
    breed = db.Column(db.String(255), nullable=True)
    gender = db.Column(db.Enum('MALE', 'FEMALE'), nullable=True)
    image = db.Column(db.String(255), nullable=True)
    last_diagnosis = db.Column(db.Date, nullable=True)
    name = db.Column(db.String(255), nullable=True)
    special_notes = db.Column(db.String(255), nullable=True)
    weight = db.Column(db.Float, nullable=True)
    member_id = db.Column(db.String(36), db.ForeignKey('members.id'), nullable=True) 