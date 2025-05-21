from datetime import datetime
from app import db

class Diagnosis(db.Model):
    __tablename__ = 'diagnoses'
    
    id = db.Column(db.Integer, primary_key=True)
    image_url = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='processing')
    result = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Diagnosis {self.id}>' 