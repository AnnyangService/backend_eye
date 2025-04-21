"""Base model class for all database models"""
from datetime import datetime
from app import db

class BaseModel(db.Model):
    """Base model that all other models should inherit from"""
    
    # Abstract model - won't create a table
    __abstract__ = True
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def save(self):
        """Save the model to the database"""
        db.session.add(self)
        db.session.commit()
        return self
    
    def delete(self):
        """Delete the model from the database"""
        db.session.delete(self)
        db.session.commit()
    
    @classmethod
    def find_by_id(cls, id):
        """Find a model by ID"""
        return cls.query.filter_by(id=id).first()
    
    @classmethod
    def find_all(cls):
        """Find all records of this model"""
        return cls.query.all() 