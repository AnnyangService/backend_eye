"""User models for authentication and authorization"""
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.ext.declarative import declarative_base
from app import db

class Role:
    """User roles enum"""
    USER = 'USER'
    ADMIN = 'ADMIN'

class User(db.Model):
    """User model for authentication and authorization"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), default=Role.USER)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __init__(self, email, password, name, role=Role.USER):
        self.email = email
        self.password = password  # This will call the password.setter
        self.name = name
        self.role = role

    @property
    def password(self):
        """Prevent password from being accessed"""
        raise AttributeError('Password is not a readable attribute')

    @password.setter
    def password(self, password):
        """Set password to a hashed password"""
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        """Check if password matches"""
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        """Check if user has admin role"""
        return self.role == Role.ADMIN
    
    def get_id(self):
        """Return the user ID as a string"""
        return str(self.id)
    
    def __repr__(self):
        return f'<User {self.email}>' 