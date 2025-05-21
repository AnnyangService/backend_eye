"""Database utility functions"""
from sqlalchemy.exc import SQLAlchemyError
from app import db

def commit_session():
    """Commit the current database session"""
    try:
        db.session.commit()
        return True
    except SQLAlchemyError as e:
        db.session.rollback()
        raise e

def add_to_db(model_instance):
    """Add a model instance to the database and commit"""
    try:
        db.session.add(model_instance)
        commit_session()
        return model_instance
    except SQLAlchemyError as e:
        db.session.rollback()
        raise e

def delete_from_db(model_instance):
    """Delete a model instance from the database and commit"""
    try:
        db.session.delete(model_instance)
        commit_session()
        return True
    except SQLAlchemyError as e:
        db.session.rollback()
        raise e 