"""CLI commands for the Flask application"""
import click
from flask import current_app
from flask.cli import with_appcontext

def register_commands(app):
    """Register CLI commands with the app"""
    # MySQL 관련 명령어 제거됨
    pass 