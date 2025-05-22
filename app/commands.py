"""CLI commands for the Flask application"""
import click
from flask.cli import with_appcontext
from app import db
from flask import current_app

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    click.echo('Error: Database operations are now handled by the API server.')
    return

def register_commands(app):
    """Register CLI commands with the app"""
    app.cli.add_command(init_db_command)

    @app.cli.command('db-info')
    @with_appcontext
    def db_info():
        """Show database connection information"""
        click.echo('Database information:')
        click.echo(f"Host: {current_app.config.get('MYSQL_HOST')}")
        click.echo(f"Port: {current_app.config.get('MYSQL_PORT')}")
        click.echo(f"Database: {current_app.config.get('MYSQL_DB')}")
        click.echo(f"Environment: {current_app.config.get('ENV', 'development')}")
        click.echo('Note: Database schema and migrations are now managed by the API server') 