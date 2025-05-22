"""CLI commands for the Flask application"""
import click
from flask.cli import with_appcontext
from app import db
from flask_migrate import upgrade, migrate, init
from flask import current_app

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    if not current_app.config.get('ENABLE_MIGRATIONS', True):
        click.echo('Error: Database migrations are disabled in production environment.')
        return
    db.drop_all()
    db.create_all()
    click.echo('Initialized the database.')

def register_commands(app):
    """Register CLI commands with the app"""
    app.cli.add_command(init_db_command)

    @app.cli.command('db-init')
    @with_appcontext
    def db_init():
        """Initialize database migrations"""
        if not current_app.config.get('ENABLE_MIGRATIONS', True):
            click.echo('Error: Database migrations are disabled in production environment.')
            return
        init()
        click.echo('Database migrations initialized')

    @app.cli.command('db-migrate')
    @with_appcontext
    def db_migrate():
        """Create a new migration"""
        if not current_app.config.get('ENABLE_MIGRATIONS', True):
            click.echo('Error: Database migrations are disabled in production environment.')
            return
        migrate()
        click.echo('New migration created')

    @app.cli.command('db-upgrade')
    @with_appcontext
    def db_upgrade():
        """Apply database migrations"""
        if not current_app.config.get('ENABLE_MIGRATIONS', True):
            click.echo('Error: Database migrations are disabled in production environment.')
            return
        upgrade()
        click.echo('Database migrations applied') 