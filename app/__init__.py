from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import config

# Initialize SQLAlchemy instance
db = SQLAlchemy()
migrate = Migrate()

def create_app(config_name='development'):
    """Application factory function to create and configure the Flask app"""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Register CLI commands
    from app.commands import register_commands
    register_commands(app)
    
    # Register blueprints
    from app.user.routes import user_bp
    app.register_blueprint(user_bp)
    
    from app.diagnosis import diagnosis_bp
    app.register_blueprint(diagnosis_bp)
    
    @app.route('/health')
    def health_check():
        """Simple health check endpoint"""
        return {'status': 'ok', 'service': 'backend-eye'}, 200
    
    return app