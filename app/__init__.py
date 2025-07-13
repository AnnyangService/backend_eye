from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app.config import config_by_name as config

# SQLAlchemy 객체 생성 (전역 변수)
db = SQLAlchemy()

def create_app(config_name='development'):
    """Application factory function to create and configure the Flask app"""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize SQLAlchemy with app
    db.init_app(app)
    
    # Import models to register them with SQLAlchemy
    from app.models import chunk
    
    # Register CLI commands
    from app.commands import register_commands
    register_commands(app)
    
    # Register Flask-RESTX API (Swagger 포함)
    from app.swagger import api_bp, register_namespaces
    register_namespaces()  # Register all namespaces
    app.register_blueprint(api_bp)
    
    @app.route('/health')
    def health_check():
        """Simple health check endpoint"""
        return {'status': 'ok', 'service': 'backend-eye'}, 200
    
    return app