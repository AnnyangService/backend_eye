from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from app.config import config_by_name as config
from app.common.response.api_response import ApiResponse

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
    migrate.init_app(app, db, render_as_batch=True)
    
    # Import models to ensure they are registered with SQLAlchemy
    from app.diagnosis.models import DiagnosisLevel1, DiagnosisLevel2, DiagnosisLevel3, DiagnosisTarget, DiagnosisRule, DiagnosisRuleDescription
    from app.member.models import Member
    from app.cat.models import Cat
    
    # Register CLI commands
    from app.commands import register_commands
    register_commands(app)
    
    # Register blueprints
    from app.diagnosis import diagnosis_bp
    app.register_blueprint(diagnosis_bp)
    
    @app.route('/health')
    def health_check():
        """Simple health check endpoint"""
        return {'status': 'ok', 'service': 'backend-eye'}, 200
    
    return app