from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Initialize SQLAlchemy instance
db = SQLAlchemy()
migrate = Migrate()

def create_app(config_name):
    """Application factory function to create and configure the Flask app"""
    app = Flask(__name__)
    
    # Load configuration
    from app.config import config_by_name
    app.config.from_object(config_by_name[config_name])
    
    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Register error handlers
    from app.global.exception.exception_handler import register_error_handlers
    register_error_handlers(app)
    
    # Register CLI commands
    from app.commands import register_commands
    register_commands(app)
    
    # Register blueprints here when created
    # from app.user.routes import user_bp
    # app.register_blueprint(user_bp)
    
    @app.route('/health')
    def health_check():
        """Simple health check endpoint"""
        return {'status': 'ok', 'service': 'backend-eye'}, 200
    
    return app 