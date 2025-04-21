"""Exception handler for Flask application"""
from flask import jsonify
from sqlalchemy.exc import SQLAlchemyError
from app.global.exception.business_exception import BusinessException, DatabaseException

def register_error_handlers(app):
    """Register error handlers with Flask app"""
    
    @app.errorhandler(BusinessException)
    def handle_business_exception(error):
        """Handle business exceptions"""
        response = {
            'error': True,
            'message': error.message,
            'code': error.error_code
        }
        return jsonify(response), error.status_code
    
    @app.errorhandler(SQLAlchemyError)
    def handle_sqlalchemy_error(error):
        """Handle SQLAlchemy errors"""
        db_error = DatabaseException(str(error))
        response = {
            'error': True,
            'message': db_error.message,
            'code': db_error.error_code
        }
        return jsonify(response), db_error.status_code
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors"""
        response = {
            'error': True,
            'message': 'Resource not found',
            'code': 'NOT_FOUND'
        }
        return jsonify(response), 404
    
    @app.errorhandler(500)
    def handle_server_error(error):
        """Handle 500 errors"""
        response = {
            'error': True,
            'message': 'Internal server error',
            'code': 'SERVER_ERROR'
        }
        return jsonify(response), 500 