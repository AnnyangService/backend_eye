from flask_restx import Api
from flask import Blueprint

# Create a blueprint for API
api_bp = Blueprint('api', __name__)

# Initialize Flask-RESTX API
api = Api(
    api_bp,
    version='1.0',
    title='Backend Eye API',
    description='API documentation for Backend Eye service',
    doc='/docs/'  # Swagger UI will be available at /docs/
)

# Import namespaces here to register them with the API
def register_namespaces():
    """Register all API namespaces"""
    from app.diagnosis.api import diagnosis_ns
    from app.chatbot.chat_api import chat_ns
    
    api.add_namespace(diagnosis_ns, path='/diagnosis')
    api.add_namespace(chat_ns, path='/chat') 