"""User API routes"""
from flask import Blueprint, request, jsonify
from app import db
from app.user.models import User
from app.common.response.api_response import ApiResponse
from app.common.exception.business_exception import BusinessException, EntityNotFoundException

# Create a blueprint
user_bp = Blueprint('user', __name__, url_prefix='/api/users')

@user_bp.route('', methods=['POST'])
def create_user():
    """Create a new user"""
    data = request.get_json()
    
    # Validate required fields
    if not all(k in data for k in ('email', 'password', 'name')):
        return ApiResponse.error("Missing required fields", "INVALID_DATA")
    
    # Check if user already exists
    if User.query.filter_by(email=data['email']).first():
        return ApiResponse.error("Email already registered", "EMAIL_EXISTS", 409)
    
    # Create new user
    try:
        user = User(
            email=data['email'],
            password=data['password'],
            name=data['name']
        )
        db.session.add(user)
        db.session.commit()
        
        # Don't return password in response
        return ApiResponse.success({
            'id': user.id,
            'email': user.email,
            'name': user.name
        }, "User created successfully", 201)
    except Exception as e:
        db.session.rollback()
        return ApiResponse.error(f"Failed to create user: {str(e)}", "SERVER_ERROR", 500)

@user_bp.route('/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get user by ID"""
    user = User.query.get(user_id)
    if not user:
        return ApiResponse.error(f"User with id {user_id} not found", "NOT_FOUND", 404)
    
    return ApiResponse.success({
        'id': user.id,
        'email': user.email,
        'name': user.name
    }) 