"""API response structure"""
from flask import jsonify
from typing import TypeVar, Optional, Any
from .error_response import ErrorResponse, ErrorCode

T = TypeVar('T')

class ApiResponse:
    """Standard API response format matching Java version"""
    
    @staticmethod
    def success(data: Optional[Any] = None) -> tuple:
        """Create a success response"""
        response = {
            "success": True,
            "data": data,
            "error": None
        }
        return jsonify(response), 200
    
    @staticmethod
    def error(error_code: str, details: Optional[dict] = None) -> tuple:
        """Create an error response"""
        error = ErrorResponse(
            code=error_code,
            message=ErrorCode.get_message(error_code),
            details=details
        )
        response = {
            "success": False,
            "data": None,
            "error": error.__dict__
        }
        return jsonify(response), 400 