"""API response structure"""
from flask import jsonify

class ApiResponse:
    """Standard API response format"""
    
    @staticmethod
    def success(data=None, message="Success", status_code=200):
        """Create a success response"""
        response = {
            "success": True,
            "message": message
        }
        if data is not None:
            response["data"] = data
        return jsonify(response), status_code
    
    @staticmethod
    def error(message="Error", error_code=None, status_code=400):
        """Create an error response"""
        response = {
            "success": False,
            "message": message
        }
        if error_code:
            response["error_code"] = error_code
        return jsonify(response), status_code 