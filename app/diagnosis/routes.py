from flask import request
from marshmallow import ValidationError
from . import diagnosis_bp
from .service import DiagnosisService
from .schemas import DiagnosisRequestSchema
from app.common.response.api_response import ApiResponse
from app.common.response.error_response import ErrorCode

diagnosis_service = DiagnosisService()
request_schema = DiagnosisRequestSchema()

@diagnosis_bp.route('/v1/diagnosis', methods=['POST'])
def diagnose():
    try:
        # Validate request data
        data = request.get_json()
        if not data:
            return ApiResponse.error(
                error_code=ErrorCode.VALIDATION_ERROR,
                details={"body": "Request body is required"}
            )
            
        # Validate against schema
        validated_data = request_schema.load(data)
        image_url = validated_data['image_url']
        cat_id = validated_data['cat_id']
        
        # Process diagnosis
        result = diagnosis_service.process_diagnosis(image_url, cat_id)
        
        return ApiResponse.success(data=result)
        
    except ValidationError as e:
        return ApiResponse.error(
            error_code=ErrorCode.VALIDATION_ERROR,
            details=e.messages
        )
    except Exception as e:
        return ApiResponse.error(
            error_code=ErrorCode.INTERNAL_ERROR
        ) 