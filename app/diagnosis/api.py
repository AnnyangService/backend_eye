from flask import request
from flask_restx import Namespace, Resource, fields
from .service import DiagnosisService
from app.common.response.api_response import ApiResponse
from app.common.response.error_response import ErrorCode

# Create namespace for diagnosis API
diagnosis_ns = Namespace('diagnosis', description='Diagnosis operations')

# Define request model for Swagger documentation
diagnosis_request_model = diagnosis_ns.model('DiagnosisRequest', {
    'image_url': fields.Url(required=True, description='URL of the image to diagnose'),
    'cat_id': fields.String(required=True, description='Category ID for diagnosis')
})

# Define response models
diagnosis_response_model = diagnosis_ns.model('DiagnosisResponse', {
    'success': fields.Boolean(description='Whether the request was successful'),
    'data': fields.Raw(description='Diagnosis result data'),
    'message': fields.String(description='Response message')
})

error_response_model = diagnosis_ns.model('ErrorResponse', {
    'success': fields.Boolean(description='Whether the request was successful'),
    'error_code': fields.String(description='Error code'),
    'message': fields.String(description='Error message'),
    'details': fields.Raw(description='Error details')
})

# Initialize services
diagnosis_service = DiagnosisService()

@diagnosis_ns.route('/v1/diagnosis')
class DiagnosisResource(Resource):
    @diagnosis_ns.doc('diagnose_image')
    @diagnosis_ns.expect(diagnosis_request_model, validate=True)
    @diagnosis_ns.marshal_with(diagnosis_response_model, code=200)
    @diagnosis_ns.marshal_with(error_response_model, code=400)
    @diagnosis_ns.marshal_with(error_response_model, code=500)
    def post(self):
        """
        Perform diagnosis on an image
        
        This endpoint accepts an image URL and category ID to perform diagnosis.
        """
        try:
            # Flask-RESTX automatically validates and parses the request
            data = request.get_json()
            
            if not data:
                return ApiResponse.error(
                    error_code=ErrorCode.VALIDATION_ERROR,
                    details={"body": "Request body is required"}
                )
            
            # Extract validated data
            image_url = data.get('image_url')
            cat_id = data.get('cat_id')
            
            # Process diagnosis
            result = diagnosis_service.process_diagnosis(image_url, cat_id)
            
            return ApiResponse.success(data=result)
            
        except Exception as e:
            return ApiResponse.error(
                error_code=ErrorCode.INTERNAL_ERROR
            ) 