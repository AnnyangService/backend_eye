from flask import request
from flask_restx import Namespace, Resource, fields
from .service import DiagnosisService

# Create namespace for diagnosis API
diagnosis_ns = Namespace('diagnosis', description='질병 진단 API')

# Define request model for Step1
step1_request_model = diagnosis_ns.model('Step1Request', {
    'image_url': fields.Url(required=True, description='분석할 이미지의 URL')
})

# Define response models for Step1
step1_data_model = diagnosis_ns.model('Step1Data', {
    'is_normal': fields.Boolean(required=True, description='정상 여부 (true: 정상, false: 이상)'),
    'confidence': fields.Float(required=True, description='신뢰도 (0.0 ~ 1.0)')
})

# Define request model for Step2
step2_request_model = diagnosis_ns.model('Step2Request', {
    'id': fields.String(required=True, description='요청 ID'),
    'password': fields.String(required=True, description='AI 서버 -> API 서버 호출시 필요한 패스워드'),
    'image_url': fields.Url(required=True, description='분석할 이미지의 URL')
})

step1_response_model = diagnosis_ns.model('Step1Response', {
    'success': fields.Boolean(required=True, description='요청 성공 여부'),
    'message': fields.String(required=True, description='응답 메시지'),
    'data': fields.Nested(step1_data_model, required=True, description='진단 결과 데이터')
})

# Define response model for Step2 (즉시 응답)
step2_response_model = diagnosis_ns.model('Step2Response', {
    'success': fields.Boolean(required=True, description='요청 성공 여부'),
    'message': fields.String(required=True, description='응답 메시지'),
    'data': fields.Raw(description='항상 null (비동기 처리로 인해 즉시 응답)')
})

error_response_model = diagnosis_ns.model('ErrorResponse', {
    'success': fields.Boolean(description='요청 성공 여부'),
    'error_code': fields.String(description='에러 코드'),
    'message': fields.String(description='에러 메시지'),
    'details': fields.Raw(description='에러 상세 정보')
})

# Initialize service
try:
    diagnosis_service = DiagnosisService()
    print("✅ DiagnosisService 초기화 성공!")
except Exception as e:
    # AI 모델 로드 실패 시 서비스 객체를 None으로 설정
    diagnosis_service = None
    import logging
    import traceback
    logger = logging.getLogger(__name__)
    logger.error(f"DiagnosisService 초기화 실패: {str(e)}")
    logger.error(f"상세 에러: {traceback.format_exc()}")
    print(f"❌ DiagnosisService 초기화 실패: {str(e)}")
    print(f"상세 에러: {traceback.format_exc()}")

@diagnosis_ns.route('/step1/')
class DiagnosisStep1Resource(Resource):
    @diagnosis_ns.doc('질병여부판단')
    @diagnosis_ns.expect(step1_request_model, validate=True)
    # @diagnosis_ns.marshal_with(step1_response_model, code=200)
    # @diagnosis_ns.marshal_with(error_response_model, code=400)
    # @diagnosis_ns.marshal_with(error_response_model, code=500)
    # @diagnosis_ns.marshal_with(error_response_model, code=503)
    def post(self):
        """
        질병분석 Step1 - 질병여부판단
        
        이미지를 분석하여 질병 여부를 판단합니다.
        """
        try:
            # 서비스 초기화 확인
            if diagnosis_service is None:
                return {
                    'success': False,
                    'error_code': 'SERVICE_UNAVAILABLE',
                    'message': 'AI 모델 서비스를 사용할 수 없습니다. 서버 관리자에게 문의하세요.',
                    'details': {'service': 'AI model not loaded'}
                }, 503
            
            # Flask-RESTX가 자동으로 검증한 데이터 가져오기
            data = request.get_json()
            
            if not data:
                return {
                    'success': False,
                    'error_code': 'VALIDATION_ERROR',
                    'message': 'Request body is required',
                    'details': {'body': 'Request body is required'}
                }, 400
            
            image_url = data.get('image_url')
            
            # URL 기본 검증
            if not image_url or image_url == "string":
                return {
                    'success': False,
                    'error_code': 'INVALID_URL',
                    'message': 'Invalid image URL provided',
                    'details': {'image_url': 'Please provide a valid image URL'}
                }, 400
            
            # Step1 진단 처리
            result = diagnosis_service.process_step1_diagnosis(image_url)
            
            # 디버깅: 서비스 결과 로그 출력
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"API에서 받은 서비스 결과: {result}")
            logger.info(f"결과 타입: {type(result)}")
            if result:
                logger.info(f"is_normal: {result.get('is_normal')} (타입: {type(result.get('is_normal'))})")
                logger.info(f"confidence: {result.get('confidence')} (타입: {type(result.get('confidence'))})")
            
            response_data = {
                'success': True,
                'message': 'Success',
                'data': result
            }
            
            logger.info(f"최종 응답 데이터: {response_data}")
            
            return response_data, 200
            
        except Exception as e:
            error_message = str(e)
            
            # AI 모델 관련 에러인지 확인
            if "AI 모델이 로드되지 않았습니다" in error_message:
                return {
                    'success': False,
                    'error_code': 'MODEL_NOT_AVAILABLE',
                    'message': error_message,
                    'details': {'model': 'AI model not loaded or failed to initialize'}
                }, 503
            
            return {
                'success': False,
                'error_code': 'INTERNAL_ERROR',
                'message': error_message,
                'details': {'error': error_message}
            }, 500 

@diagnosis_ns.route('/step2/')
class DiagnosisStep2Resource(Resource):
    @diagnosis_ns.doc('질병분석 Step2')
    @diagnosis_ns.expect(step2_request_model, validate=True)
    def post(self):
        """
        질병분석 Step2
        
        Step2 진단을 위한 요청을 처리합니다.
        """
        try:
            # 서비스 초기화 확인
            if diagnosis_service is None:
                return {
                    'success': False,
                    'error_code': 'SERVICE_UNAVAILABLE',
                    'message': 'AI 모델 서비스를 사용할 수 없습니다. 서버 관리자에게 문의하세요.',
                    'details': {'service': 'AI model not loaded'}
                }, 503
            
            # Flask-RESTX가 자동으로 검증한 데이터 가져오기
            data = request.get_json()
            
            if not data:
                return {
                    'success': False,
                    'error_code': 'VALIDATION_ERROR',
                    'message': 'Request body is required',
                    'details': {'body': 'Request body is required'}
                }, 400
            
            # 필수 필드 검증
            required_fields = ['id', 'password', 'image_url']
            for field in required_fields:
                if not data.get(field):
                    return {
                        'success': False,
                        'error_code': 'VALIDATION_ERROR',
                        'message': f'{field} is required',
                        'details': {field: f'{field} is required'}
                    }, 400
            
            # Step2 비동기 진단 처리 (즉시 응답)
            result = diagnosis_service.process_step2_diagnosis_async(
                data['id'], 
                data['password'], 
                data['image_url']
            )
            
            response_data = result  # 이미 올바른 형식으로 반환됨
            
            return response_data, 200
            
        except Exception as e:
            error_message = str(e)
            
            return {
                'success': False,
                'error_code': 'INTERNAL_ERROR',
                'message': error_message,
                'details': {'error': error_message}
            }, 500 