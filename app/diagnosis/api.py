from flask import request
from flask_restx import Namespace, Resource, fields
from .service import DiagnosisService

# Create namespace for diagnosis API
diagnosis_ns = Namespace('diagnosis', description='질병 진단 API')

# Define request model for Step1
step1_request_model = diagnosis_ns.model('Step1Request', {
    'image_url': fields.Url(required=True, description='분석할 이미지의 URL', example="image.png")
})

# Define response models for Step1
step1_data_model = diagnosis_ns.model('Step1Data', {
    'is_normal': fields.Boolean(required=True, description='정상 여부 (true: 정상, false: 이상)', example=False),
    'confidence': fields.Float(required=True, description='신뢰도 (0.0 ~ 1.0)', example=0.87)
})

# Define request model for Step2
step2_request_model = diagnosis_ns.model('Step2Request', {
    'id': fields.String(required=True, description='요청 ID', example="01JTTKJYG28CFYMBKXC0Q80F61"),
    'password': fields.String(required=True, description='AI 서버 -> API 서버 호출시 필요한 패스워드', example="a123456789!"),
    'image_url': fields.Url(required=True, description='분석할 이미지의 URL', example="image.png")
})

step1_response_model = diagnosis_ns.model('Step1Response', {
    'success': fields.Boolean(required=True, description='요청 성공 여부', example=True),
    'message': fields.String(required=True, description='응답 메시지', example="Success"),
    'data': fields.Nested(step1_data_model, required=True, description='진단 결과 데이터')
})

# Define response model for Step2 (즉시 응답)
step2_response_model = diagnosis_ns.model('Step2Response', {
    'success': fields.Boolean(required=True, description='요청 성공 여부', example=True),
    'message': fields.String(required=True, description='응답 메시지', example="Success"),
    'data': fields.Raw(description='항상 null (비동기 처리로 인해 즉시 응답)', example=None)
})

# Define request model for Step3
step3_attribute_model = diagnosis_ns.model('Step3Attribute', {
    'id': fields.Integer(required=True, description='룰 ID', example=1),
    'description': fields.String(required=True, description='속성 설명', example="미세한 분비물")
})

step3_request_model = diagnosis_ns.model('Step3Request', {
    'secondStepDiagnosisResult': fields.String(required=True, description='2단계 진단 결과 (inflammation 또는 corneal)', 
                                              enum=['inflammation', 'corneal'], example="inflammation"),
    'attributes': fields.List(fields.Nested(step3_attribute_model), required=True, 
                             description='진단 속성 리스트',
                             example=[
                                 {
                                     "id": 1,
                                     "description": "주로 눈문을 많이 흘리고 미세하게 분비물이 있어요."
                                 },
                                 {
                                     "id": 2,
                                     "description": "천천히 진행돼요."
                                 },
                                 {
                                     "id": 3,
                                     "description": "눈물을 흘리고 눈 뜨는 데 어려움이 없어요. 각막 표면이 매끄러워요."
                                 },
                                 {
                                     "id": 4,
                                     "description": "한쪽 눈에서 발생해요."
                                 }
                             ])
})

# Define response model for Step3
step3_data_model = diagnosis_ns.model('Step3Data', {
    'category': fields.String(required=True, description='진단 카테고리', example="알레르기성 결막염"),
    'description': fields.String(required=True, description='LLM이 생성한 진단 결과', 
                                example="환자의 증상과 관찰된 징후를 종합적으로 분석한 결과, 알레르기성 결막염으로 진단됩니다. 안구 표면의 점상 출혈과 결막 부종, 그리고 눈물 분비량 감소가 주요 소견으로 확인되었습니다. 적절한 항염 치료와 함께 알레르기 원인 회피가 권장됩니다.")
})

step3_response_model = diagnosis_ns.model('Step3Response', {
    'success': fields.Boolean(required=True, description='요청 성공 여부', example=True),
    'message': fields.String(required=True, description='응답 메시지', example="Success"),
    'data': fields.Nested(step3_data_model, required=True, description='진단 결과 데이터')
})

error_response_model = diagnosis_ns.model('ErrorResponse', {
    'success': fields.Boolean(description='요청 성공 여부', example=False),
    'error_code': fields.String(description='에러 코드', example="VALIDATION_ERROR"),
    'message': fields.String(description='에러 메시지', example="secondStepDiagnosisResult is required"),
    'details': fields.Raw(description='에러 상세 정보', example={"secondStepDiagnosisResult": "This field is required"})
})

# Initialize service
try:
    diagnosis_service = DiagnosisService()
    print("✅ DiagnosisService 초기화 성공!")
    
    # 모델 정보 출력
    model_info = diagnosis_service.get_model_info()
    print(f"🤖 AI Model Information:")
    if model_info.get('step1_model'):
        step1_info = model_info['step1_model']
        print(f"   Step1: {step1_info['model_architecture']} ({'✅ Loaded' if step1_info['model_loaded'] else '❌ Failed'})")
    if model_info.get('step2_model'):
        step2_info = model_info['step2_model']
        print(f"   Step2: {step2_info['model_architecture']} ({'✅ Loaded' if step2_info['model_loaded'] else '❌ Failed'})")
        
except Exception as e:
    # AI 모델 로드 실패 시 서비스 객체를 None으로 설정
    diagnosis_service = None
    import logging
    import traceback
    logger = logging.getLogger(__name__)
    logger.error(f"DiagnosisService 초기화 실패: {str(e)}")
    logger.error(f"상세 에러: {traceback.format_exc()}")
    print(f"❌ DiagnosisService 초기화 실패: {str(e)}")

@diagnosis_ns.route('/info/')
class DiagnosisInfoResource(Resource):
    @diagnosis_ns.doc('모델 정보 조회')
    def get(self):
        """
        현재 로드된 AI 모델 정보를 조회합니다.
        """
        try:
            if diagnosis_service is None:
                return {
                    'success': False,
                    'error_code': 'SERVICE_UNAVAILABLE',
                    'message': 'AI 모델 서비스를 사용할 수 없습니다.',
                    'details': {'service': 'AI model not loaded'}
                }, 503
            
            model_info = diagnosis_service.get_model_info()
            
            return {
                'success': True,
                'message': 'AI model information retrieved successfully',
                'data': model_info
            }, 200
            
        except Exception as e:
            return {
                'success': False,
                'error_code': 'INTERNAL_ERROR',
                'message': str(e),
                'details': {'error': str(e)}
            }, 500

@diagnosis_ns.route('/step1/')
class DiagnosisStep1Resource(Resource):
    @diagnosis_ns.doc('질병여부판단')
    @diagnosis_ns.expect(step1_request_model, validate=True)
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
                    'message': 'AI 모델 서비스를 사용할 수 없습니다.',
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
            
            response_data = {
                'success': True,
                'message': 'Success',
                'data': result
            }
            
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
                    'message': 'AI 모델 서비스를 사용할 수 없습니다.',
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

@diagnosis_ns.route('/step3/')
class DiagnosisStep3Resource(Resource):
    @diagnosis_ns.doc('질병분석 Step3')
    @diagnosis_ns.expect(step3_request_model, validate=True)
    @diagnosis_ns.marshal_with(step3_response_model, code=200)
    @diagnosis_ns.marshal_with(error_response_model, code=400)
    @diagnosis_ns.marshal_with(error_response_model, code=500)
    @diagnosis_ns.marshal_with(error_response_model, code=503)
    def post(self):
        """
        질병분석 Step3 - 세부 진단
        
        2단계 진단 결과와 속성 정보를 기반으로 세부 진단을 수행합니다.
        """
        try:
            # 서비스 초기화 확인
            if diagnosis_service is None:
                return {
                    'success': False,
                    'error_code': 'SERVICE_UNAVAILABLE',
                    'message': 'AI 모델 서비스를 사용할 수 없습니다.',
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
            second_step_result = data.get('secondStepDiagnosisResult')
            attributes = data.get('attributes')
            
            if not second_step_result:
                return {
                    'success': False,
                    'error_code': 'VALIDATION_ERROR',
                    'message': 'secondStepDiagnosisResult is required',
                    'details': {'secondStepDiagnosisResult': 'This field is required'}
                }, 400
            
            if not attributes or not isinstance(attributes, list):
                return {
                    'success': False,
                    'error_code': 'VALIDATION_ERROR',
                    'message': 'attributes is required and must be a list',
                    'details': {'attributes': 'This field is required and must be a list'}
                }, 400
            
            # attributes 유효성 검증
            for i, attr in enumerate(attributes):
                if not isinstance(attr, dict):
                    return {
                        'success': False,
                        'error_code': 'VALIDATION_ERROR',
                        'message': f'attributes[{i}] must be an object',
                        'details': {'attributes': f'Item at index {i} must be an object'}
                    }, 400
                
                if 'id' not in attr or 'description' not in attr:
                    return {
                        'success': False,
                        'error_code': 'VALIDATION_ERROR',
                        'message': f'attributes[{i}] must have id and description fields',
                        'details': {'attributes': f'Item at index {i} must have id and description fields'}
                    }, 400
            
            # Step3 진단 처리
            result = diagnosis_service.process_step3_diagnosis(second_step_result, attributes)
            
            response_data = {
                'success': True,
                'message': 'Success',
                'data': result
            }
            
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