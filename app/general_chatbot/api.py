from flask import request
from flask_restx import Namespace, Resource, fields
from .service import GeneralChatService
import logging

# Create namespace for general chat API
general_chat_ns = Namespace('chat', description='일반 챗봇 API')

# Define request model for general chat
general_chat_request_model = general_chat_ns.model('GeneralChatRequest', {
    'query': fields.String(required=True, description='사용자 질문', example="고양이가 눈을 자주 비비고 있어요"),
    'previous_question': fields.String(description='가장 최근 이전 질문', example="고양이 눈이 빨갛게 보여요"),
    'previous_answer': fields.String(description='가장 최근 이전 답변', example="눈이 빨갛게 보이는 것은 염증의 증상일 수 있습니다."),
    'two_turn_question': fields.String(description='전전 질문', example="고양이 눈에 이상이 있나요?"),
    'two_turn_answer': fields.String(description='전전 답변', example="현재 증상으로는 정확한 진단이 어렵습니다.")
})

# Define response models
general_chat_data_model = general_chat_ns.model('GeneralChatData', {
    'answer': fields.String(required=True, description='생성된 답변', 
                           example="고양이가 눈을 자주 비비는 것은 여러 원인이 있을 수 있습니다. 하지만 정확한 진단은 수의사와의 상담을 통해 이루어져야 합니다."),
    'error': fields.String(description='오류 메시지 (있는 경우)', example=None)
})

general_chat_response_model = general_chat_ns.model('GeneralChatResponse', {
    'success': fields.Boolean(required=True, description='요청 성공 여부', example=True),
    'message': fields.String(required=True, description='응답 메시지', example="Success"),
    'data': fields.Nested(general_chat_data_model, required=True, description='챗봇 응답 데이터')
})

error_response_model = general_chat_ns.model('ErrorResponse', {
    'success': fields.Boolean(description='요청 성공 여부', example=False),
    'error_code': fields.String(description='에러 코드', example="VALIDATION_ERROR"),
    'message': fields.String(description='에러 메시지', example="query is required"),
    'details': fields.Raw(description='에러 상세 정보', example={"query": "This field is required"})
})

# Initialize service
try:
    general_chat_service = GeneralChatService()
    logger = logging.getLogger(__name__)
    logger.info("일반 챗봇 서비스 초기화 완료")
except Exception as e:
    general_chat_service = None
    import logging
    import traceback
    logger = logging.getLogger(__name__)
    logger.error(f"일반 챗봇 서비스 초기화 실패: {str(e)}")
    logger.error(f"상세 에러: {traceback.format_exc()}")

@general_chat_ns.route('/general')
class GeneralChatResource(Resource):
    @general_chat_ns.doc('일반 채팅')
    @general_chat_ns.expect(general_chat_request_model, validate=True)
    @general_chat_ns.response(200, 'Success', general_chat_response_model)
    @general_chat_ns.response(400, 'Validation Error', error_response_model)
    @general_chat_ns.response(500, 'Internal Server Error', error_response_model)
    def post(self):
        """
        일반 채팅 - 수의사 어시스턴트
        
        사용자의 일반적인 질문에 대해 수의사 어시스턴트가 답변을 생성합니다.
        이전 대화 기록을 참고하여 맥락을 유지합니다.
        """
        try:
            # 서비스 초기화 확인
            if general_chat_service is None:
                return {
                    'success': False,
                    'error_code': 'SERVICE_UNAVAILABLE',
                    'message': '일반 챗봇 서비스를 사용할 수 없습니다.',
                    'details': {'service': 'General chat service not loaded'}
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
            required_fields = ['query']
            for field in required_fields:
                if not data.get(field):
                    return {
                        'success': False,
                        'error_code': 'VALIDATION_ERROR',
                        'message': f'{field} is required',
                        'details': {field: f'{field} is required'}
                    }, 400
            
            # 일반 채팅 처리
            result = general_chat_service.generate_response(data)
            
            # 에러가 있는 경우
            if result.get('error'):
                return {
                    'success': False,
                    'error_code': 'CHAT_ERROR',
                    'message': result['error'],
                    'details': {'error': result['error']}
                }, 500
            
            response_data = {
                'success': True,
                'message': 'Success',
                'data': {
                    'answer': result['answer'],
                    'error': result.get('error')
                }
            }
            
            return response_data, 200
            
        except Exception as e:
            error_message = str(e)
            
            return {
                'success': False,
                'error_code': 'INTERNAL_ERROR',
                'message': error_message,
                'details': {'error': error_message}
            }, 500
