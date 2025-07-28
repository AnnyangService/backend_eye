from flask import request
from flask_restx import Namespace, Resource, fields
from .chat_service import InitialChatService, RAGChatService
import logging

# Create namespace for chat API
chat_ns = Namespace('chat', description='챗봇 API')

# Define request model for initial chat
initial_chat_request_model = chat_ns.model('InitialChatRequest', {
    'query': fields.String(required=True, description='사용자 질문', example="이 질병은 어떻게 치료하나요?"),
    'diagnosis_result': fields.String(required=True, description='진단 결과', example="비궤양성 각막염"),
    'summary': fields.String(required=True, description='진단 요약', 
                            example="🔍 진단 결과: 비궤양성 각막염\n• 분비물 특성: 비궤양성 각막염 (85.7% 유사)\n• 진행 속도: 비궤양성 각막염 (84.5% 유사)\n• 주요 증상: 비궤양성 각막염 (90.6% 유사)\n• 발생 패턴: 결막염 (82.7% 유사)\n\n📊 전체 유사도 분석:\n• 비궤양성 각막염: 85.5%\n• 안검염: 85.1%\n• 결막염: 84.7%"),
    'details': fields.String(required=True, description='진단 상세 정보', 
                            example="# 비궤양성 각막염 진단 보고서\n\n제공된 정보를 바탕으로 비궤양성 각막염이 의심됩니다. 눈물 과다 분비, 미세한 분비물, 서서히 진행되는 양상 등이 주요 근거입니다. 하지만 정확한 진단은 임상 검사를 통해 확정해야 하며, 추가적인 검사가 필요할 수 있습니다.")
})

# Define request model for RAG chat
rag_chat_request_model = chat_ns.model('RAGChatRequest', {
    'query': fields.String(required=True, description='사용자 질문', example="이 질병은 어떻게 치료하나요?"),
    'previous_question': fields.String(description='가장 최근 이전 질문', example="이 질병은 무엇인가요?"),
    'previous_answer': fields.String(description='가장 최근 이전 답변', example="비궤양성 각막염은 각막의 염증성 질환입니다."),
    'two_turn_question': fields.String(description='전전 질문', example="증상이 심각한가요?"),
    'two_turn_answer': fields.String(description='전전 답변', example="현재 증상은 중간 정도의 심각도를 보입니다.")
})

# Define response models
chat_data_model = chat_ns.model('ChatData', {
    'answer': fields.String(required=True, description='생성된 답변', 
                           example="비궤양성 각막염의 치료는 주로 항염증제와 인공눈물을 사용합니다. 하지만 정확한 치료는 수의사와의 상담을 통해 결정해야 합니다."),
    'error': fields.String(description='오류 메시지 (있는 경우)', example=None)
})

rag_chat_data_model = chat_ns.model('RAGChatData', {
    'answer': fields.String(required=True, description='생성된 답변', 
                           example="비궤양성 각막염의 치료는 주로 항염증제와 인공눈물을 사용합니다. 하지만 정확한 치료는 수의사와의 상담을 통해 결정해야 합니다."),
    'retrieved_documents': fields.List(fields.Raw, description='검색된 문서들'),
    'error': fields.String(description='오류 메시지 (있는 경우)', example=None)
})

# Define response models
initial_chat_response_model = chat_ns.model('InitialChatResponse', {
    'success': fields.Boolean(required=True, description='요청 성공 여부', example=True),
    'message': fields.String(required=True, description='응답 메시지', example="Success"),
    'data': fields.Nested(chat_data_model, required=True, description='챗봇 응답 데이터')
})

rag_chat_response_model = chat_ns.model('RAGChatResponse', {
    'success': fields.Boolean(required=True, description='요청 성공 여부', example=True),
    'message': fields.String(required=True, description='응답 메시지', example="Success"),
    'data': fields.Nested(rag_chat_data_model, required=True, description='챗봇 응답 데이터')
})

error_response_model = chat_ns.model('ErrorResponse', {
    'success': fields.Boolean(description='요청 성공 여부', example=False),
    'error_code': fields.String(description='에러 코드', example="VALIDATION_ERROR"),
    'message': fields.String(description='에러 메시지', example="query is required"),
    'details': fields.Raw(description='에러 상세 정보', example={"query": "This field is required"})
})

# Initialize services
try:
    initial_chat_service = InitialChatService()
    rag_chat_service = RAGChatService()
    
    logger = logging.getLogger(__name__)
    logger.info("챗봇 서비스 초기화 완료")
        
except Exception as e:
    # 서비스 초기화 실패 시 서비스 객체를 None으로 설정
    initial_chat_service = None
    rag_chat_service = None
    import logging
    import traceback
    logger = logging.getLogger(__name__)
    logger.error(f"챗봇 서비스 초기화 실패: {str(e)}")
    logger.error(f"상세 에러: {traceback.format_exc()}")

@chat_ns.route('/first')
class InitialChatResource(Resource):
    @chat_ns.doc('최초 채팅')
    @chat_ns.expect(initial_chat_request_model, validate=True)
    @chat_ns.response(200, 'Success', initial_chat_response_model)
    @chat_ns.response(400, 'Validation Error', error_response_model)
    @chat_ns.response(500, 'Internal Server Error', error_response_model)
    def post(self):
        """
        최초 채팅 - 진단 결과 기반
        
        진단 결과를 컨텍스트로 사용하여 사용자 질문에 답변을 생성합니다.
        """
        try:
            # 서비스 초기화 확인
            if initial_chat_service is None:
                return {
                    'success': False,
                    'error_code': 'SERVICE_UNAVAILABLE',
                    'message': '챗봇 서비스를 사용할 수 없습니다.',
                    'details': {'service': 'Chat service not loaded'}
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
            required_fields = ['query', 'diagnosis_result', 'summary', 'details']
            for field in required_fields:
                if not data.get(field):
                    return {
                        'success': False,
                        'error_code': 'VALIDATION_ERROR',
                        'message': f'{field} is required',
                        'details': {field: f'{field} is required'}
                    }, 400
            
            # 최초 채팅 처리
            result = initial_chat_service.generate_response(data)
            
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

@chat_ns.route('/second')
class RAGChatResource(Resource):
    @chat_ns.doc('이후 채팅')
    @chat_ns.expect(rag_chat_request_model, validate=True)
    @chat_ns.response(200, 'Success', rag_chat_response_model)
    @chat_ns.response(400, 'Validation Error', error_response_model)
    @chat_ns.response(500, 'Internal Server Error', error_response_model)
    def post(self):
        """
        이후 채팅 - RAG 검색 기반
        
        이전 대화 기록과 RAG 검색을 통해 사용자 질문에 답변을 생성합니다.
        """
        try:
            # 서비스 초기화 확인
            if rag_chat_service is None:
                return {
                    'success': False,
                    'error_code': 'SERVICE_UNAVAILABLE',
                    'message': '챗봇 서비스를 사용할 수 없습니다.',
                    'details': {'service': 'Chat service not loaded'}
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
            if not data.get('query'):
                return {
                    'success': False,
                    'error_code': 'VALIDATION_ERROR',
                    'message': 'query is required',
                    'details': {'query': 'This field is required'}
                }, 400
            
            # RAG 채팅 처리
            result = rag_chat_service.generate_response(data)
            
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
                    'retrieved_documents': result.get('retrieved_documents', []),
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