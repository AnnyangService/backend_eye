"""
일반 챗봇 서비스
수의사 어시스턴트로서 일반적인 질문에 답변을 제공합니다.
"""

import logging
import os
from typing import Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)

def setup_gemini_api():
    """제미나이 API 설정 및 클라이언트 반환"""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        logger.error("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
        raise ValueError("GOOGLE_API_KEY 환경변수를 .env 파일에 설정해주세요.")
    
    # SDK client 초기화
    client = genai.Client(api_key=api_key)
    logger.info("제미나이 API 클라이언트 설정 완료")
    return client

def get_generation_config():
    """생성 설정 반환"""
    return types.GenerateContentConfig(
        temperature=0.7,  # 챗봇은 적당한 창의성
        top_p=0.8,
        top_k=40,
        max_output_tokens=2048,
    )

class GeneralChatService:
    """일반 챗봇용 서비스 (수의사 어시스턴트)"""
    
    def __init__(self):
        """초기화"""
        # 제미나이 API 클라이언트 및 설정 초기화
        try:
            self.client = setup_gemini_api()
            self.model_id = "gemini-2.0-flash"
            self.generation_config = get_generation_config()
            self.api_available = True
            logger.info("제미나이 API 연결 성공")
        except Exception as e:
            logger.error(f"제미나이 API 연결 실패: {str(e)}")
            self.client = None
            self.model_id = None
            self.generation_config = None
            self.api_available = False
        
        logger.info("GeneralChatService 초기화 완료")
    
    def _call_gemini(self, prompt: str) -> str:
        """제미나이 API 호출"""
        if not self.api_available or not self.client:
            logger.warning("제미나이 API를 사용할 수 없습니다. 기본 응답을 반환합니다.")
            return "죄송합니다. 현재 AI 서비스를 사용할 수 없습니다. 잠시 후 다시 시도해주세요."
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=self.generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"제미나이 API 호출 중 오류: {str(e)}")
            return f"AI 응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    def generate_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 질문에 대한 답변 생성 (일반 챗봇)
        
        Args:
            request_data (dict): 요청 데이터
                {
                    "query": str,                    # 현재 질문
                    "previous_question": str,         # 이전 질문 (선택사항)
                    "previous_answer": str,           # 이전 답변 (선택사항)
                    "two_turn_question": str,         # 전전 질문 (선택사항)
                    "two_turn_answer": str            # 전전 답변 (선택사항)
                }
        
        Returns:
            dict: {
                "answer": str,                        # 생성된 답변
                "error": str                          # 오류 메시지 (있는 경우)
            }
        """
        try:
            # 입력 데이터 검증
            query = request_data.get('query', '').strip()
            if not query:
                return {
                    "answer": "질문을 입력해주세요.",
                    "error": "질문이 비어있습니다."
                }
            
            logger.info(f"일반 챗봇 요청 처리 시작: {query[:50]}...")
            
            # 이전 대화 기록 포맷팅
            conversation_history = self._format_conversation_history(request_data)
            
            # 프롬프트 생성
            prompt = self._build_prompt(query, conversation_history)
            
            # 생성된 프롬프트 로그 출력 (전체)
            logger.info("=" * 50)
            logger.info("생성된 프롬프트:")
            logger.info("=" * 50)
            logger.info(prompt)
            logger.info("=" * 50)
            
            # 제미나이 API 호출
            answer = self._call_gemini(prompt)
            
            logger.info("일반 챗봇 응답 생성 완료")
            
            return {
                "answer": answer,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"일반 챗봇 응답 생성 중 오류: {str(e)}")
            return {
                "answer": "죄송합니다. 응답 생성 중 오류가 발생했습니다.",
                "error": str(e)
            }
    
    def _format_conversation_history(self, request_data: Dict[str, Any]) -> str:
        """이전 대화 기록을 포맷팅"""
        conversation_parts = []
        
        # 전전 대화 (있는 경우)
        two_turn_question = request_data.get('two_turn_question', '').strip()
        two_turn_answer = request_data.get('two_turn_answer', '').strip()
        
        if two_turn_question and two_turn_answer:
            conversation_parts.append(f"사용자: {two_turn_question}")
            conversation_parts.append(f"어시스턴트: {two_turn_answer}")
        
        # 이전 대화 (있는 경우)
        previous_question = request_data.get('previous_question', '').strip()
        previous_answer = request_data.get('previous_answer', '').strip()
        
        if previous_question and previous_answer:
            conversation_parts.append(f"사용자: {previous_question}")
            conversation_parts.append(f"어시스턴트: {previous_answer}")
        
        if conversation_parts:
            return "\n".join(conversation_parts)
        else:
            return "이전 대화 기록이 없습니다."
    
    def _build_prompt(self, query: str, conversation_history: str) -> str:
        """프롬프트 생성 (일반 챗봇용)"""
        
        prompt = f"""[기본 프롬프트 - 시스템 역할 및 지침]
당신은 전문 수의사 어시스턴트입니다. 사용자의 질문에 정확하고 책임감 있게 답변하십시오.
제공하는 정보는 참고용이며, 최종 진단 및 치료는 반드시 수의사와의 상담을 통해 이루어져야 합니다.
아래의 이전 대화 기록을 참고하여 사용자의 질문에 답변하세요.

[이전 대화 기록 - 최근 2턴]
{conversation_history}

[사용자 질문]
{query}

위의 이전 대화 기록을 참고하여 사용자의 질문에 답변해주세요. 답변은 다음과 같은 원칙을 따라주세요:
1. 정확하고 유용한 정보를 제공하되, 최종 진단은 수의사에게 맡겨야 한다는 점을 명시하세요.
2. 이전 대화 맥락을 고려하여 일관성 있는 답변을 제공하세요.
3. 보호자가 이해하기 쉽고 실용적인 조언을 제공하세요.
4. 긴급한 상황이나 심각한 증상의 경우 즉시 수의사 진료를 권고하세요.
5. 가능하다면 사용자의 질문에 유용한 답변을 제공해야 합니다."""
        
        return prompt
