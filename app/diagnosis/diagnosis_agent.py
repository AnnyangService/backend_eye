"""
Step3 진단 결과 LLM 보고서 생성 에이전트

염증류/각막류 진단 결과를 받아서 LLM을 통해 의료 보고서를 생성합니다.
현재는 임시로 포맷팅된 출력을 제공하며, 추후 LLM 호출 기능을 추가할 예정입니다.
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
    
    # SDK cleint 초기화
    client = genai.Client(api_key=api_key)
    logger.info("제미나이 API 클라이언트 설정 완료")
    return client

def get_generation_config():
    """생성 설정 반환"""
    return types.GenerateContentConfig(
        temperature=0.3,  # 의료 보고서는 일관성 있게
        top_p=0.8,
        top_k=40,
        max_output_tokens=2048,
    )

class MedicalDiagnosisAgent:
    """의료 진단 보고서 생성 에이전트"""
    
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
        
        # 질병 규칙 정의
        self.disease_rules = {
            '안검염': {
                '분비물 특성': '비늘 같은 각질, 기름기 있는 분비물',
                '진행 속도': '점진적, 만성적',
                '주요 증상': '속눈썹 주변 각질, 눈꺼풀 붙음',
                '발생 패턴': '양안'
            },
            '비궤양성 각막염': {
                '분비물 특성': '미세한 분비물, 주로 눈물',
                '진행 속도': '점진적',
                '주요 증상': '눈 뜨는데 어려움 없음, 각막 표면 매끄러움, 눈물 흘림',
                '발생 패턴': '단안 또는 양안'
            },
            '결막염': {
                '분비물 특성': '수양성, 점액성, 화농성',
                '진행 속도': '급성, 점진적',
                '주요 증상': '눈을 비비는 행동, 충혈, 부종',
                '발생 패턴': '양안(알레르기성), 단안(감염성)'
            },
            '각막궤양': {
                '분비물 특성': '화농성, 점액성 분비물',
                '진행 속도': '급성, 빠른 진행',
                '주요 증상': '눈을 뜨기 힘듦, 눈물, 심한 통증, 시력 저하',
                '발생 패턴': '단안'
            },
            '각막부골편': {
                '분비물 특성': '수양성, 경미한 분비물',
                '진행 속도': '급성, 반복성',
                '주요 증상': '아침에 심함, 갑작스러운 심한 통증, 이물감, 깜빡임 시 통증',
                '발생 패턴': '단안 또는 양안'
            }
        }
        logger.info("MedicalDiagnosisAgent 초기화")
    
    def _call_gemini(self, prompt: str) -> str:
        """제미나이 API 호출"""
        if not self.api_available or not self.client:
            logger.warning("제미나이 API를 사용할 수 없습니다. 기본 응답을 반환합니다.")
            return "LLM 서비스를 사용할 수 없습니다."
        
        try:
            print(self.model_id)
            print(self.generation_config)
            print(prompt)
            print(self.client)
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=self.generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"제미나이 API 호출 중 오류: {str(e)}")
            return f"LLM 호출 실패: {str(e)}"
    
    
    def generate_report(self, diagnosis_result: Dict[str, Any]) -> str:
        """
        진단 결과를 받아서 의료 보고서를 생성
        
        Args:
            diagnosis_result (dict): 진단 결과
                {
                    "category": str,
                    "attribute_analysis": {
                        "속성명": {
                            "user_input": str,
                            "most_similar_disease": str,
                            "similarity": float,
                            "all_similarities": dict
                        }
                    }
                }
        
        Returns:
            str: 생성된 의료 보고서
        """
        try:
            if not diagnosis_result or 'category' not in diagnosis_result:
                return "진단 결과가 없습니다."
            
            category = diagnosis_result.get('category', '알 수 없음')
            attribute_analysis = diagnosis_result.get('attribute_analysis', {})
            
            logger.debug(f"보고서 생성 시작: {category}")
            
            if not attribute_analysis:
                logger.warning("상세 분석 정보가 없음")
                return f"진단 결과: {category}\n상세 분석 정보가 없습니다."
            
            # LLM을 사용한 자연스러운 보고서 생성
            if not self.api_available:
                logger.error("LLM API를 사용할 수 없습니다.")
                return "LLM API를 사용할 수 없어 보고서를 생성할 수 없습니다. API 설정을 확인해주세요."
            
            return self._generate_llm_report(category, attribute_analysis)
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류: {str(e)}")
            return f"보고서 생성 실패: {str(e)}"
    
    def _generate_llm_report(self, category: str, attribute_analysis: Dict[str, Any]) -> str:
        """LLM을 사용하여 자연스러운 의료 보고서 생성"""
        
        # 1단계: 컨텍스트 정보 구성
        context_info = self._build_context_info(category, attribute_analysis)
        
        # 2단계: 첫 번째 프롬프트 - 수의학 전문가 역할 설정 및 기본 가이드라인
        first_prompt = f"""
당신은 경험이 풍부한 수의학 안과 전문의입니다. 반려동물의 안과 질환 진단 결과를 바탕으로 보호자에게 설명할 의료 보고서를 작성해야 합니다.

중요한 주의사항:
1. 이 진단 결과는 참고용이며, 반드시 수의사의 직접 진료를 받아야 합니다.
2. 보호자가 과도하게 걱정하거나 맹신하지 않도록 신중하게 설명해야 합니다.
3. 의학적 전문 용어보다는 이해하기 쉬운 표현을 사용해야 합니다.
4. 추가 검사나 치료의 필요성을 언급해야 합니다.

다음은 반려동물의 안과 질환 분류와 주요 특징입니다:
{self._get_disease_knowledge()}

이 정보를 바탕으로 진단 결과를 설명할 준비가 되었나요? 준비되었다면 "준비완료"라고 답변해주세요.
"""
        
        first_response = self._call_gemini(first_prompt)
        logger.debug(f"첫 번째 프롬프트 응답: {first_response}")
        
        # 3단계: 두 번째 프롬프트 - 구체적인 진단 결과 분석 및 보고서 생성
        second_prompt = f"""
이제 구체적인 진단 결과를 분석하여 보호자를 위한 의료 보고서를 작성해주세요.

{context_info}

위의 정보를 바탕으로 다음 형식으로 의료 보고서를 작성해주세요:

1. 진단 결과 요약
2. 증상 분석 설명 (각 관찰된 증상이 어떤 질병의 특징과 얼마나 유사한지)
3. 주의사항 및 권고사항

보고서는 보호자가 이해하기 쉽고, 적절한 수준의 안심과 주의를 줄 수 있도록 작성해주세요.
"""
        
        final_response = self._call_gemini(second_prompt)
        logger.info(f"LLM 기반 보고서 생성 완료: {category}")
        
        return final_response
    
    def _build_context_info(self, category: str, attribute_analysis: Dict[str, Any]) -> str:
        """진단 결과에 대한 컨텍스트 정보 구성"""
        
        context_parts = [f"진단된 질병 분류: {category}"]
        
        # 각 속성별 분석 결과 정리
        for attr_name, attr_data in attribute_analysis.items():
            user_input = attr_data.get('user_input', '정보 없음')
            most_similar = attr_data.get('most_similar_disease', '알 수 없음')
            similarity = attr_data.get('similarity', 0.0)
            all_similarities = attr_data.get('all_similarities', {})
            
            # 관찰된 증상과 가장 유사한 질병
            context_parts.append(f"\n[{attr_name}]")
            context_parts.append(f"- 관찰된 증상: '{user_input}'")
            context_parts.append(f"- 가장 유사한 질병: {most_similar} (유사도: {similarity:.1%})")
            
            # 질병별 특징적 증상 정보 추가
            if most_similar in self.disease_rules and attr_name in self.disease_rules[most_similar]:
                disease_symptom = self.disease_rules[most_similar][attr_name]
                context_parts.append(f"- {most_similar}의 {attr_name} 특징: {disease_symptom}")
            
            # 다른 질병과의 유사도 비교
            if all_similarities and len(all_similarities) > 1:
                other_diseases = [(disease, sim) for disease, sim in all_similarities.items() 
                                if disease != most_similar]
                if other_diseases:
                    other_diseases.sort(key=lambda x: x[1], reverse=True)
                    top_others = other_diseases[:2]  # 상위 2개만
                    other_comparisons = [f"{disease} ({sim:.1%})" for disease, sim in top_others]
                    context_parts.append(f"- 다른 질병과의 유사도: {', '.join(other_comparisons)}")
        
        return "\n".join(context_parts)
    
    def _get_disease_knowledge(self) -> str:
        """질병 정보를 문자열로 구성"""
        
        knowledge_parts = []
        for disease, rules in self.disease_rules.items():
            knowledge_parts.append(f"\n[{disease}]")
            for attr, description in rules.items():
                knowledge_parts.append(f"- {attr}: {description}")
        
        return "\n".join(knowledge_parts)
    
 

