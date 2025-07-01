"""
Step3 진단 결과 LLM 보고서 생성 에이전트

염증류/각막류 진단 결과를 받아서 LLM을 통해 의료 보고서를 생성합니다.
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
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=self.generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"제미나이 API 호출 중 오류: {str(e)}")
            return f"LLM 호출 실패: {str(e)}"
    
    
    def generate_report(self, diagnosis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        진단 결과를 받아서 의료 보고서를 생성 (요약과 상세 정보 분리)
        
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
            dict: {
                "summary": str,      # 핵심 요약
                "details": str,      # 상세 보고서
                "attribute_analysis": dict  # 속성별 분석 결과
            }
        """
        try:
            if not diagnosis_result or 'category' not in diagnosis_result:
                return {
                    "summary": "진단 결과가 없습니다.",
                    "details": "진단 결과가 없습니다.",
                    "attribute_analysis": {}
                }
            
            category = diagnosis_result.get('category', '알 수 없음')
            attribute_analysis = diagnosis_result.get('attribute_analysis', {})
            
            logger.debug(f"보고서 생성 시작: {category}")
            
            if not attribute_analysis:
                logger.warning("상세 분석 정보가 없음")
                return {
                    "summary": f"진단 결과: {category}",
                    "details": f"진단 결과: {category}\n상세 분석 정보가 없습니다.",
                    "attribute_analysis": {}
                }
            
            # 요약 생성 (LLM 없이)
            summary = self._generate_summary(category, attribute_analysis)
            
            # 상세 보고서 생성 (LLM 1번째 호출)
            if not self.api_available:
                logger.error("LLM API를 사용할 수 없습니다.")
                details = "LLM API를 사용할 수 없어 상세 보고서를 생성할 수 없습니다. API 설정을 확인해주세요."
                enhanced_analysis = attribute_analysis
            else:
                details = self._generate_disease_details(category, attribute_analysis)
                enhanced_analysis = self._generate_attribute_analysis(category, attribute_analysis)
            
            return {
                "summary": summary,
                "details": details,
                "attribute_analysis": enhanced_analysis
            }
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류: {str(e)}")
            return {
                "summary": f"보고서 생성 실패: {str(e)}",
                "details": f"보고서 생성 실패: {str(e)}",
                "attribute_analysis": {}
            }
    
    def _generate_disease_details(self, category: str, attribute_analysis: Dict[str, Any]) -> str:
        """LLM을 사용하여 진단된 질병의 특성과 규칙을 기반으로 상세 보고서 생성 (2단계 프롬프트 체이닝)"""
        
        # 진단된 질병의 특성 정보 가져오기
        disease_info = self.disease_rules.get(category, {})
        
        # 컨텍스트 정보 구성 (summary와 겹치지 않도록)
        context_info = self._build_disease_context(category, disease_info, attribute_analysis)
        
        # 1단계: 초기 보고서 생성
        first_prompt = f"""
당신은 반려동물의 안과 질환을 분석하는 전문가입니다. 진단 결과를 바탕으로 보호자에게 설명할 상세 보고서를 작성해주세요.

다음 원칙을 반드시 지켜주세요:
1. 이 진단 결과는 참고용이며, 반드시 수의사의 직접 진료를 받아야 합니다.
2. 보호자가 과도하게 걱정하거나 맹신하지 않도록 신중하게 설명해야 합니다.
3. 의학적 전문 용어보다는 이해하기 쉬운 표현을 사용해야 합니다.
4. 추가 검사나 치료의 필요성을 언급해야 합니다.

다음 형식으로 작성해주세요:

# {category} 진단 보고서

(간단한 진단 결과 요약 - 2-3문장)

# 1. {category}란?

## [질병 개요]
({category}에 대한 기본 설명)

## [주요 증상]
({category}의 특징적인 증상들)

# 2. 관찰된 증상과의 일치도

## [일치하는 증상]
(관찰된 증상 중 {category}와 일치하는 부분)

## [주의할 증상]
(관찰된 증상 중 {category}와 다른 부분이나 주의점)

# 3. 주의사항 및 권고사항

## [치료 방향]
(치료에 대한 권고사항)

## [예후 및 관리]
(예후와 관리 방법)

진단 정보:
{context_info}

다음은 반려동물의 안과 질환 분류와 주요 특징입니다:
{self._get_disease_knowledge()}

위의 정보를 바탕으로 {category}에 대한 상세 보고서를 작성해주세요.
"""
        
        first_response = self._call_gemini(first_prompt)
        
        # 2단계: 첫 번째 응답을 받아서 다듬기
        second_prompt = f"""
다음은 반려동물의 안과 질환 진단 보고서 초안입니다:

{first_response}

반드시 다음 형식으로 정확히 작성해주세요:

# {category} 진단 보고서

(간단한 진단 결과 요약 - 2-3문장)

# 1. {category}란?

## [질병 개요]
({category}에 대한 기본 설명)

## [주요 증상]
({category}의 특징적인 증상들)

# 2. 관찰된 증상과의 일치도

## [일치하는 증상]
(관찰된 증상 중 {category}와 일치하는 부분)

## [주의할 증상]
(관찰된 증상 중 {category}와 다른 부분이나 주의점)

# 3. 주의사항 및 권고사항

## [치료 방향]
(치료에 대한 권고사항)

## [예후 및 관리]
(예후와 관리 방법)

중요한 지시사항:
- 반드시 위의 형식을 정확히 따라주세요
- 대제목은 #으로 시작하고 1. 2. 3. 숫자를 붙여주세요
- 소제목은 ##으로 시작하고 []로 감싸주세요
- "~~했음", "~~임" 같은 말투는 사용하지 마세요
- "다음은 개선된 보고서입니다" 같은 문장은 사용하지 마세요
- 동물병원 수의사 컨셉은 제거하고 객관적인 분석 보고서로 작성해주세요
- 보호자가 이해하기 쉽고 적절한 수준의 안심과 주의를 줄 수 있도록 작성해주세요
"""
        
        final_response = self._call_gemini(second_prompt)
        
        return final_response
    
    def _build_disease_context(self, category: str, disease_info: Dict[str, str], attribute_analysis: Dict[str, Any]) -> str:
        """진단된 질병의 특성과 규칙을 기반으로 컨텍스트 정보 구성 (details용)"""
        
        context_parts = [f"진단된 질병: {category}"]
        
        # 진단된 질병의 특성 정보
        if disease_info:
            context_parts.append("\n[질병의 주요 특징]")
            for attr_name, description in disease_info.items():
                context_parts.append(f"• {attr_name}: {description}")
        
        # 관찰된 증상과의 일치도 분석
        context_parts.append("\n[관찰된 증상과의 일치도]")
        for attr_name, attr_data in attribute_analysis.items():
            user_input = attr_data.get('user_input', '정보 없음')
            most_similar = attr_data.get('most_similar_disease', '알 수 없음')
            similarity = attr_data.get('similarity', 0.0)
            
            context_parts.append(f"• {attr_name}: '{user_input}' → {most_similar}과 {similarity:.1%} 일치")
        
        return "\n".join(context_parts)
    
    def _generate_attribute_analysis(self, category: str, attribute_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """LLM을 사용하여 속성별 상세 분석 생성 (attribute_analysis용, 2단계 프롬프트 체이닝)"""
        
        enhanced_analysis = {}
        
        for attr_name, attr_data in attribute_analysis.items():
            user_input = attr_data.get('user_input', '정보 없음')
            most_similar = attr_data.get('most_similar_disease', '알 수 없음')
            similarity = attr_data.get('similarity', 0.0)
            all_similarities = attr_data.get('all_similarities', {})
            
            # 1단계: 초기 증상 분석
            first_prompt = f"""
당신은 고양이의 눈 질병 증상을 분석하는 전문가입니다. 특정 증상이 어떤 질병과 일치하는지 분석해주세요.

분석 원칙:
1. 객관적이고 정확한 분석을 제공해야 합니다.
2. 의학적 전문 용어보다는 이해하기 쉬운 표현을 사용해야 합니다.
3. 추가로 확인해야 할 증상이나 주의사항을 언급해야 합니다.

분석 대상 속성: {attr_name}
관찰된 증상: {user_input}
가장 유사한 질병: {most_similar} (유사도: {similarity:.1%})

다른 질병과의 유사도:
{self._format_similarities(all_similarities)}

다음은 반려동물의 안과 질환 분류와 주요 특징입니다:
{self._get_disease_knowledge()}

이 증상에 대해 다음 내용을 중심으로 분석해주세요:
- 이 증상이 {most_similar}과 얼마나 일치하는지
- 다른 질병과의 차이점
"""
            
            try:
                first_response = self._call_gemini(first_prompt)
                
                # 2단계: 첫 번째 응답을 받아서 다듬기
                second_prompt = f"""
다음은 증상 분석 초안입니다:

{first_response}

반드시 다음 형식으로 정확히 작성해주세요:

# 1. {category}와 유사성

제공된 정보에 따르면 {most_similar}과 {similarity:.1%} 일치합니다.

## [일치하는 점]
(이 증상이 {most_similar}과 일치하는 부분을 설명)

## [주의할 점]
(이 증상에서 {most_similar}과 다른 부분이나 주의해야 할 점을 설명)

# 2. 다른 질병과의 차이점

## [차이점 분석]
(다른 질병들과의 차이점을 설명)

중요한 지시사항:
- 반드시 위의 형식을 정확히 따라주세요
- 대제목은 #으로 시작하고 1. 2. 숫자를 붙여주세요
- 소제목은 ##으로 시작하고 []로 감싸주세요
- "~~했음", "~~임" 같은 말투는 사용하지 마세요
- "다음은 개선된 분석입니다" 같은 문장은 사용하지 마세요
- 간결하고 명확하게 작성해주세요
"""
                
                llm_analysis = self._call_gemini(second_prompt)
                enhanced_analysis[attr_name] = {
                    "llm_analysis": llm_analysis
                }
                
            except Exception as e:
                enhanced_analysis[attr_name] = {
                    "llm_analysis": f"분석 실패: {str(e)}"
                }
        
        return enhanced_analysis
    
    def _format_similarities(self, all_similarities: Dict[str, float]) -> str:
        """유사도 정보를 포맷팅"""
        if not all_similarities:
            return "정보 없음"
        
        formatted = []
        for disease, similarity in sorted(all_similarities.items(), key=lambda x: x[1], reverse=True):
            formatted.append(f"• {disease}: {similarity:.1%}")
        
        return "\n".join(formatted)
    
    def _build_context_info(self, category: str, attribute_analysis: Dict[str, Any]) -> str:
        """진단 결과에 대한 컨텍스트 정보 구성 (기존 메서드 유지)"""
        
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
    
    def _generate_summary(self, category: str, attribute_analysis: Dict[str, Any]) -> str:
        """핵심 진단 요약 생성"""
        
        summary_parts = [f"🔍 진단 결과: {category}"]
        
        # 각 속성별 분석 결과 요약
        for attr_name, attr_data in attribute_analysis.items():
            most_similar = attr_data.get('most_similar_disease', '알 수 없음')
            similarity = attr_data.get('similarity', 0.0)
            summary_parts.append(f"• {attr_name}: {most_similar} ({similarity:.1%} 유사)")
        
        # 전체 유사도 분석 (상위 3개 질병)
        if attribute_analysis:
            all_diseases = {}
            for attr_data in attribute_analysis.values():
                all_similarities = attr_data.get('all_similarities', {})
                for disease, sim in all_similarities.items():
                    if disease not in all_diseases:
                        all_diseases[disease] = []
                    all_diseases[disease].append(sim)
            
            # 각 질병의 평균 유사도 계산
            avg_similarities = {}
            for disease, similarities in all_diseases.items():
                avg_similarities[disease] = sum(similarities) / len(similarities)
            
            # 상위 3개 질병 정렬
            top_diseases = sorted(avg_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
            
            summary_parts.append("")
            summary_parts.append("📊 전체 유사도 분석:")
            for disease, avg_sim in top_diseases:
                summary_parts.append(f"• {disease}: {avg_sim:.1%}")
        
        return "\n".join(summary_parts)
    
    def _get_disease_knowledge(self) -> str:
        """질병 정보를 문자열로 구성"""
        
        knowledge_parts = []
        for disease, rules in self.disease_rules.items():
            knowledge_parts.append(f"\n[{disease}]")
            for attr, description in rules.items():
                knowledge_parts.append(f"- {attr}: {description}")
        
        return "\n".join(knowledge_parts)
    
 

