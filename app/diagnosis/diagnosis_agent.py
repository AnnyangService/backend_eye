"""
Step3 진단 결과 LLM 보고서 생성 에이전트

염증류/각막류 진단 결과를 받아서 LLM을 통해 의료 보고서를 생성합니다.
현재는 임시로 포맷팅된 출력을 제공하며, 추후 LLM 호출 기능을 추가할 예정입니다.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MedicalDiagnosisAgent:
    """의료 진단 보고서 생성 에이전트"""
    
    def __init__(self):
        """초기화"""
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
            
            # LLM 보고서 생성 (현재는 임시 구현)
            report_parts = []
            report_parts.append(f"{category}으로 진단됩니다.")
            
            if not attribute_analysis:
                logger.warning("상세 분석 정보가 없음")
                return " ".join(report_parts)
            
            # 각 속성별 분석 결과를 자연스럽게 설명
            analysis_details = []
            for attr_name, attr_data in attribute_analysis.items():
                user_input = attr_data.get('user_input', '정보 없음')
                most_similar = attr_data.get('most_similar_disease', '알 수 없음')
                similarity = attr_data.get('similarity', 0.0)
                all_similarities = attr_data.get('all_similarities', {})
                
                # 로깅용 상세 정보
                logger.debug(f"[{attr_name}] 환자 증상: '{user_input}' → {most_similar} ({similarity:.1%})")
                
                # 구체적인 증상 설명 생성
                if most_similar in self.disease_rules and attr_name in self.disease_rules[most_similar]:
                    disease_symptom = self.disease_rules[most_similar][attr_name]
                    detail = f"환자가 입력한 {attr_name}은 {most_similar}의 증상인 '{disease_symptom}'와 {similarity:.1%} 유사합니다"
                else:
                    # fallback: 기존 방식
                    detail = f"환자가 입력한 {attr_name}은 {most_similar}의 특징적인 증상과 {similarity:.1%} 유사합니다"
                
                # 두 번째로 유사한 질병과 간단한 비교 추가
                if all_similarities and len(all_similarities) > 1:
                    other_diseases = [(disease, sim) for disease, sim in all_similarities.items() 
                                    if disease != most_similar]
                    if other_diseases:
                        other_diseases.sort(key=lambda x: x[1], reverse=True)
                        top_other_disease, top_other_sim = other_diseases[0]
                        detail += f". {top_other_disease}와는 {top_other_sim:.1%} 유사합니다"
                
                analysis_details.append(detail)
            
            if analysis_details:
                report_parts.append("분석 결과: " + ", ".join(analysis_details) + ".")
            
            # TODO: 여기서 실제 LLM 호출하여 더 자연스러운 보고서 생성
            logger.info(f"보고서 생성 완료: {category}")
            
            final_report = " ".join(report_parts)
            logger.debug(f"생성된 보고서: {final_report}")
            
            return final_report
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류: {str(e)}")
            return f"보고서 생성 실패: {str(e)}"
    
    def print_report(self, diagnosis_result: Dict[str, Any]) -> None:
        """
        진단 결과 보고서를 콘솔에 출력
        
        Args:
            diagnosis_result (dict): 진단 결과
        """
        report = self.generate_report(diagnosis_result)
        print(report)
    
    def call_llm_for_report(self, diagnosis_result: Dict[str, Any]) -> str:
        """
        LLM을 호출하여 자연어 의료 보고서 생성 (미구현)
        
        Args:
            diagnosis_result (dict): 진단 결과
            
        Returns:
            str: LLM이 생성한 의료 보고서
        """
        # TODO: LLM 호출 구현 예정
        logger.info("LLM 호출 기능은 추후 구현 예정")
        return "LLM 기반 보고서 생성 기능은 아직 구현되지 않았습니다."

# 편의 함수들
def generate_medical_report(diagnosis_result: Dict[str, Any]) -> str:
    """의료 보고서 생성 편의 함수"""
    agent = MedicalDiagnosisAgent()
    return agent.generate_report(diagnosis_result)

def print_medical_report(diagnosis_result: Dict[str, Any]) -> None:
    """의료 보고서 출력 편의 함수"""
    agent = MedicalDiagnosisAgent()
    agent.print_report(diagnosis_result) 