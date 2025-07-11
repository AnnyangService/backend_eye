"""
키워드 추출 유틸리티
문서에서 키워드를 추출하는 기능
"""

from typing import List
from .keywords import GLOBAL_MEDICAL_KEYWORDS


def extract_keywords_from_text(text: str) -> List[str]:
    """
    텍스트에서 키워드를 추출합니다.
    
    Args:
        text (str): 분석할 텍스트
        
    Returns:
        List[str]: 추출된 키워드 리스트
    """
    extracted_keywords = []
    
    # 모든 키워드와 동의어를 확인
    for keyword, synonyms in GLOBAL_MEDICAL_KEYWORDS.items():
        # 메인 키워드 확인
        if keyword in text:
            extracted_keywords.append(keyword)
            continue
            
        # 동의어 확인
        for synonym in synonyms:
            if synonym in text:
                extracted_keywords.append(keyword)
                break
    
    return list(set(extracted_keywords))  # 중복 제거 