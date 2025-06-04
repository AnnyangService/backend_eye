"""
유사도 계산 유틸리티

진단 시스템에서 사용되는 다양한 유사도 계산 함수들을 제공합니다.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    두 임베딩 간의 코사인 유사도를 계산합니다.
    
    Args:
        embedding1 (np.ndarray): 첫 번째 임베딩
        embedding2 (np.ndarray): 두 번째 임베딩
        
    Returns:
        float: 코사인 유사도 (0.0 ~ 1.0)
    """
    try:
        # 벡터 정규화
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 코사인 유사도 계산
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # 0~1 범위로 정규화 (코사인 유사도는 -1~1 범위이므로)
        similarity = (similarity + 1) / 2
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"코사인 유사도 계산 실패: {str(e)}")
        return 0.0

def calculate_euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    두 임베딩 간의 유클리드 거리를 계산합니다.
    
    Args:
        embedding1 (np.ndarray): 첫 번째 임베딩
        embedding2 (np.ndarray): 두 번째 임베딩
        
    Returns:
        float: 유클리드 거리
    """
    try:
        return float(np.linalg.norm(embedding1 - embedding2))
    except Exception as e:
        logger.error(f"유클리드 거리 계산 실패: {str(e)}")
        return float('inf')

def calculate_manhattan_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    두 임베딩 간의 맨하탄 거리를 계산합니다.
    
    Args:
        embedding1 (np.ndarray): 첫 번째 임베딩
        embedding2 (np.ndarray): 두 번째 임베딩
        
    Returns:
        float: 맨하탄 거리
    """
    try:
        return float(np.sum(np.abs(embedding1 - embedding2)))
    except Exception as e:
        logger.error(f"맨하탄 거리 계산 실패: {str(e)}")
        return float('inf') 