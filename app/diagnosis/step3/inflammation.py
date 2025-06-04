"""
염증류 진단 로직
"""

import logging
import numpy as np
import os
from typing import Dict, List
from ..util.similarity_calculator import calculate_cosine_similarity

logger = logging.getLogger(__name__)

class InflammationDiagnosis:
    """염증류 진단 클래스"""
    
    def __init__(self):
        """초기화"""
        self.vectorizer = None
        # 염증류 질병 목록
        self.inflammation_diseases = ['결막염', '비궤양성 각막염', '안검염']
        # ID별 속성 매핑
        self.attribute_mapping = {
            1: '분비물 특성',
            2: '진행 속도', 
            3: '주요 증상',
            4: '발생 패턴'
        }
        # rule_embeddings 디렉토리 경로
        self.embeddings_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'rule_embeddings')
        logger.info("InflammationDiagnosis 초기화")
    
    def _get_vectorizer(self):
        """RuleVectorizer 인스턴스를 가져옵니다."""
        if self.vectorizer is None:
            try:
                from ..util.rule_vectorizer import RuleVectorizer
                self.vectorizer = RuleVectorizer(use_medical_model=True)
                logger.info("RuleVectorizer 초기화 완료")
            except Exception as e:
                logger.error(f"RuleVectorizer 초기화 실패: {str(e)}")
                raise Exception(f"벡터화 모델 로드 실패: {str(e)}")
        return self.vectorizer
    
    def _load_disease_embedding(self, disease_name: str, attribute_name: str) -> np.ndarray:
        """
        특정 질병의 특정 속성 임베딩을 로드합니다.
        
        Args:
            disease_name (str): 질병명
            attribute_name (str): 속성명
            
        Returns:
            np.ndarray: 임베딩 벡터
        """
        try:
            filename = f"{disease_name}_{attribute_name}.npy"
            filepath = os.path.join(self.embeddings_dir, filename)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"임베딩 파일을 찾을 수 없습니다: {filepath}")
            
            embedding = np.load(filepath)
            logger.debug(f"임베딩 로드 완료: {filename} (차원: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 로드 실패: {disease_name}_{attribute_name} - {str(e)}")
            raise Exception(f"임베딩 로드 실패: {str(e)}")
    
    def diagnose(self, attributes):
        """
        염증류 세부 진단 수행
        
        Args:
            attributes (list): 진단 속성 리스트 [{"id": int, "description": str}, ...]
            
        Returns:
            dict: 진단 결과 {"category": str, "attribute_analysis": dict}
        """
        try:
            logger.info("염증류 진단 시작")
            logger.debug(f"입력 속성: {attributes}")
            
            # RuleVectorizer 초기화
            vectorizer = self._get_vectorizer()
            
            # 속성을 딕셔너리로 변환하여 쉽게 접근
            attr_dict = {attr.get('id'): attr.get('description', '') for attr in attributes if attr.get('id') is not None}
            
            # 상세 분석 데이터 저장용
            detailed_analysis = {
                'category': None,
                'attribute_analysis': {},  # 각 속성별로 가장 유사한 질병과 유사도
                'disease_scores': {},  # 각 질병별 종합 점수 (디버깅용)
                'overall_score': 0.0
            }
            
            # 임시로 모든 속성별 유사도를 저장할 딕셔너리
            temp_similarities = {}
            
            # 각 속성별로 모든 질병과의 유사도 계산
            for attr_id, description in attr_dict.items():
                if attr_id in self.attribute_mapping:
                    attribute_name = self.attribute_mapping[attr_id]
                    
                    # 해당 속성에 대한 모든 질병의 유사도 저장
                    temp_similarities[attribute_name] = {
                        'user_input': description,
                        'disease_similarities': {}
                    }
                    
                    # 사용자 입력 임베딩
                    try:
                        user_embedding = vectorizer.vectorize_text(description)
                    except Exception as e:
                        logger.warning(f"사용자 입력 임베딩 실패 ({attribute_name}): {str(e)}")
                        continue
                    
                    # 각 염증류 질병과 비교
                    for disease_name in self.inflammation_diseases:
                        try:
                            # 질병의 해당 속성 임베딩 로드
                            disease_embedding = self._load_disease_embedding(disease_name, attribute_name)
                            
                            # 유사도 계산 (util 함수 사용)
                            similarity = calculate_cosine_similarity(user_embedding, disease_embedding)
                            
                            # 임시 저장
                            temp_similarities[attribute_name]['disease_similarities'][disease_name] = similarity
                            
                        except Exception as e:
                            logger.warning(f"  {disease_name} - {attribute_name} 유사도 계산 실패: {str(e)}")
                            temp_similarities[attribute_name]['disease_similarities'][disease_name] = 0.0
                    
                    similarities_str = ", ".join([f"{disease}: {sim:.3f}" for disease, sim in temp_similarities[attribute_name]['disease_similarities'].items()])
                    logger.info(f"[{attribute_name}] 사용자 입력: '{description}' → 유사도: {similarities_str}")
            
            # 각 속성별로 가장 유사한 질병 찾기
            for attribute_name, attr_data in temp_similarities.items():
                if attr_data['disease_similarities']:
                    # 가장 높은 유사도를 가진 질병 찾기
                    best_disease_for_attr = max(attr_data['disease_similarities'].keys(), 
                                              key=lambda x: attr_data['disease_similarities'][x])
                    best_similarity_for_attr = attr_data['disease_similarities'][best_disease_for_attr]
                    
                    detailed_analysis['attribute_analysis'][attribute_name] = {
                        'user_input': attr_data['user_input'],
                        'most_similar_disease': best_disease_for_attr,
                        'similarity': best_similarity_for_attr,
                        'all_similarities': attr_data['disease_similarities']  # 디버깅용
                    }
            
            # 각 질병별 평균 유사도 계산
            for disease_name in self.inflammation_diseases:
                total_similarity = 0.0
                valid_attributes = 0
                attribute_scores = {}
                
                for attribute_name, attr_data in temp_similarities.items():
                    if disease_name in attr_data['disease_similarities']:
                        similarity = attr_data['disease_similarities'][disease_name]
                        attribute_scores[attribute_name] = similarity
                        total_similarity += similarity
                        valid_attributes += 1
                
                if valid_attributes > 0:
                    avg_similarity = total_similarity / valid_attributes
                    detailed_analysis['disease_scores'][disease_name] = {
                        'avg_similarity': avg_similarity,
                        'attribute_scores': attribute_scores,
                        'valid_attributes': valid_attributes
                    }
                    logger.info(f"{disease_name} 평균 유사도: {avg_similarity:.4f}")
            
            # 가장 높은 유사도를 가진 질병 선택
            if not detailed_analysis['disease_scores']:
                raise Exception("유효한 진단 결과를 생성할 수 없습니다.")
            
            best_disease = max(detailed_analysis['disease_scores'].keys(), 
                             key=lambda x: detailed_analysis['disease_scores'][x]['avg_similarity'])
            best_score = detailed_analysis['disease_scores'][best_disease]['avg_similarity']
            
            # 상세 분석에 최종 결과 저장
            detailed_analysis['category'] = best_disease
            detailed_analysis['overall_score'] = best_score
            
            result = {
                "category": best_disease,
                "attribute_analysis": detailed_analysis['attribute_analysis']
            }
            
            logger.info(f"염증류 진단 완료: {best_disease} (유사도: {best_score:.4f})")
            
            # MedicalDiagnosisAgent를 통해 보고서 생성
            try:
                from ..diagnosis_agent import MedicalDiagnosisAgent
                agent = MedicalDiagnosisAgent()
                report = agent.generate_report(result)
                logger.info("의료 보고서 생성 완료")
                
                return {
                    "category": best_disease,
                    "description": report
                }
                
            except Exception as e:
                logger.warning(f"보고서 생성 실패, 기본 결과 반환: {str(e)}")
                return {
                    "category": best_disease,
                    "description": f"{best_disease}으로 진단되었습니다. (유사도: {best_score:.1%})"
                }
            
        except Exception as e:
            logger.error(f"염증류 진단 중 오류: {str(e)}")
            # 임시 응답 반환 (오류 시)
            return {
                "category": "진단_실패",
                "description": f"염증류 진단 처리 중 오류가 발생했습니다: {str(e)}"
            }
    