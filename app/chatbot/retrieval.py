"""
RAG 검색 서비스 (PostgreSQL 벡터 검색)
사용자 입력을 임베딩하고 PostgreSQL의 벡터 검색 기능을 사용하여 유사한 문서를 검색합니다.
"""

import os
import json
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sqlalchemy import text
from flask import current_app
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class RAGRetrievalService:
    """
    RAG 검색 서비스 (PostgreSQL 벡터 검색)
    사용자 질문을 임베딩하고 PostgreSQL의 pgvector 확장을 사용하여 유사한 문서 청크를 검색합니다.
    """
    
    def __init__(self, model_name: str = "madatnlp/km-bert", tokenizer_name: str = "snunlp/KR-BERT-char16424"):
        """
        RAGRetrievalService 초기화
        
        Args:
            model_name (str): 사용할 모델명 (기본값: km-bert)
            tokenizer_name (str): 사용할 토크나이저명
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self._load_model()
    
    def _load_model(self):
        """모델과 토크나이저를 로드합니다."""
        try:
            logger.info(f"모델 로드 시작: {self.model_name}")
            logger.info(f"토크나이저 로드 시작: {self.tokenizer_name}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            logger.info(f"토크나이저 로드 완료: {type(self.tokenizer).__name__}")
            
            # 모델 로드
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"모델 로드 완료, 디바이스: {self.device}")
            
            # 모델 정보 출력
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"총 파라미터 수: {total_params:,}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            raise Exception(f"RAGRetrievalService 초기화 실패: {str(e)}")
    
    def _get_db_session(self):
        """SQLAlchemy 세션을 반환합니다."""
        from app import db
        return db.session
    
    def generate_query_embedding(self, query: str, max_length: int = 512) -> np.ndarray:
        """
        사용자 질문의 임베딩을 생성합니다.
        
        Args:
            query (str): 사용자 질문
            max_length (int): 최대 토큰 길이
            
        Returns:
            np.ndarray: 질문 임베딩 벡터
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise Exception("모델이 로드되지 않았습니다.")
            
            # 토크나이징
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                max_length=max_length,
                padding=True,
                truncation=True
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 추론 실행
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # [CLS] 토큰의 임베딩을 사용
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                return cls_embedding[0]  # 배치 차원 제거
                
        except Exception as e:
            logger.error(f"질문 임베딩 생성 실패: {str(e)}")
            raise Exception(f"질문 임베딩 생성 실패: {str(e)}")
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        사용자 질문과 유사한 청크들을 PostgreSQL 벡터 검색으로 찾습니다.
        
        Args:
            query (str): 사용자 질문
            top_k (int): 반환할 최대 청크 수
            
        Returns:
            List[Dict[str, Any]]: 유사한 청크들과 유사도 점수
        """
        try:
            # 질문 임베딩 생성
            query_embedding = self.generate_query_embedding(query)
            
            # pgvector는 문자열 형태의 '[1,2,3,...]'을 vector로 파싱함
            embedding_str = "[" + ",".join(str(x) for x in query_embedding.tolist()) + "]"
            session = self._get_db_session()
            
            # embedding 파라미터를 쿼리문에 직접 문자열로 삽입
            sql = f"""
                SELECT 
                    id,
                    content,
                    keywords,
                    source,
                    1 - (embedding <=> '{embedding_str}'::vector) as similarity
                FROM documents 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT :top_k;
            """
            result = session.execute(text(sql), {'top_k': top_k})
            
            results = []
            for row in result.fetchall():
                results.append({
                    'chunk': {
                        'id': row[0],
                        'content': row[1],
                        'keywords': row[2] or {},
                        'source': row[3]
                    },
                    'similarity': float(row[4]),
                    'chunk_id': str(row[0])
                })
            
            logger.info(f"질문 '{query}'에 대한 벡터 검색 완료: {len(results)}개 결과")
            
            return results
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {str(e)}")
            raise Exception(f"벡터 검색 실패: {str(e)}")
    

