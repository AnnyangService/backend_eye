"""
RAG용 임베딩 생성 모듈
km-bert를 사용하여 청킹된 문서의 임베딩을 생성하고 별도로 저장합니다.
"""

import torch
import numpy as np
import json
import os
import logging
from typing import List, Dict, Any
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class RAGEmbeddingGenerator:
    """
    RAG 시스템을 위한 임베딩 생성기
    km-bert를 사용하여 청킹된 문서의 임베딩을 생성하고 별도로 저장합니다.
    """
    
    def __init__(self, model_name: str = "madatnlp/km-bert", tokenizer_name: str = "snunlp/KR-BERT-char16424"):
        """
        RAGEmbeddingGenerator 초기화
        
        Args:
            model_name (str): 사용할 모델명 (기본값: km-bert)
            tokenizer_name (str): 사용할 토크나이저명
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # embeddings 디렉토리 생성
        self.embeddings_dir = "C:/Users/jjj53/Desktop/Hi_MeoW/backend_eye/app/rag/embeddings"
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
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
            raise Exception(f"RAGEmbeddingGenerator 초기화 실패: {str(e)}")
    
    def generate_embeddings_batch(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        여러 텍스트의 임베딩을 배치로 생성합니다.
        
        Args:
            texts (List[str]): 임베딩을 생성할 텍스트 리스트
            max_length (int): 최대 토큰 길이
            
        Returns:
            np.ndarray: 임베딩 벡터들 (N x 768)
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise Exception("모델이 로드되지 않았습니다.")
            
            if not texts:
                return np.array([])
            
            # 토크나이징 (배치 처리)
            inputs = self.tokenizer(
                texts,
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
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                return cls_embeddings
                
        except Exception as e:
            logger.error(f"배치 임베딩 생성 실패: {str(e)}")
            raise Exception(f"배치 임베딩 생성 실패: {str(e)}")
    
    def save_embeddings(self, chunk_ids: List[str], embeddings: np.ndarray) -> Dict[str, str]:
        """
        임베딩들을 개별 .npy 파일로 저장합니다.
        
        Args:
            chunk_ids (List[str]): 청킹 ID 리스트
            embeddings (np.ndarray): 임베딩 배열 (N x 768)
            
        Returns:
            Dict[str, str]: 청킹 ID와 임베딩 파일 경로의 매핑
        """
        try:
            embedding_paths = {}
            
            for i, chunk_id in enumerate(chunk_ids):
                # 파일명 생성 (특수문자 처리)
                safe_filename = chunk_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                embedding_file = os.path.join(self.embeddings_dir, f"{safe_filename}.npy")
                
                # 임베딩 저장
                np.save(embedding_file, embeddings[i])
                embedding_paths[chunk_id] = embedding_file
                
                logger.debug(f"임베딩 저장: {chunk_id} -> {embedding_file}")
            
            logger.info(f"총 {len(chunk_ids)}개의 임베딩을 저장했습니다.")
            return embedding_paths
            
        except Exception as e:
            logger.error(f"임베딩 저장 실패: {str(e)}")
            raise Exception(f"임베딩 저장 실패: {str(e)}")
    
    def generate_and_save_embeddings(self, chunks: List[Dict[str, Any]], max_length: int = 512) -> Dict[str, str]:
        """
        청킹된 문서들의 임베딩을 생성하고 저장합니다.
        
        Args:
            chunks (List[Dict[str, Any]]): 청킹된 문서 리스트
            max_length (int): 최대 토큰 길이
            
        Returns:
            Dict[str, str]: 청킹 ID와 임베딩 파일 경로의 매핑
        """
        try:
            if not chunks:
                return {}
            
            # content 텍스트들과 ID 추출
            contents = [chunk['content'] for chunk in chunks]
            chunk_ids = [chunk['id'] for chunk in chunks]
            
            # 배치로 임베딩 생성
            embeddings = self.generate_embeddings_batch(contents, max_length)
            
            # 임베딩 저장
            embedding_paths = self.save_embeddings(chunk_ids, embeddings)
            
            logger.info(f"총 {len(chunks)}개의 청킹에 대한 임베딩을 생성하고 저장했습니다.")
            return embedding_paths
            
        except Exception as e:
            logger.error(f"청킹 임베딩 생성 및 저장 실패: {str(e)}")
            raise Exception(f"청킹 임베딩 생성 및 저장 실패: {str(e)}")
    
    def process_chunks_file(self, chunks_file: str, save_mapping: bool = True):
        """
        저장된 청킹 데이터를 불러와서 임베딩을 생성하고 저장합니다.
        
        Args:
            chunks_file (str): 청킹 데이터 파일 경로
            save_mapping (bool): 청킹 ID와 임베딩 파일 경로 매핑을 저장할지 여부
        """
        try:
            # 청킹 데이터 불러오기
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            logger.info(f"청킹 데이터를 불러왔습니다: {chunks_file}")
            print(f"📄 청킹 데이터를 불러왔습니다: {len(chunks)}개")
            
            # 임베딩 생성 및 저장
            embedding_paths = self.generate_and_save_embeddings(chunks)
            
            print(f"✅ 총 {len(embedding_paths)}개의 임베딩을 생성하고 저장했습니다.")
            print(f"📁 임베딩 저장 경로: {self.embeddings_dir}/")
            
            return embedding_paths
            
        except Exception as e:
            logger.error(f"청킹 데이터 임베딩 생성 실패: {str(e)}")
            raise Exception(f"청킹 데이터 임베딩 생성 실패: {str(e)}")
    
    def load_embedding(self, chunk_id: str) -> np.ndarray:
        """
        특정 청킹의 임베딩을 로드합니다.
        
        Args:
            chunk_id (str): 청킹 ID
            
        Returns:
            np.ndarray: 임베딩 벡터
        """
        try:
            # 파일명 생성
            safe_filename = chunk_id.replace('/', '_').replace('\\', '_').replace(':', '_')
            embedding_file = os.path.join(self.embeddings_dir, f"{safe_filename}.npy")
            
            if not os.path.exists(embedding_file):
                raise FileNotFoundError(f"임베딩 파일을 찾을 수 없습니다: {embedding_file}")
            
            # 임베딩 로드
            embedding = np.load(embedding_file)
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 로드 실패: {str(e)}")
            raise Exception(f"임베딩 로드 실패: {str(e)}")


if __name__ == "__main__":
    # 청킹 데이터 파일 경로
    chunks_file = "C:/Users/jjj53/Desktop/Hi_MeoW/backend_eye/app/rag/chunks/chunks.json"
    
    if os.path.exists(chunks_file):
        # 임베딩 생성기 초기화
        embedding_generator = RAGEmbeddingGenerator()
        
        # 임베딩 생성 및 저장
        embedding_paths = embedding_generator.process_chunks_file(chunks_file)
        
        print(f"✅ 총 {len(embedding_paths)}개의 임베딩을 생성하고 저장했습니다.")
        
    else:
        print(f"❌ 청킹 데이터 파일을 찾을 수 없습니다: {chunks_file}")
        print("먼저 chunk_storage.py를 실행하여 청킹 데이터를 생성해주세요.")