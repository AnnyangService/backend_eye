import torch
import numpy as np
import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Union
from transformers import AutoModel, AutoTokenizer
from flask import current_app

logger = logging.getLogger(__name__)

class RuleVectorizer:
    """
    KLUE RoBERTa 또는 KM-BERT를 사용하여 진단 룰을 벡터화하는 클래스
    """
    
    def __init__(self, model_name: str = "klue/roberta-small", use_medical_model: bool = False):
        """
        RuleVectorizer 초기화
        
        Args:
            model_name (str): 사용할 모델명
            use_medical_model (bool): 의료 특화 모델 사용 여부
        """
        if use_medical_model:
            self.model_name = "madatnlp/km-bert"
            self.tokenizer_name = "snunlp/KR-BERT-char16424"
            print("🏥 한국어 의료 특화 KM-BERT 모델을 사용합니다.")
        else:
            self.model_name = model_name
            self.tokenizer_name = model_name
            print("🤖 일반 언어모델을 사용합니다.")
        
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
            
            # 토크나이저 로드 (KM-BERT의 경우 별도 토크나이저 사용)
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
            
            # 모델 타입 출력
            if "km-bert" in self.model_name.lower():
                logger.info("🏥 한국어 의료 도메인 특화 모델 로드 완료")
            else:
                logger.info("🤖 일반 언어모델 로드 완료")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            raise Exception(f"RuleVectorizer 초기화 실패: {str(e)}")
    
    def vectorize_text(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        단일 텍스트를 벡터화합니다.
        
        Args:
            text (str): 벡터화할 텍스트
            max_length (int): 최대 토큰 길이
            
        Returns:
            np.ndarray: 벡터화된 임베딩 (768차원)
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise Exception("모델이 로드되지 않았습니다.")
            
            # 토크나이징
            inputs = self.tokenizer(
                text,
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
                
                # [CLS] 토큰의 임베딩을 사용 (첫 번째 토큰)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # 배치 차원 제거하여 1차원 벡터로 반환
                return cls_embedding.squeeze()
                
        except Exception as e:
            logger.error(f"텍스트 벡터화 실패: {str(e)}")
            raise Exception(f"텍스트 벡터화 실패: {str(e)}")
    
    def vectorize_texts(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        여러 텍스트를 배치로 벡터화합니다.
        
        Args:
            texts (List[str]): 벡터화할 텍스트 리스트
            max_length (int): 최대 토큰 길이
            
        Returns:
            np.ndarray: 벡터화된 임베딩들 (N x 768)
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
            logger.error(f"텍스트 배치 벡터화 실패: {str(e)}")
            raise Exception(f"텍스트 배치 벡터화 실패: {str(e)}")
    
    def vectorize_rule_description(self, rule_name: str, description: str, target_name: Optional[str] = None) -> Dict:
        """
        진단 룰의 설명을 벡터화합니다.
        
        Args:
            rule_name (str): 룰 이름
            description (str): 룰 설명
            target_name (Optional[str]): 진단 대상 이름
            
        Returns:
            Dict: 벡터화 결과
        """
        try:
            # 컨텍스트가 있는 텍스트 생성
            if target_name:
                context_text = f"[{target_name}] {rule_name}: {description}"
            else:
                context_text = f"{rule_name}: {description}"
            
            # 벡터화 수행
            embedding = self.vectorize_text(context_text)
            
            # 결과 반환
            result = {
                'rule_name': rule_name,
                'description': description,
                'target_name': target_name,
                'context_text': context_text,
                'embedding': embedding.tolist(),  # JSON 직렬화를 위해 리스트로 변환
                'embedding_dimension': len(embedding),
                'model_name': self.model_name
            }
            
            logger.info(f"룰 벡터화 완료: '{rule_name}' (차원: {len(embedding)})")
            return result
            
        except Exception as e:
            logger.error(f"룰 설명 벡터화 실패: {str(e)}")
            raise Exception(f"룰 설명 벡터화 실패: {str(e)}")
    
    def vectorize_multiple_rules(self, rules_data: List[Dict]) -> List[Dict]:
        """
        여러 룰을 배치로 벡터화합니다.
        
        Args:
            rules_data (List[Dict]): 룰 데이터 리스트
                각 딕셔너리는 'rule_name', 'description', 'target_name'(선택) 키를 포함
                
        Returns:
            List[Dict]: 벡터화된 룰 리스트
        """
        try:
            if not rules_data:
                return []
            
            # 컨텍스트 텍스트 생성
            context_texts = []
            for rule_data in rules_data:
                rule_name = rule_data.get('rule_name', '')
                description = rule_data.get('description', '')
                target_name = rule_data.get('target_name')
                
                if target_name:
                    context_text = f"[{target_name}] {rule_name}: {description}"
                else:
                    context_text = f"{rule_name}: {description}"
                
                context_texts.append(context_text)
            
            # 배치 벡터화
            embeddings = self.vectorize_texts(context_texts)
            
            # 결과 생성
            results = []
            for i, rule_data in enumerate(rules_data):
                result = {
                    'rule_name': rule_data.get('rule_name', ''),
                    'description': rule_data.get('description', ''),
                    'target_name': rule_data.get('target_name'),
                    'context_text': context_texts[i],
                    'embedding': embeddings[i].tolist(),
                    'embedding_dimension': len(embeddings[i]),
                    'model_name': self.model_name
                }
                results.append(result)
            
            logger.info(f"배치 룰 벡터화 완료: {len(results)}개 룰 처리")
            return results
            
        except Exception as e:
            logger.error(f"배치 룰 벡터화 실패: {str(e)}")
            raise Exception(f"배치 룰 벡터화 실패: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """모델 정보를 반환합니다."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'tokenizer_type': type(self.tokenizer).__name__ if self.tokenizer else None,
            'max_position_embeddings': getattr(self.model.config, 'max_position_embeddings', None) if self.model else None,
            'hidden_size': getattr(self.model.config, 'hidden_size', None) if self.model else None
        }
    
    def cosine_similarity(self, embedding1: Union[np.ndarray, List], embedding2: Union[np.ndarray, List]) -> float:
        """
        두 임베딩 간의 코사인 유사도를 계산합니다.
        
        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩
            
        Returns:
            float: 코사인 유사도 (-1 ~ 1)
        """
        try:
            # numpy 배열로 변환
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 코사인 유사도 계산
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"코사인 유사도 계산 실패: {str(e)}")
            return 0.0

    def save_embedding_to_file(self, text: str, output_dir: str = "embeddings", filename: str = None):
        """
        텍스트를 벡터화하고 파일로 저장합니다.
        
        Args:
            text (str): 벡터화할 텍스트
            output_dir (str): 저장할 디렉토리
            filename (str): 파일명 (None이면 자동 생성)
        
        Returns:
            Dict: 저장 결과 정보
        """
        try:
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 벡터화 수행
            embedding = self.vectorize_text(text)
            
            # 파일명 생성
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_text = safe_text.replace(' ', '_')
                filename = f"embedding_{timestamp}_{safe_text}"
            
            # JSON 파일로 저장
            json_filepath = os.path.join(output_dir, f"{filename}.json")
            result_data = {
                'text': text,
                'embedding': embedding.tolist(),
                'embedding_dimension': len(embedding),
                'model_name': self.model_name,
                'device': str(self.device),
                'created_at': datetime.now().isoformat(),
                'text_length': len(text)
            }
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            # NumPy 파일로도 저장 (임베딩만)
            npy_filepath = os.path.join(output_dir, f"{filename}.npy")
            np.save(npy_filepath, embedding)
            
            # 결과 정보
            result_info = {
                'success': True,
                'text': text,
                'text_length': len(text),
                'embedding_dimension': len(embedding),
                'json_file': json_filepath,
                'npy_file': npy_filepath,
                'model_name': self.model_name,
                'device': str(self.device)
            }
            
            print(f"✅ 임베딩 저장 완료!")
            print(f"   텍스트: {text}")
            print(f"   차원: {len(embedding)}")
            print(f"   JSON 파일: {json_filepath}")
            print(f"   NumPy 파일: {npy_filepath}")
            
            return result_info
            
        except Exception as e:
            error_msg = f"임베딩 파일 저장 실패: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'text': text
            }

    def analyze_text_tokens(self, text: str) -> Dict:
        """
        텍스트의 토큰 분석을 수행합니다.
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            Dict: 토큰 분석 결과
        """
        try:
            if self.tokenizer is None:
                raise Exception("토크나이저가 로드되지 않았습니다.")
            
            # 토크나이징
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=False
            )
            
            # 토큰 ID를 텍스트로 변환
            token_ids = inputs['input_ids'][0].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            
            # 특수 토큰 제거한 실제 토큰들
            actual_tokens = [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]
            
            return {
                'original_text': text,
                'token_ids': token_ids,
                'tokens': tokens,
                'actual_tokens': actual_tokens,
                'token_count': len(actual_tokens),
                'attention_mask': inputs['attention_mask'][0].tolist()
            }
            
        except Exception as e:
            logger.error(f"토큰 분석 실패: {str(e)}")
            raise Exception(f"토큰 분석 실패: {str(e)}")
    
    def detailed_similarity_analysis(self, text1: str, text2: str) -> Dict:
        """
        두 텍스트의 상세한 유사도 분석을 수행합니다.
        
        Args:
            text1 (str): 첫 번째 텍스트
            text2 (str): 두 번째 텍스트
            
        Returns:
            Dict: 상세 분석 결과
        """
        try:
            # 각 텍스트의 토큰 분석
            token_analysis1 = self.analyze_text_tokens(text1)
            token_analysis2 = self.analyze_text_tokens(text2)
            
            # 공통 토큰 찾기
            tokens1_set = set(token_analysis1['actual_tokens'])
            tokens2_set = set(token_analysis2['actual_tokens'])
            common_tokens = tokens1_set.intersection(tokens2_set)
            
            # 임베딩 계산
            embedding1 = self.vectorize_text(text1)
            embedding2 = self.vectorize_text(text2)
            
            # 유사도 계산
            cosine_sim = self.cosine_similarity(embedding1, embedding2)
            
            # 유클리드 거리 계산
            euclidean_distance = np.linalg.norm(embedding1 - embedding2)
            
            # 맨하탄 거리 계산
            manhattan_distance = np.sum(np.abs(embedding1 - embedding2))
            
            # 임베딩 통계
            embedding1_stats = {
                'mean': float(np.mean(embedding1)),
                'std': float(np.std(embedding1)),
                'min': float(np.min(embedding1)),
                'max': float(np.max(embedding1)),
                'norm': float(np.linalg.norm(embedding1))
            }
            
            embedding2_stats = {
                'mean': float(np.mean(embedding2)),
                'std': float(np.std(embedding2)),
                'min': float(np.min(embedding2)),
                'max': float(np.max(embedding2)),
                'norm': float(np.linalg.norm(embedding2))
            }
            
            return {
                'text1': text1,
                'text2': text2,
                'token_analysis': {
                    'text1_tokens': token_analysis1['actual_tokens'],
                    'text2_tokens': token_analysis2['actual_tokens'],
                    'common_tokens': list(common_tokens),
                    'common_token_count': len(common_tokens),
                    'text1_unique_tokens': list(tokens1_set - tokens2_set),
                    'text2_unique_tokens': list(tokens2_set - tokens1_set),
                    'token_overlap_ratio': len(common_tokens) / max(len(tokens1_set), len(tokens2_set)) if max(len(tokens1_set), len(tokens2_set)) > 0 else 0
                },
                'similarity_metrics': {
                    'cosine_similarity': float(cosine_sim),
                    'euclidean_distance': float(euclidean_distance),
                    'manhattan_distance': float(manhattan_distance)
                },
                'embedding_stats': {
                    'text1': embedding1_stats,
                    'text2': embedding2_stats
                },
                'analysis': {
                    'embedding_dimension': len(embedding1),
                    'model_name': self.model_name
                }
            }
            
        except Exception as e:
            logger.error(f"상세 유사도 분석 실패: {str(e)}")
            raise Exception(f"상세 유사도 분석 실패: {str(e)}")

def handle_medical_symptom_test_interactive(vectorizer):
    """질문-답변식 의료 증상 테스트 (표 기반 진단)"""
    print("\n🏥 의료 증상 테스트 (질문-답변식)")
    print("-" * 40)
    
    # 표 기반 진단 룰 정의 (각 열이 하나의 질환)
    disease_rules = [
        {
            'name': '안검염',
            '분비물 특성': '비늘 같은 각질, 기름기 있는 분비물',
            '진행 속도': '점진적, 만성적',
            '주요 증상': '속눈썹 주변 각질, 눈꺼풀 붙음',
            '발생 패턴': '양안'
        },
        {
            'name': '비궤양성 각막염',
            '분비물 특성': '미세한 분비물, 주로 눈물',
            '진행 속도': '점진적',
            '주요 증상': '눈 뜨는데 어려움 없음, 각막 표면 매끄러움, 눈물 흘림',
            '발생 패턴': '단안 또는 양안'
        },
        {
            'name': '결막염',
            '분비물 특성': '수양성, 점액성, 화농성',
            '진행 속도': '급성, 점진적',
            '주요 증상': '눈을 비비는 행동, 충혈, 부종',
            '발생 패턴': '양안(알레르기성), 단안(감염성)'
        },
        {
            'name': '각막궤양',
            '분비물 특성': '화농성, 점액성 분비물',
            '진행 속도': '급성, 빠른 진행',
            '주요 증상': '눈을 뜨기 힘듦, 눈물, 심한 통증, 시력 저하',
            '발생 패턴': '단안'
        },
        {
            'name': '각막부골편',
            '분비물 특성': '수양성, 경미한 분비물',
            '진행 속도': '급성, 반복성',
            '주요 증상': '아침에 심함, 갑작스러운 심한 통증, 이물감, 깜빡임 시 통증',
            '발생 패턴': '단안 또는 양안'
        }
    ]
    
    # 질문 순서 및 키
    questions = [
        ('분비물 특성', '분비물의 특성(예: 비늘 같은 각질, 미세한 분비물, 수양성/점액성/화농성 등)을 입력하세요:'),
        ('진행 속도', '진행 속도(예: 점진적, 만성적, 급성 등)을 입력하세요:'),
        ('주요 증상', '주요 증상(예: 각질, 눈꺼풀, 눈 뜨기 어려움, 충혈 등)을 입력하세요:'),
        ('발생 패턴', '발생 패턴(예: 양안, 단안, 양안(알레르기성), 단안(감염성) 등)을 입력하세요:')
    ]
    
    # 사용자 답변 수집
    user_answers = {}
    for key, question in questions:
        while True:
            answer = input(f"{question} ").strip()
            if answer:
                user_answers[key] = answer
                break
            else:
                print("❌ 입력이 필요합니다.")
    
    print("\n⚡ 입력하신 증상 정보:")
    for k, v in user_answers.items():
        print(f"   {k}: {v}")
    print("\n표 기반 진단 룰과 비교 중...")
    
    # 각 질환별로 항목별 유사도 계산 및 합산
    results = []
    for rule in disease_rules:
        total_score = 0
        detail_scores = {}
        for key in user_answers:
            user_text = user_answers[key]
            rule_text = rule[key]
            sim = vectorizer.cosine_similarity(
                vectorizer.vectorize_text(user_text),
                vectorizer.vectorize_text(rule_text)
            )
            detail_scores[key] = sim
            total_score += sim
        avg_score = total_score / len(user_answers)
        results.append({
            'disease': rule['name'],
            'avg_score': avg_score,
            'detail_scores': detail_scores
        })
    
    # 유사도 순 정렬
    results.sort(key=lambda x: x['avg_score'], reverse=True)
    
    print("\n✨ 진단 결과 (유사도 순):")
    for i, res in enumerate(results, 1):
        emoji = "🔥" if i == 1 and res['avg_score'] >= 0.6 else ("✅" if res['avg_score'] >= 0.5 else "🟡")
        print(f"  {i}. {emoji} {res['disease']} (평균 유사도: {res['avg_score']:.3f})")
        for k, v in res['detail_scores'].items():
            print(f"     - {k} 유사도: {v:.3f}")
    
    # 최종 추천
    top = results[0]
    if top['avg_score'] >= 0.6:
        print(f"\n💡 최종 진단 제안: '{top['disease']}' 가능성이 높습니다.")
    elif top['avg_score'] >= 0.5:
        print(f"\n💡 최종 진단 제안: '{top['disease']}'와 유사하지만 추가 검토가 필요합니다.")
    else:
        print(f"\n💡 최종 진단 제안: 명확한 진단이 어려우니 추가 정보가 필요합니다.")
    print("\n(참고: 이 결과는 AI 기반 참고용입니다. 전문의 상담을 권장합니다.)")

def handle_rule_embedding(vectorizer):
    """진단 룰 임베딩 생성 및 저장"""
    print("\n🧠 진단 룰 임베딩 생성")
    print("-" * 40)
    
    # 업데이트된 진단 룰 정의
    disease_rules = [
        {
            'name': '안검염',
            '분비물 특성': '비늘 같은 각질, 기름기 있는 분비물',
            '진행 속도': '점진적, 만성적',
            '주요 증상': '속눈썹 주변 각질, 눈꺼풀 붙음',
            '발생 패턴': '양안'
        },
        {
            'name': '비궤양성 각막염',
            '분비물 특성': '미세한 분비물, 주로 눈물',
            '진행 속도': '점진적',
            '주요 증상': '눈 뜨는데 어려움 없음, 각막 표면 매끄러움, 눈물 흘림',
            '발생 패턴': '단안 또는 양안'
        },
        {
            'name': '결막염',
            '분비물 특성': '수양성, 점액성, 화농성',
            '진행 속도': '급성, 점진적',
            '주요 증상': '눈을 비비는 행동, 충혈, 부종',
            '발생 패턴': '양안(알레르기성), 단안(감염성)'
        },
        {
            'name': '각막궤양',
            '분비물 특성': '화농성, 점액성 분비물',
            '진행 속도': '급성, 빠른 진행',
            '주요 증상': '눈을 뜨기 힘듦, 눈물, 심한 통증, 시력 저하',
            '발생 패턴': '단안'
        },
        {
            'name': '각막부골편',
            '분비물 특성': '수양성, 경미한 분비물',
            '진행 속도': '급성, 반복성',
            '주요 증상': '아침에 심함, 갑작스러운 심한 통증, 이물감, 깜빡임 시 통증',
            '발생 패턴': '단안 또는 양안'
        }
    ]
    
    print(f"📋 총 {len(disease_rules)}개의 질병 룰을 임베딩합니다...")
    
    # 출력 디렉토리 생성
    output_dir = "rule_embeddings"
    os.makedirs(output_dir, exist_ok=True)
    
    all_rule_embeddings = []
    
    # 각 질병별로 룰 임베딩 생성
    for disease in disease_rules:
        disease_name = disease['name']
        print(f"\n🔍 {disease_name} 룰 임베딩 중...")
        
        disease_embeddings = {
            'disease_name': disease_name,
            'rule_embeddings': {},
            'combined_embedding': None
        }
        
        # 각 항목별 임베딩 생성
        rule_texts = []
        for category, description in disease.items():
            if category != 'name':
                rule_text = f"{category}: {description}"
                rule_texts.append(rule_text)
                
                # 개별 항목 임베딩
                embedding = vectorizer.vectorize_text(rule_text)
                disease_embeddings['rule_embeddings'][category] = {
                    'text': rule_text,
                    'embedding': embedding.tolist(),
                    'dimension': len(embedding)
                }
                print(f"   ✅ {category} 임베딩 완료 (차원: {len(embedding)})")
        
        # 전체 룰을 합친 임베딩 생성
        combined_text = f"{disease_name}: " + ", ".join([f"{k}({v})" for k, v in disease.items() if k != 'name'])
        combined_embedding = vectorizer.vectorize_text(combined_text)
        disease_embeddings['combined_embedding'] = {
            'text': combined_text,
            'embedding': combined_embedding.tolist(),
            'dimension': len(combined_embedding)
        }
        print(f"   🔥 {disease_name} 통합 임베딩 완료 (차원: {len(combined_embedding)})")
        
        # 개별 질병 파일 저장
        disease_file = os.path.join(output_dir, f"{disease_name}_embeddings.json")
        with open(disease_file, 'w', encoding='utf-8') as f:
            json.dump(disease_embeddings, f, ensure_ascii=False, indent=2)
        
        all_rule_embeddings.append(disease_embeddings)
    
    # 전체 룰 임베딩 파일 저장
    all_embeddings_data = {
        'model_name': vectorizer.model_name,
        'created_at': datetime.now().isoformat(),
        'total_diseases': len(disease_rules),
        'embedding_dimension': len(combined_embedding),
        'disease_embeddings': all_rule_embeddings
    }
    
    all_file = os.path.join(output_dir, "all_disease_embeddings.json")
    with open(all_file, 'w', encoding='utf-8') as f:
        json.dump(all_embeddings_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 모든 룰 임베딩 완료!")
    print(f"📁 저장 위치: {output_dir}/")
    print(f"📊 총 {len(disease_rules)}개 질병, 임베딩 차원: {len(combined_embedding)}")
    print(f"🗃️ 개별 파일: {len(disease_rules)}개")
    print(f"🗂️ 통합 파일: all_disease_embeddings.json")
    
    # 임베딩 통계 정보
    print(f"\n📈 임베딩 통계:")
    for disease_data in all_rule_embeddings:
        disease_name = disease_data['disease_name']
        combined_emb = np.array(disease_data['combined_embedding']['embedding'])
        print(f"   {disease_name}:")
        print(f"     - 평균: {np.mean(combined_emb):.6f}")
        print(f"     - 표준편차: {np.std(combined_emb):.6f}")
        print(f"     - 놈: {np.linalg.norm(combined_emb):.6f}")
    
    return all_embeddings_data

def handle_detailed_analysis(vectorizer):
    """두 텍스트 상세 분석 처리"""
    print("\n🔍 두 텍스트 상세 분석 모드")
    print("-" * 40)
    
    while True:
        print("\n" + "-" * 40)
        print("두 개의 텍스트를 입력해주세요:")
        
        # 첫 번째 텍스트 입력
        text1 = input("📝 첫 번째 텍스트 (뒤로가기: 'back'): ").strip()
        if text1.lower() == 'back':
            break
        
        if not text1:
            print("❌ 첫 번째 텍스트를 입력해주세요.")
            continue
        
        # 두 번째 텍스트 입력
        text2 = input("📝 두 번째 텍스트: ").strip()
        if not text2:
            print("❌ 두 번째 텍스트를 입력해주세요.")
            continue
        
        print(f"⚡ 두 텍스트를 상세 분석 중...")
        
        try:
            # 상세 분석
            analysis_result = vectorizer.detailed_similarity_analysis(text1, text2)
            
            # 결과 출력
            print(f"\n✨ 상세 분석 완료!")
            print(f"📊 결과:")
            print(f"   텍스트 1: {text1}")
            print(f"   텍스트 2: {text2}")
            
            # 토큰 분석 결과
            print(f"\n📈 토큰 분석:")
            print(f"   텍스트 1 토큰: {analysis_result['token_analysis']['text1_tokens']}")
            print(f"   텍스트 2 토큰: {analysis_result['token_analysis']['text2_tokens']}")
            print(f"   공통 토큰 수: {analysis_result['token_analysis']['common_token_count']}")
            print(f"   공통 토큰: {analysis_result['token_analysis']['common_tokens']}")
            print(f"   토큰 겹침 비율: {analysis_result['token_analysis']['token_overlap_ratio']:.3f}")
            
            if analysis_result['token_analysis']['common_token_count'] > 0:
                print(f"\n🔍 왜 유사도가 높은지 분석:")
                print(f"   - 공통 토큰이 {analysis_result['token_analysis']['common_token_count']}개 있습니다")
                print(f"   - 공통 토큰: {', '.join(analysis_result['token_analysis']['common_tokens'])}")
                print(f"   - 이러한 공통 토큰들이 유사도를 높이는 주요 원인으로 보입니다")
            
            # 고유 토큰 분석
            print(f"\n🔄 고유 토큰 분석:")
            print(f"   텍스트 1만의 토큰: {analysis_result['token_analysis']['text1_unique_tokens']}")
            print(f"   텍스트 2만의 토큰: {analysis_result['token_analysis']['text2_unique_tokens']}")
            
            # 유사도 해석
            cosine_sim = analysis_result['similarity_metrics']['cosine_similarity']
            print(f"\n📊 유사도 해석:")
            print(f"   코사인 유사도: {cosine_sim:.6f}")
            
            if cosine_sim >= 0.9:
                interpretation = "매우 유사함 🟢"
            elif cosine_sim >= 0.7:
                interpretation = "유사함 🟡"
            elif cosine_sim >= 0.5:
                interpretation = "보통 🟠"
            elif cosine_sim >= 0.3:
                interpretation = "다소 다름 🔴"
            else:
                interpretation = "매우 다름 ⚫"
            
            print(f"   해석: {interpretation}")
            print(f"   유클리드 거리: {analysis_result['similarity_metrics']['euclidean_distance']:.6f}")
            print(f"   맨하탄 거리: {analysis_result['similarity_metrics']['manhattan_distance']:.6f}")
            
            # 임베딩 통계
            print(f"\n📊 임베딩 통계:")
            print(f"   텍스트 1 임베딩 - 평균: {analysis_result['embedding_stats']['text1']['mean']:.6f}, 표준편차: {analysis_result['embedding_stats']['text1']['std']:.6f}")
            print(f"   텍스트 2 임베딩 - 평균: {analysis_result['embedding_stats']['text2']['mean']:.6f}, 표준편차: {analysis_result['embedding_stats']['text2']['std']:.6f}")
        
        except Exception as e:
            print(f"❌ 상세 분석 실패: {str(e)}")

def main():
    """메인 함수 - 진단 룰 임베딩 생성 및 저장"""
    print("🧠 진단 룰 임베딩 생성 도구")
    print("=" * 60)
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 의료 특화 모델 사용 설정
        use_medical = True
        print("📥 KM-BERT (의료 특화) 모델 로딩 중...")
        
        # RuleVectorizer 초기화
        vectorizer = RuleVectorizer(use_medical_model=use_medical)
        
        # 모델 정보 출력
        model_info = vectorizer.get_model_info()
        print(f"✅ 모델 로드 완료:")
        print(f"   모델명: {model_info['model_name']}")
        print(f"   디바이스: {model_info['device']}")
        print(f"   임베딩 차원: {model_info['hidden_size']}")
        print()
        
        # 진단 룰 정의
        disease_rules = [
            {
                'name': '안검염',
                '분비물 특성': '비늘 같은 각질, 기름기 있는 분비물',
                '진행 속도': '점진적, 만성적',
                '주요 증상': '속눈썹 주변 각질, 눈꺼풀 붙음',
                '발생 패턴': '양안'
            },
            {
                'name': '비궤양성 각막염',
                '분비물 특성': '미세한 분비물, 주로 눈물',
                '진행 속도': '점진적',
                '주요 증상': '눈 뜨는데 어려움 없음, 각막 표면 매끄러움, 눈물 흘림',
                '발생 패턴': '단안 또는 양안'
            },
            {
                'name': '결막염',
                '분비물 특성': '수양성, 점액성, 화농성',
                '진행 속도': '급성, 점진적',
                '주요 증상': '눈을 비비는 행동, 충혈, 부종',
                '발생 패턴': '양안(알레르기성), 단안(감염성)'
            },
            {
                'name': '각막궤양',
                '분비물 특성': '화농성, 점액성 분비물',
                '진행 속도': '급성, 빠른 진행',
                '주요 증상': '눈을 뜨기 힘듦, 눈물, 심한 통증, 시력 저하',
                '발생 패턴': '단안'
            },
            {
                'name': '각막부골편',
                '분비물 특성': '수양성, 경미한 분비물',
                '진행 속도': '급성, 반복성',
                '주요 증상': '아침에 심함, 갑작스러운 심한 통증, 이물감, 깜빡임 시 통증',
                '발생 패턴': '단안 또는 양안'
            }
        ]
        
        print(f"📋 총 {len(disease_rules)}개의 질병 룰을 임베딩합니다...")
        
        # 출력 디렉토리 생성
        output_dir = "rule_embeddings"
        os.makedirs(output_dir, exist_ok=True)
        
        # 각 질병별로 룰 임베딩 생성 및 저장
        for disease in disease_rules:
            disease_name = disease['name']
            print(f"\n🔍 {disease_name} 룰 임베딩 중...")
            
            # 각 항목별 임베딩 생성
            for category, description in disease.items():
                if category != 'name':
                    rule_text = f"{category}: {description}"
                    
                    # 임베딩 생성
                    embedding = vectorizer.vectorize_text(rule_text)
                    
                    # NumPy 파일로 저장
                    filename = f"{disease_name}_{category}.npy"
                    filepath = os.path.join(output_dir, filename)
                    np.save(filepath, embedding)
                    
                    print(f"   ✅ {category} 임베딩 저장: {filename} (차원: {len(embedding)})")
            
            # 전체 룰을 합친 임베딩 생성
            combined_text = f"{disease_name}: " + ", ".join([f"{k}({v})" for k, v in disease.items() if k != 'name'])
            combined_embedding = vectorizer.vectorize_text(combined_text)
            
            # 통합 임베딩도 NumPy 파일로 저장
            combined_filename = f"{disease_name}_combined.npy"
            combined_filepath = os.path.join(output_dir, combined_filename)
            np.save(combined_filepath, combined_embedding)
            
            print(f"   🔥 {disease_name} 통합 임베딩 저장: {combined_filename} (차원: {len(combined_embedding)})")
        
        print(f"\n✅ 모든 룰 임베딩 완료!")
        print(f"📁 저장 위치: {output_dir}/")
        print(f"📊 총 {len(disease_rules)}개 질병")
        print(f"📁 총 {len(disease_rules) * 5}개 임베딩 파일 생성 (.npy 형식)")
        print(f"🎉 임베딩 생성 작업이 성공적으로 완료되었습니다!")
        
    except KeyboardInterrupt:
        print("\n👋 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 