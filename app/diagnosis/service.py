import os
import urllib.parse
import logging
import requests
from PIL import Image
from flask import current_app
from .ai_model import Step1Model

# 로거 설정
logger = logging.getLogger(__name__)

class DiagnosisService:
    def __init__(self):
        # 이미지 저장 디렉토리 설정
        self.images_dir = os.path.join(current_app.instance_path, 'images')
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir, exist_ok=True)
        
        # AI 모델 초기화
        self.step1_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """AI 모델을 초기화합니다."""
        try:
            logger.info("Step1 AI 모델 초기화 시작...")
            self.step1_model = Step1Model()
            logger.info("Step1 AI 모델 초기화 완료")
        except Exception as e:
            logger.error(f"AI 모델 초기화 실패: {str(e)}")
            # 모델 로드 실패 시 None으로 설정
            self.step1_model = None
            # 에러를 다시 발생시켜서 서비스 초기화 시점에 문제를 알림
            raise Exception(f"AI 모델 로드 실패: {str(e)}")
    
    def _download_image(self, image_url):
        """
        이미지 다운로드
        
        Args:
            image_url (str): 다운로드할 이미지 URL
            
        Returns:
            str: 로컬에 저장된 이미지 파일 경로
        """
        # URL에서 파일명 추출
        # S3 이후 수정 필요
        parsed_url = urllib.parse.urlparse(image_url)
        filename = os.path.basename(parsed_url.path)
        if not filename or '.' not in filename:
            filename = 'image.jpg'
        
        local_path = os.path.join(self.images_dir, filename)
        
        # 이미 파일이 있으면 기존 파일 사용
        if os.path.exists(local_path):
            logger.info(f"기존 이미지 파일 사용: {local_path}")
            return local_path
        
        # 이미지 다운로드
        logger.info(f"이미지 다운로드 시작: {image_url}")
        
        # config에서 timeout 설정 가져오기
        timeout = current_app.config.get('IMAGE_DOWNLOAD_TIMEOUT', 30)
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()
        
        # Content-Type 검증
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            error_msg = f"유효하지 않은 이미지 타입: {content_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 파일 저장
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"이미지 다운로드 완료: {local_path}")
        return local_path
    
    def _validate_image(self, image):
        """이미지 유효성 검사"""
        # 고도화시 추가 검증 로직 구현 필요
        # config에서 이미지 크기 제한 가져오기
        min_size = current_app.config.get('MIN_IMAGE_SIZE', 100)
        max_size = current_app.config.get('MAX_IMAGE_SIZE', 4096)
        
        if image.width < min_size or image.height < min_size:
            error_msg = f"이미지 크기가 너무 작습니다 (최소 {min_size}x{min_size})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 이미지 크기가 너무 큰 경우 제한
        if image.width > max_size or image.height > max_size:
            error_msg = f"이미지 크기가 너무 큽니다 (최대 {max_size}x{max_size})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"이미지 검증 완료: {image.width}x{image.height}")
    
    def process_step1_diagnosis(self, image_url):
        """
        Step1: 질병여부판단
        
        POST /diagnosis/step1/ 요청을 처리하는 메인 함수
        
        Args:
            image_url (str): 분석할 이미지의 URL
            
        Returns:
            dict: {
                "is_normal": bool,    # 정상 여부
                "confidence": float   # 신뢰도 (0.0 ~ 1.0)
            }
        """
        logger.info(f"Step1 진단 시작: {image_url}")
        
        try:
            # 1. 이미지 다운로드
            local_path = self._download_image(image_url)
            
            # 2. 이미지 로드 및 검증
            image = Image.open(local_path)
            self._validate_image(image)
            
            logger.info(f"이미지 로드 성공: {local_path} ({image.width}x{image.height})")
            
            # 3. AI 모델 분석
            if self.step1_model and self.step1_model.is_model_loaded():
                logger.info(f"AI 모델 분석 시작: {local_path}")
                
                # 실제 AI 모델 추론
                prediction_result = self.step1_model.predict(image)
                
                result = {
                    "is_normal": prediction_result['is_normal'],
                    "confidence": prediction_result['confidence']
                }
                
                logger.info(f"AI 분석 결과 - 정상: {result['is_normal']}, 신뢰도: {result['confidence']}")
                logger.info(f"Step1 진단 완료: {result}")
                return result
                
            else:
                # AI 모델이 로드되지 않은 경우 에러 발생
                error_msg = "AI 모델이 로드되지 않았습니다. 서버 관리자에게 문의하세요."
                logger.error(error_msg)
                raise Exception(error_msg)
            
        except requests.RequestException as e:
            error_msg = f"이미지 다운로드 실패: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Step1 진단 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

