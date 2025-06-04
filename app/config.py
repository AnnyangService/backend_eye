import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    """애플리케이션 기본 설정 클래스"""
    
    # Flask 설정
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DEBUG = False
    
    # 데이터베이스 설정
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # AI 모델 설정
    STEP1_MODEL_PATH = os.environ.get('STEP1_MODEL_PATH') or 'app/diagnosis/models/step1'
    STEP2_MODEL_PATH = os.environ.get('STEP2_MODEL_PATH') or 'app/diagnosis/models/step2'
    
    # 이미지 처리 설정
    MAX_IMAGE_SIZE = int(os.environ.get('MAX_IMAGE_SIZE', '4096'))
    MIN_IMAGE_SIZE = int(os.environ.get('MIN_IMAGE_SIZE', '100'))
    IMAGE_DOWNLOAD_TIMEOUT = int(os.environ.get('IMAGE_DOWNLOAD_TIMEOUT', '30'))
    
    # 로깅 설정
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Swagger 설정
    RESTX_MASK_SWAGGER = False

class DevelopmentConfig(Config):
    """개발 환경 설정"""
    DEBUG = True
    SQLALCHEMY_ECHO = True  # SQL 쿼리 로깅
    
    # API 서버 설정 (Step2 결과 콜백용)
    API_SERVER_URL = 'http://host.docker.internal:8080'
    API_SERVER_CALLBACK_ENDPOINT = '/diagnosis/step2'

class ProductionConfig(Config):
    """프로덕션 환경 설정"""
    DEBUG = False
    SQLALCHEMY_ECHO = False
    
    # API 서버 설정 (추후 설정)
    API_SERVER_URL = os.environ.get('API_SERVER_URL')  # 환경변수에서만 가져옴
    API_SERVER_CALLBACK_ENDPOINT = '/diagnosis/step2'

# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
} 