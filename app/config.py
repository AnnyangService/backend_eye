import os
from dotenv import load_dotenv

if(os.environ.get('FLASK_ENV') == 'production'):
    # 프로덕션 환경에서는 .env.production 파일을 로드
    load_dotenv('.env.production')
else:
    # 개발 환경에서는 .env 파일을 로드
    load_dotenv('.env')

class Config:
    """애플리케이션 기본 설정 클래스"""
    
    # Flask 설정
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DEBUG = False
    
    # SQLAlchemy 공통 설정
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
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
    
    # 데이터베이스 설정 (개발환경)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://postgres:password@postgres:5432/eye_diagnosis'
    
    # API 서버 설정 (Step2 결과 콜백용)
    # .env 파일에서 가져오되, 없으면 개발환경 기본값 사용
    API_SERVER_URL = os.environ.get('API_SERVER_URL') or 'http://host.docker.internal:8080'
    API_SERVER_CALLBACK_ENDPOINT = os.environ.get('API_SERVER_CALLBACK_ENDPOINT') or '/diagnosis/step2'

class ProductionConfig(Config):
    """프로덕션 환경 설정"""
    DEBUG = False
    
    # 데이터베이스 설정 (프로덕션)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')  # 반드시 환경변수로 설정
    
    # API 서버 설정 (프로덕션에서는 반드시 .env에서 설정해야 함)
    API_SERVER_URL = os.environ.get('API_SERVER_URL')  # 기본값 없음 - 반드시 설정 필요
    API_SERVER_CALLBACK_ENDPOINT = os.environ.get('API_SERVER_CALLBACK_ENDPOINT') or '/diagnosis/step2'

# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
} 