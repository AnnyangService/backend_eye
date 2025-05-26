import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    """애플리케이션 설정 클래스"""
    
    # Flask 설정
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # 데이터베이스 설정
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # AI 모델 설정
    STEP1_MODEL_PATH = os.environ.get('STEP1_MODEL_PATH') or 'app/diagnosis/models/step1'
    STEP2_MODEL_PATH = os.environ.get('STEP2_MODEL_PATH') or 'app/diagnosis/models/step2'
    
    # API 서버 설정 (Step2 결과 콜백용)
    API_SERVER_URL = os.environ.get('API_SERVER_URL') or 'http://localhost:8080'
    API_SERVER_CALLBACK_ENDPOINT = os.environ.get('API_SERVER_CALLBACK_ENDPOINT') or '/diagnosis/step2/callback'
    
    # 이미지 처리 설정
    MAX_IMAGE_SIZE = int(os.environ.get('MAX_IMAGE_SIZE', '4096'))
    MIN_IMAGE_SIZE = int(os.environ.get('MIN_IMAGE_SIZE', '100'))
    IMAGE_DOWNLOAD_TIMEOUT = int(os.environ.get('IMAGE_DOWNLOAD_TIMEOUT', '30'))
    
    # 로깅 설정
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Swagger 설정
    RESTX_MASK_SWAGGER = False  # Swagger UI에서 X-Fields 헤더 숨기기

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_ECHO = True  # SQL 쿼리 로깅
    API_SERVER_URL = os.environ.get('DEV_API_SERVER_URL') or 'http://localhost:8080'

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL', 'sqlite:///test.db')
    API_SERVER_URL = os.environ.get('TEST_API_SERVER_URL') or 'http://localhost:8080'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_ECHO = False
    API_SERVER_URL = os.environ.get('PROD_API_SERVER_URL') or 'https://api.production-server.com'

# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig
} 