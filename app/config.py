import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key')
    DEBUG = False
    TESTING = False
    
    # MySQL configuration
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306))
    MYSQL_USER = os.environ.get('MYSQL_USER', 'admin')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '1234')
    MYSQL_DB = os.environ.get('MYSQL_DB', 'hi_meow')
    
    # SQLAlchemy configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    ENABLE_MIGRATIONS = False  # API 서버에서 데이터베이스 스키마 관리

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_ECHO = True  # SQL 쿼리 로깅

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL', 'sqlite:///test.db')

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_ECHO = False

# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig
} 