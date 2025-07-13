"""CLI commands for the Flask application"""
import click
from flask import current_app
from flask.cli import with_appcontext
from sqlalchemy import text
from app import db

def register_commands(app):
    """Register CLI commands with the app"""
    
    @app.cli.command('init-db')
    @with_appcontext
    def init_db():
        """데이터베이스 테이블을 생성합니다."""
        try:
            # pgvector 확장 활성화
            db.session.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
            db.session.commit()
            
            # documents 테이블을 vector 타입으로 생성
            create_documents_table_sql = """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(384),
                keywords JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
            db.session.execute(text(create_documents_table_sql))
            db.session.commit()
            
            click.echo('데이터베이스 테이블이 성공적으로 생성되었습니다.')
        except Exception as e:
            click.echo(f'데이터베이스 초기화 중 오류가 발생했습니다: {e}')
    
    @app.cli.command('drop-db')
    @with_appcontext
    def drop_db():
        """데이터베이스 테이블을 삭제합니다."""
        try:
            db.drop_all()
            click.echo('데이터베이스 테이블이 성공적으로 삭제되었습니다.')
        except Exception as e:
            click.echo(f'데이터베이스 삭제 중 오류가 발생했습니다: {e}')
    
    @app.cli.command('reset-db')
    @with_appcontext
    def reset_db():
        """데이터베이스를 초기화합니다 (삭제 후 재생성)."""
        try:
            db.drop_all()
            
            # pgvector 확장 활성화
            db.session.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
            db.session.commit()
            
            # documents 테이블을 vector 타입으로 생성
            create_documents_table_sql = """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(384),
                keywords JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
            db.session.execute(text(create_documents_table_sql))
            db.session.commit()
            
            click.echo('데이터베이스가 성공적으로 초기화되었습니다.')
        except Exception as e:
            click.echo(f'데이터베이스 초기화 중 오류가 발생했습니다: {e}') 