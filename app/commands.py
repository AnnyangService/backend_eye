"""CLI commands for the Flask application"""
import click
import json
import numpy as np
import os
import logging
from flask import current_app
from flask.cli import with_appcontext
from sqlalchemy import text
from app import db
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

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
                embedding vector(768),
                keywords JSONB,
                source VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
            db.session.execute(text(create_documents_table_sql))
            db.session.commit()
            
            logger.info('데이터베이스 테이블이 성공적으로 생성되었습니다.')
        except Exception as e:
            logger.error(f'데이터베이스 초기화 중 오류가 발생했습니다: {e}')
    
    @app.cli.command('drop-db')
    @with_appcontext
    def drop_db():
        """데이터베이스 테이블을 삭제합니다."""
        try:
            db.drop_all()
            logger.info('데이터베이스 테이블이 성공적으로 삭제되었습니다.')
        except Exception as e:
            logger.error(f'데이터베이스 삭제 중 오류가 발생했습니다: {e}')
    
    @app.cli.command('reset-db')
    @with_appcontext
    def reset_db():
        """데이터베이스를 초기화합니다 (삭제 후 재생성)."""
        try:
            db.drop_all()
            
            # pgvector 확장 활성화
            db.session.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
            db.session.commit()
            
            # documents 테이블을 vector 타입으로 생성 (source 컬럼 추가)
            create_documents_table_sql = """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(768),
                keywords JSONB,
                source VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
            db.session.execute(text(create_documents_table_sql))
            db.session.commit()
            
            logger.info('데이터베이스가 성공적으로 초기화되었습니다.')
        except Exception as e:
            logger.error(f'데이터베이스 초기화 중 오류가 발생했습니다: {e}')
    
    @app.cli.command('load-chunks')
    @with_appcontext
    def load_chunks():
        """청크 데이터와 임베딩을 PostgreSQL에 저장합니다."""
        try:
            # 청크 데이터 파일 경로
            chunks_file = "app/rag/chunks/chunks.json"
            embeddings_dir = "app/rag/embeddings"
            
            # 청크 데이터 로드
            if not os.path.exists(chunks_file):
                logger.error(f'청크 파일을 찾을 수 없습니다: {chunks_file}')
                return
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            logger.info(f'청크 데이터를 로드했습니다: {len(chunks)}개')
            
            # 기존 데이터 삭제
            db.session.execute(text('DELETE FROM documents;'))
            db.session.commit()
            logger.info('기존 데이터를 삭제했습니다.')
            
            # 청크 데이터 저장
            saved_count = 0
            for chunk in chunks:
                chunk_id = chunk['id']
                content = chunk['content']
                keywords = chunk.get('keywords', [])
                
                # 임베딩 파일 경로
                embedding_file = os.path.join(embeddings_dir, f"{chunk_id}.npy")
                
                if os.path.exists(embedding_file):
                    # 임베딩 로드
                    embedding = np.load(embedding_file)
                    
                    # PostgreSQL vector 형식으로 변환
                    embedding_str = f"[{','.join(map(str, embedding))}]"
                    
                    # 데이터베이스에 저장 (source에 chunk_id 저장)
                    insert_sql = """
                    INSERT INTO documents (content, embedding, keywords, source)
                    VALUES (:content, :embedding, :keywords, :source)
                    """
                    db.session.execute(text(insert_sql), {
                        'content': content,
                        'embedding': embedding_str,
                        'keywords': json.dumps(keywords),
                        'source': chunk_id
                    })
                    
                    saved_count += 1
                    logger.info(f'저장됨: {chunk_id}')
                else:
                    logger.warning(f'임베딩 파일을 찾을 수 없음: {embedding_file}')
            
            db.session.commit()
            logger.info(f'총 {saved_count}개의 청크를 데이터베이스에 저장했습니다.')
            
        except Exception as e:
            db.session.rollback()
            logger.error(f'청크 데이터 저장 중 오류가 발생했습니다: {e}')
    
    @app.cli.command('show-db-status')
    @with_appcontext
    def show_db_status():
        """데이터베이스 상태를 확인합니다."""
        try:
            # 테이블 개수 확인
            count_result = db.session.execute(text('SELECT COUNT(*) as count FROM documents;'))
            count = count_result.fetchone()[0]
            logger.info(f'총 문서 개수: {count}개')
            
            # 샘플 데이터 확인 (source, keywords, content_preview, created_at 포함)
            sample_result = db.session.execute(text('''
                SELECT id, 
                       source,
                       LEFT(content, 50) as content_preview, 
                       keywords,
                       created_at 
                FROM documents 
                LIMIT 5;
            '''))
            
            samples = sample_result.fetchall()
            logger.info('샘플 데이터:')
            for sample in samples:
                logger.info(f'  ID: {sample[0]}, source: {sample[1]}, 내용: {sample[2]}..., 키워드: {sample[3]}, 생성일: {sample[4]}')
            
            if count > 0:
                logger.info('임베딩이 포함된 데이터가 저장되어 있습니다.')
            else:
                logger.info('저장된 데이터가 없습니다.')
            
        except Exception as e:
            logger.error(f'데이터베이스 상태 확인 중 오류가 발생했습니다: {e}')
    