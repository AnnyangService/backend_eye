from datetime import datetime
from app import db

class Chunk(db.Model):
    """텍스트 청크 모델"""
    __tablename__ = 'documents'
    
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)  # 원본 텍스트
    embedding = db.Column('embedding', db.Text, nullable=True)  # 임베딩 벡터 (384차원)
    keywords = db.Column(db.JSON, nullable=True)  # 키워드 배열
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Chunk {self.id}: {self.content[:50]}...>'
    
    def to_dict(self):
        """모델을 딕셔너리로 변환"""
        return {
            'id': self.id,
            'content': self.content,
            'embedding': self.embedding,  # vector 데이터
            'keywords': self.keywords,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    __table_args__ = {
        'postgresql_using': 'btree'
    } 