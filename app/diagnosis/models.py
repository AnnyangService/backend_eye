# AI 서버용 - 읽기 전용 데이터베이스 모델들
from datetime import datetime
from app import db

class DiagnosisLevel1(db.Model):
    __tablename__ = 'diagnosis_level_1'
    
    id = db.Column(db.String(36), primary_key=True)
    cat_id = db.Column(db.String(36), nullable=False)
    image_url = db.Column(db.Text, nullable=False)
    is_normal = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    updated_at = db.Column(db.DateTime, nullable=False)
    
    # Relationships for reading
    level2_diagnoses = db.relationship('DiagnosisLevel2', backref='level1_diagnosis', lazy=True)

class DiagnosisLevel2(db.Model):
    __tablename__ = 'diagnosis_level_2'
    
    id = db.Column(db.String(36), primary_key=True)
    parent_diagnosis_id = db.Column(db.String(36), db.ForeignKey('diagnosis_level_1.id'), nullable=False)
    category = db.Column(db.String(255), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    updated_at = db.Column(db.DateTime, nullable=False)
    
    # Relationships for reading
    level3_diagnoses = db.relationship('DiagnosisLevel3', backref='level2_diagnosis', lazy=True)

class DiagnosisLevel3(db.Model):
    __tablename__ = 'diagnosis_level_3'
    
    id = db.Column(db.String(36), primary_key=True)
    parent_diagnosis_id = db.Column(db.String(36), db.ForeignKey('diagnosis_level_2.id'), nullable=False)
    category = db.Column(db.String(255), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    updated_at = db.Column(db.DateTime, nullable=False)

class DiagnosisTarget(db.Model):
    __tablename__ = 'diagnosis_target'
    
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    updated_at = db.Column(db.DateTime, nullable=False)
    
    # Relationships for reading
    rule_descriptions = db.relationship('DiagnosisRuleDescription', backref='diagnosis_target', lazy=True)

class DiagnosisRule(db.Model):
    __tablename__ = 'diagnosis_rule'
    
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    updated_at = db.Column(db.DateTime, nullable=False)
    
    # Relationships for reading
    rule_descriptions = db.relationship('DiagnosisRuleDescription', backref='diagnosis_rule', lazy=True)

class DiagnosisRuleDescription(db.Model):
    __tablename__ = 'diagnosis_rule_description'
    
    id = db.Column(db.String(36), primary_key=True)
    diagnosis_target_id = db.Column(db.String(36), db.ForeignKey('diagnosis_target.id'), nullable=False)
    diagnosis_rule_id = db.Column(db.String(36), db.ForeignKey('diagnosis_rule.id'), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    updated_at = db.Column(db.DateTime, nullable=False) 