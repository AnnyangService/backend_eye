import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import logging
from flask import current_app

logger = logging.getLogger(__name__)

class DiagnosisModel:
    """질병 진단 AI 모델 클래스 (PyTorch Mobile)"""
    
    def __init__(self, model_path=None, model_type="step1"):
        self.device = torch.device('cpu')  # Mobile 모델은 CPU에서 실행
        self.model = None
        self.img_size = 224
        
        # 모델 타입에 따라 클래스 이름 설정
        self.diagnosis_type = model_type
        if model_type == "step1":
            self.class_names = ['abnormal', 'normal']  # 0: abnormal, 1: normal
        elif model_type == "step2":
            self.class_names = ['corneal', 'inflammation']  # 0: corneal, 1: inflammation
        else:
            self.class_names = ['class_0', 'class_1']  # 기본값
        
        # 모델 경로 설정
        if model_path is None:
            try:
                if model_type == "step1":
                    model_path = current_app.config.get('STEP1_MODEL_PATH')
                elif model_type == "step2":
                    model_path = current_app.config.get('STEP2_MODEL_PATH')
                
                if model_path is None:
                    model_path = os.path.join(os.path.dirname(__file__), 'models', model_type)
            except RuntimeError:
                model_path = os.path.join(os.path.dirname(__file__), 'models', model_type)
        
        self.model_path = model_path
        
        # 이미지 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # PyTorch Mobile 모델 로드
        self._load_mobile_model()
    
    def _get_mobile_model_path(self):
        """모바일 모델 경로를 가져옵니다."""
        if os.path.isfile(self.model_path):
            base_path = os.path.dirname(self.model_path)
            mobile_path = os.path.join(base_path, f"{self.diagnosis_type}_mobile.ptl")
        else:
            mobile_path = os.path.join(os.path.dirname(self.model_path), f"{self.diagnosis_type}_mobile.ptl")
        
        return mobile_path
    
    def _load_mobile_model(self):
        """PyTorch Mobile 모델을 로드합니다."""
        mobile_path = self._get_mobile_model_path()
        
        if not os.path.exists(mobile_path):
            error_msg = f"PyTorch Mobile model not found at {mobile_path}. Please run utils/convert_to_mobile.py first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            self.model = torch.jit.load(mobile_path, map_location=self.device)
            self.model.eval()
            logger.info(f"PyTorch Mobile model loaded: {self.diagnosis_type}")
            
        except Exception as e:
            error_msg = f"Failed to load PyTorch Mobile model: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def preprocess_image(self, image):
        """이미지를 모델 입력 형태로 전처리합니다."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def predict(self, image):
        """이미지에 대해 질병 여부를 예측합니다."""
        if self.model is None:
            raise Exception("PyTorch Mobile model not loaded")
        
        # 이미지 전처리
        input_tensor = self.preprocess_image(image)
        
        # 추론 실행
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 모델 타입에 따라 결과 해석
        if self.diagnosis_type == "step1":
            is_normal = (predicted_class == 1)  # 0: abnormal, 1: normal
            result = {
                'is_normal': is_normal,
                'confidence': float(confidence),
                'predicted_class': self.class_names[predicted_class],
                'predicted_class_index': predicted_class,
                'probabilities': {
                    'normal': float(probabilities[0][1]),
                    'abnormal': float(probabilities[0][0])
                },
                'model_type': 'PyTorch Mobile'
            }
        elif self.diagnosis_type == "step2":
            category = self.class_names[predicted_class]
            result = {
                'category': category,
                'confidence': float(confidence),
                'predicted_class': self.class_names[predicted_class],
                'predicted_class_index': predicted_class,
                'probabilities': {
                    'corneal': float(probabilities[0][0]),
                    'inflammation': float(probabilities[0][1])
                },
                'model_type': 'PyTorch Mobile'
            }
        else:
            result = {
                'predicted_class': self.class_names[predicted_class],
                'confidence': float(confidence),
                'predicted_class_index': predicted_class,
                'model_type': 'PyTorch Mobile'
            }
        
        return result
    
    def is_model_loaded(self):
        """모델이 로드되었는지 확인합니다."""
        return self.model is not None
    
    def get_model_info(self):
        """모델 정보를 반환합니다."""
        return {
            'model_type': 'PyTorch Mobile',
            'diagnosis_type': self.diagnosis_type,
            'class_names': self.class_names,
            'device': str(self.device),
            'model_loaded': self.is_model_loaded(),
            'model_path': self._get_mobile_model_path()
        }


class MobileModelConverter:
    """기존 PyTorch 모델을 PyTorch Mobile로 변환하는 유틸리티"""
    
    @staticmethod
    def convert_to_mobile(original_model_path, mobile_model_path, model_type="step1"):
        """
        기존 모델을 PyTorch Mobile 형식으로 변환
        
        Args:
            original_model_path: 원본 모델 경로
            mobile_model_path: 변환된 모바일 모델 저장 경로
            model_type: 모델 타입 ("step1" 또는 "step2")
        """
        try:
            logger.info(f"Converting {original_model_path} to mobile format...")
            
            # 모바일 모델 생성
            mobile_model = DiagnosisModel(
                model_path=original_model_path,
                model_type=model_type,
                use_mobile=True
            )
            
            # 모바일 모델 저장
            mobile_model.save_mobile_model(mobile_model_path)
            
            logger.info(f"Conversion completed: {mobile_model_path}")
            
            return mobile_model_path
            
        except Exception as e:
            logger.error(f"Model conversion failed: {str(e)}")
            raise Exception(f"Model conversion failed: {str(e)}")
    
    @staticmethod
    def compare_model_sizes(original_path, mobile_path):
        """원본 모델과 모바일 모델의 크기 비교"""
        try:
            original_size = os.path.getsize(original_path) / (1024 * 1024)
            mobile_size = os.path.getsize(mobile_path) / (1024 * 1024)
            
            reduction_percent = ((original_size - mobile_size) / original_size) * 100
            
            comparison = {
                'original_size_mb': round(original_size, 2),
                'mobile_size_mb': round(mobile_size, 2),
                'size_reduction_percent': round(reduction_percent, 1),
                'size_reduction_mb': round(original_size - mobile_size, 2)
            }
            
            logger.debug(f"Model size comparison: {comparison}")
            return comparison
            
        except Exception as e:
            logger.error(f"Size comparison failed: {str(e)}")
            return None


# 경량화된 모바일 전용 클래스 (기존 LiteMobileDiagnosisModel과 호환)
class LiteMobileDiagnosisModel:
    """최소 의존성으로 구현된 PyTorch Mobile 모델"""
    
    def __init__(self, mobile_model_path, class_names):
        """
        Args:
            mobile_model_path: .ptl 모바일 모델 파일 경로
            class_names: 클래스 이름 리스트
        """
        self.model = torch.jit.load(mobile_model_path)
        self.model.eval()
        self.class_names = class_names
        self.img_size = 224
        self.model_type = 'PyTorch Mobile Lite'
        
        # 전처리 파라미터
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
    
    def preprocess_image(self, image):
        """최소한의 이미지 전처리"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # PIL을 사용한 리사이즈
        image = image.resize((self.img_size, self.img_size))
        
        # 텐서로 변환
        tensor = transforms.ToTensor()(image)
        
        # 정규화
        tensor = transforms.Normalize(self.mean, self.std)(tensor)
        
        # 배치 차원 추가
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(self, image):
        """경량화된 모바일 추론"""
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'predicted_class_index': int(predicted_class),
            'probabilities': {name: float(prob) for name, prob in zip(self.class_names, probabilities[0])},
            'model_type': self.model_type
        }
    
    def is_model_loaded(self):
        """모델이 로드되었는지 확인합니다."""
        return self.model is not None 