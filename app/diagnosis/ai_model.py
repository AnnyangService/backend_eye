import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import logging
from flask import current_app
from efficientnet_pytorch import EfficientNet

logger = logging.getLogger(__name__)

class DiagnosisModel:
    """질병 진단 AI 모델 클래스 (PyTorch)"""
    
    def __init__(self, model_path=None, model_type="step1"):
        self.device = torch.device('cpu')
        self.model = None
        self.img_size = 224
        
        # 모델 타입에 따라 클래스 이름 설정
        self.diagnosis_type = model_type
        if model_type == "step1":
            self.class_names = ['abnormal', 'normal']  # 0: abnormal, 1: normal
            num_classes = 2
        elif model_type == "step2":
            self.class_names = ['corneal', 'inflammation']  # 0: corneal, 1: inflammation
            num_classes = 2
        else:
            self.class_names = ['class_0', 'class_1']  # 기본값
            num_classes = 2
        
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
        
        # 모델 로드
        self._load_model(num_classes)
    
    def _load_model(self, num_classes):
        """PyTorch 모델을 로드합니다."""
        if not os.path.exists(self.model_path):
            error_msg = f"Model not found at {self.model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # EfficientNet 모델 생성
            self.model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
            self.model._dropout = torch.nn.Dropout(0.5)
            
            # 체크포인트 로드
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info(f"Model loaded: {self.diagnosis_type}")
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
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
            raise Exception("Model not loaded")
        
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
                'model_type': 'PyTorch'
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
                'model_type': 'PyTorch'
            }
        else:
            result = {
                'predicted_class': self.class_names[predicted_class],
                'confidence': float(confidence),
                'predicted_class_index': predicted_class,
                'model_type': 'PyTorch'
            }
        
        return result
    
    def is_model_loaded(self):
        """모델이 로드되었는지 확인합니다."""
        return self.model is not None
    
    def get_model_info(self):
        """모델 정보를 반환합니다."""
        return {
            'model_type': 'PyTorch',
            'diagnosis_type': self.diagnosis_type,
            'class_names': self.class_names,
            'device': str(self.device),
            'model_loaded': self.is_model_loaded(),
            'model_path': self.model_path
        } 