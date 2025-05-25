import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
import os
import logging
from flask import current_app

logger = logging.getLogger(__name__)

class Step1Model:
    """Step1 질병여부판단 AI 모델 클래스"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_type = "efficientnet-b2"
        self.img_size = 224
        self.num_classes = 2  # normal, abnormal
        self.class_names = ['abnormal', 'normal']  # 0: abnormal, 1: normal
        
        # 모델 경로 설정
        if model_path is None:
            try:
                # Flask 앱 컨텍스트에서 config 가져오기
                model_path = current_app.config.get('STEP1_MODEL_PATH')
            except RuntimeError:
                # Flask 앱 컨텍스트가 없는 경우 기본 경로 사용
                model_path = os.path.join(os.path.dirname(__file__), 'models', 'step1')
        
        self.model_path = model_path
        
        # 이미지 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 모델 로드
        self._load_model()
    
    def _load_model(self):
        """AI 모델을 로드합니다."""
        try:
            # EfficientNet 모델 생성
            model = EfficientNet.from_pretrained(self.model_type, num_classes=self.num_classes)
            
            # 모델 파일 경로 확인
            if os.path.isfile(self.model_path):
                # 파일 경로인 경우
                model_file = self.model_path
            else:
                # 디렉토리 경로인 경우 best_model.pth 또는 step1 파일 찾기
                possible_files = [
                    os.path.join(self.model_path, 'best_model.pth'),
                    os.path.join(self.model_path, 'step1'),
                    os.path.join(self.model_path, 'model.pth')
                ]
                
                model_file = None
                for file_path in possible_files:
                    if os.path.exists(file_path):
                        model_file = file_path
                        break
                
                if model_file is None:
                    raise FileNotFoundError(f"Model file not found in {self.model_path}")
            
            logger.info(f"Loading model from: {model_file}")
            
            # 파일 크기 확인
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            logger.info(f"Model file size: {file_size:.2f} MB")
            
            # 모델 로드
            checkpoint = torch.load(model_file, map_location=self.device)
            
            # checkpoint 정보 로깅
            logger.info(f"Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
                
                # 훈련 정보가 있다면 출력
                if 'epoch' in checkpoint:
                    logger.info(f"Model epoch: {checkpoint['epoch']}")
                if 'best_acc' in checkpoint:
                    logger.info(f"Best accuracy: {checkpoint['best_acc']}")
                if 'loss' in checkpoint:
                    logger.info(f"Loss: {checkpoint['loss']}")
            
            # checkpoint가 dict인 경우 (state_dict가 포함된 경우)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                # checkpoint가 직접 state_dict인 경우
                state_dict = checkpoint
            
            # state_dict 정보 로깅
            logger.info(f"State dict keys count: {len(state_dict.keys())}")
            logger.info(f"First few keys: {list(state_dict.keys())[:5]}")
            
            # DataParallel로 저장된 모델인 경우 처리
            if list(state_dict.keys())[0].startswith('module.'):
                logger.info("Removing 'module.' prefix from DataParallel model")
                # DataParallel로 저장된 모델의 키에서 'module.' 제거
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('module.', '')
                    new_state_dict[new_key] = value
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            
            # 모델 파라미터 수 확인
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total model parameters: {total_params:,}")
            
            self.model = model
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise Exception(f"Model loading failed: {str(e)}")
    
    def preprocess_image(self, image):
        """이미지를 모델 입력 형태로 전처리합니다."""
        try:
            # PIL Image로 변환 (이미 PIL Image인 경우 그대로 사용)
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            # RGB로 변환 (RGBA나 다른 모드인 경우)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 전처리 적용
            tensor = self.transform(image)
            
            # 배치 차원 추가
            tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise Exception(f"Image preprocessing failed: {str(e)}")
    
    def predict(self, image):
        """이미지에 대해 질병 여부를 예측합니다."""
        try:
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
            
            # 원시 출력값도 로깅
            logger.info(f"Raw model outputs: {outputs[0].tolist()}")
            logger.info(f"Softmax probabilities: {probabilities[0].tolist()}")
            logger.info(f"Predicted class index: {predicted_class}")
            logger.info(f"Class names mapping: {dict(enumerate(self.class_names))}")
            
            # 결과 해석
            is_normal = (predicted_class == 1)  # 0: abnormal, 1: normal
            
            result = {
                'is_normal': is_normal,
                'confidence': float(confidence),
                'predicted_class': self.class_names[predicted_class],
                'predicted_class_index': predicted_class,
                'probabilities': {
                    'normal': float(probabilities[0][1]),      # 인덱스 1이 normal
                    'abnormal': float(probabilities[0][0])     # 인덱스 0이 abnormal
                },
                'raw_outputs': outputs[0].tolist()  # 디버깅용
            }
            
            logger.info(f"Prediction result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")
    
    def is_model_loaded(self):
        """모델이 로드되었는지 확인합니다."""
        return self.model is not None 