import os
import urllib.parse
import logging
import requests
import threading
import torch
from PIL import Image
from flask import current_app
from .ai_model import DiagnosisModel

# 로거 설정
logger = logging.getLogger(__name__)

class DiagnosisService:
    def __init__(self):
        # 이미지 저장 디렉토리 설정
        try:
            # Flask 앱 컨텍스트가 있는 경우
            self.images_dir = os.path.join(current_app.instance_path, 'images')
        except RuntimeError:
            # Flask 앱 컨텍스트가 없는 경우 기본 경로 사용
            self.images_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'instance', 'images')
        
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir, exist_ok=True)
        
        # AI 모델 초기화
        self.step1_model = None
        self.step2_model = None
        self._initialize_model()
    
    def _get_model_path(self, step_type):
        """모델 경로를 가져옵니다."""
        config_key = f'{step_type.upper()}_MODEL_PATH'
        try:
            model_path = current_app.config.get(config_key)
            if model_path is None:
                model_path = os.path.join(os.path.dirname(__file__), 'models', step_type)
        except RuntimeError:
            # Flask 앱 컨텍스트가 없는 경우 기본 경로 사용
            model_path = os.path.join(os.path.dirname(__file__), 'models', step_type)
        return model_path

    def _initialize_model(self):
        """AI 모델을 초기화합니다."""
        try:
            logger.info("AI 모델 초기화 시작...")
            
            step1_model_path = self._get_model_path('step1')
            logger.info(f"Step1 모델 경로: {step1_model_path}")
            self.step1_model = DiagnosisModel(model_path=step1_model_path, model_type="step1")
            
            # Step1 모델 동적 양자화 적용 (항상 시도)
            if self.step1_model.is_model_loaded():
                self._apply_dynamic_quantization(self.step1_model, "Step1")
            
            logger.info("Step1 모델 로드 완료")
            
            step2_model_path = self._get_model_path('step2')
            logger.info(f"Step2 모델 경로: {step2_model_path}")
            self.step2_model = DiagnosisModel(model_path=step2_model_path, model_type="step2")
            
            # Step2 모델 동적 양자화 적용 (항상 시도)
            if self.step2_model.is_model_loaded():
                self._apply_dynamic_quantization(self.step2_model, "Step2")
            
            logger.info("Step2 모델 로드 완료")
            logger.info("AI 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"AI 모델 초기화 실패: {str(e)}")
            import traceback
            logger.error(f"상세 에러: {traceback.format_exc()}")
            # 모델 로드 실패 시 None으로 설정
            self.step1_model = None
            self.step2_model = None
            # 에러를 다시 발생시켜서 서비스 초기화 시점에 문제를 알림
            raise Exception(f"AI 모델 로드 실패: {str(e)}")
    
    def _apply_dynamic_quantization(self, model_wrapper, model_name):
        """
        모델에 동적 양자화를 적용합니다. 실패 시 원본 모델을 유지합니다.
        
        Args:
            model_wrapper: DiagnosisModel 인스턴스
            model_name: 모델 이름 (로깅용)
        """
        try:
            logger.info(f"{model_name} 동적 양자화 시작...")
            
            # 양자화 전 모델 크기 측정
            original_size = self._get_model_size(model_wrapper.model)
            
            # 동적 양자화 적용
            quantized_model = torch.quantization.quantize_dynamic(
                model_wrapper.model,
                {torch.nn.Linear, torch.nn.Conv2d},  # 양자화할 레이어 타입
                dtype=torch.qint8  # 8비트 정수로 양자화
            )
            
            # 양자화된 모델로 교체
            model_wrapper.model = quantized_model
            
            # 양자화 후 모델 크기 측정
            quantized_size = self._get_model_size(quantized_model)
            
            # 크기 감소 비율 계산
            size_reduction = ((original_size - quantized_size) / original_size) * 100
            
            logger.info(f"{model_name} 동적 양자화 완료:")
            logger.info(f"  - 원본 크기: {original_size:.2f} MB")
            logger.info(f"  - 양자화 후 크기: {quantized_size:.2f} MB")
            logger.info(f"  - 크기 감소: {size_reduction:.1f}%")
            
        except Exception as e:
            logger.warning(f"{model_name} 동적 양자화 실패 (원본 모델 유지): {str(e)}")
            # 양자화 실패 시 원본 모델을 그대로 사용
    
    def _get_model_size(self, model):
        """
        모델의 메모리 크기를 MB 단위로 계산합니다.
        
        Args:
            model: PyTorch 모델
            
        Returns:
            float: 모델 크기 (MB)
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def get_model_info(self):
        """현재 로드된 모델 정보를 반환합니다."""
        info = {
            "service_info": {
                "step1_model_loaded": self.step1_model is not None,
                "step2_model_loaded": self.step2_model is not None,
                "quantization_attempted": True,  # 항상 양자화 시도
                "images_directory": self.images_dir
            }
        }
        
        if self.step1_model is not None:
            step1_info = self.step1_model.get_model_info()
            step1_info["quantized"] = self._is_model_quantized(self.step1_model.model)
            info["step1_model"] = step1_info
        
        if self.step2_model is not None:
            step2_info = self.step2_model.get_model_info()
            step2_info["quantized"] = self._is_model_quantized(self.step2_model.model)
            info["step2_model"] = step2_info
        
        return info
    
    def _is_model_quantized(self, model):
        """모델이 양자화되었는지 확인합니다."""
        for module in model.modules():
            if hasattr(module, '_packed_params') or 'quantized' in str(type(module)).lower():
                return True
        return False
    
    def _download_image(self, image_url):
        """
        이미지 다운로드
        
        Args:
            image_url (str): 다운로드할 이미지 URL
            
        Returns:
            str: 로컬에 저장된 이미지 파일 경로
        """
        # URL에서 파일명 추출
        # S3 이후 수정 필요
        parsed_url = urllib.parse.urlparse(image_url)
        filename = os.path.basename(parsed_url.path)
        if not filename or '.' not in filename:
            filename = 'image.jpg'
        
        local_path = os.path.join(self.images_dir, filename)
        
        # 이미 파일이 있으면 기존 파일 사용
        if os.path.exists(local_path):
            logger.debug(f"기존 이미지 파일 사용: {local_path}")
            return local_path
        
        # 이미지 다운로드
        logger.debug(f"이미지 다운로드 시작: {image_url}")
        
        # config에서 timeout 설정 가져오기
        try:
            timeout = current_app.config.get('IMAGE_DOWNLOAD_TIMEOUT', 30)
        except RuntimeError:
            timeout = 30
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()
        
        # Content-Type 검증
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            error_msg = f"유효하지 않은 이미지 타입: {content_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 파일 저장
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        logger.debug(f"이미지 다운로드 완료: {local_path}")
        return local_path
    
    def _validate_image(self, image):
        """이미지 유효성 검사"""
        # 고도화시 추가 검증 로직 구현 필요
        # config에서 이미지 크기 제한 가져오기
        try:
            min_size = current_app.config.get('MIN_IMAGE_SIZE', 100)
            max_size = current_app.config.get('MAX_IMAGE_SIZE', 4096)
        except RuntimeError:
            min_size = 100
            max_size = 4096
        
        if image.width < min_size or image.height < min_size:
            error_msg = f"이미지 크기가 너무 작습니다 (최소 {min_size}x{min_size})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 이미지 크기가 너무 큰 경우 제한
        if image.width > max_size or image.height > max_size:
            error_msg = f"이미지 크기가 너무 큽니다 (최대 {max_size}x{max_size})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"이미지 검증 완료: {image.width}x{image.height}")
    
    def process_step1_diagnosis(self, image_url):
        """
        Step1: 질병여부판단
        
        POST /diagnosis/step1/ 요청을 처리하는 메인 함수
        
        Args:
            image_url (str): 분석할 이미지의 URL
            
        Returns:
            dict: {
                "is_normal": bool,    # 정상 여부
                "confidence": float   # 신뢰도 (0.0 ~ 1.0)
            }
        """
        logger.info(f"Step1 진단 시작: {image_url}")
        
        try:
            # 1. 이미지 다운로드
            local_path = self._download_image(image_url)
            
            # 2. 이미지 로드 및 검증
            image = Image.open(local_path)
            self._validate_image(image)
            
            logger.debug(f"이미지 로드 성공: {local_path} ({image.width}x{image.height})")
            
            # 3. AI 모델 분석
            if self.step1_model and self.step1_model.is_model_loaded():
                logger.debug(f"AI 모델 분석 시작: {local_path}")
                
                # 실제 AI 모델 추론
                prediction_result = self.step1_model.predict(image)
                
                result = {
                    "is_normal": prediction_result['is_normal'],
                    "confidence": prediction_result['confidence']
                }
                
                logger.info(f"Step1 진단 완료 - 정상: {result['is_normal']}, 신뢰도: {result['confidence']:.3f}")
                return result
                
            else:
                # AI 모델이 로드되지 않은 경우 에러 발생
                error_msg = "AI 모델이 로드되지 않았습니다."
                logger.error(error_msg)
                raise Exception(error_msg)
            
        except requests.RequestException as e:
            error_msg = f"이미지 다운로드 실패: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Step1 진단 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def process_step2_diagnosis(self, request_id, password, image_url):
        """
        Step2: 진단 처리
        
        POST /diagnosis/step2/ 요청을 처리하는 메인 함수
        
        Args:
            request_id (str): 요청 ID
            password (str): AI 서버 -> API 서버 호출시 필요한 패스워드
            image_url (str): 분석할 이미지의 URL
            
        Returns:
            dict: {
                "category": str,      # 진단 카테고리
                "confidence": float   # 신뢰도 (0.0 ~ 1.0)
            }
        """
        logger.info(f"Step2 진단 시작: ID={request_id}, URL={image_url}")
        
        try:
            # 1. 이미지 다운로드
            local_path = self._download_image(image_url)
            
            # 2. 이미지 로드 및 검증
            image = Image.open(local_path)
            self._validate_image(image)
            
            logger.debug(f"이미지 로드 성공: {local_path} ({image.width}x{image.height})")
            
            # 3. AI 모델 분석
            if self.step2_model and self.step2_model.is_model_loaded():
                logger.debug(f"Step2 AI 모델 분석 시작: {local_path}")
                
                # 실제 AI 모델 추론
                prediction_result = self.step2_model.predict(image)
                
                result = {
                    "category": prediction_result['category'],
                    "confidence": prediction_result['confidence']
                }
                
                logger.info(f"Step2 진단 완료 - 카테고리: {result['category']}, 신뢰도: {result['confidence']:.3f}")
                return result
                
            else:
                # AI 모델이 로드되지 않은 경우 에러 발생
                error_msg = "Step2 AI 모델이 로드되지 않았습니다."
                logger.error(error_msg)
                raise Exception(error_msg)
            
        except requests.RequestException as e:
            error_msg = f"이미지 다운로드 실패: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Step2 진단 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def process_step2_diagnosis_async(self, request_id, password, image_url):
        """
        Step2: 비동기 진단 처리
        
        즉시 응답을 반환하고 백그라운드에서 추론을 실행한 후 API 서버로 결과를 전송합니다.
        
        Args:
            request_id (str): 요청 ID
            password (str): AI 서버 -> API 서버 호출시 필요한 패스워드
            image_url (str): 분석할 이미지의 URL
            
        Returns:
            dict: 즉시 응답 (data는 항상 null)
        """
        logger.info(f"Step2 비동기 진단 요청 접수: ID={request_id}")
        
        # 현재 Flask 앱 인스턴스를 백그라운드 스레드로 전달
        app = current_app._get_current_object()
        
        # 백그라운드 스레드에서 실제 추론 실행
        thread = threading.Thread(
            target=self._process_step2_background,
            args=(app, request_id, password, image_url)
        )
        thread.daemon = True
        thread.start()
        
        # 즉시 응답 반환
        return {
            "success": True,
            "message": "Success",
            "data": None
        }
    
    def _process_step2_background(self, app, request_id, password, image_url):
        """
        백그라운드에서 Step2 추론을 실행하고 결과를 API 서버로 전송합니다.
        """
        # Flask 앱 컨텍스트 설정
        with app.app_context():
            try:
                logger.debug(f"Step2 백그라운드 추론 시작: ID={request_id}")
                
                # 1. 이미지 다운로드
                local_path = self._download_image(image_url)
                
                # 2. 이미지 로드 및 검증
                image = Image.open(local_path)
                self._validate_image(image)
                
                logger.debug(f"이미지 로드 성공: {local_path} ({image.width}x{image.height})")
                
                # 3. AI 모델 분석
                if self.step2_model and self.step2_model.is_model_loaded():
                    logger.debug(f"Step2 AI 모델 분석 시작: {local_path}")
                    
                    # 실제 AI 모델 추론
                    prediction_result = self.step2_model.predict(image)
                    
                    # 4. API 서버로 성공 결과 전송
                    callback_data = {
                        "id": request_id,
                        "password": password,
                        "category": prediction_result['category'],
                        "confidence": prediction_result['confidence'],
                        "error": False,
                        "message": None
                    }
                    
                    self._send_callback_to_api_server(app, callback_data)
                    
                    logger.info(f"Step2 백그라운드 처리 완료: ID={request_id}, 카테고리={prediction_result['category']}")
                    
                else:
                    # AI 모델이 로드되지 않은 경우
                    error_msg = "Step2 AI 모델이 로드되지 않았습니다."
                    logger.error(error_msg)
                    
                    # 에러를 API 서버로 전송
                    self._send_error_to_api_server(app, request_id, password, error_msg)
                    
            except Exception as e:
                error_msg = f"Step2 백그라운드 처리 중 오류 발생: {str(e)}"
                logger.error(error_msg)
                
                # 에러를 API 서버로 전송
                self._send_error_to_api_server(app, request_id, password, error_msg)
    
    def _send_callback_to_api_server(self, app, callback_data):
        """
        Step2 결과를 API 서버로 전송합니다.
        
        Args:
            app: Flask 앱 인스턴스
            callback_data (dict): 전송할 데이터
        """
        try:
            # API 서버 URL 구성
            api_server_url = app.config.get('API_SERVER_URL')
            callback_endpoint = app.config.get('API_SERVER_CALLBACK_ENDPOINT')
            
            callback_url = f"{api_server_url}{callback_endpoint}"
            
            logger.debug(f"API 서버로 콜백 전송 시작: {callback_url}")
            
            # POST 요청 전송
            timeout = app.config.get('IMAGE_DOWNLOAD_TIMEOUT', 30)
            
            response = requests.post(
                callback_url,
                json=callback_data,
                headers={'Content-Type': 'application/json'},
                timeout=timeout
            )
            response.raise_for_status()
            
            logger.info(f"API 서버 콜백 전송 성공: {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"API 서버 콜백 전송 실패: {str(e)}")
        except Exception as e:
            logger.error(f"콜백 전송 중 예상치 못한 오류: {str(e)}")

    def _send_error_to_api_server(self, app, request_id, password, error_message):
        """
        에러를 API 서버로 전송합니다.
        
        Args:
            app: Flask 앱 인스턴스
            request_id (str): 요청 ID
            password (str): 패스워드
            error_message (str): 에러 메시지
        """
        try:
            callback_data = {
                "id": request_id,
                "password": password,
                "category": None,
                "confidence": None,
                "error": True,
                "message": error_message
            }
            
            logger.debug(f"에러 콜백 전송: ID={request_id}, 에러={error_message}")
            self._send_callback_to_api_server(app, callback_data)
            
        except Exception as e:
            logger.error(f"에러 콜백 전송 중 오류: {str(e)}")

