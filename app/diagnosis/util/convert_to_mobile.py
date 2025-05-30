#!/usr/bin/env python3
"""
기존 PyTorch 모델을 PyTorch Mobile 형식으로 변환
"""

import os
import sys
import logging
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.parent  # diagnosis 폴더로 이동
sys.path.append(str(current_dir))

from model import MobileModelConverter, DiagnosisModel, LiteMobileDiagnosisModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# 모델 경로 설정 (여기서 수정하세요!)
# ========================================

# Step1 모델 설정
STEP1_INPUT_PATH = "../models/step1"  # 기존 step1 모델 경로
STEP1_OUTPUT_PATH = "../models/step1_mobile.ptl"  # 변환될 mobile 모델 저장 경로

# Step2 모델 설정  
STEP2_INPUT_PATH = "../models/step2"  # 기존 step2 모델 경로
STEP2_OUTPUT_PATH = "../models/step2_mobile.ptl"  # 변환될 mobile 모델 저장 경로

# 크기 비교 여부
COMPARE_SIZES = True

def convert_model(model_type, input_path, output_path):
    """개별 모델 변환 함수"""
    try:
        # 절대 경로로 변환
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(script_dir, input_path)
        output_path = os.path.join(script_dir, output_path)
        
        # 입력 경로 확인
        if not os.path.exists(input_path):
            logger.error(f"❌ Input path not found: {input_path}")
            return False
        
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"🔄 Starting conversion for {model_type} model...")
        logger.info(f"   Input: {input_path}")
        logger.info(f"   Output: {output_path}")
        
        # 모델 변환
        mobile_path = MobileModelConverter.convert_to_mobile(
            original_model_path=input_path,
            mobile_model_path=output_path,
            model_type=model_type
        )
        
        logger.info(f"✅ {model_type} conversion completed successfully!")
        logger.info(f"   Mobile model saved to: {mobile_path}")
        
        # 크기 비교 (요청된 경우)
        if COMPARE_SIZES:
            try:
                import torch
                
                # 원본 모델 크기 계산을 위해 임시로 원본 모델 로드
                original_model = DiagnosisModel(
                    model_path=input_path,
                    model_type=model_type,
                    use_mobile=False
                )
                
                # 임시 파일로 원본 모델 저장
                temp_original_path = output_path.replace('.ptl', '_temp_original.pt')
                torch.jit.save(original_model.model, temp_original_path)
                
                # 크기 비교
                comparison = MobileModelConverter.compare_model_sizes(
                    temp_original_path, mobile_path
                )
                
                if comparison:
                    print(f"\n📊 {model_type.upper()} Model Size Comparison:")
                    print(f"   Original model: {comparison['original_size_mb']} MB")
                    print(f"   Mobile model: {comparison['mobile_size_mb']} MB")
                    print(f"   Size reduction: {comparison['size_reduction_percent']}% ({comparison['size_reduction_mb']} MB saved)")
                
                # 임시 파일 삭제
                if os.path.exists(temp_original_path):
                    os.remove(temp_original_path)
                    
            except Exception as e:
                logger.warning(f"Could not compare sizes for {model_type}: {str(e)}")
        
        # 변환된 모델 테스트
        logger.info(f"🧪 Testing converted {model_type} mobile model...")
        test_mobile_model(mobile_path, model_type)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ {model_type} conversion failed: {str(e)}")
        return False

def test_mobile_model(mobile_path, model_type):
    """변환된 모바일 모델을 테스트"""
    try:
        import torch
        from PIL import Image
        import numpy as np
        
        # 클래스 이름 설정
        if model_type == "step1":
            class_names = ['abnormal', 'normal']
        else:
            class_names = ['corneal', 'inflammation']
        
        # 모바일 모델 로드
        mobile_model = LiteMobileDiagnosisModel(mobile_path, class_names)
        
        # 더미 이미지로 테스트
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # 추론 테스트
        result = mobile_model.predict(dummy_image)
        
        logger.info(f"✅ {model_type} mobile model test successful!")
        logger.info(f"   Test prediction: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
        
    except Exception as e:
        logger.warning(f"❌ {model_type} mobile model test failed: {str(e)}")

def main():
    """메인 실행 함수"""
    print("🚀 PyTorch Mobile Model Converter")
    print("=" * 50)
    
    success_count = 0
    total_count = 2
    
    # Step1 모델 변환
    print(f"\n📱 Converting Step1 Model...")
    if convert_model("step1", STEP1_INPUT_PATH, STEP1_OUTPUT_PATH):
        success_count += 1
    
    # Step2 모델 변환
    print(f"\n📱 Converting Step2 Model...")
    if convert_model("step2", STEP2_INPUT_PATH, STEP2_OUTPUT_PATH):
        success_count += 1
    
    # 결과 요약
    print("\n" + "=" * 50)
    print(f"🎯 Conversion Summary:")
    print(f"   ✅ Successful: {success_count}/{total_count}")
    print(f"   ❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print(f"\n🎉 All models converted successfully!")
        print(f"   Step1 Mobile: {os.path.join(os.path.dirname(__file__), STEP1_OUTPUT_PATH)}")
        print(f"   Step2 Mobile: {os.path.join(os.path.dirname(__file__), STEP2_OUTPUT_PATH)}")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Update your Flask app to use MobileDiagnosisService")
        print(f"   2. Set mobile model paths in your config:")
        print(f"      STEP1_MOBILE_MODEL_PATH = '{STEP1_OUTPUT_PATH}'")
        print(f"      STEP2_MOBILE_MODEL_PATH = '{STEP2_OUTPUT_PATH}'")
    else:
        print(f"\n⚠️  Some conversions failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 