"""
Step3 진단 처리 모듈

이 모듈은 step3 세부 진단 로직을 담당합니다.
- inflammation: 염증류 진단 로직
- corneal: 각막류 진단 로직
"""

from .inflammation import InflammationDiagnosis
from .corneal import CornealDiagnosis

__all__ = ['InflammationDiagnosis', 'CornealDiagnosis'] 