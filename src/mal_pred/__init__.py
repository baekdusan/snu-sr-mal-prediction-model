"""
mal_pred 패키지 초기화 모듈.

핵심 클래스를 외부에 노출하여 `from mal_pred import MALPredictor` 형태로
간단히 사용할 수 있게 한다.
"""

from .predictor import MALPredictor

__all__ = ["MALPredictor"]


