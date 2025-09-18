"""
General Chatbot Module
일반적인 수의사 어시스턴트 챗봇 기능을 제공합니다.
"""

from .api import general_chat_ns
from .service import GeneralChatService

__all__ = ['general_chat_ns', 'GeneralChatService']
