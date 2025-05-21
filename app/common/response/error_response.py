from dataclasses import dataclass
from typing import Optional

@dataclass
class ErrorResponse:
    code: str
    message: str
    details: Optional[dict] = None

class ErrorCode:
    # Common errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_FOUND = "NOT_FOUND"
    
    # Diagnosis specific errors
    IMAGE_PROCESSING_ERROR = "IMAGE_PROCESSING_ERROR"
    INVALID_IMAGE_URL = "INVALID_IMAGE_URL"
    
    @staticmethod
    def get_message(code: str) -> str:
        messages = {
            ErrorCode.VALIDATION_ERROR: "Validation error occurred",
            ErrorCode.INTERNAL_ERROR: "Internal server error",
            ErrorCode.NOT_FOUND: "Resource not found",
            ErrorCode.IMAGE_PROCESSING_ERROR: "Error processing image",
            ErrorCode.INVALID_IMAGE_URL: "Invalid image URL"
        }
        return messages.get(code, "Unknown error") 