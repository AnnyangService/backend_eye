"""Base business exception class for all application exceptions"""

class BusinessException(Exception):
    """Base exception class for business logic errors"""
    
    def __init__(self, message, error_code=None, status_code=400):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)

class DatabaseException(BusinessException):
    """Exception raised for database errors"""
    
    def __init__(self, message="Database operation failed", error_code="DB_ERROR", status_code=500):
        super().__init__(message, error_code, status_code)

class EntityNotFoundException(BusinessException):
    """Exception raised when an entity is not found"""
    
    def __init__(self, entity="Entity", error_code="NOT_FOUND", status_code=404):
        message = f"{entity} not found"
        super().__init__(message, error_code, status_code) 