class FinSightError(Exception):
    """Base custom exception for all FinSight application errors."""
    def __init__(self, message: str, error_type: str = "InternalError") -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type