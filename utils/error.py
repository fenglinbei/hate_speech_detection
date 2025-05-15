class ApiError(Exception):
    """
    Exception raised when an API request fails.
    """
    def __init__(self, message, status_code=None, response_body=None, *args, **kwargs):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body

    def __str__(self):
        return f"API Error: Status Code {self.status_code}, Response Body: {self.response_body}"
    
class ProcessError(Exception):
    """
    Exception raised when review process fails.
    """
    def __init__(self, message: str, error_code: int):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
    
    def __str__(self):
        return f"Process Error: Error Code {self.error_code}, Message: {self.message}"
