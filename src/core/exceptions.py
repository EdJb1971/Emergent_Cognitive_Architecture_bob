from fastapi import HTTPException, status

class APIException(HTTPException):
    """
    Custom exception for consistent API error responses.
    """
    def __init__(self, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, detail: str = "An unexpected error occurred"):
        super().__init__(status_code=status_code, detail=detail)

class LLMServiceException(APIException):
    """
    Specific exception for errors originating from the LLM Integration Service.
    """
    def __init__(self, detail: str = "LLM service error", status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(status_code=status_code, detail=detail)

class ConfigurationError(APIException):
    """
    Specific exception for configuration related errors.
    """
    def __init__(self, detail: str = "Configuration error", status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(status_code=status_code, detail=detail)

class AgentServiceException(APIException):
    """
    Specific exception for errors originating from any specialized AI agent.
    """
    def __init__(self, agent_id: str, detail: str = "Agent service error", status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(status_code=status_code, detail=f"[{agent_id}] {detail}")
        self.agent_id = agent_id
