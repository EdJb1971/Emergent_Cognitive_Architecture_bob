import logging
from typing import Annotated
from uuid import UUID

from fastapi import Header, HTTPException, status, Depends

from src.core.config import settings
from src.core.exceptions import APIException

logger = logging.getLogger(__name__)

# SEC-004, CQ-001 Fix: Define a fixed system user ID.
# In a production multi-user system, this would be retrieved from a secure user database
# after successful authentication (e.g., JWT token validation, OAuth2).
# For this fix, we simulate a single system user associated with the single API_KEY.
SYSTEM_USER_ID = UUID("a1b2c3d4-e5f6-7890-1234-567890abcdef") # A fixed UUID for the system user

async def get_api_key_user_id(x_api_key: Annotated[str, Header(alias=settings.API_KEY_HEADER_NAME)]) -> UUID:
    """
    Authenticates requests using a single API key provided in the 'X-API-Key' header.
    This API key maps to a predefined system user ID.
    
    Raises:
        HTTPException: If the API key is invalid or missing.
        APIException: For internal configuration errors.
    """
    # SEC-003, CQ-002 Fix: Validate against a single configured API_KEY instead of a list.
    if not settings.API_KEY:
        logger.critical("API_KEY is not configured. Authentication will fail.")
        raise APIException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API key configuration error.")

    if x_api_key != settings.API_KEY:
        logger.warning(f"Unauthorized access attempt with invalid API key: {x_api_key[:5]}...")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    
    # SEC-004, CQ-001 Fix: Return the fixed SYSTEM_USER_ID directly.
    # This replaces the insecure derivation from a hash and the placeholder comments.
    logger.debug(f"API key authenticated. Assigning SYSTEM_USER_ID: {SYSTEM_USER_ID}")
    return SYSTEM_USER_ID

# Define a reusable dependency for API key authentication
APIKeyAuth = Depends(get_api_key_user_id)
