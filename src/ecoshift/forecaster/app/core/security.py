import logging
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
import hmac

from ecoshift.forecaster.app.core.config import settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key_header: Optional[str] = Security(API_KEY_HEADER)) -> str:
    if not hasattr(settings, "API_KEY") or not settings.API_KEY:
        return "unprotected"

    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"API Key not found in the header '{API_KEY_HEADER.name}'",
        )

    is_valid = hmac.compare_digest(api_key_header, settings.API_KEY)

    if not is_valid:
        logger.warning("Login try unauthorized with invalid API key")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access Refused : Invalid API Key"
        )

    return api_key_header