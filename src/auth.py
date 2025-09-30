from src.settings import AppSettings

from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException
from starlette.status import HTTP_403_FORBIDDEN
from loguru import logger

# Load global settings (e.g., secrets, API keys) from a centralized config object
settings = AppSettings()

api_key_header = APIKeyHeader(name="access-token", auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    FastAPI dependency that checks if the provided API key matches the configured one.

    This function will be called automatically when declared as a dependency in route handlers.
    If the key is valid, it is returned. Otherwise, an HTTP 403 error is raised.

    Customization: Modify `settings.OTOBO_AI_API_KEY` to control which API key is allowed access.
    """
    
    logger.error(f"API KEY : {api_key_header} =?= {settings.OTOBO_AI_API_KEY}")
    
    if api_key_header == settings.OTOBO_AI_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate AI API KEY"
        )
