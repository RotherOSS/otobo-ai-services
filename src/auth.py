from src.settings import AppSettings

from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException
from starlette.status import HTTP_403_FORBIDDEN

settings = AppSettings()

api_key_header = APIKeyHeader(name="access_token", auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """checks whether the api key is correct

    Args:
        api_key_header (str, optional): _description_. Defaults to Security(api_key_header).

    Raises:
        HTTPException: _description_

    Returns:
        _type_: _description_
    """
    if api_key_header == settings.AI_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate AI API KEY"
        )
