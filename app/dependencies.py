"""
Shared FastAPI dependencies.

API Key authentication
──────────────────────
All /api/* routes use `verify_api_key` to prevent unintended access from
other machines on the same LAN / Tailscale network.

• Clients must pass the header:  X-API-Key: <key>
• The key is configured in app/config.py → Settings.api_key
• If api_key is set to "" authentication is disabled (open access).

The Web UI itself embeds the key in every page (via a <meta> tag in base.html)
so the in-browser JavaScript can include the header transparently.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    """Dependency: raise 401 when the API key is wrong or missing."""
    if not settings.api_key:
        # Auth is disabled — allow all requests.
        return
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Set the header: X-API-Key: <key>",
            headers={"WWW-Authenticate": "ApiKey"},
        )
