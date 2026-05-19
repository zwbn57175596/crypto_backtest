import hashlib
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        password = os.environ.get("WEB_PASSWORD")
        if not password:
            return await call_next(request)

        path = request.url.path
        if path == "/login" or path.startswith("/static/"):
            return await call_next(request)

        expected = hashlib.sha256(password.encode()).hexdigest()
        if request.cookies.get("session") == expected:
            return await call_next(request)

        if path.startswith("/api/"):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return RedirectResponse(url="/login", status_code=302)
