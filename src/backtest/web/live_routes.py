import hashlib
import os

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request

router = APIRouter()

_LOGIN_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>登录</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e0e0e0;
         display: flex; justify-content: center; align-items: center; height: 100vh; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 32px; width: 320px; }}
  h2 {{ font-size: 18px; color: #00d4aa; margin-bottom: 24px; text-align: center; }}
  input {{ width: 100%; padding: 10px 12px; background: #0f1117; color: #e0e0e0;
          border: 1px solid #333; border-radius: 6px; font-size: 14px; margin-bottom: 16px; }}
  button {{ width: 100%; padding: 10px; background: #00d4aa; color: #0f1117;
           border: none; border-radius: 6px; font-size: 14px; cursor: pointer; font-weight: 600; }}
  .error {{ color: #ef4444; font-size: 13px; margin-bottom: 12px; text-align: center; }}
</style>
</head>
<body>
<div class="card">
  <h2>实盘监控</h2>
  {error}
  <form method="POST" action="/login">
    <input type="password" name="password" placeholder="密码" autofocus>
    <button type="submit">登录</button>
  </form>
</div>
</body>
</html>"""


@router.get("/login", response_class=HTMLResponse)
def login_page():
    return HTMLResponse(_LOGIN_HTML.format(error=""))


@router.post("/login")
async def do_login(request: Request):
    form = await request.form()
    password = str(form.get("password", ""))
    expected_password = os.environ.get("WEB_PASSWORD", "")
    if not expected_password or password == expected_password:
        resp = RedirectResponse(url="/live", status_code=302)
        if expected_password:
            session_val = hashlib.sha256(expected_password.encode()).hexdigest()
            resp.set_cookie("session", session_val, httponly=True, samesite="lax")
        return resp
    return HTMLResponse(
        _LOGIN_HTML.format(error='<p class="error">密码错误</p>'),
        status_code=401,
    )
