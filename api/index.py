import os
import sys

BACKEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "backend")
)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app import app
from database import ensure_db

try:
    ensure_db()
except Exception as e:
    print(f"[AttendX] DB initialization warning: {e}")


class ApiPathMiddleware:
    """Restore the real /api/... path after Vercel's rewrite to /api/index.py.

    vercel.json passes the original path in the ?__p= query parameter, which is
    the most reliable signal. Header-based fallbacks are kept for safety.
    """

    def __init__(self, application):
        self.application = application

    def __call__(self, environ, start_response):
        from urllib.parse import parse_qsl, urlencode

        query = environ.get("QUERY_STRING", "")
        params = parse_qsl(query, keep_blank_values=True)
        real = None
        rest = []
        for k, v in params:
            if k == "__p" and real is None:
                real = v
            else:
                rest.append((k, v))

        if real is not None:
            environ["PATH_INFO"] = "/api/" + real.strip("/") if real.strip("/") else "/api"
            environ["QUERY_STRING"] = urlencode(rest)
        else:
            current = environ.get("PATH_INFO", "")
            if current in ("/api/index.py", "/api/index"):
                raw = (
                    environ.get("HTTP_X_VERCEL_FORWARDED_PATH")
                    or environ.get("HTTP_X_MATCHED_PATH")
                    or environ.get("REQUEST_URI")
                    or environ.get("RAW_URI")
                )
                if raw:
                    clean = raw.split("?")[0]
                    if clean and clean not in ("/api/index.py", "/api/index"):
                        environ["PATH_INFO"] = clean

        return self.application(environ, start_response)


app.wsgi_app = ApiPathMiddleware(app.wsgi_app)