import os
import sys

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app import app
from database import init_db

try:
    init_db()
except Exception as e:
    print(f"[AttendX] DB initialization warning: {e}")


class ApiPathMiddleware:
    def __init__(self, application):
        self.application = application

    def __call__(self, environ, start_response):
        raw_uri = environ.get("REQUEST_URI") or environ.get("RAW_URI") or environ.get("HTTP_X_MATCHED_PATH")
        if raw_uri:
            clean_path = raw_uri.split("?")[0]
            if clean_path and clean_path not in ("/api/index.py", "/api/index"):
                environ["PATH_INFO"] = clean_path

        return self.application(environ, start_response)


app.wsgi_app = ApiPathMiddleware(app.wsgi_app)
