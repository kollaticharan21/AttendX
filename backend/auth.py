import os
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, g
from werkzeug.security import check_password_hash, generate_password_hash
from database import db_session, User

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "attendx_srkr_college_super_secret_jwt_key_2026")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRATION_HOURS = int(os.environ.get("TOKEN_EXPIRATION_HOURS", 24))


def generate_token(user: User) -> str:
    """Generate a JWT token for the authenticated user."""
    payload = {
        "user_id": user.id,
        "reg_number": user.reg_number,
        "role": user.role,
        "full_name": user.full_name,
        "department": user.department,
        "section": user.section,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])


def jwt_required(f):
    """Decorator to enforce valid JWT authentication on protected endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization", None)
        if not auth_header:
            return jsonify({"success": False, "message": "Authorization header is missing"}), 401

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return jsonify({"success": False, "message": "Invalid Authorization header format. Expected 'Bearer <token>'"}), 401

        token = parts[1]
        try:
            payload = decode_token(token)
            session = db_session()
            user = session.query(User).filter_by(id=payload.get("user_id")).first()
            if not user:
                session.close()
                return jsonify({"success": False, "message": "User not found or deactivated"}), 401
            
            g.current_user = user
            session.close()
        except jwt.ExpiredSignatureError:
            return jsonify({"success": False, "message": "Token has expired. Please log in again."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"success": False, "message": "Invalid token. Authorization denied."}), 401
        except Exception as e:
            return jsonify({"success": False, "message": f"Authentication failed: {str(e)}"}), 401

        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to enforce admin role access on sensitive endpoints."""
    @wraps(f)
    @jwt_required
    def decorated_function(*args, **kwargs):
        if not hasattr(g, "current_user") or g.current_user.role != "admin":
            return jsonify({"success": False, "message": "Admin privileges required for this action."}), 403
        return f(*args, **kwargs)
    return decorated_function