import os
import json
import numpy as np
from datetime import datetime, date
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Date,
    DateTime,
    ForeignKey,
    Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, scoped_session
from werkzeug.security import generate_password_hash

import tempfile
import shutil

Base = declarative_base()

if os.environ.get("VERCEL"):
    tmp_dir = tempfile.gettempdir()
    tmp_db = os.path.join(tmp_dir, "attendx.db")
    seed_db = os.path.join(os.path.abspath(os.path.dirname(__file__)), "attendx.db")
    if os.path.exists(seed_db) and not os.path.exists(tmp_db):
        try:
            shutil.copy2(seed_db, tmp_db)
        except Exception as _e:
            pass
    DEFAULT_DATABASE_URL = f"sqlite:///{tmp_db.replace(chr(92), '/')}"
else:
    DEFAULT_DATABASE_URL = "sqlite:///attendx.db"

DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    reg_number = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="student")  # 'admin' or 'student'
    department = Column(String(32), nullable=True, index=True)    # e.g. 'CSE', 'IT', 'ECE'
    section = Column(String(16), nullable=True, index=True)       # e.g. 'A', 'B', 'C'
    year = Column(String(1), nullable=True, default="2", index=True)  # academic year 1-4
    full_name = Column(String(128), nullable=False)
    profile_photo_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    face_embeddings = relationship(
        "FaceEmbedding", back_populates="user", cascade="all, delete-orphan"
    )
    attendance_records = relationship(
        "AttendanceRecord", back_populates="user", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "reg_number": self.reg_number,
            "role": self.role,
            "department": self.department,
            "section": self.section,
            "year": self.year or "2",
            "full_name": self.full_name,
            "name": self.full_name,  # for frontend compatibility
            "profile_photo_path": self.profile_photo_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    embedding = Column(Text, nullable=False)  # JSON-serialized 512-d float list
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="face_embeddings")

    def set_vector(self, vector: np.ndarray):
        """Store 512-d numpy vector as a normalized JSON array."""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        self.embedding = json.dumps(vector.tolist())

    def get_vector(self) -> np.ndarray:
        """Retrieve stored embedding as a float32 numpy vector."""
        arr = np.array(json.loads(self.embedding), dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    department = Column(String(32), nullable=False, index=True)
    section = Column(String(16), nullable=False, index=True)
    year = Column(String(1), nullable=False, default="2", index=True)
    date = Column(Date, nullable=False, default=date.today, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    status = Column(String(20), nullable=False)  # 'Present' or 'Absent'

    user = relationship("User", back_populates="attendance_records")

    __table_args__ = (
        Index("idx_attendance_dept_sec_year_date", "department", "section", "year", "date"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "reg_number": self.user.reg_number if self.user else None,
            "student_name": self.user.full_name if self.user else None,
            "department": self.department,
            "section": self.section,
            "year": self.year or "2",
            "date": self.date.isoformat() if self.date else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "status": self.status,
        }


def init_db():
    Base.metadata.create_all(bind=engine)
    _add_legacy_year_columns()
    seed_default_admin()


def _add_legacy_year_columns():
    """Add year columns to databases created before academic-year support."""
    if not DATABASE_URL.startswith("sqlite"):
        return
    with engine.begin() as connection:
        for table in ("users", "attendance_records"):
            columns = {row[1] for row in connection.exec_driver_sql(f"PRAGMA table_info({table})")}
            if "year" not in columns:
                connection.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN year VARCHAR(1) NOT NULL DEFAULT '2'"
                )


def seed_default_admin():
    session = db_session()
    try:
        admin = session.query(User).filter_by(role="admin").first()
        if not admin:
            admin_reg = os.environ.get("DEFAULT_ADMIN_REG", "ADMIN01")
            admin_pass = os.environ.get("DEFAULT_ADMIN_PASS", "Admin@123")
            new_admin = User(
                reg_number=admin_reg,
                password_hash=generate_password_hash(admin_pass),
                role="admin",
                department="CSE",
                section="A",
                full_name="System Administrator",
                profile_photo_path=None
            )
            session.add(new_admin)
            session.commit()
            print(f"[AttendX DB] Seeded default admin: {admin_reg} / {admin_pass}")
    except Exception as e:
        session.rollback()
        print(f"[AttendX DB] Warning during admin seeding: {e}")
    finally:
        session.close()