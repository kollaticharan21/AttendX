import os
import sys
import json
import pytest
import numpy as np
from io import BytesIO
from PIL import Image

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

os.environ["DATABASE_URL"] = "sqlite:///test_attendx.db"
os.environ["JWT_SECRET_KEY"] = "test_secret_key_12345"

from app import app
from database import init_db, db_session, User, FaceEmbedding, AttendanceRecord
from face_engine import face_engine, FaceRecognitionEngine


@pytest.fixture(scope="module")
def test_client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        with app.app_context():
            init_db()
        yield client

    db_session.remove()
    if os.path.exists("test_attendx.db"):
        try:
            os.remove("test_attendx.db")
        except Exception:
            pass


def create_dummy_image_bytes(color=(200, 100, 50), size=(300, 300)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestAuthentication:
    def test_admin_login_success(self, test_client):
        res = test_client.post("/login", json={
            "reg_number": "ADMIN01",
            "password": "Admin@123"
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "token" in data
        assert data["role"] == "admin"

    def test_login_invalid_password(self, test_client):
        res = test_client.post("/login", json={
            "reg_number": "ADMIN01",
            "password": "WrongPassword!"
        })
        assert res.status_code == 401
        data = res.get_json()
        assert data["success"] is False

    def test_login_missing_fields(self, test_client):
        res = test_client.post("/login", json={})
        assert res.status_code == 400


class TestStudentManagement:
    admin_token = None

    @pytest.fixture(autouse=True)
    def setup_admin_token(self, test_client):
        res = test_client.post("/login", json={
            "reg_number": "ADMIN01",
            "password": "Admin@123"
        })
        self.admin_token = res.get_json()["token"]

    def test_register_student(self, test_client):
        img_bytes = create_dummy_image_bytes()
        res = test_client.post(
            "/register-student",
            headers={"Authorization": f"Bearer {self.admin_token}"},
            data={
                "reg_number": "21B01A0599",
                "full_name": "Test Student 99",
                "department": "CSE",
                "section": "A",
                "photo": (BytesIO(img_bytes), "test_student.jpg")
            },
            content_type="multipart/form-data"
        )
        assert res.status_code == 201
        data = res.get_json()
        assert data["success"] is True
        assert data["user"]["reg_number"] == "21B01A0599"

    def test_register_person_multiple_photos(self, test_client):
        img_bytes = create_dummy_image_bytes()
        res = test_client.post(
            "/register-person",
            headers={"Authorization": f"Bearer {self.admin_token}"},
            data={
                "reg_number": "21B01A0588",
                "name": "Charan Teja",
                "department": "CSE",
                "section": "A",
                "photos": [
                    (BytesIO(img_bytes), "photo1.jpg"),
                    (BytesIO(img_bytes), "photo2.jpg"),
                    (BytesIO(img_bytes), "photo3.jpg")
                ]
            },
            content_type="multipart/form-data"
        )
        assert res.status_code == 201
        data = res.get_json()
        assert data["success"] is True
        assert data["embeddings_extracted"] >= 1

    def test_get_people_by_section(self, test_client):
        res = test_client.get("/people?department=CSE&section=A")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert any(p["reg_number"] == "21B01A0599" for p in data["people"])

    def test_get_photo_by_name(self, test_client):
        res = test_client.get("/photo/Test Student 99")
        assert res.status_code == 200

    def test_get_photo_fallback_avatar(self, test_client):
        res = test_client.get("/photo/NonExistentStudent")
        assert res.status_code == 200
        assert b"<svg" in res.data


class TestFaceRecognitionLogic:
    def test_cosine_similarity_identical_vectors(self):
        v = np.random.randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)
        sim = FaceRecognitionEngine.cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-4

    def test_matching_against_roster(self):
        v1 = np.random.randn(512).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.random.randn(512).astype(np.float32)
        v2 = v2 / np.linalg.norm(v2)

        roster = [
            {"user_id": 101, "reg_number": "21B01A0501", "name": "Alice", "embedding": v1},
            {"user_id": 102, "reg_number": "21B01A0502", "name": "Bob", "embedding": v2}
        ]

        engine = FaceRecognitionEngine(model_name="buffalo_l")
        detected_v = v1 + np.random.randn(512) * 0.01
        detected_v = (detected_v / np.linalg.norm(detected_v)).astype(np.float32)

        res = engine.match_faces_against_roster([detected_v], roster, threshold=0.42)
        assert 101 in res["matched_user_ids"]
        assert 102 not in res["matched_user_ids"]
        assert res["unknown_count"] == 0


class TestAttendanceMarking:
    def test_mark_attendance_endpoint(self, test_client):
        login_res = test_client.post("/login", json={
            "reg_number": "ADMIN01",
            "password": "Admin@123"
        })
        token = login_res.get_json()["token"]

        img_bytes = create_dummy_image_bytes()
        res = test_client.post(
            "/mark-attendance",
            headers={"Authorization": f"Bearer {token}"},
            data={
                "department": "CSE",
                "section": "A",
                "photos": [
                    (BytesIO(img_bytes), "classroom_view1.jpg"),
                    (BytesIO(img_bytes), "classroom_view2.jpg")
                ]
            },
            content_type="multipart/form-data"
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "summary" in data
        assert data["summary"]["photos_processed"] == 2


if __name__ == "__main__":
    pytest.main(["-v", __file__])