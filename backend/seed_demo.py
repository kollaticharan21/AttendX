import os
import sys
import numpy as np
from werkzeug.security import generate_password_hash

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from database import init_db, db_session, User, FaceEmbedding, AttendanceRecord

SAMPLE_STUDENTS = [
    {"reg_number": "21B01A0501", "full_name": "K. Rajesh Kumar", "department": "CSE", "section": "A"},
    {"reg_number": "21B01A0502", "full_name": "P. Sneha Latha", "department": "CSE", "section": "A"},
    {"reg_number": "21B01A0503", "full_name": "M. Sai Tarun", "department": "CSE", "section": "A"},
    {"reg_number": "21B01A0504", "full_name": "V. Ananya Rao", "department": "CSE", "section": "A"},
    {"reg_number": "21B01A0505", "full_name": "G. Varun Teja", "department": "CSE", "section": "A"},
    {"reg_number": "21B01A1201", "full_name": "B. Akhil Varma", "department": "IT", "section": "A"},
    {"reg_number": "21B01A1202", "full_name": "T. Divya Sri", "department": "IT", "section": "A"}
]


def seed_demo_data():
    init_db()
    session = db_session()

    print("\n--- Seeding AttendX Demo Data ---")
    seeded_count = 0

    for s_info in SAMPLE_STUDENTS:
        existing = session.query(User).filter_by(reg_number=s_info["reg_number"]).first()
        if not existing:
            student = User(
                reg_number=s_info["reg_number"],
                password_hash=generate_password_hash(s_info["reg_number"]),
                role="student",
                department=s_info["department"],
                section=s_info["section"],
                full_name=s_info["full_name"],
                profile_photo_path=None
            )
            session.add(student)
            session.flush()

            np.random.seed(int(s_info["reg_number"][-4:]))
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)

            face_emb = FaceEmbedding(user_id=student.id)
            face_emb.set_vector(emb)
            session.add(face_emb)

            seeded_count += 1
            print(f"  ✓ Seeded student: {student.full_name} ({student.reg_number}) in {student.department}-{student.section}")

    session.commit()
    session.close()

    print(f"\n[Done] Successfully seeded {seeded_count} student(s).")
    print(f"Admin Credentials:\n  Reg No:   ADMIN01\n  Password: Admin@123\n")


if __name__ == "__main__":
    seed_demo_data()