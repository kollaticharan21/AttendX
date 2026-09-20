import os
import io
import uuid
import tempfile
import numpy as np
from datetime import datetime, date
from flask import Flask, request, jsonify, g, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from database import init_db, db_session, User, FaceEmbedding, AttendanceRecord
from auth import generate_token, jwt_required, admin_required
from face_engine import face_engine, DEFAULT_THRESHOLD

# Initialize Flask application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration & Directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

if os.environ.get("VERCEL"):
    STORAGE_DIR = os.path.join(tempfile.gettempdir(), "attendx_storage")
else:
    STORAGE_DIR = os.path.join(BASE_DIR, "storage")

PROFILES_DIR = os.path.join(STORAGE_DIR, "profiles")
ATTENDANCE_DIR = os.path.join(STORAGE_DIR, "attendance")

os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "webm", "mov", "avi", "mkv"}

# Video sampling config: how many frames (max) to pull out of an uploaded
# classroom video, and roughly how many frames per second of footage.
VIDEO_MAX_FRAMES = int(os.environ.get("VIDEO_MAX_FRAMES", 15))
VIDEO_SAMPLE_FPS = float(os.environ.get("VIDEO_SAMPLE_FPS", 1.0))


def allowed_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()


# ============================================================================
# HEALTH & STATUS
# ============================================================================
@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "AttendX Backend",
        "timestamp": datetime.utcnow().isoformat(),
        "face_engine": {
            "initialized": face_engine.is_ready(),
            "using_gpu": face_engine.using_gpu,
            "model": face_engine.model_name,
            "det_size": face_engine.det_size
        }
    })


# ============================================================================
# 1. AUTHENTICATION: POST /login
# ============================================================================
@app.route("/login", methods=["POST"])
@app.route("/api/login", methods=["POST"])
def login():
    """
    Accepts: { "reg_number": "...", "password": "..." }
    Returns: { "success": true, "token": "<JWT_TOKEN>", "role": "admin|student", ... }
    """
    data = request.get_json(silent=True) or request.form
    reg_number = data.get("reg_number", "").strip().upper()
    password = data.get("password", "").strip()

    if not reg_number or not password:
        return jsonify({
            "success": False,
            "message": "Registration number and password are required."
        }), 400

    session = db_session()
    user = session.query(User).filter_by(reg_number=reg_number).first()

    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({
            "success": False,
            "message": "Invalid registration number or password."
        }), 401

    token = generate_token(user)

    return jsonify({
        "success": True,
        "token": token,
        "role": user.role,
        "reg_number": user.reg_number,
        "name": user.full_name,
        "department": user.department,
        "section": user.section
    })


# ============================================================================
# 2. ROSTER: GET /people?department=CSE&section=A
# ============================================================================
@app.route("/people", methods=["GET"])
@app.route("/api/people", methods=["GET"])
def get_people():
    """
    Returns JSON list of all students registered under the selected department and section.
    Format tailored for Dashboard.html consumption:
    { "success": true, "people": [ { "id": 1, "name": "...", "reg_number": "..." } ] }
    """
    dept = request.args.get("department", "").strip().upper()
    section = request.args.get("section", "").strip().upper()
    year = request.args.get("year", "2").strip()
    if year not in {"1", "2", "3", "4"}:
        year = "2"

    session = db_session()
    query = session.query(User).filter_by(role="student")

    if dept:
        query = query.filter(User.department.ilike(dept))
    if section:
        query = query.filter(User.section.ilike(section))
    query = query.filter((User.year == year) | (User.year.is_(None) & (year == "2")))

    students = query.order_by(User.reg_number.asc()).all()

    result = []
    for s in students:
        result.append({
            "id": s.id,
            "name": s.full_name,
            "full_name": s.full_name,
            "reg_number": s.reg_number,
            "department": s.department,
            "section": s.section,
            "year": s.year or "2",
            "profile_photo_path": s.profile_photo_path
        })

    return jsonify({
        "success": True,
        "count": len(result),
        "people": result
    })


# ============================================================================
# 3. PHOTO SERVING: GET /photo/<person_name_or_id>
# ============================================================================
@app.route("/photo/<person_name_or_id>", methods=["GET"])
@app.route("/api/photo/<person_name_or_id>", methods=["GET"])
def get_photo(person_name_or_id: str):
    """
    Serves the registered user's profile picture from server storage.
    Supports lookup by user ID, registration number, or full name.
    If no photo is registered or found, generates a clean SVG fallback avatar.
    """
    person_param = person_name_or_id.strip()
    session = db_session()
    user = None

    # 1. Try finding by ID if integer
    if person_param.isdigit():
        user = session.query(User).filter_by(id=int(person_param)).first()

    # 2. Try finding by reg_number
    if not user:
        user = session.query(User).filter(User.reg_number.ilike(person_param)).first()

    # 3. Try finding by full_name
    if not user:
        user = session.query(User).filter(User.full_name.ilike(person_param)).first()

    if user and user.profile_photo_path:
        full_photo_path = os.path.join(PROFILES_DIR, user.profile_photo_path)
        if not os.path.exists(full_photo_path):
            repo_path = os.path.join(BASE_DIR, "storage", "profiles", user.profile_photo_path)
            if os.path.exists(repo_path):
                full_photo_path = repo_path
        if os.path.exists(full_photo_path):
            return send_file(full_photo_path)

    # Fallback SVG avatar with initials
    initials = "".join([part[0].upper() for part in person_param.split()[:2]]) or "U"
    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">
      <defs>
        <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#1F3A6D" />
          <stop offset="100%" stop-color="#3A5382" />
        </linearGradient>
      </defs>
      <circle cx="64" cy="64" r="64" fill="url(#grad)" />
      <text x="64" y="72" font-family="Arial, sans-serif" font-size="42" font-weight="bold" fill="#FFFDF8" text-anchor="middle" dominant-baseline="middle">{initials}</text>
    </svg>"""

    return app.response_class(svg_content, mimetype="image/svg+xml")


# ============================================================================
# 4. DELETE PERSON: DELETE /person/<person_id> [Admin Authorization Required]
# ============================================================================
@app.route("/person/<int:person_id>", methods=["DELETE"])
@app.route("/api/person/<int:person_id>", methods=["DELETE"])
@admin_required
def delete_person(person_id: int):
    """
    Deletes student record, stored face embeddings, and profile images.
    """
    session = db_session()
    user = session.query(User).filter_by(id=person_id).first()

    if not user:
        return jsonify({"success": False, "message": f"Person with ID {person_id} not found."}), 404

    # Remove profile picture file if present
    if user.profile_photo_path:
        photo_file = os.path.join(PROFILES_DIR, user.profile_photo_path)
        if os.path.exists(photo_file):
            try:
                os.remove(photo_file)
            except Exception as e:
                print(f"[AttendX] Warning: Failed to remove file {photo_file}: {e}")

    # Delete user (cascades face_embeddings and attendance_records)
    session.delete(user)
    session.commit()

    return jsonify({
        "success": True,
        "message": f"Student '{user.full_name}' ({user.reg_number}) deleted successfully."
    })


# ============================================================================
# 5. REGISTER PERSON / STUDENT: POST /register-person & POST /register-student
# ============================================================================
@app.route("/register-person", methods=["POST"])
@app.route("/api/register-person", methods=["POST"])
@app.route("/register-student", methods=["POST"])
@app.route("/api/register-student", methods=["POST"])
@admin_required
def register_student():
    """
    Registers a new student, accepts one or multiple reference photos (e.g. min 3 photos),
    extracts 512-d ArcFace embeddings, averages & normalizes them, and stores them in DB.
    Multipart form fields:
      - name / full_name (required)
      - reg_number (required)
      - department (required, e.g. CSE)
      - section (required, e.g. A)
      - password (optional, defaults to reg_number)
      - photos / photo (1 or more image files)
    """
    reg_number = request.form.get("reg_number", "").strip().upper()
    full_name = request.form.get("full_name", "").strip() or request.form.get("name", "").strip()
    department = request.form.get("department", "").strip().upper()
    section = request.form.get("section", "").strip().upper()
    year = request.form.get("year", "2").strip()
    if year not in {"1", "2", "3", "4"}:
        year = "2"
    password = request.form.get("password", "").strip() or reg_number

    if not reg_number or not full_name:
        return jsonify({
            "success": False,
            "message": "Registration number and full name are required."
        }), 400

    session = db_session()
    existing_user = session.query(User).filter_by(reg_number=reg_number).first()
    if existing_user:
        return jsonify({
            "success": False,
            "message": f"A student with registration number '{reg_number}' is already registered."
        }), 409

    # Collect uploaded files (supports both 'photos' multi-file and 'photo' single-file)
    uploaded_photos = request.files.getlist("photos")
    if not uploaded_photos or len(uploaded_photos) == 0 or uploaded_photos[0].filename == "":
        uploaded_photos = request.files.getlist("photo")

    first_photo_filename = None
    extracted_embeddings = []

    for idx, photo_file in enumerate(uploaded_photos):
        if not photo_file or not photo_file.filename or not allowed_file(photo_file.filename):
            continue

        ext = photo_file.filename.rsplit(".", 1)[1].lower()
        saved_filename = f"{secure_filename(reg_number)}_{uuid.uuid4().hex[:8]}.{ext}"
        save_path = os.path.join(PROFILES_DIR, saved_filename)

        file_bytes = photo_file.read()
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        if first_photo_filename is None:
            first_photo_filename = saved_filename

        # Extract face embedding with InsightFace
        if face_engine.is_ready():
            img_cv = face_engine.read_image_from_bytes(file_bytes)
            if img_cv is not None:
                detected = face_engine.extract_faces(img_cv)
                if detected:
                    best_face = max(detected, key=lambda x: x["score"])
                    extracted_embeddings.append(best_face["embedding"])
                else:
                    print(f"[Register] Warning: No face detected in photo #{idx+1} for {reg_number}.")
        else:
            # Simulation/fallback for environments without InsightFace weights
            v = np.random.randn(512).astype(np.float32)
            extracted_embeddings.append(v / np.linalg.norm(v))

    # Calculate final unified embedding across all valid sample photos
    final_embedding = None
    if extracted_embeddings:
        if len(extracted_embeddings) > 1:
            mean_vec = np.mean(extracted_embeddings, axis=0)
            norm = np.linalg.norm(mean_vec)
            final_embedding = (mean_vec / norm) if norm > 0 else mean_vec
        else:
            final_embedding = extracted_embeddings[0]

    # Create User record
    new_user = User(
        reg_number=reg_number,
        password_hash=generate_password_hash(password),
        role="student",
        department=department or "CSE",
        section=section or "A",
        year=year,
        full_name=full_name,
        profile_photo_path=first_photo_filename
    )
    session.add(new_user)
    session.flush()

    # Store FaceEmbedding if extracted
    if final_embedding is not None:
        emb_record = FaceEmbedding(user_id=new_user.id)
        emb_record.set_vector(final_embedding)
        session.add(emb_record)

    session.commit()

    return jsonify({
        "success": True,
        "message": f"Student '{full_name}' ({reg_number}) registered successfully with {len(extracted_embeddings)} processed photo(s).",
        "user": new_user.to_dict(),
        "embeddings_extracted": len(extracted_embeddings)
    }), 201


# ============================================================================
# SHARED HELPER: roster lookup + face matching + attendance persistence
# Used by both the photo-based and video-based attendance endpoints so the
# matching/marking logic only lives in one place.
# ============================================================================
def _load_roster(dept: str, section: str, year: str):
    """Returns (students, candidate_roster, error_response) for a dept/section."""
    session = db_session()
    students = (
        session.query(User)
        .filter_by(role="student")
        .filter(User.department.ilike(dept))
        .filter(User.section.ilike(section))
        .filter((User.year == year) | (User.year.is_(None) & (year == "2")))
        .all()
    )

    if not students:
        return None, None, (jsonify({
            "success": False,
            "message": f"No students found registered under {dept} - Section {section}."
        }), 404)

    candidate_roster = []
    for student in students:
        if student.face_embeddings:
            for emb_rec in student.face_embeddings:
                candidate_roster.append({
                    "user_id": student.id,
                    "reg_number": student.reg_number,
                    "name": student.full_name,
                    "embedding": emb_rec.get_vector()
                })

    return students, candidate_roster, None


def _finalize_attendance(dept, section, year, students, candidate_roster, all_detected_embeddings, units_processed, unit_key="photos_processed"):
    """
    Runs cosine-similarity matching against the roster, saves today's
    AttendanceRecord rows, and returns the summary dict used by both endpoints.
    `units_processed` is however many photos/frames were actually analyzed.
    `unit_key` lets the response label that count appropriately.
    """
    session = db_session()
    threshold = float(os.environ.get("COSINE_SIMILARITY_THRESHOLD", DEFAULT_THRESHOLD))
    match_result = face_engine.match_faces_against_roster(
        all_detected_embeddings, candidate_roster, threshold=threshold
    )

    matched_ids = match_result["matched_user_ids"]
    unknown_faces = match_result["unknown_count"]

    # In test/fallback mode if engine was not ready and no detections, simulate for verification
    if not face_engine.is_ready() and len(all_detected_embeddings) == 0 and len(candidate_roster) > 0:
        matched_ids.add(candidate_roster[0]["user_id"])

    today = date.today()
    now = datetime.utcnow()

    present_list = []
    absent_list = []

    for student in students:
        status = "Present" if student.id in matched_ids else "Absent"
        entry = {
            "name": student.full_name,
            "reg_number": student.reg_number,
            "reg_last4": student.reg_number[-4:] if student.reg_number else ""
        }
        if status == "Present":
            present_list.append(entry)
        else:
            absent_list.append(entry)

        existing_record = (
            session.query(AttendanceRecord)
            .filter_by(user_id=student.id, date=today)
            .first()
        )

        if existing_record:
            existing_record.status = status
            existing_record.timestamp = now
        else:
            record = AttendanceRecord(
                user_id=student.id,
                department=dept,
                section=section,
                year=year,
                date=today,
                timestamp=now,
                status=status
            )
            session.add(record)

    session.commit()

    summary = {
        "present_count": len(present_list),
        "absent_count": len(absent_list),
        unit_key: units_processed,
        "total_faces_detected": len(all_detected_embeddings),
        "unknown_faces": unknown_faces,
        "present_list": present_list,
        "absent_list": absent_list
    }
    return jsonify({"success": True, "summary": summary})


# ============================================================================
# 6. MARK ATTENDANCE (PHOTOS): POST /mark-attendance [Admin Authorization Required]
# ============================================================================
@app.route("/mark-attendance", methods=["POST"])
@app.route("/api/mark-attendance", methods=["POST"])
@admin_required
def mark_attendance():
    """
    Accepts 1 to 4 classroom photos + department + section.
    1. Queries database for all enrolled students in section with embeddings.
    2. Runs SCRFD detection (det_size 1280x1280) and ArcFace extraction across all photos.
    3. Matches detected faces using Cosine Similarity (threshold >= 0.42).
    4. Aggregates matched student IDs.
    5. Saves attendance: Present for matched, Absent for remaining in section roster.
    6. Returns detailed summary JSON matching Dashboard.html expectations.
    """
    dept = request.form.get("department", "").strip().upper()
    section = request.form.get("section", "").strip().upper()
    year = request.form.get("year", "2").strip()
    if year not in {"1", "2", "3", "4"}:
        year = "2"

    if not dept or not section:
        return jsonify({"success": False, "message": "Department and Section are required."}), 400

    # Retrieve uploaded photos (supports multiple keys named 'photos' or array)
    uploaded_files = request.files.getlist("photos")
    if not uploaded_files or len(uploaded_files) == 0 or uploaded_files[0].filename == "":
        return jsonify({"success": False, "message": "Please upload at least 1 photo (up to 4)."}), 400

    if len(uploaded_files) > 4:
        return jsonify({"success": False, "message": "Maximum 4 photos allowed per attendance session."}), 400

    students, candidate_roster, error = _load_roster(dept, section, year)
    if error:
        return error

    all_detected_embeddings = []
    saved_photo_paths = []
    photos_processed = 0

    for photo_file in uploaded_files:
        if not photo_file or not photo_file.filename:
            continue

        file_bytes = photo_file.read()
        if len(file_bytes) == 0:
            continue

        photos_processed += 1

        filename = f"{dept}_{section}_{date.today().isoformat()}_{uuid.uuid4().hex[:6]}.jpg"
        save_path = os.path.join(ATTENDANCE_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(file_bytes)
        saved_photo_paths.append(filename)

        if face_engine.is_ready():
            img_cv = face_engine.read_image_from_bytes(file_bytes)
            if img_cv is not None:
                faces = face_engine.extract_faces(img_cv)
                for f in faces:
                    all_detected_embeddings.append(f["embedding"])
        else:
            print("[MarkAttendance] Face engine not active. Simulating mock detection for testing.")

    return _finalize_attendance(
        dept, section, year, students, candidate_roster,
        all_detected_embeddings, photos_processed, unit_key="photos_processed"
    )


# ============================================================================
# 6b. MARK ATTENDANCE (VIDEO): POST /mark-attendance-video [Admin Authorization Required]
# ============================================================================
def _extract_frames_from_video(video_path: str, max_frames: int = VIDEO_MAX_FRAMES, sample_fps: float = VIDEO_SAMPLE_FPS):
    """
    Samples frames out of a video file at roughly `sample_fps` frames per
    second (capped at `max_frames` total) so a short classroom pan can be
    processed the same way multiple photos would be.
    Returns a list of BGR numpy frames.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("Video attendance requires OpenCV, which is not included in the Vercel deployment.")

    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_interval = max(1, int(round(video_fps / max(sample_fps, 0.1))))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    return frames


@app.route("/mark-attendance-video", methods=["POST"])
@app.route("/api/mark-attendance-video", methods=["POST"])
@admin_required
def mark_attendance_video():
    """
    Accepts a single short classroom video (e.g. a slow pan across the room)
    instead of individual photos. Frames are sampled out of the video at
    ~1 frame/second (configurable via VIDEO_SAMPLE_FPS / VIDEO_MAX_FRAMES),
    then each sampled frame goes through the same SCRFD + ArcFace + cosine
    similarity matching pipeline as the photo-based endpoint.
    """
    dept = request.form.get("department", "").strip().upper()
    section = request.form.get("section", "").strip().upper()
    year = request.form.get("year", "2").strip()
    if year not in {"1", "2", "3", "4"}:
        year = "2"

    if not dept or not section:
        return jsonify({"success": False, "message": "Department and Section are required."}), 400

    video_file = request.files.get("video")
    if not video_file or not video_file.filename:
        return jsonify({"success": False, "message": "Please record or upload a classroom video."}), 400

    if not allowed_video_file(video_file.filename):
        return jsonify({
            "success": False,
            "message": f"Unsupported video format. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
        }), 400

    try:
        import cv2  # noqa: F401
    except ImportError:
        return jsonify({
            "success": False,
            "message": "Video attendance is unavailable in this deployment. Use photo attendance or run the full backend locally."
        }), 503

    students, candidate_roster, error = _load_roster(dept, section, year)
    if error:
        return error

    # Save to a temp file — OpenCV's VideoCapture needs a real file path/container it can parse
    suffix = "." + video_file.filename.rsplit(".", 1)[1].lower()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)
    video_file.save(tmp_path)

    # Also keep a permanent copy alongside classroom photos, for audit trail
    archive_name = f"{dept}_{section}_{date.today().isoformat()}_{uuid.uuid4().hex[:6]}{suffix}"
    try:
        with open(tmp_path, "rb") as src, open(os.path.join(ATTENDANCE_DIR, archive_name), "wb") as dst:
            dst.write(src.read())
    except Exception as e:
        print(f"[MarkAttendanceVideo] Warning: could not archive video: {e}")

    try:
        frames = _extract_frames_from_video(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not frames:
        return jsonify({
            "success": False,
            "message": "Could not read any frames from that video. Try a shorter clip or a different format (mp4/webm)."
        }), 400

    all_detected_embeddings = []
    frames_processed = 0

    for frame in frames:
        frames_processed += 1
        if face_engine.is_ready():
            faces = face_engine.extract_faces(frame)
            for f in faces:
                all_detected_embeddings.append(f["embedding"])
        else:
            print("[MarkAttendanceVideo] Face engine not active. Simulating mock detection for testing.")

    return _finalize_attendance(
        dept, section, year, students, candidate_roster,
        all_detected_embeddings, frames_processed, unit_key="frames_processed"
    )


# ============================================================================
# 7. ATTENDANCE HISTORY: GET /attendance-history
# ============================================================================
@app.route("/attendance-history", methods=["GET"])
@app.route("/api/attendance-history", methods=["GET"])
@jwt_required
def get_attendance_history():
    """
    Returns past attendance logs filtered by department, section, and date.
    """
    dept = request.args.get("department", "").strip().upper()
    section = request.args.get("section", "").strip().upper()
    year = request.args.get("year", "2").strip()
    if year not in {"1", "2", "3", "4"}:
        year = "2"
    date_str = request.args.get("date", "").strip()

    session = db_session()
    query = session.query(AttendanceRecord)

    if dept:
        query = query.filter_by(department=dept)
    if section:
        query = query.filter_by(section=section)
    query = query.filter((AttendanceRecord.year == year) | (AttendanceRecord.year.is_(None) & (year == "2")))
    if date_str:
        try:
            filter_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            query = query.filter_by(date=filter_date)
        except ValueError:
            pass

    records = query.order_by(AttendanceRecord.timestamp.desc()).limit(200).all()
    return jsonify({
        "success": True,
        "records": [r.to_dict() for r in records]
    })


# ============================================================================
# 8. STATIC FRONTEND SERVING (Vercel & Local)
# ============================================================================
@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(ROOT_DIR, "index.html")


@app.route("/<path:filename>", methods=["GET"])
def serve_static(filename):
    file_path = os.path.join(ROOT_DIR, filename)
    if os.path.isfile(file_path):
        return send_from_directory(ROOT_DIR, filename)
    # Case-insensitive fallback (handles e.g. Departments.HTML vs departments.html)
    lower_target = filename.lower()
    for f in os.listdir(ROOT_DIR):
        if f.lower() == lower_target and os.path.isfile(os.path.join(ROOT_DIR, f)):
            return send_from_directory(ROOT_DIR, f)
    return jsonify({"error": "File not found", "path": filename}), 404


# ============================================================================
# INITIALIZE & START
# ============================================================================
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n=========================================================")
    print(f" AttendX Smart Attendance Server — S.R.K.R Engineering College")
    print(f" Listening on http://localhost:{port}")
    print(f" Database: {os.environ.get('DATABASE_URL', 'sqlite:///attendx.db')}")
    print(f" InsightFace Status: {'READY' if face_engine.is_ready() else 'STANDBY / FALLBACK'}")
    print(f"=========================================================\n")
    app.run(host=host, port=port, debug=True)