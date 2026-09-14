# AttendX — Smart Attendance Management System
**S.R.K.R Engineering College**

AttendX is a high-accuracy, production-ready facial recognition attendance management system built for lecture halls and large classrooms. It utilizes **InsightFace (buffalo_l)** with **SCRFD** (multi-face detector scaled to `det_size=(1280, 1280)`) and **ArcFace** (512-dimensional normalized embeddings) paired with Cosine Similarity matching (default threshold: `0.42`).

---

## 📁 Project Architecture

```
attendx_backend/
├── app.py                     # Flask REST API with face recognition & JWT authentication
├── database.py                # SQLAlchemy ORM models (User, FaceEmbedding, AttendanceRecord)
├── auth.py                    # JWT token generation, password hashing & access decorators
├── face_engine.py             # InsightFace buffalo_l wrapper (SCRFD 1280x1280 + ArcFace)
├── train_faces.py             # Batch training utility (extracts to .npy & .pkl, syncs to DB)
├── seed_demo.py               # Pre-populates admin & sample student rosters with embeddings
├── test_backend.py            # Comprehensive unit and integration test suite
├── requirements.txt           # Production Python dependencies
├── .env.example               # Environment variables configuration template
├── storage/                   # File storage for student profile photos and classroom snapshots
│   ├── profiles/
│   └── attendance/
├── dataset/                   # Input directory for batch face training
│   └── <student_reg_number>/  # e.g., 21B01A0501/photo1.jpg
└── frontend/                  # Complete client UI
    ├── Login.html             # Sign-in page (JWT authentication)
    ├── Departments.html       # Academic department selector
    ├── Section.html           # Class section selector
    └── Dashboard.html         # Classroom roster, camera attendance & multi-photo upload
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites
- **Python 3.10 or 3.11** is recommended for compatibility with InsightFace and ONNX Runtime.
- **C++ Build Tools** (Visual Studio C++ Build Tools on Windows) if compiling packages from source.

### 2. Environment Setup

Open PowerShell or Terminal in this directory:
```powershell
# Create a virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\Activate.ps1

# (Or on Linux/macOS)
# source venv/bin/activate
```

### 3. Install Dependencies

#### Option A: CPU Runtime (Standard)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: NVIDIA GPU Acceleration (CUDA / cuDNN)
To run SCRFD and ArcFace on an NVIDIA GPU for ultra-fast processing of multiple 1280x1280 images:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```
> **GPU Requirements**: NVIDIA GPU with CUDA 11.8 or 12.x and matching cuDNN installed. `face_engine.py` automatically detects CUDA; if GPU initialization fails, it seamlessly falls back to CPU without crashing.

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default parameters in `.env`:
- `PORT=5000`
- `DATABASE_URL=sqlite:///attendx.db` (or PostgreSQL `postgresql://user:pass@localhost:5432/attendx`)
- `COSINE_SIMILARITY_THRESHOLD=0.42`
- `DEFAULT_ADMIN_REG=ADMIN01`
- `DEFAULT_ADMIN_PASS=Admin@123`

### 5. Seed Initial Data
Initialize tables and populate default admin (`ADMIN01` / `Admin@123`) and sample students:
```bash
python seed_demo.py
```

### 6. Start the Backend Server
```bash
python app.py
```
The server will boot on `http://localhost:5000`.

---

## 🧑‍🎓 Batch Model Training (`train_faces.py`)

To train/extract embeddings for your student photo dataset:

1. Place student reference photos in the `dataset/` directory organized by registration number:
   ```
   dataset/
   ├── 21B01A0501/
   │   ├── photo1.jpg
   │   └── photo2.jpg
   ├── 21B01A0502/
   │   └── id_photo.jpg
   └── 21B01A0503/
       └── snapshot.jpg
   ```
2. Run the batch trainer:
   ```bash
   # Extract embeddings to .npy and .pkl
   python train_faces.py --dataset-dir ./dataset

   # Extract embeddings AND sync directly into the AttendX database
   python train_faces.py --dataset-dir ./dataset --sync-db
   ```
Outputs produced:
- `student_embeddings.npy`: `(N, 512)` float32 matrix of L2-normalized vectors.
- `student_ids.pkl`: Pickled Python list of `N` registration numbers corresponding to rows.

---

## 🌐 Connecting the Frontend

Open `frontend/Login.html` in your browser:
1. Log in with the default administrator credentials:
   - **User ID / Reg No**: `ADMIN01`
   - **Password**: `Admin@123`
2. Select an academic department (e.g., **CSE**).
3. Select a section (e.g., **Section A**).
4. View enrolled students on `Dashboard.html`.
5. Click the orange **Take Attendance** button to either capture via live webcam or upload 1 to 4 classroom group photos.
6. The system runs SCRFD detection on all faces in the classroom photos, computes cosine similarity against enrolled section members, and marks students as **Present** or **Absent**.

---

## 📡 REST API Documentation

### 1. `POST /login`
Authenticates a student or administrator.
- **Request Body**:
  ```json
  {
    "reg_number": "ADMIN01",
    "password": "Admin@123"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "role": "admin",
    "reg_number": "ADMIN01",
    "name": "System Administrator"
  }
  ```

### 2. `GET /people?department=CSE&section=A`
Fetches all students in a specified section.
- **Response**:
  ```json
  {
    "success": true,
    "count": 5,
    "people": [
      {
        "id": 1,
        "name": "K. Rajesh Kumar",
        "reg_number": "21B01A0501",
        "department": "CSE",
        "section": "A"
      }
    ]
  }
  ```

### 3. `GET /photo/<person_name_or_id>`
Serves student profile picture (lookup by ID, registration number, or name). If no photo exists, dynamically serves an SVG avatar with student initials.

### 4. `DELETE /person/<person_id>` `[Admin Only]`
Deletes student record, stored face embeddings, attendance records, and file from disk.
- **Headers**: `Authorization: Bearer <JWT_TOKEN>`

### 5. `POST /mark-attendance` `[Admin Only]`
Marks attendance from 1 to 4 classroom group photos.
- **Headers**: `Authorization: Bearer <JWT_TOKEN>`
- **Content-Type**: `multipart/form-data`
- **Fields**:
  - `department`: e.g. `CSE`
  - `section`: e.g. `A`
  - `photos`: 1 to 4 image files
- **Response**:
  ```json
  {
    "success": true,
    "summary": {
      "present_count": 4,
      "absent_count": 1,
      "photos_processed": 2,
      "total_faces_detected": 5,
      "unknown_faces": 1,
      "present_list": ["K. Rajesh Kumar", "P. Sneha Latha", "M. Sai Tarun", "V. Ananya Rao"],
      "absent_list": ["G. Varun Teja"]
    }
  }
  ```

### 6. `POST /register-student` `[Admin Only]`
Registers a student with photo and auto-generates 512-d face embedding.
- **Headers**: `Authorization: Bearer <JWT_TOKEN>`
- **Content-Type**: `multipart/form-data`
- **Fields**: `reg_number`, `full_name`, `department`, `section`, `password` (optional), `photo` (file).

### 7. `GET /attendance-history` `[Authenticated]`
Retrieves attendance records filtered by `department`, `section`, and `date` (YYYY-MM-DD).

---

## 🧪 Running Automated Tests

```bash
pytest test_backend.py -v
```
All tests validate authentication, database cascading, L2 normalization, vector cosine similarity math, and multipart attendance submission.
