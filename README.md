# AttendX

## Live deployment

The website is hosted on Vercel, and its API is hosted on Render. Face recognition
needs InsightFace and ONNX Runtime, which are not included in the Vercel function.
The Render Blueprint in `render.yaml` creates the full Python backend and a
persistent disk for the database, uploaded photos, and face-recognition model.

### First-time setup

1. In Render, choose **New → Blueprint**, connect this GitHub repository, and
   select the `main` branch. Render reads `render.yaml` and builds the backend
   from `backend/Dockerfile`.
2. Set the prompted `DEFAULT_ADMIN_PASS` to a strong password you can keep.
   Render generates `JWT_SECRET_KEY` for you. The service uses registration
   number `ADMIN01` for its initial administrator.
3. Wait for the service deployment to finish. Open
   `https://attendx-backend.onrender.com/api/health` and verify that the
   response reports `"status":"healthy"` and
   `"face_engine":{"initialized":true,...}`.
4. Only after Render is healthy, update the Vercel rewrite in `vercel.json` to
   send `/api/*` requests to `https://attendx-backend.onrender.com/api/:path*`
   and deploy that Vercel change. This keeps the website from routing requests
   to a backend before it is ready.
5. Sign in to the live site as `ADMIN01` with the password chosen in step 2.
   Register students with their photos on the live site so the new backend can
   create their face embeddings before taking attendance.

The new backend starts with its own database; local databases, attendance
records, and profile photos are deliberately not copied or uploaded. Keep
student photos and face data private. Render's `standard` service and persistent
disk are paid resources. Do not downgrade to an ephemeral/free instance for
attendance data.

### Local development

The frontend continues to use `http://127.0.0.1:5000` when opened from
`localhost`; run the backend locally with the dependencies in
`backend/requirements.txt`.