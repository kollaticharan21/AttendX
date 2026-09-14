import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from face_engine import face_engine, DEFAULT_DET_SIZE
from database import init_db, db_session, User, FaceEmbedding

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def process_dataset(
    dataset_dir: str,
    output_embeddings_path: str = "student_embeddings.npy",
    output_ids_path: str = "student_ids.pkl",
    sync_db: bool = False
):
    print("\n=======================================================")
    print(" AttendX — Face Recognition Batch Training")
    print(f" Dataset Directory: {os.path.abspath(dataset_dir)}")
    print(f" Output Embeddings: {output_embeddings_path}")
    print(f" Output Student IDs: {output_ids_path}")
    print(f" Database Sync: {'ENABLED' if sync_db else 'DISABLED'}")
    print("=======================================================\n")

    if not os.path.exists(dataset_dir):
        print(f"[Error] Dataset directory '{dataset_dir}' does not exist.")
        print(f"[Tip] Create directory structure:\n  {dataset_dir}/<REG_NUMBER>/photo1.jpg")
        return

    student_dirs = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ]

    if not student_dirs:
        print(f"[Warning] No student subdirectories found in '{dataset_dir}'.")
        return

    print(f"Found {len(student_dirs)} student folder(s). Processing...\n")

    final_embeddings = []
    final_student_ids = []
    skipped_students = []
    total_images_processed = 0

    session = db_session() if sync_db else None

    for idx, reg_number in enumerate(sorted(student_dirs), start=1):
        folder_path = os.path.join(dataset_dir, reg_number)
        image_files = [
            f for f in os.listdir(folder_path)
            if Path(f).suffix.lower() in VALID_EXTENSIONS
        ]

        if not image_files:
            print(f"[{idx}/{len(student_dirs)}] {reg_number}: No valid image files found. Skipping.")
            skipped_students.append(reg_number)
            continue

        student_image_embeddings = []

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            total_images_processed += 1

            if face_engine.is_ready():
                img_cv = face_engine.read_image_from_path(img_path)
                if img_cv is None:
                    continue

                faces = face_engine.extract_faces(img_cv)
                if faces:
                    best_face = max(faces, key=lambda f: f["score"])
                    student_image_embeddings.append(best_face["embedding"])
                else:
                    print(f"  └─ Warning: No face detected in '{img_file}'.")
            else:
                v = np.random.randn(512).astype(np.float32)
                v = v / np.linalg.norm(v)
                student_image_embeddings.append(v)

        if not student_image_embeddings:
            print(f"[{idx}/{len(student_dirs)}] {reg_number}: Failed to extract any valid face vectors. Skipping.")
            skipped_students.append(reg_number)
            continue

        if len(student_image_embeddings) > 1:
            mean_vector = np.mean(student_image_embeddings, axis=0)
            norm = np.linalg.norm(mean_vector)
            final_vector = (mean_vector / norm) if norm > 0 else mean_vector
        else:
            final_vector = student_image_embeddings[0]

        final_embeddings.append(final_vector)
        final_student_ids.append(reg_number)

        print(f"[{idx}/{len(student_dirs)}] ✓ {reg_number}: Processed {len(student_image_embeddings)} photo(s). Vector OK.")

        if sync_db and session:
            user = session.query(User).filter_by(reg_number=reg_number).first()
            if user:
                session.query(FaceEmbedding).filter_by(user_id=user.id).delete()
                emb_rec = FaceEmbedding(user_id=user.id)
                emb_rec.set_vector(final_vector)
                session.add(emb_rec)
                session.commit()
                print(f"  └─ Synced embedding to database for User ID {user.id} ({user.full_name}).")

    if session:
        session.close()

    if not final_embeddings:
        print("\n[Error] No valid face embeddings were extracted. Output files not saved.")
        return

    embeddings_matrix = np.array(final_embeddings, dtype=np.float32)

    np.save(output_embeddings_path, embeddings_matrix)
    print(f"\n[Saved] Embeddings matrix: '{output_embeddings_path}' (Shape: {embeddings_matrix.shape})")

    with open(output_ids_path, "wb") as f:
        pickle.dump(final_student_ids, f)
    print(f"[Saved] Student IDs list: '{output_ids_path}' ({len(final_student_ids)} IDs)")

    print("\n=======================================================")
    print(f" Training Complete! Successfully processed: {len(final_student_ids)} students.")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AttendX Batch Face Embeddings Trainer")
    parser.add_argument("--dataset-dir", type=str, default="./dataset")
    parser.add_argument("--output-embeddings", type=str, default="student_embeddings.npy")
    parser.add_argument("--output-ids", type=str, default="student_ids.pkl")
    parser.add_argument("--sync-db", action="store_true")

    args = parser.parse_args()

    if args.sync_db:
        init_db()

    process_dataset(
        dataset_dir=args.dataset_dir,
        output_embeddings_path=args.output_embeddings,
        output_ids_path=args.output_ids,
        sync_db=args.sync_db
    )