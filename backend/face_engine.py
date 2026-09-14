import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

DEFAULT_DET_SIZE = (1280, 1280)
DEFAULT_THRESHOLD = float(os.environ.get("COSINE_SIMILARITY_THRESHOLD", 0.42))
MODEL_NAME = os.environ.get("INSIGHTFACE_MODEL", "buffalo_l")


class FaceRecognitionEngine:
    """
    High-accuracy face recognition engine utilizing InsightFace buffalo_l:
    - SCRFD multi-face detector configured with det_size=(1280, 1280) for rear classroom rows
    - ArcFace 512-dimensional unit-normalized embeddings
    - Cosine similarity matching with configurable threshold (default: 0.42)
    """

    def __init__(self, model_name: str = MODEL_NAME, det_size: Tuple[int, int] = DEFAULT_DET_SIZE):
        self.model_name = model_name
        self.det_size = det_size
        self.app = None
        self.is_initialized = False
        self.using_gpu = False
        self._init_model()

    def _init_model(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print(f"[FaceEngine] Initializing InsightFace '{self.model_name}' with det_size={self.det_size}...")

            try:
                # Try GPU (ctx_id=0)
                app = FaceAnalysis(name=self.model_name, providers=providers)
                app.prepare(ctx_id=0, det_size=self.det_size)
                self.app = app
                self.using_gpu = True
                self.is_initialized = True
                print("[FaceEngine] InsightFace initialized successfully with GPU acceleration.")
            except Exception as gpu_err:
                print(f"[FaceEngine] GPU initialization skipped ({gpu_err}). Falling back to CPU...")
                # Fallback to CPU (ctx_id=-1)
                app = FaceAnalysis(name=self.model_name, providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=self.det_size)
                self.app = app
                self.using_gpu = False
                self.is_initialized = True
                print("[FaceEngine] InsightFace initialized successfully on CPU.")

        except ImportError:
            print("[FaceEngine] WARNING: 'insightface' or 'onnxruntime' is not installed.")
            print("[FaceEngine] Engine will run in simulation/fallback mode until dependencies are installed.")
            self.app = None
            self.is_initialized = False
        except Exception as e:
            print(f"[FaceEngine] Error loading InsightFace model: {e}")
            self.app = None
            self.is_initialized = False

    def is_ready(self) -> bool:
        return self.is_initialized and self.app is not None

    def read_image_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        try:
            import cv2
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[FaceEngine] Error decoding image bytes: {e}")
            return None

    def read_image_from_path(self, path: str) -> Optional[np.ndarray]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return self.read_image_from_bytes(f.read())
        except Exception as e:
            print(f"[FaceEngine] Error reading image from {path}: {e}")
            return None

    def extract_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        if not self.is_ready():
            raise RuntimeError("InsightFace engine is not initialized.")

        if img is None or img.size == 0:
            return []

        raw_faces = self.app.get(img)
        results = []

        for face in raw_faces:
            emb = face.embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            results.append({
                "bbox": face.bbox.tolist() if hasattr(face, "bbox") else [],
                "score": float(face.det_score) if hasattr(face, "det_score") else 1.0,
                "embedding": emb,
            })

        return results

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def match_faces_against_roster(
        self,
        detected_embeddings: List[np.ndarray],
        roster: List[Dict[str, Any]],
        threshold: float = DEFAULT_THRESHOLD
    ) -> Dict[str, Any]:
        matched_user_ids = set()
        matches = []
        unknown_count = 0

        if not roster or not detected_embeddings:
            return {
                "matched_user_ids": matched_user_ids,
                "matches": matches,
                "unknown_count": len(detected_embeddings)
            }

        roster_embs = np.stack([r["embedding"] for r in roster])  # (M, 512)

        for detected_emb in detected_embeddings:
            similarities = np.dot(roster_embs, detected_emb)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            if best_score >= threshold:
                matched_student = roster[best_idx]
                matched_user_ids.add(matched_student["user_id"])
                matches.append({
                    "user_id": matched_student["user_id"],
                    "reg_number": matched_student["reg_number"],
                    "name": matched_student["name"],
                    "similarity": round(best_score, 4)
                })
            else:
                unknown_count += 1

        return {
            "matched_user_ids": matched_user_ids,
            "matches": matches,
            "unknown_count": unknown_count
        }


face_engine = FaceRecognitionEngine()