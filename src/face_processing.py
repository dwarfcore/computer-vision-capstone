"""
face_processing.py - Face detection, encoding, and identity matching.

Uses DeepFace (pure Python / ONNX — zero C++ compilation required) for:
  - Face detection via OpenCV's built-in DNN detector (no extra download)
  - 128-d embedding extraction via DeepFace's Facenet model (ONNX)
  - Identity matching via vector DB cosine similarity

Models download automatically on first use into ~/.deepface/weights/
Thread-safe: model is loaded once at startup and reused.
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
import threading
import os

# Cosine similarity threshold: embeddings with similarity >= this are a match.
# Range 0–1. Raise to reduce false positives, lower to catch more faces.
DEFAULT_MATCH_THRESHOLD = 0.65

_model = None
_model_lock = threading.Lock()

# OpenCV face detector (ships with opencv-python, no extra install)
_face_cascade = None


def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _get_model():
    """Lazy-load DeepFace Facenet model (downloads ~90 MB on first call)."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from deepface import DeepFace
            from deepface.models.FacialRecognition import FacialRecognition
        except ImportError:
            raise ImportError(
                "deepface is not installed.\n"
                "Run: pip install deepface\n"
                "(Pure Python — no C++ compiler required)"
            )
        # Pre-warm the model so first detection isn't slow
        DeepFace.build_model("Facenet")
        _model = DeepFace
        return _model


@dataclass
class DetectedFace:
    """One face found in a frame."""
    bbox: Tuple[int, int, int, int]  # (top, right, bottom, left)
    embedding: np.ndarray            # 128-d float32 Facenet vector
    person_id: Optional[int] = None
    name: str = "Unknown"
    confidence: float = 0.0          # cosine similarity (0–1)
    is_monitored: bool = False


def _detect_faces_opencv(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect face bounding boxes using OpenCV Haar cascade (ships with opencv-python).
    Returns list of (top, right, bottom, left).
    """
    cascade = _get_cascade()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detections = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    locations = []
    if len(detections) > 0:
        for (x, y, w, h) in detections:
            # Convert (x, y, w, h) → (top, right, bottom, left)
            locations.append((y, x + w, y + h, x))
    return locations


def _embed_face(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Crop the face region and produce a 128-d Facenet embedding via DeepFace.
    Returns None if embedding fails (too small, bad crop, etc.).
    """
    DeepFace = _get_model()
    top, right, bottom, left = bbox
    h, w = frame_bgr.shape[:2]

    # Add padding and clamp to frame
    pad = 20
    y1 = max(0, top - pad)
    y2 = min(h, bottom + pad)
    x1 = max(0, left - pad)
    x2 = min(w, right + pad)

    face_crop = frame_bgr[y1:y2, x1:x2]
    if face_crop.size == 0 or face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
        return None

    try:
        result = DeepFace.represent(
            img_path=face_crop,
            model_name="Facenet",
            enforce_detection=False,  # don't crash if detector struggles with crop
            detector_backend="skip",  # we already cropped the face
        )
        if result:
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            # L2-normalise for cosine similarity via dot product
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
    except Exception:
        pass
    return None


def detect_and_encode(
    frame_rgb: np.ndarray,
    model: str = "Facenet",  # kept for API compatibility
    upsample: int = 1        # kept for API compatibility
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
    """
    Detect all faces in an RGB frame and return bounding boxes + embeddings.

    Returns:
        locations  - list of (top, right, bottom, left)
        encodings  - list of 128-d numpy arrays
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    locations = _detect_faces_opencv(frame_bgr)

    encodings = []
    valid_locations = []
    for loc in locations:
        emb = _embed_face(frame_bgr, loc)
        if emb is not None:
            encodings.append(emb)
            valid_locations.append(loc)

    return valid_locations, encodings


def match_face(
    embedding: np.ndarray,
    vector_db,
    sql_conn,
    threshold: float = DEFAULT_MATCH_THRESHOLD
) -> Tuple[Optional[int], float]:
    """
    Search vector DB for closest match to embedding.

    Embeddings are L2-normalised Facenet 128-d vectors.
    ChromaDB with cosine metric returns distance in [0, 2]:
        0 = identical, 1 = orthogonal, 2 = opposite.

    cosine_similarity = 1 - (distance / 2), range [0, 1].
    We match if similarity >= threshold (default 0.65).

    Returns:
        (person_id, confidence) — person_id is None if no match.
        confidence is the cosine similarity (0–1).
    """
    if vector_db.count() == 0:
        return None, 0.0

    hits = vector_db.search(embedding, top_k=1)
    if not hits:
        return None, 0.0

    person_id, chroma_distance = hits[0]
    similarity = 1.0 - (chroma_distance / 2.0)

    if similarity < threshold:
        return None, 0.0

    return person_id, float(similarity)


def process_frame(
    frame_bgr: np.ndarray,
    vector_db,
    sql_conn,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    model: str = "hog",
    scale: float = 0.5            # downscale for detection speed
) -> List[DetectedFace]:
    """
    Full pipeline: detect → encode → match.

    Returns list of DetectedFace (one per face in frame).
    """
    from db_sqlite import get_person

    # Downscale for faster detection
    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations, encodings = detect_and_encode(rgb, model=model)

    # Scale locations back up
    inv = 1.0 / scale
    locations = [
        (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
        for t, r, b, l in locations
    ]

    results: List[DetectedFace] = []
    for loc, enc in zip(locations, encodings):
        person_id, confidence = match_face(enc, vector_db, sql_conn, threshold)

        name = "Unknown"
        is_monitored = False
        if person_id is not None:
            person = get_person(sql_conn, person_id)
            if person:
                name = person["name"] or "Unknown"
                is_monitored = bool(person["is_monitored"])

        results.append(DetectedFace(
            bbox=loc,
            embedding=enc,
            person_id=person_id,
            name=name,
            confidence=confidence,
            is_monitored=is_monitored,
        ))

    return results


def draw_detections(frame_bgr: np.ndarray, faces: List[DetectedFace]) -> np.ndarray:
    """Overlay bounding boxes and labels onto the frame."""
    out = frame_bgr.copy()
    for face in faces:
        top, right, bottom, left = face.bbox
        color = (0, 0, 220) if face.is_monitored else (0, 200, 80)
        thickness = 3 if face.is_monitored else 2
        cv2.rectangle(out, (left, top), (right, bottom), color, thickness)

        label = f"{face.name}"
        if face.confidence > 0:
            label += f"  {face.confidence:.0%}"
        if face.is_monitored:
            label = f"⚠ {label}"

        # Label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (left, bottom - h - 8), (left + w + 4, bottom), color, -1)
        cv2.putText(out, label, (left + 2, bottom - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return out


def crop_face(frame_bgr: np.ndarray, bbox: Tuple[int,int,int,int], padding: int = 20) -> np.ndarray:
    """Return a cropped + padded face image."""
    top, right, bottom, left = bbox
    h, w = frame_bgr.shape[:2]
    top    = max(0, top - padding)
    left   = max(0, left - padding)
    bottom = min(h, bottom + padding)
    right  = min(w, right + padding)
    return frame_bgr[top:bottom, left:right]
