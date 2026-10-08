"""
enroll.py — Employee face enrollment using InsightFace.

Usage
-----
python -m ai_models.face_recognition.enroll \
    --photos-dir data/employee_photos \
    --output    data/embeddings/employees.pkl

Expected directory layout for --photos-dir:
    data/employee_photos/
        <employee_id>/
            photo1.jpg
            photo2.jpg
            ...

For each employee, all photos are processed, their 512-D face embeddings are
L2-normalised, then averaged and re-normalised to produce a single robust
embedding per employee.  The result is saved as a pickle file:

    { "<employee_id>": np.ndarray(shape=(512,), dtype=float32), ... }
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# InsightFace model name — uses the buffalo_l pack by default (ArcFace R100).
# Set INSIGHTFACE_MODEL env-var or pass --model to override.
DEFAULT_MODEL = "buffalo_l"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_insightface_app(model_name: str, ctx_id: int = 0):
    """Initialise and return an InsightFace FaceAnalysis app."""
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError(
            "InsightFace is not installed. Run: pip install insightface"
        ) from exc

    app = FaceAnalysis(name=model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def _get_embedding(app, image_path: Path) -> np.ndarray | None:
    """
    Return the L2-normalised face embedding from *image_path*, or None if no
    face is detected.  When multiple faces are found, the largest bounding-box
    face is used (most likely to be the subject of a cropped ID photo).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        log.warning("Could not read image: %s — skipping.", image_path)
        return None

    faces = app.get(img)
    if not faces:
        log.warning("No face detected in %s — skipping.", image_path)
        return None

    if len(faces) > 1:
        log.debug("Multiple faces in %s — using the largest.", image_path)
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

    emb = faces[0].normed_embedding  # already L2-normalised by InsightFace
    return emb.astype(np.float32)


def _average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Average a list of L2-normalised embeddings and re-normalise the result
    so the centroid is itself a unit vector suitable for cosine similarity.
    """
    mean = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-8:
        return mean  # degenerate case — return as-is
    return (mean / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def enroll_employees(
    photos_dir: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL,
    ctx_id: int = 0,
) -> dict[str, np.ndarray]:
    """
    Enroll all employees found under *photos_dir*.

    Parameters
    ----------
    photos_dir:   Root directory containing one sub-folder per employee_id.
    output_path:  Destination pickle file path.
    model_name:   InsightFace model pack name.
    ctx_id:       GPU device id (0 for first GPU, -1 for CPU).

    Returns
    -------
    dict mapping employee_id -> averaged, L2-normalised embedding.
    """
    photos_dir = Path(photos_dir)
    output_path = Path(output_path)

    if not photos_dir.exists():
        raise FileNotFoundError(f"Photos directory not found: {photos_dir}")

    employee_dirs = [d for d in sorted(photos_dir.iterdir()) if d.is_dir()]
    if not employee_dirs:
        raise ValueError(f"No employee sub-directories found in: {photos_dir}")

    log.info("Loading InsightFace model '%s' ...", model_name)
    app = _load_insightface_app(model_name, ctx_id=ctx_id)

    embeddings_store: dict[str, np.ndarray] = {}
    failed_employees: list[str] = []

    for emp_dir in employee_dirs:
        employee_id = emp_dir.name
        photo_paths = [
            p for p in sorted(emp_dir.iterdir())
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not photo_paths:
            log.warning("No supported images for employee '%s' — skipping.", employee_id)
            failed_employees.append(employee_id)
            continue

        log.info("Enrolling employee '%s' from %d photo(s) ...", employee_id, len(photo_paths))

        embs: list[np.ndarray] = []
        for photo_path in photo_paths:
            emb = _get_embedding(app, photo_path)
            if emb is not None:
                embs.append(emb)

        if not embs:
            log.warning(
                "No valid embeddings extracted for employee '%s' — skipping.", employee_id
            )
            failed_employees.append(employee_id)
            continue

        embeddings_store[employee_id] = _average_embeddings(embs)
        log.info(
            "  Enrolled '%s' using %d/%d photo(s).",
            employee_id,
            len(embs),
            len(photo_paths),
        )

    # Persist to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(embeddings_store, fh, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(
        "Enrollment complete: %d enrolled, %d skipped. Saved to '%s'.",
        len(embeddings_store),
        len(failed_employees),
        output_path,
    )

    if failed_employees:
        log.warning("Skipped employees: %s", ", ".join(failed_employees))

    return embeddings_store


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enroll employees by generating face embeddings with InsightFace."
    )
    parser.add_argument(
        "--photos-dir",
        type=Path,
        default=Path("data/employee_photos"),
        help="Root directory of employee photo sub-folders (default: data/employee_photos).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/embeddings/employees.pkl"),
        help="Destination pickle file (default: data/embeddings/employees.pkl).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"InsightFace model pack name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device id; use -1 to force CPU (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    enroll_employees(
        photos_dir=args.photos_dir,
        output_path=args.output,
        model_name=args.model,
        ctx_id=args.gpu,
    )
