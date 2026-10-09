"""Project folders under ``<data_dir>/projects/<id>/``.

Layout::

    project.json
    original.<ext>
    layers/*.png
    cache/*.npz
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from .config import get_settings
from .models import Calibration, Project

_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class ProjectNotFound(KeyError):
    pass


def projects_root() -> Path:
    root = get_settings().data_dir / "projects"
    root.mkdir(parents=True, exist_ok=True)
    return root


def project_dir(project_id: str) -> Path:
    if not _ID_RE.match(project_id):
        raise ProjectNotFound(project_id)
    d = projects_root() / project_id
    if not (d / "project.json").exists():
        raise ProjectNotFound(project_id)
    return d


def layers_dir(project_id: str) -> Path:
    d = project_dir(project_id) / "layers"
    d.mkdir(exist_ok=True)
    return d


def cache_dir(project_id: str) -> Path:
    d = project_dir(project_id) / "cache"
    d.mkdir(exist_ok=True)
    return d


def create_project(data: bytes, filename: str) -> Project:
    ext = Path(filename).suffix.lower() or ".png"
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"unsupported image type {ext!r}")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    pid = uuid.uuid4().hex[:12]
    d = projects_root() / pid
    d.mkdir(parents=True)
    image_file = f"original{ext}"
    (d / image_file).write_bytes(data)
    h, w = img.shape[:2]
    project = Project(
        id=pid,
        image_file=image_file,
        width=w,
        height=h,
        calibration=Calibration(scale=get_settings().default_scale),
    )
    (d / "project.json").write_text(project.model_dump_json(indent=2))
    return project


def load_project(project_id: str) -> Project:
    d = project_dir(project_id)
    return Project.model_validate_json((d / "project.json").read_text())


def save_project(project: Project) -> None:
    d = project_dir(project.id)
    tmp = d / "project.json.tmp"
    tmp.write_text(project.model_dump_json(indent=2))
    tmp.replace(d / "project.json")


def list_projects() -> list[Project]:
    out = []
    for p in sorted(projects_root().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            out.append(Project.model_validate_json((p / "project.json").read_text()))
        except (OSError, ValueError):
            continue
    return out


def image_path(project: Project) -> Path:
    return project_dir(project.id) / project.image_file


def load_image_bgr(project: Project) -> np.ndarray:
    data = image_path(project).read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode stored image")
    return img


def cache_load(project_id: str, name: str, key: str) -> dict[str, np.ndarray] | None:
    d = cache_dir(project_id)
    keyfile = d / f"{name}.key"
    npz = d / f"{name}.npz"
    if not (keyfile.exists() and npz.exists()) or keyfile.read_text() != key:
        return None
    with np.load(npz) as z:
        return {k: z[k] for k in z.files}


def cache_save(project_id: str, name: str, key: str, arrays: dict[str, np.ndarray]) -> None:
    d = cache_dir(project_id)
    np.savez_compressed(d / f"{name}.npz", **arrays)
    (d / f"{name}.key").write_text(key)


def write_meta(project_id: str, name: str, meta: dict) -> None:
    (cache_dir(project_id) / f"{name}.json").write_text(json.dumps(meta))
