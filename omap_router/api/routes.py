"""HTTP endpoints (base ``/api``)."""

from __future__ import annotations

import math

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .. import storage
from ..config import TerrainClass, get_settings
from ..models import (
    AnalyzeResponse,
    CalibrationRequest,
    Control,
    CropRequest,
    EditPolygon,
    Leg,
    Project,
    ProjectSummary,
)
from ..pipeline import run
from ..pipeline.render import LAYER_NAMES

router = APIRouter(prefix="/api")

_MEDIA = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _load(pid: str) -> Project:
    try:
        return storage.load_project(pid)
    except storage.ProjectNotFound:
        raise HTTPException(404, f"project {pid} not found") from None


@router.post("/projects", response_model=Project)
async def create_project(file: UploadFile = File(...)) -> Project:
    data = await file.read()
    try:
        return storage.create_project(data, file.filename or "upload.png")
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@router.get("/projects", response_model=list[ProjectSummary])
def list_projects() -> list[ProjectSummary]:
    return [ProjectSummary(id=p.id, image_file=p.image_file, width=p.width, height=p.height) for p in storage.list_projects()]


@router.get("/projects/{pid}", response_model=Project)
def get_project(pid: str) -> Project:
    return _load(pid)


@router.put("/projects/{pid}/crop", response_model=Project)
def set_crop(pid: str, body: CropRequest) -> Project:
    p = _load(pid)
    if body.crop is not None:
        x0, y0, x1, y1 = body.crop
        x0, x1 = sorted((max(0, x0), min(p.width, x1)))
        y0, y1 = sorted((max(0, y0), min(p.height, y1)))
        if x1 - x0 < 10 or y1 - y0 < 10:
            raise HTTPException(400, "crop too small")
        p.crop = (x0, y0, x1, y1)
    else:
        p.crop = None
    storage.save_project(p)
    return p


@router.put("/projects/{pid}/color-samples", response_model=Project)
def set_color_samples(pid: str, body: dict[str, list[tuple[int, int]]]) -> Project:
    p = _load(pid)
    valid = {c.name for c in TerrainClass if c != TerrainClass.UNKNOWN}
    bad = [k for k in body if k.upper() not in valid]
    if bad:
        raise HTTPException(400, f"unknown classes: {bad}")
    p.color_samples = {k.upper(): v for k, v in body.items() if v}
    storage.save_project(p)
    return p


@router.post("/projects/{pid}/analyze", response_model=AnalyzeResponse)
def analyze(pid: str) -> AnalyzeResponse:
    p, layers = run.analyze(_load(pid))
    return AnalyzeResponse(project=p, layers=layers)


@router.get("/projects/{pid}/image")
def get_image(pid: str) -> FileResponse:
    p = _load(pid)
    path = storage.image_path(p)
    return FileResponse(path, media_type=_MEDIA.get(path.suffix.lower(), "application/octet-stream"))


@router.get("/projects/{pid}/layers/{name}.png")
def get_layer(pid: str, name: str) -> FileResponse:
    if name not in LAYER_NAMES:
        raise HTTPException(404, f"unknown layer {name}")
    _load(pid)
    path = storage.layers_dir(pid) / f"{name}.png"
    if not path.exists():
        raise HTTPException(404, "layer not computed yet; run analyze")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.post("/projects/{pid}/controls/detect", response_model=Project)
def detect_controls(pid: str) -> Project:
    return run.detect_controls(_load(pid))


@router.put("/projects/{pid}/controls", response_model=Project)
def set_controls(pid: str, body: list[Control]) -> Project:
    p = _load(pid)
    ids = [c.id for c in body]
    if len(ids) != len(set(ids)):
        raise HTTPException(400, "duplicate control ids")
    p.controls = body
    p.legs = []
    storage.save_project(p)
    return p


@router.put("/projects/{pid}/calibration", response_model=Project)
def set_calibration(pid: str, body: CalibrationRequest) -> Project:
    p = _load(pid)
    cfg = get_settings()
    cal = p.calibration
    if body.r0 is not None:
        if body.r0 <= 1:
            raise HTTPException(400, "r0 must be > 1 px")
        cal.r0 = body.r0
        cal.r0_source = "user"
        cal.px_per_mm = run.px_per_mm_from_r0(body.r0, cfg)
    if body.p1 is not None or body.p2 is not None or body.meters is not None:
        if body.p1 is None or body.p2 is None or not body.meters or body.meters <= 0:
            raise HTTPException(400, "calibration by measurement needs p1, p2 and meters > 0")
        dist = math.hypot(body.p2[0] - body.p1[0], body.p2[1] - body.p1[1])
        if dist < 1:
            raise HTTPException(400, "points are too close")
        ppm = cal.px_per_mm or cfg.default_px_per_mm
        cal.scale = max(1, int(round(body.meters / dist * ppm * 1000)))
        cal.scale_confirmed = True
    elif body.scale is not None:
        if body.scale <= 0:
            raise HTTPException(400, "scale must be positive")
        cal.scale = body.scale
        cal.scale_confirmed = True
    run.update_leg_lengths(p, cfg)
    storage.save_project(p)
    return p


@router.put("/projects/{pid}/edits", response_model=Project)
def set_edits(pid: str, body: list[EditPolygon]) -> Project:
    p = _load(pid)
    p.edits = body
    storage.save_project(p)
    return p


@router.post("/projects/{pid}/route", response_model=list[Leg])
def compute_route(pid: str) -> list[Leg]:
    p = run.route(_load(pid))
    return p.legs
