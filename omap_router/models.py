"""Pydantic data models shared by the pipeline, storage and API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Point = tuple[float, float]


class Control(BaseModel):
    id: int  # stable id
    order: int | None = None  # 0 = start, 1..n controls, last = finish
    kind: Literal["start", "control", "finish"] = "control"
    x: float  # pixel coords in original image
    y: float
    r: float = 0.0  # detected radius px
    code: int | None = None  # printed control code, e.g. 126 in "1-126"
    source: Literal["auto", "user"] = "auto"


class Calibration(BaseModel):
    r0: float | None = None  # dominant control circle radius px
    px_per_mm: float | None = None
    scale: int = 4000
    scale_confirmed: bool = False
    r0_source: Literal["auto", "user", "default"] = "auto"


class EditPolygon(BaseModel):
    id: str
    kind: Literal["forbid", "allow"]
    points: list[Point]  # pixel coords


class Leg(BaseModel):
    from_id: int
    to_id: int
    path: list[Point] = Field(default_factory=list)  # pixel coords, simplified
    length_px: float = 0.0
    length_m: float = 0.0
    cost: float = 0.0
    ok: bool = True
    message: str | None = None  # ok=False if unreachable


class Project(BaseModel):
    id: str
    image_file: str
    width: int
    height: int
    crop: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 (exclusive)
    color_samples: dict[str, list[tuple[int, int]]] = Field(default_factory=dict)
    calibration: Calibration = Field(default_factory=Calibration)
    controls: list[Control] = Field(default_factory=list)
    edits: list[EditPolygon] = Field(default_factory=list)
    legs: list[Leg] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)  # warnings from last run


# --- API request/response bodies ---


class CropRequest(BaseModel):
    crop: tuple[int, int, int, int] | None


class CalibrationRequest(BaseModel):
    scale: int | None = None
    p1: Point | None = None
    p2: Point | None = None
    meters: float | None = None
    r0: float | None = None  # user-marked control circle radius (fallback)


class AnalyzeResponse(BaseModel):
    project: Project
    layers: list[str]


class ProjectSummary(BaseModel):
    id: str
    image_file: str
    width: int
    height: int
