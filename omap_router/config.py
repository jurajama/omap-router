"""Tunable settings and defaults.

Every size-dependent threshold is expressed in printed millimetres (``_mm``) and
converted to pixels with ``px_per_mm`` derived from the detected control circle
radius (see ``pipeline.controls``). Values that are inherently pixel based
(anti-aliasing, filter kernels) carry a ``_px`` suffix.
"""

from __future__ import annotations

from enum import IntEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TerrainClass(IntEnum):
    WHITE = 0
    YELLOW = 1
    BEIGE = 2  # paved area
    GREEN = 3  # vegetation
    OLIVE = 4  # out of bounds
    GREY = 5  # building
    BLACK = 6
    PURPLE = 7  # course overprint
    BROWN = 8
    BLUE = 9  # water
    UNKNOWN = 10


RGB = tuple[int, int, int]

# Several reference shades per class: printed maps vary a lot between print
# runs, scanners and phone cameras. Values are RGB.
DEFAULT_REFERENCE_COLORS: dict[str, list[RGB]] = {
    "WHITE": [(252, 250, 246), (255, 255, 255)],
    "YELLOW": [(240, 187, 91), (247, 190, 80), (252, 222, 160)],
    "BEIGE": [(233, 202, 171), (238, 205, 173), (222, 196, 166)],
    "GREEN": [(152, 196, 155), (143, 216, 109), (75, 96, 55)],
    "OLIVE": [(172, 158, 55), (143, 170, 36)],
    "GREY": [(149, 144, 141), (123, 119, 114), (180, 180, 180)],
    "BLACK": [(40, 30, 13), (20, 20, 20)],
    "PURPLE": [(165, 70, 185), (167, 87, 187), (190, 110, 210), (195, 95, 140), (160, 85, 115)],  # thin purple over yellow / olive
    "BROWN": [(182, 141, 86), (162, 134, 104), (200, 168, 132)],
    "BLUE": [(2, 171, 235), (0, 211, 211), (72, 224, 225)],
}

# Colours used when rendering the label layer (RGB).
CLASS_DISPLAY_COLORS: dict[TerrainClass, RGB] = {
    TerrainClass.WHITE: (255, 255, 255),
    TerrainClass.YELLOW: (247, 190, 80),
    TerrainClass.BEIGE: (235, 205, 170),
    TerrainClass.GREEN: (110, 200, 100),
    TerrainClass.OLIVE: (160, 160, 40),
    TerrainClass.GREY: (130, 130, 130),
    TerrainClass.BLACK: (0, 0, 0),
    TerrainClass.PURPLE: (200, 40, 200),
    TerrainClass.BROWN: (170, 110, 50),
    TerrainClass.BLUE: (0, 170, 230),
    TerrainClass.UNKNOWN: (255, 0, 0),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OMAP_")

    data_dir: Path = Path("/data")

    # --- preprocess ---
    bilateral_d_px: int = 5
    bilateral_sigma_color: float = 25.0
    bilateral_sigma_space: float = 5.0
    white_percentile: float = 95.0

    # --- segment ---
    reference_colors: dict[str, list[RGB]] = Field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_REFERENCE_COLORS.items()}
    )
    max_delta_e: float = 28.0
    unknown_fill_iterations: int = 3
    sample_patch_px: int = 5

    # --- overprint ---
    purple_dilate_px: int = 1  # catch anti-aliased overprint edges
    hatch_enabled: bool = True  # purple cross-hatching -> forbidden
    hatch_min_holes: int = 6
    hatch_hole_min_mm: float = 0.25  # side of a hatch cell
    hatch_hole_max_mm: float = 2.5
    hatch_min_size_mm: float = 3.0  # smaller purple components are never hatches
    hatch_window_mm: float = 2.0  # density window for fine/fragmented hatching
    hatch_density_min_ratio: float = 0.2
    hatch_open_mm: float = 2.5

    # --- controls (ISSprOM printed sizes) ---
    control_diameter_mm: float = 6.0
    finish_inner_diameter_mm: float = 5.0
    finish_outer_diameter_mm: float = 7.0
    finish_radius_tol_ratio: float = 0.2
    finish_center_tol_ratio: float = 0.2  # of r0
    start_side_mm: float = 7.0
    start_side_tol_ratio: float = 0.25
    hough_min_radius_px: int = 5
    hough_max_radius_ratio: float = 0.125  # of min(H, W)
    hough_param1: float = 100.0
    hough_param2: float = 10.0
    ring_support_min_ratio: float = 0.5
    ring_tolerance_ratio: float = 0.15
    ring_tolerance_min_px: float = 2.0
    ring_tight_tolerance_ratio: float = 0.07  # after centre/radius polish
    ring_sectors: int = 8
    ring_sector_min_ratio: float = 0.75  # sectors that must contain ring ink
    filled_interior_max_ratio: float = 0.5
    ring_inner_max_ratio: float = 0.15  # purple just inside the ring
    ring_outer_max_ratio: float = 0.3  # purple just outside the ring
    ring_max_thickness_ratio: float = 0.3  # ring stroke width / radius
    min_circles_for_r0: int = 3
    relative_support_min_ratio: float = 0.7  # of the median support of detected controls
    refine_radius_ratio: float = 0.15
    leg_support_min_ratio: float = 0.6
    ocr_annulus_inner_ratio: float = 1.0
    ocr_annulus_outer_ratio: float = 2.5
    default_px_per_mm: float = 6.0  # used only when no circle is detected

    # --- black lines ---
    thick_line_mm: float = 0.35
    building_outline_px: int = 2
    line_min_bbox_mm: float = 2.0
    line_min_elongation_ratio: float = 3.0

    # --- terrain ---
    outside_map_enabled: bool = True  # blank paper around the map is impassable
    map_gap_mm: float = 5.0  # wider featureless white areas touching the edge are outside the map
    water_min_width_mm: float = 0.5  # thinner blue (north lines) stays passable
    cell_mm: float = 0.12
    cost_beige: float = 1.0
    cost_white: float = 1.2
    cost_yellow: float = 1.2
    cost_brown: float = 1.2
    cost_black_thin: float = 1.2
    cost_green: float = 3.0
    cost_unknown: float = 1.5
    cost_purple: float = 1.2  # only if purple survives replacement

    # --- routing ---
    snap_min_component_mm2: float = 4.0  # ignore passable pockets smaller than this when snapping

    # --- scale ---
    default_scale: int = 4000


@lru_cache
def get_settings() -> Settings:
    return Settings()


def class_cost(cfg: Settings) -> dict[TerrainClass, float]:
    """Per-class cost; ``inf`` means forbidden."""
    inf = float("inf")
    return {
        TerrainClass.WHITE: cfg.cost_white,
        TerrainClass.YELLOW: cfg.cost_yellow,
        TerrainClass.BEIGE: cfg.cost_beige,
        TerrainClass.GREEN: cfg.cost_green,
        TerrainClass.OLIVE: inf,
        TerrainClass.GREY: inf,
        TerrainClass.BLACK: cfg.cost_black_thin,
        TerrainClass.PURPLE: cfg.cost_purple,
        TerrainClass.BROWN: cfg.cost_brown,
        TerrainClass.BLUE: inf,
        TerrainClass.UNKNOWN: cfg.cost_unknown,
    }
