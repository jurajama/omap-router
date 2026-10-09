# omap-router

Web application that takes a scanned or photographed **sprint orienteering map**
(ISSprOM style, JPG/PNG/WebP), finds the course (start, controls, finish), builds
a passability/cost model of the terrain and computes a least-cost route for
every leg. Routes prefer roads and paved areas and never cross forbidden areas
(buildings, olive out-of-bounds, water, impassable walls/fences, purple
cross-hatched out-of-bounds areas, blank paper outside the map, and polygons you
draw). If forbidden areas separate two controls, the leg is reported as having
no route instead of being routed around the map. You review and correct everything in the browser.

## Run with Docker

```sh
docker build -t omap-router .
docker run -p 8000:8000 -v ~/maps:/data omap-router
```

Open <http://localhost:8000>. Projects are stored under `~/maps/projects/<id>/`.
The image builds natively on arm64 and amd64 (no architecture-specific wheels).
The UI loads Leaflet and Leaflet-Geoman from unpkg.com, so the browser needs
internet access.

## Run locally (development)

```sh
uv sync
OMAP_DATA_DIR=./data uv run uvicorn omap_router.api.app:app --reload --port 8000
uv run pytest
```

Optional: install `tesseract-ocr` for the OCR fallback that reads control numbers
when leg lines do not determine the order.

## Workflow in the UI

1. **Project** – upload a map image (analysis starts automatically) or open an
   existing project. Optionally set a crop rectangle (two clicks) and re-analyze.
2. **Layers** – toggle `labels`, `forbidden`, `outside`, `purple`, `hatch`, `barriers`, `cost`,
   routes and controls; adjust overlay opacity.
3. **Colors** – if the classification is off (check the `labels` layer), pick a
   class, click on the map to add colour samples and press *Re-analyze*.
4. **Controls** – *Detect controls* finds circles, start triangle and finish and
   orders them by following the leg lines. Drag markers to move, right-click to
   delete, *Add by click* to add, change the kind in the table, reorder with
   ▲/▼, untick *use* to drop a false detection from the course. If no circles are
   found, use *Mark circle* (click a circle's centre, then its ring) to tell the
   app the circle size.
5. **Edits** – draw *forbid* (red) or *allow* (green) polygons with the map
   toolbar, e.g. for an out-of-bounds area the analysis missed, or *allow* over
   an automatically detected area that is actually passable. Allow polygons
   override every automatic mask, including purple hatching. Edits are stored
   separately and survive re-analysis. *Clear all* removes every polygon.
6. **Scale** – distances need the map scale. The default 1:4000 is an assumption
   (lengths are labelled *est.*). Enter the real scale or use *Measure…*: click
   two points and enter their distance in metres.
7. **Route** – *Compute routes* draws the legs and lists their lengths;
   unreachable legs are shown as red dashed lines.

## How it works

The map scale and scan resolution are unknown, so everything is calibrated from
the control circles: ISSprOM control circles are printed 6 mm wide, so once the
dominant circle radius `r0` (pixels) is known, `px_per_mm = 2·r0 / 6`. All size
thresholds in [`config.py`](omap_router/config.py) are in printed millimetres.

| Stage | Module | What it does |
|---|---|---|
| 8.1 | `pipeline/preprocess.py` | crop, bilateral filter, white balance |
| 8.2 | `pipeline/segment.py` | nearest reference colour in Lab (several shades per class, user samples override), unknown pixels filled by 3×3 majority |
| 8.3 | `pipeline/overprint.py` | purple mask; purple pixels replaced by the nearest non-purple label; purple cross-hatching detected as out-of-bounds (components enclosing many small cells, or dense purple texture with many enclosed gaps, opened to drop attached leg lines) |
| 8.4 | `pipeline/controls.py` | wide Hough search → ring verification (support, empty inside/outside, thin stroke, sector coverage) → radius consensus `r0` → narrow re-search; finish = concentric ring pair; start = chamfer-matched rotated triangle; order = path through the leg-line graph that visits every point (DFS, threshold relaxed if lines are cut), OCR fallback |
| 8.5 | `pipeline/lines.py` | black strokes skeletonised, width = area / length per segment, thick strokes become barriers; building outlines and small symbols ignored |
| 8.6 | `pipeline/terrain.py` | forbidden = grey ∪ olive ∪ water (thin blue north lines removed) ∪ barriers ∪ purple hatching ∪ outside the map ∪ forbid polygons − allow polygons; outside the map = white paper not enclosed by map content (content closed over gaps up to 5 mm, holes filled); cost grid where a cell is forbidden only if all its pixels are, or if a barrier crosses it |
| 8.7 | `pipeline/routing.py` | `skimage.graph.MCP_Geometric` per leg between snapped endpoints, simplified with shapely and re-checked against the forbidden cells |

Default costs: paved 1.0; white/yellow/brown/thin black 1.2; green 3.0; unknown 1.5.

## API

Base `/api`, JSON unless noted. Pixel coordinates are always original-image `(x, y)`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/projects` | multipart upload (`file`) → Project |
| GET | `/projects`, `/projects/{id}` | list / get |
| PUT | `/projects/{id}/crop` | `{"crop": [x0, y0, x1, y1] \| null}` |
| PUT | `/projects/{id}/color-samples` | `{"BEIGE": [[x, y], …], …}` |
| POST | `/projects/{id}/analyze` | run 8.1–8.6 → `{project, layers}` |
| GET | `/projects/{id}/image` | original image |
| GET | `/projects/{id}/layers/{name}.png` | RGBA overlay: `labels`, `forbidden`, `outside`, `purple`, `hatch`, `barriers`, `cost` |
| POST | `/projects/{id}/controls/detect` | run 8.4 |
| PUT | `/projects/{id}/controls` | replace the control list |
| PUT | `/projects/{id}/calibration` | `{"scale": 4000}`, `{"p1": [x, y], "p2": [x, y], "meters": 120}` or `{"r0": 17.5}` |
| PUT | `/projects/{id}/edits` | replace the forbid/allow polygons |
| POST | `/projects/{id}/route` | compute legs → list of legs |

## Tests

`uv run pytest` runs unit tests on synthetic images (circle detection for random
radii 8–60 px without a size hint, barrier widths, narrow passages, diagonal
leaks, road preference, the invariant that no path crosses a forbidden cell), API
tests with FastAPI's `TestClient`, and fixture tests. The repository contains no
real map data: a synthetic ISSprOM-like map is generated in
[`tests/synthetic.py`](tests/synthetic.py). To test against real maps, add
`tests/fixtures/<name>/image.png` plus any of `forbidden.png` (white =
forbidden), `passages.png` (white = narrow passages that must stay open) and
`controls.json` (`{"r0": 17.5, "course": [{"kind": "start", "x": 712, "y": 231}, …]}`).

## Known limitations

- A featureless white area wider than ~5 mm that reaches the image edge is
  treated as outside the map. If it is real terrain, draw an *allow* polygon
  over it, increase `OMAP_MAP_GAP_MM`, or set `OMAP_OUTSIDE_MAP_ENABLED=false`.

- Hatch detection follows the drawn grid, so the edge of a detected area can be
  about half a grid cell inside or outside the printed boundary, and very
  small hatched patches (under ~3 mm) are ignored. Correct these with forbid or
  allow polygons. Set `OMAP_HATCH_ENABLED=false` to turn detection off.
- Routing uses an 8-connected grid, which overestimates oblique distances by up
  to ~8 %, so a road is preferred when it is roughly 10–20 % longer than the
  direct line, depending on its direction.
- The OCR fallback misreads some control numbers. It is used only when it yields
  a complete and consistent 1…n numbering; otherwise you order the controls by hand.
- Very low resolution scans (control circle radius below ~8 px) make the
  thick/thin line distinction unreliable; use forbid polygons for missed walls.
