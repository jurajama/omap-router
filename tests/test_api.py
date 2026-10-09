import cv2
import pytest
from fastapi.testclient import TestClient

from omap_router.config import get_settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OMAP_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    from omap_router.api.app import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture
def project(client, synth):
    ok, png = cv2.imencode(".png", synth.image)
    assert ok
    r = client.post("/api/projects", files={"file": ("map.png", png.tobytes(), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "leaflet" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_upload_rejects_garbage(client):
    r = client.post("/api/projects", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_full_workflow(client, project, synth):
    pid = project["id"]
    assert project["width"] == synth.image.shape[1]
    assert any(p["id"] == pid for p in client.get("/api/projects").json())
    assert client.get(f"/api/projects/{pid}/image").status_code == 200

    r = client.post(f"/api/projects/{pid}/analyze")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["layers"]) >= {"labels", "forbidden", "purple", "barriers", "cost"}
    assert body["project"]["calibration"]["r0"] == pytest.approx(synth.r0, rel=0.1)
    for name in body["layers"]:
        lr = client.get(f"/api/projects/{pid}/layers/{name}.png")
        assert lr.status_code == 200 and lr.headers["content-type"] == "image/png"
    assert client.get(f"/api/projects/{pid}/layers/nope.png").status_code == 404

    p = client.post(f"/api/projects/{pid}/controls/detect").json()
    kinds = [c["kind"] for c in sorted(p["controls"], key=lambda c: c["order"])]
    assert kinds == [k for k, *_ in synth.course]

    legs = client.post(f"/api/projects/{pid}/route").json()
    assert len(legs) == len(synth.course) - 1
    assert all(leg["ok"] for leg in legs)
    assert all(leg["length_m"] > 0 for leg in legs)

    # scale: setting it confirms and rescales lengths
    before = legs[0]["length_m"]
    p = client.put(f"/api/projects/{pid}/calibration", json={"scale": 2000}).json()
    assert p["calibration"]["scale_confirmed"]
    assert p["legs"][0]["length_m"] == pytest.approx(before / 2, rel=1e-6)
    # measure: a 100 px line is 50 m
    p = client.put(f"/api/projects/{pid}/calibration", json={"p1": [0, 0], "p2": [100, 0], "meters": 50}).json()
    expected = 50 / 100 * p["calibration"]["px_per_mm"] * 1000
    assert p["calibration"]["scale"] == round(expected)


def test_edits_persist_and_change_route(client, project, synth):
    pid = project["id"]
    client.post(f"/api/projects/{pid}/analyze")
    client.post(f"/api/projects/{pid}/controls/detect")
    legs = client.post(f"/api/projects/{pid}/route").json()
    first = legs[0]
    # forbid a polygon covering the middle of the first route
    mid = first["path"][len(first["path"]) // 2]
    x, y = mid
    poly = [(x - 25, y - 25), (x + 25, y - 25), (x + 25, y + 25), (x - 25, y + 25)]
    r = client.put(f"/api/projects/{pid}/edits", json=[{"id": "e1", "kind": "forbid", "points": poly}])
    assert r.status_code == 200
    # re-analyze keeps the edits
    p = client.post(f"/api/projects/{pid}/analyze").json()["project"]
    assert len(p["edits"]) == 1 and p["edits"][0]["id"] == "e1"
    assert len(p["controls"]) == len(synth.course)
    legs2 = client.post(f"/api/projects/{pid}/route").json()
    new = legs2[0]["path"]
    assert new != first["path"]
    for px, py in new:
        assert not (x - 25 < px < x + 25 and y - 25 < py < y + 25)


def test_user_controls_and_unreachable(client, project):
    pid = project["id"]
    client.post(f"/api/projects/{pid}/analyze")
    controls = [
        {"id": 1, "order": 0, "kind": "start", "x": 300, "y": 220, "r": 18, "source": "user"},
        {"id": 2, "order": 1, "kind": "finish", "x": 700, "y": 640, "r": 18, "source": "user"},  # in the lake
    ]
    p = client.put(f"/api/projects/{pid}/controls", json=controls).json()
    assert len(p["controls"]) == 2
    legs = client.post(f"/api/projects/{pid}/route").json()
    assert len(legs) == 1  # the lake point snaps to the shore, so the leg is routable
    assert legs[0]["ok"]
    # enclose the finish completely with forbid polygons
    controls[1].update(x=700, y=450)
    client.put(f"/api/projects/{pid}/controls", json=controls)
    wall = [
        {"id": "w1", "kind": "forbid", "points": [(560, 330), (840, 330), (840, 340), (560, 340)]},
        {"id": "w2", "kind": "forbid", "points": [(560, 330), (570, 330), (570, 560), (560, 560)]},
        {"id": "w3", "kind": "forbid", "points": [(830, 330), (840, 330), (840, 560), (830, 560)]},
    ]
    client.put(f"/api/projects/{pid}/edits", json=wall)
    legs = client.post(f"/api/projects/{pid}/route").json()
    assert not legs[0]["ok"]
    assert legs[0]["message"]


def test_calibration_validation(client, project):
    pid = project["id"]
    assert client.put(f"/api/projects/{pid}/calibration", json={"p1": [0, 0], "meters": 5}).status_code == 400
    assert client.put(f"/api/projects/{pid}/calibration", json={"scale": -1}).status_code == 400
    p = client.put(f"/api/projects/{pid}/calibration", json={"r0": 20}).json()
    assert p["calibration"]["r0_source"] == "user"
    assert p["calibration"]["px_per_mm"] == pytest.approx(40 / 6)


def test_crop_and_color_samples(client, project):
    pid = project["id"]
    p = client.put(f"/api/projects/{pid}/crop", json={"crop": [0, 0, 880, 600]}).json()
    assert p["crop"] == [0, 0, 880, 600]
    p = client.put(f"/api/projects/{pid}/color-samples", json={"beige": [[430, 100]]}).json()
    assert p["color_samples"] == {"BEIGE": [[430, 100]]}
    assert client.put(f"/api/projects/{pid}/color-samples", json={"pink": [[1, 1]]}).status_code == 400
    r = client.post(f"/api/projects/{pid}/analyze")
    assert r.status_code == 200
    # layers are padded back to the original image size
    import numpy as np

    data = client.get(f"/api/projects/{pid}/layers/labels.png").content
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img.shape[:2] == (700, 900)
    assert (img[650:, :, 3] == 0).all()  # outside the crop is transparent


def test_missing_project(client):
    assert client.get("/api/projects/deadbeef00").status_code == 404
    assert client.get("/api/projects/../../etc").status_code == 404
