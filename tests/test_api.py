"""API uçtan uca testi (sentetik görüntü ile)."""
import cv2
from fastapi.testclient import TestClient

from hr_extractor.web import app
from tests.test_synthetic import make_image


def test_extract_endpoint():
    img = make_image([(x, 100 + (x % 30), 110 + (x % 30)) for x in range(40, 700, 8)])
    ok, buf = cv2.imencode(".png", img)
    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "ok"
    r = c.post("/api/extract?debug=true", files={"file": ("s.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200
    d = r.json()
    assert len(d["samples"]) > 50 and d["annotated_png"]
    r = c.post("/api/extract.xlsx", files={"file": ("s.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200 and r.content[:2] == b"PK"


def test_bad_file():
    r = TestClient(app).post("/api/extract", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
