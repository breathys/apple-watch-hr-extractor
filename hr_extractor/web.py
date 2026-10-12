"""REST API (FastAPI). Vercel'de /api/* altında, yerelde uvicorn ile çalışır.

Yerel:  uvicorn hr_extractor.web:app --reload --port 8000   (arayüz: http://localhost:8000)
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from . import __version__
from .core import annotate, extract
from .export import _default, to_excel

API_KEY = os.getenv("HR_API_KEY")       # tanımlıysa X-API-Key başlığı zorunlu
MAX_BYTES = 4_400_000                   # Vercel istek gövdesi sınırı ~4.5 MB
PUBLIC = Path(__file__).resolve().parent.parent / "public"

app = FastAPI(title="Nabız Veri Çıkarıcı API", version=__version__,
              docs_url="/api/docs", openapi_url="/api/openapi.json")


def _check(key: str | None) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Geçersiz API anahtarı")


async def _run(file: UploadFile):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Görüntü 4.4 MB'tan büyük. Daha küçük bir dosya deneyin.")
    try:
        return data, extract(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Dosya görüntü olarak açılamadı. PNG veya JPG yükleyin.") from e


def _records(df):
    return json.loads(df.to_json(orient="records")) if not df.empty else []


@app.get("/api/health")
def health():
    from .ocr import _BACKEND
    return {"status": "ok", "version": __version__, "ocr": _BACKEND}


@app.post("/api/extract")
async def extract_json(file: UploadFile = File(...), debug: bool = Query(False),
                       x_api_key: str | None = Header(None)):
    """Ekran görüntüsünü işle; tüm tabloları tek JSON'da döndür."""
    _check(x_api_key)
    data, r = await _run(file)
    out = {
        "file": file.filename,
        "summary": r.summary(),
        "zones": _records(r.zones_df()),
        "samples": _records(r.samples()),
        "per_minute": _records(r.per_minute()),
        "recovery": _records(r.recovery_samples()),
        "warnings": r.warnings,
    }
    if debug:
        ok, png = cv2.imencode(".png", annotate(data, r))
        out["annotated_png"] = base64.b64encode(png.tobytes()).decode() if ok else None
    return JSONResponse(json.loads(json.dumps(out, default=_default)))


@app.post("/api/extract.xlsx")
async def extract_xlsx(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    _check(x_api_key)
    _, r = await _run(file)
    name = (file.filename or "nabiz").rsplit(".", 1)[0]
    return Response(to_excel([(file.filename or "image", r)]),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{name}.xlsx"'})


@app.post("/api/extract.csv")
async def extract_csv(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    _check(x_api_key)
    _, r = await _run(file)
    return Response(r.samples().to_csv(index=False), media_type="text/csv")


# Yerel geliştirmede arayüzü de sun. Vercel'de public/ klasörünü CDN servis eder.
if not os.getenv("VERCEL"):
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(PUBLIC / "index.html")
