"""REST API (SaaS entegrasyonu için).

Çalıştırma:  uvicorn api:app --host 0.0.0.0 --port 8000
Örnek:       curl -F "file=@ekran.png" http://localhost:8000/extract
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from hr_extractor import __version__, extract
from hr_extractor.export import to_excel, to_json

API_KEY = os.getenv("HR_API_KEY")  # tanımlıysa her istekte X-API-Key başlığı zorunlu
MAX_BYTES = 15 * 1024 * 1024

app = FastAPI(title="Apple Watch HR Extractor", version=__version__)


def _check(key: str | None) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Geçersiz API anahtarı")


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Dosya çok büyük (en fazla 15 MB)")
    return data


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/extract")
async def extract_json(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    _check(x_api_key)
    try:
        r = extract(await _read(file))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse(json.loads(to_json(r)))


@app.post("/extract.xlsx")
async def extract_xlsx(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    _check(x_api_key)
    r = extract(await _read(file))
    return Response(to_excel([(file.filename or "image", r)]),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": 'attachment; filename="nabiz.xlsx"'})


@app.post("/extract.csv")
async def extract_csv(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    _check(x_api_key)
    r = extract(await _read(file))
    return Response(r.samples().to_csv(index=False), media_type="text/csv")
