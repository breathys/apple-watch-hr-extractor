"""Metin bulma ve OCR yardımcıları.

Ekran görüntüsündeki her metin parçasını önce OpenCV ile (bağlantılı bileşenler)
kelime kutularına ayırır, sonra her kutuyu tek tek Tesseract'a verir.
Renkli (yeşil, turuncu, pembe) ve soluk gri yazılar da bu sayede okunur.
"""
from __future__ import annotations

from dataclasses import dataclass

import os
import threading
from pathlib import Path

import cv2
import numpy as np

# OCR motoru: tesserocr (pip ile gelir, sistem kurulumu gerekmez — Vercel gibi
# sunucusuz ortamlarda çalışır). Yoksa sistemdeki tesseract + pytesseract kullanılır.
TESSDATA = str(Path(__file__).parent / "tessdata")
try:
    import tesserocr
    from PIL import Image
    _BACKEND = "tesserocr"
except ImportError:  # pragma: no cover
    import pytesseract
    _BACKEND = "pytesseract"
if os.getenv("HR_OCR_BACKEND") == "pytesseract":
    import pytesseract  # noqa: F811
    _BACKEND = "pytesseract"

_local = threading.local()


def _api():
    api = getattr(_local, "api", None)
    if api is None:
        api = tesserocr.PyTessBaseAPI(path=TESSDATA + "/", lang="eng",
                                      psm=tesserocr.PSM.SINGLE_LINE)
        _local.api = api
    return api


def _recognize(img: np.ndarray, whitelist: str | None, psm: int) -> tuple[str, float]:
    if _BACKEND == "tesserocr":
        api = _api()
        api.SetPageSegMode(psm)
        api.SetVariable("tessedit_char_whitelist", whitelist or "")
        api.SetImage(Image.fromarray(img))
        text = " ".join(api.GetUTF8Text().split())
        return text, (float(api.MeanTextConf()) if text else -1.0)
    cfg = f"--psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    parts, confs = [], []
    for t, c in zip(data["text"], data["conf"]):
        if t.strip():
            parts.append(t.strip())
            confs.append(float(c))
    return " ".join(parts), (float(np.mean(confs)) if confs else -1.0)


@dataclass
class Word:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


def background_color(img: np.ndarray) -> np.ndarray:
    """Ekranın baskın arka plan rengi (BGR)."""
    small = cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)
    px = small.reshape(-1, 3).astype(int)
    # En sık görülen (kabaca nicemlenmiş) renk
    q = (px // 8) * 8
    vals, counts = np.unique(q, axis=0, return_counts=True)
    return vals[counts.argmax()].astype(float) + 4


def foreground_mask(img: np.ndarray, bg: np.ndarray, thr: int = 35) -> np.ndarray:
    """Arka plandan belirgin şekilde farklı pikseller = 255."""
    diff = np.abs(img.astype(int) - bg.astype(int)).max(axis=2)
    return np.where(diff > thr, 255, 0).astype(np.uint8)


def contrast_image(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Arka plana uzaklık (0-255). Renkli/gri tüm yazılar parlak çıkar."""
    diff = np.abs(img.astype(int) - bg.astype(int)).max(axis=2)
    return np.clip(diff * 1.6, 0, 255).astype(np.uint8)


def _ocr_crop(contrast: np.ndarray, x: int, y: int, w: int, h: int,
              whitelist: str | None = None, psm: int = 7) -> tuple[str, float]:
    H, W = contrast.shape
    # Birkaç dolgu/ölçek kombinasyonu dene; ilk dolu sonuç kullanılır
    for pad, target in ((max(4, h // 2), 56.0), (4, 40.0), (6, 72.0)):
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = 255 - contrast[y0:y1, x0:x1]  # koyu yazı, açık zemin
        scale = max(1.0, target / max(h, 1))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        text, conf = _recognize(crop, whitelist, psm)
        if text:
            return text, conf
    return "", -1.0


def find_words(mask: np.ndarray, unit: float = 1.0, min_h: int = 8,
               max_h: int = 60) -> list[tuple[int, int, int, int]]:
    """Yatay genişletme (dilation) ile harfleri kelime kutularına birleştir.

    ``unit``: ekran ölçeği (736 px genişlikli iPhone görüntüsü = 1.0).
    """
    kw = max(3, int(round(9 * unit)))
    kh = max(1, int(round(3 * unit)))
    dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        # Genişletme payını geri al
        x, w = x + kw // 2, w - 2 * (kw // 2)
        y, h = y + kh // 2, h - 2 * (kh // 2)
        if min_h * unit <= h <= max_h * unit and w >= 3:
            boxes.append((int(x), int(y), int(w), int(h)))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def read_words(mask: np.ndarray, contrast: np.ndarray,
               region: tuple[int, int, int, int] | None = None,
               whitelist: str | None = None, unit: float = 1.0) -> list[Word]:
    """Bölgedeki (x0, y0, x1, y1) tüm kelimeleri oku."""
    x0, y0 = 0, 0
    msub, csub = mask, contrast
    if region is not None:
        x0, y0, x1, y1 = region
        msub, csub = mask[y0:y1, x0:x1], contrast[y0:y1, x0:x1]
    words = []
    for (x, y, w, h) in find_words(msub, unit=unit):
        text, conf = _ocr_crop(csub, x, y, w, h, whitelist=whitelist)
        if text:
            words.append(Word(text, x + x0, y + y0, w, h, conf))
    return words


def read_box(contrast: np.ndarray, x: int, y: int, w: int, h: int,
             whitelist: str | None = None) -> tuple[str, float]:
    return _ocr_crop(contrast, x, y, w, h, whitelist=whitelist)
