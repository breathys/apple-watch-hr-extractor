"""Apple Watch / Fitness uygulaması kalp atım ekran görüntüsünden veri çıkarımı.

İşlem adımları (hepsi OpenCV + NumPy, metinler Tesseract OCR):

1. Arka plan rengi bulunur, ön plan maskesi ve kontrast görüntüsü üretilir.
2. Grafikler bulunur: tam genişlikte yatay taban çizgisi + üzerinden yükselen
   dikey ızgara çizgileri olan her bölge bir grafiktir.
3. Eksen etiketleri OCR ile okunur (zaman: ``14:02`` / ``1 MIN``; nabız: ``165`` / ``75``)
   ve piksel -> değer dönüşümü kalibre edilir.
4. Renkli çubuklar (her çubuk bir zaman dilimindeki min-maks nabız aralığı)
   sütun sütun taranır ve gerçek değerlere çevrilir.
5. Özet alanlar (ortalama nabız, nabız bölgeleri, toparlanma nabzı) OCR ile okunur.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .ocr import Word, background_color, contrast_image, foreground_mask, read_words

TIME_RE = re.compile(r"(\d{1,2})[:.](\d{2})")
MIN_RE = re.compile(r"(\d+)\s*MIN", re.I)
NUM_RE = re.compile(r"\d{2,3}")
DUR_RE = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})")


# --------------------------------------------------------------------------- #
# Veri yapıları
# --------------------------------------------------------------------------- #
@dataclass
class Chart:
    kind: str                      # "workout" | "recovery"
    left: int
    right: int
    top: int
    base: int
    gridlines: list[int]
    y_max: float | None = None     # üst etiket (ör. 165)
    y_min: float | None = None     # alt etiket (ör. 75)
    x_labels: list[tuple[int, str]] = field(default_factory=list)
    start_minute: float | None = None   # günün dakikası (ör. 14:02 -> 842)
    minutes_per_px: float | None = None
    bars: pd.DataFrame | None = None


@dataclass
class Zone:
    zone: int
    duration: str
    seconds: int
    range_text: str
    low: int | None
    high: int | None


@dataclass
class Result:
    avg_hr: int | None
    zones: list[Zone]
    workout: Chart | None
    recovery: Chart | None
    recovery_label_bpm: int | None
    warnings: list[str]
    image_size: tuple[int, int]

    # ---- türetilmiş tablolar ---------------------------------------------
    def samples(self) -> pd.DataFrame:
        """Antrenman grafiğindeki her çubuk (ham ölçüm aralıkları)."""
        if self.workout is None or self.workout.bars is None:
            return pd.DataFrame()
        return self.workout.bars.copy()

    def recovery_samples(self) -> pd.DataFrame:
        if self.recovery is None or self.recovery.bars is None:
            return pd.DataFrame()
        return self.recovery.bars.copy()

    def per_second(self, step_s: int = 5) -> pd.DataFrame:
        """Çubuk orta noktalarından doğrusal enterpolasyonla eşit aralıklı seri."""
        df = self.samples()
        if df.empty:
            return df
        t = df["t_sec"].to_numpy()
        grid = np.arange(t.min(), t.max() + 1, step_s)
        out = pd.DataFrame({"t_sec": grid})
        for col in ("hr_min", "hr_max", "hr_mid"):
            out[col] = np.round(np.interp(grid, t, df[col].to_numpy()), 1)
        if self.workout and self.workout.start_minute is not None:
            out["clock"] = [_clock(self.workout.start_minute * 60 + s) for s in grid]
            out = out[["clock", "t_sec", "hr_min", "hr_max", "hr_mid"]]
        return out

    def per_minute(self) -> pd.DataFrame:
        df = self.per_second(5)
        if df.empty:
            return df
        df["minute"] = (df["t_sec"] // 60).astype(int)
        g = df.groupby("minute").agg(hr_min=("hr_min", "min"), hr_max=("hr_max", "max"),
                                     hr_avg=("hr_mid", "mean")).reset_index()
        g["hr_avg"] = g["hr_avg"].round(1)
        if self.workout and self.workout.start_minute is not None:
            g.insert(0, "clock", [_clock(self.workout.start_minute * 60 + m * 60)[:5]
                                  for m in g["minute"]])
        return g

    def zones_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(z) for z in self.zones])

    def summary(self) -> dict[str, Any]:
        s: dict[str, Any] = {"apple_avg_hr": self.avg_hr}
        df = self.samples()
        if not df.empty:
            ps = self.per_second(1)
            dur = int(df["t_sec"].max() - df["t_sec"].min())
            s.update({
                "start": df["clock"].iloc[0] if "clock" in df else None,
                "end": df["clock"].iloc[-1] if "clock" in df else None,
                "duration_min": round(dur / 60, 1),
                "bars_detected": int(len(df)),
                "hr_max_est": int(round(df["hr_max"].max())),
                "hr_min_est": int(round(df["hr_min"].min())),
                "hr_avg_est": round(float(ps["hr_mid"].mean()), 1),
            })
        total = sum(z.seconds for z in self.zones)
        if total:
            s["zones_total_min"] = round(total / 60, 1)
            s["zone_share_pct"] = {f"Z{z.zone}": round(100 * z.seconds / total, 1)
                                   for z in self.zones}
        if self.zones and not df.empty:
            s["zone_est_from_chart_min"] = self._zone_est()
        rec = self.recovery_samples()
        if not rec.empty:
            s["recovery_start_hr"] = int(round(rec["hr_mid"].iloc[0]))
            s["recovery_last_hr"] = int(round(rec["hr_mid"].iloc[-1]))
            s["recovery_change_bpm"] = int(round(rec["hr_mid"].iloc[-1] - rec["hr_mid"].iloc[0]))
        if self.recovery_label_bpm is not None:
            s["recovery_label_bpm"] = self.recovery_label_bpm
        return s

    def _zone_est(self) -> dict[str, float]:
        """Grafikten bölge süresi tahmini.

        Her çubuk, ölçüm aralığı (pitch) kadar süreyi temsil eder ve bu süre çubuğun
        min-maks aralığına eşit dağıtılarak bölgelere paylaştırılır.
        """
        df = self.samples()
        t = df["t_sec"].to_numpy()
        pitch = float(np.median(np.diff(t))) if len(t) > 1 else 60.0
        out = {f"Z{z.zone}": 0.0 for z in self.zones}
        for lo_b, hi_b in zip(df["hr_min"], df["hr_max"]):
            span = max(hi_b - lo_b, 1.0)
            lo_b, hi_b = (lo_b + hi_b) / 2 - span / 2, (lo_b + hi_b) / 2 + span / 2
            for z in self.zones:
                lo = z.low if z.low is not None else 0
                hi = (z.high + 1) if z.high is not None else 999
                ov = max(0.0, min(hi, hi_b) - max(lo, lo_b))
                out[f"Z{z.zone}"] += pitch * ov / span
        return {k: round(v / 60, 1) for k, v in out.items()}

    def to_dict(self) -> dict[str, Any]:
        def chart(c: Chart | None):
            if c is None:
                return None
            d = {k: v for k, v in asdict(c).items() if k != "bars"}
            d["bars"] = c.bars.to_dict(orient="records") if c.bars is not None else []
            return d
        return {
            "summary": self.summary(),
            "zones": [asdict(z) for z in self.zones],
            "workout_chart": chart(self.workout),
            "recovery_chart": chart(self.recovery),
            "warnings": self.warnings,
        }


def _clock(total_seconds: float) -> str:
    total_seconds = int(round(total_seconds)) % 86400
    return f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"


# --------------------------------------------------------------------------- #
# Grafik bulma
# --------------------------------------------------------------------------- #
def _grid_mask(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    d = np.abs(img.astype(int) - bg.astype(int)).max(axis=2)
    return (hsv[..., 1] < 45) & (d > 10) & (d < 110)


def color_mask(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Doygun renkli (çubuk) pikseller."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    d = np.abs(img.astype(int) - bg.astype(int)).max(axis=2)
    return (hsv[..., 1] > 70) & (hsv[..., 2] > 70) & (d > 40)


def tip_mask(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Koyu ama doygun çubuk uçları (ör. Bölge 5'in koyu pembe tepesi).

    Tek başına gürültüye açık olduğundan yalnızca çubuk gövdesine bitişikse kullanılır.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    d = np.abs(img.astype(int) - bg.astype(int)).max(axis=2)
    return (hsv[..., 1] > 110) & (hsv[..., 2] > 30) & (d > 18)


def find_charts(img: np.ndarray, bg: np.ndarray, unit: float) -> list[Chart]:
    H, W = img.shape[:2]
    g = _grid_mask(img, bg)
    gc = g | color_mask(img, bg)   # çubuklar ızgara çizgisini kesebilir
    min_vlen = int(30 * unit)
    rows = g.sum(axis=1)
    # İnce yatay çizgi: kendisi dolu, 4 px üstü/altı boş (durum çubuğu gibi bölgeler elenir)
    cand = [y for y in range(int(60 * unit), H - 5)
            if 0.6 * W < rows[y] < 0.995 * W and rows[y - 4] < 0.3 * W and rows[y + 4] < 0.3 * W]
    charts: list[Chart] = []
    used = -1
    for y in cand:
        if y <= used:
            continue
        xs = np.where(g[y])[0]
        left, right = int(xs.min()), int(xs.max())
        # tabandan yukarı doğru kesintisiz gri koşu uzunluğu
        tops = {}
        gap_tol = max(4, int(8 * unit))
        for x in range(left, right + 1):
            colg = gc[:, max(0, x - 1):x + 2].any(axis=1)
            yy, last, misses = y - 1, y, 0
            while yy > 0 and y - yy < 400 * unit:
                if colg[yy]:
                    last, misses = yy, 0
                else:
                    misses += 1
                    if misses > gap_tol:
                        break
                yy -= 1
            run = g[last:y, max(0, x - 1):x + 2].any(axis=1).sum()
            if y - last >= min_vlen and run >= 0.5 * (y - last):
                tops[x] = last
        if len(tops) < 2:
            continue
        # bitişik sütunları tek ızgara çizgisinde topla
        cols = sorted(tops)
        groups: list[list[int]] = [[cols[0]]]
        for x in cols[1:]:
            (groups[-1].append(x) if x - groups[-1][-1] <= 2 else groups.append([x]))
        lines = [int(round(np.mean(gr))) for gr in groups]
        if len(lines) < 2:
            continue
        top = int(np.median([min(tops[x] for x in gr) for gr in groups]))
        charts.append(Chart(kind="", left=left, right=right, top=top, base=y, gridlines=lines))
        used = y + int(5 * unit)
    return charts


# --------------------------------------------------------------------------- #
# Kalibrasyon
# --------------------------------------------------------------------------- #
def _merge_min_labels(xl: list[Word], contrast: np.ndarray, unit: float) -> list[Word]:
    """"1 MIN" gibi etiketlerde rakam ayrı kutu olarak kalırsa rakamla yeniden oku."""
    from .ocr import read_box
    out = []
    used: set[int] = set()
    xl = sorted(xl, key=lambda w: w.x)
    for i, w in enumerate(xl):
        if i in used:
            continue
        nxt = xl[i + 1] if i + 1 < len(xl) else None
        if nxt is not None and re.fullmatch(r"(?i)M[I1l]N", nxt.text.strip()) and nxt.x - w.x2 < 20 * unit \
                and not MIN_RE.search(w.text):
            txt, _ = read_box(contrast, w.x, w.y, w.w, w.h, whitelist="0123456789")
            d = _digits(txt)
            if d:
                out.append(Word(f"{d} MIN", w.x, w.y, nxt.x2 - w.x, max(w.h, nxt.h), 90.0))
                used.add(i + 1)
                continue
        out.append(w)
    return out


def _calibrate(chart: Chart, words: list[Word], unit: float, warnings: list[str],
               contrast: np.ndarray) -> None:
    h = chart.base - chart.top
    # y etiketleri: sağda, üst çizginin hemen üstü ve tabanın hemen altı
    right_zone = chart.right - 80 * unit
    top_lbl = [w for w in words if w.cx > right_zone and chart.top - 40 * unit < w.cy < chart.top + 5 * unit
               and NUM_RE.fullmatch(_digits(w.text) or "x")]
    bot_lbl = [w for w in words if w.cx > right_zone and chart.base < w.cy < chart.base + 35 * unit
               and NUM_RE.fullmatch(_digits(w.text) or "x")]
    if top_lbl:
        chart.y_max = float(_digits(top_lbl[0].text))
    if bot_lbl:
        chart.y_min = float(_digits(bot_lbl[0].text))
    if chart.y_max is None or chart.y_min is None or chart.y_max <= chart.y_min:
        warnings.append(f"{chart.kind}: nabız ekseni etiketleri okunamadı")

    # x etiketleri: tabanın altında, ızgara çizgilerinin hizasında
    xl = [w for w in words if chart.base < w.cy < chart.base + 35 * unit and w.cx < right_zone]
    xl = _merge_min_labels(xl, contrast, unit)
    pts: list[tuple[float, float]] = []   # (x_px, dakika)
    start_abs = None
    for w in sorted(xl, key=lambda w: w.x):
        gl = min(chart.gridlines, key=lambda g: abs(g - w.x))
        if abs(gl - w.x) > 15 * unit:
            continue
        t = w.text.replace(" ", "")
        m = TIME_RE.search(t)
        if m:
            minute = int(m.group(1)) * 60 + int(m.group(2))
            if start_abs is None:
                start_abs = minute
            elif minute < start_abs:      # gece yarısını geçiş
                minute += 1440
            pts.append((gl, minute))
            chart.x_labels.append((gl, m.group(0)))
            continue
        m = MIN_RE.search(w.text) or (re.fullmatch(r"\d", t) and re.match(r"(\d)", t))
        if m:
            pts.append((gl, (start_abs or 0) + int(m.group(1))))
            chart.x_labels.append((gl, f"{m.group(1)} MIN"))
    # Etiketi okunamayan ilk ızgara = başlangıç (kurtarma grafiği "0" dakika)
    if chart.kind == "recovery" and start_abs is not None and not any(abs(p[0] - chart.gridlines[0]) < 3 for p in pts):
        pts.append((chart.gridlines[0], start_abs))
    pts = sorted(set(pts))
    if len(pts) >= 2:
        xs, ms = np.array(pts).T
        a, b = np.polyfit(xs, ms, 1)
        # Mantık denetimi: ızgaralar eşit aralıklı olmalı
        chart.minutes_per_px = float(a)
        chart.start_minute = float(a * chart.left + b)
        if start_abs is None:
            chart.start_minute = None
    elif chart.kind == "recovery" and len(chart.gridlines) >= 2:
        # Apple: kurtarma grafiği 3 dakikalık, ızgaralar 1 dk aralıklı
        sp = np.median(np.diff(chart.gridlines))
        chart.minutes_per_px = 1.0 / sp
        chart.start_minute = float(start_abs) if start_abs is not None else None
        warnings.append("recovery: zaman ekseni varsayılan 1 dk/ızgara ile ölçeklendi")
    else:
        warnings.append(f"{chart.kind}: zaman ekseni etiketleri okunamadı")


def _digits(s: str) -> str:
    s = s.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    m = re.search(r"\d+", s)
    return m.group(0) if m else ""


# --------------------------------------------------------------------------- #
# Çubuk çıkarımı
# --------------------------------------------------------------------------- #
def plot_mask(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Grafikteki veri işaretleri: renkli çubuklar + soluk/beyazımsı tekil noktalar."""
    d = np.abs(img.astype(int) - bg.astype(int)).max(axis=2)
    return color_mask(img, bg) | (d > 50)


def _split_runs(profile: np.ndarray, unit: float) -> list[tuple[int, int]]:
    """Sütun profilini (her sütundaki işaret pikseli sayısı) tek tek çubuklara böl.

    Bitişik çubuklar arasında kenar yumuşatmasından kalan 1-3 piksellik "vadi"
    olur; bu vadiler kesim noktasıdır.
    """
    runs = []
    x, W = 0, len(profile)
    while x < W:
        if profile[x] == 0:
            x += 1
            continue
        s = x
        while x < W and profile[x] > 0:
            x += 1
        runs.append((s, x))
    out = []
    max_w = 6 * unit
    for s, e in runs:
        seg = profile[s:e]
        if e - s <= max_w + 1:
            out.append((s, e))
            continue
        # vadiler: komşularının yarısından az olan yerel minimumlar
        cuts = [s]
        i = 1
        while i < len(seg) - 1:
            left_peak = seg[max(0, i - 4):i].max()
            right_peak = seg[i + 1:i + 5].max()
            if seg[i] < 0.7 * min(left_peak, right_peak) and seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1]:
                # düz vadide ortadaki sütun
                j = i
                while j + 1 < len(seg) - 1 and seg[j + 1] == seg[i]:
                    j += 1
                cuts.append(s + (i + j) // 2 + 1)
                i = j + 2
                continue
            i += 1
        cuts.append(e)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a >= 1:
                out.append((a, b))
    return out


def _merge_close(bars: list[tuple]) -> list[tuple]:
    """Çubuk aralığının (pitch) %70'inden yakın iki parça aynı çubuktur; birleştir."""
    if len(bars) < 4:
        return bars
    bars = sorted(bars)
    d = np.diff([b[0] for b in bars])
    pitch = float(np.median(d[d > 0])) if (d > 0).any() else 0
    out = [list(bars[0])]
    for b in bars[1:]:
        p = out[-1]
        if pitch and b[0] - p[0] < 0.7 * pitch:
            hp, hb = p[2] - p[1] + 1, b[2] - b[1] + 1
            p[0] = (p[0] * hp + b[0] * hb) / (hp + hb)
            p[1], p[2] = min(p[1], b[1]), max(p[2], b[2])
            p[3] = p[3] if hp >= hb else b[3]
        else:
            out.append(list(b))
    return [tuple(b) for b in out]


def extract_bars(img: np.ndarray, bg: np.ndarray, chart: Chart, unit: float) -> pd.DataFrame:
    pm = plot_mask(img, bg)
    tm = tip_mask(img, bg)
    y0 = max(0, chart.top - int(6 * unit))
    y1 = chart.base - 1
    x0 = chart.left + 1
    sub = pm[y0:y1, x0:chart.right]
    # ızgara çizgisi sütunlarında yalnızca renkli/parlak işaretler kalır (d>50 zaten eler)
    profile = sub.sum(axis=0)
    bars = []
    for a, b in _split_runs(profile, unit):
        block = sub[:, a:b]
        if block.sum() < max(3, int(4 * unit * unit)):   # JPEG gürültüsü
            continue
        prof = block.sum(axis=0)
        core = block[:, prof >= 0.5 * prof.max()]    # yumuşatılmış kenar sütunlarını at
        ys = np.where(core.any(axis=1))[0]
        if len(ys) == 0:
            continue
        top_px, bot_px = ys.min() + y0, ys.max() + y0
        cols = np.where(prof >= 0.5 * prof.max())[0]
        # gövdeye bitişik koyu uçları ekle
        tcols = tm[:, x0 + a + cols.min():x0 + a + cols.max() + 1]
        while top_px - 1 >= y0 and tcols[top_px - 1].any():
            top_px -= 1
        while bot_px + 1 < y1 and tcols[bot_px + 1].any():
            bot_px += 1
        w = prof[cols].astype(float)
        cx = x0 + a + float((cols * w).sum() / w.sum())
        seg = img[top_px:bot_px + 1, x0 + a:x0 + b]
        hue = cv2.cvtColor(seg, cv2.COLOR_BGR2HSV)[..., 0]
        bars.append((cx, top_px, bot_px, float(np.median(hue))))
    if not bars:
        return pd.DataFrame()
    bars = _merge_close(bars)
    arr = np.array(bars)
    df = pd.DataFrame(arr, columns=["x_px", "top_px", "bottom_px", "hue"])

    if chart.y_max is not None and chart.y_min is not None:
        k = (chart.y_max - chart.y_min) / (chart.base - chart.top)
        # Piksel merkezleri: üst kenar = top, alt kenar = bottom + 1
        df["hr_max"] = np.round(chart.y_min + (chart.base - df["top_px"] - 0.5) * k, 1)
        df["hr_min"] = np.round(chart.y_min + (chart.base - df["bottom_px"] - 0.5) * k, 1)
        df["hr_mid"] = np.round((df["hr_max"] + df["hr_min"]) / 2, 1)
    if chart.minutes_per_px is not None:
        rel_min = (df["x_px"] - chart.left) * chart.minutes_per_px
        df["t_sec"] = np.round(rel_min * 60).astype(int)
        if chart.start_minute is not None:
            df.insert(0, "clock", [_clock(chart.start_minute * 60 + s) for s in df["t_sec"]])
    df["hue"] = df["hue"].round(0)
    return df


# --------------------------------------------------------------------------- #
# Özet metinler
# --------------------------------------------------------------------------- #
def _parse_zones(words: list[Word], unit: float) -> list[Zone]:
    zones = []
    labels = [w for w in words if re.match(r"(?i)z[o0]ne\s*\|?\s*([1-5])", w.text)]
    for lab in labels:
        n = int(re.match(r"(?i)z[o0]ne\s*\|?\s*([1-5])", lab.text).group(1))
        row = [w for w in words if abs(w.cy - lab.cy) < 14 * unit and w is not lab]
        dur = next((w for w in sorted(row, key=lambda w: w.x) if DUR_RE.fullmatch(w.text.strip(" |"))), None)
        rng = next((w for w in sorted(row, key=lambda w: -w.x) if "BPM" in w.text.upper().replace("8", "B")), None)
        secs, dtext = 0, ""
        if dur:
            m = DUR_RE.fullmatch(dur.text.strip(" |"))
            hh, mm, ss = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
            secs, dtext = hh * 3600 + mm * 60 + ss, m.group(0)
        lo = hi = None
        rtext = ""
        if rng:
            rtext = rng.text.replace(" ", "")
            nums = [int(v) for v in re.findall(r"\d{2,3}", rtext)]
            if rtext.startswith("<") and nums:
                hi = nums[0]
            elif "+" in rtext and nums:
                lo = nums[0]
            elif len(nums) >= 2:
                lo, hi = nums[0], nums[1]
        zones.append(Zone(n, dtext, secs, rtext, lo, hi))
    zones.sort(key=lambda z: z.zone)
    # "<131" sonraki bölgenin alt sınırı ile tutarlı hale getir
    for i, z in enumerate(zones):
        if z.low is None and z.zone == 1:
            z.low = 0
        if z.high is not None and z.zone == 1 and i + 1 < len(zones) and zones[i + 1].low:
            z.high = zones[i + 1].low - 1
    return zones


def _parse_avg(words: list[Word], img: np.ndarray, contrast: np.ndarray, unit: float) -> int | None:
    from .ocr import read_box
    lab = next((w for w in words if re.search(r"(?i)avg", w.text) and w.cy < img.shape[0] * 0.4
                and "BPM" not in w.text.upper()), None)
    if lab is None:
        return None
    below = sorted([w for w in words if lab.y2 < w.y < lab.y2 + 70 * unit], key=lambda w: w.y)
    if not below:
        return None
    w = below[0]
    return _read_bpm(contrast, w)


def _read_bpm(contrast: np.ndarray, w: Word) -> int | None:
    """"127BPM" biçimli kutudan sayıyı oku."""
    from .ocr import read_box
    for txt in (read_box(contrast, w.x, w.y, w.w, w.h, whitelist="0123456789BPM")[0], w.text):
        m = re.search(r"(\d{2,3})\s*B", txt.replace("8PM", "BPM"))
        if m and 30 <= int(m.group(1)) <= 230:
            return int(m.group(1))
    return None


# --------------------------------------------------------------------------- #
# Ana fonksiyon
# --------------------------------------------------------------------------- #
def extract(image: np.ndarray | str | bytes) -> Result:
    """Ekran görüntüsünden tüm kalp atım verilerini çıkar.

    ``image``: dosya yolu, ham bayt (yükleme) veya BGR NumPy dizisi.
    """
    img = _load(image)
    H, W = img.shape[:2]
    unit = W / 736.0
    if abs(unit - 1.0) > 0.02:
        # Ölçeği standartlaştır (OCR ve eşikler 736 px genişlik için ayarlı)
        img = cv2.resize(img, (736, int(round(H / unit))), interpolation=cv2.INTER_AREA if unit > 1 else cv2.INTER_CUBIC)
        H, W = img.shape[:2]
        unit = 1.0
    warnings: list[str] = []

    bg = background_color(img)
    mask = foreground_mask(img, bg, 25)
    contrast = contrast_image(img, bg)

    charts = find_charts(img, bg, unit)
    # Grafik içlerini OCR'dan hariç tut (hız + gürültü)
    ocr_mask = mask.copy()
    for c in charts:
        ocr_mask[c.top - 2:c.base + 2, c.left:c.right - int(70 * unit)] = 0
    words = read_words(ocr_mask, contrast, region=(0, int(H * 0.06), W, H), unit=unit)

    workout = recovery = None
    post_lbl = next((w for w in words if re.search(r"(?i)post|recovery", w.text)), None)
    for c in charts:
        if post_lbl is not None and c.top > post_lbl.y:
            c.kind = "recovery"
            recovery = recovery or c
        else:
            c.kind = "workout"
            workout = workout or c
    if workout is None:
        warnings.append("Antrenman nabız grafiği bulunamadı")
    for c in (workout, recovery):
        if c is not None:
            _calibrate(c, words, unit, warnings, contrast)
            c.bars = extract_bars(img, bg, c, unit)
            if c.bars.empty:
                warnings.append(f"{c.kind}: çubuk bulunamadı")

    rec_bpm = None
    if recovery is not None:
        lbl = [w for w in words if recovery.base + 20 * unit < w.cy < recovery.base + 60 * unit
               and "BPM" in w.text.upper().replace("8", "B")]
        if lbl:
            rec_bpm = _read_bpm(contrast, lbl[0])

    return Result(
        avg_hr=_parse_avg(words, img, contrast, unit),
        zones=_parse_zones(words, unit),
        workout=workout,
        recovery=recovery,
        recovery_label_bpm=rec_bpm,
        warnings=warnings,
        image_size=(W, H),
    )


def _load(image: np.ndarray | str | bytes) -> np.ndarray:
    if isinstance(image, np.ndarray):
        img = image
    elif isinstance(image, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Görüntü okunamadı")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def annotate(image: np.ndarray | str | bytes, result: Result) -> np.ndarray:
    """Tespit edilen grafik çerçevelerini ve çubukları gösteren kontrol görüntüsü."""
    img = _load(image).copy()
    H, W = img.shape[:2]
    if abs(W / 736 - 1) > 0.2:
        img = cv2.resize(img, (736, int(round(H * 736 / W))))
    for c, col in ((result.workout, (0, 255, 255)), (result.recovery, (255, 0, 255))):
        if c is None:
            continue
        cv2.rectangle(img, (c.left, c.top), (c.right, c.base), col, 1)
        for g in c.gridlines:
            cv2.line(img, (g, c.base), (g, c.base + 8), col, 2)
        if c.bars is not None and not c.bars.empty:
            for _, b in c.bars.iterrows():
                x = int(round(b.x_px))
                cv2.line(img, (x, int(b.top_px)), (x, int(b.bottom_px)), (255, 255, 255), 1)
                cv2.circle(img, (x, int(b.top_px)), 2, (0, 0, 255), -1)
                cv2.circle(img, (x, int(b.bottom_px)), 2, (255, 0, 0), -1)
    return img
