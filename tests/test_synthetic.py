"""Bilinen değerlerle çizilmiş sentetik bir grafik üzerinde uçtan uca test."""
import cv2
import numpy as np
import pytest

from hr_extractor import extract

BG = (28, 28, 28)
GRID = (52, 50, 50)
LABEL = (140, 140, 140)
Y_TOP, Y_BASE, X0, X1 = 432, 529, 26, 708
HR_MAX, HR_MIN = 170, 70
T0, MPP = 14 * 60, 0.15  # 14:00 başlangıç, piksel başına 0.15 dk


def y_of(hr):
    return int(round(Y_BASE - (hr - HR_MIN) / (HR_MAX - HR_MIN) * (Y_BASE - Y_TOP)))


def make_image(bars):
    img = np.full((1600, 736, 3), BG, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Heart Rate", (28, 390), font, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(img, str(HR_MAX), (668, 420), font, 0.6, LABEL, 1, cv2.LINE_AA)
    cv2.putText(img, str(HR_MIN), (680, 552), font, 0.6, LABEL, 1, cv2.LINE_AA)
    cv2.line(img, (X0, Y_BASE), (X1, Y_BASE), GRID, 2)
    for gx, lbl in ((X0, "14:00"), (X0 + 200, "14:30"), (X0 + 400, "15:00")):
        cv2.line(img, (gx, Y_TOP), (gx, Y_BASE), GRID, 2)
        cv2.putText(img, lbl, (gx + 2, 552), font, 0.6, LABEL, 1, cv2.LINE_AA)
    for x, lo, hi in bars:
        cv2.rectangle(img, (x - 2, y_of(hi)), (x + 1, y_of(lo)), (200, 210, 60), -1)
    return img


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(0)
    xs = np.arange(40, 700, 8)
    lo = rng.integers(90, 140, len(xs))
    hi = lo + rng.integers(2, 25, len(xs))
    bars = list(zip(xs, lo, hi))
    return extract(make_image(bars)), bars


def test_axes(result):
    r, _ = result
    assert r.workout is not None
    assert r.workout.y_max == HR_MAX and r.workout.y_min == HR_MIN
    assert r.workout.minutes_per_px == pytest.approx(MPP, rel=0.03)


def test_bars(result):
    r, bars = result
    df = r.samples()
    assert len(df) == len(bars)
    px_bpm = (HR_MAX - HR_MIN) / (Y_BASE - Y_TOP)
    assert np.abs(df["hr_max"].to_numpy() - [b[2] for b in bars]).max() <= 2 * px_bpm + 0.5
    assert np.abs(df["hr_min"].to_numpy() - [b[1] for b in bars]).max() <= 2 * px_bpm + 0.5
    assert df["clock"].iloc[0].startswith("14:0")
