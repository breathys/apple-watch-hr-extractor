"""Gerçek ekran görüntüsü testi (tests/fixtures/*.jpg|png varsa çalışır)."""
from pathlib import Path

import pytest

from hr_extractor import extract

FIX = sorted(Path(__file__).parent.glob("fixtures/*.[jp][pn]g"))


@pytest.mark.skipif(not FIX, reason="tests/fixtures içinde örnek görüntü yok")
@pytest.mark.parametrize("path", FIX, ids=[p.name for p in FIX])
def test_real_screenshot(path):
    r = extract(str(path))
    assert r.workout is not None and not r.samples().empty
    assert r.avg_hr is not None
    assert len(r.zones) == 5
    s = r.summary()
    assert abs(s["hr_avg_est"] - r.avg_hr) < 8
