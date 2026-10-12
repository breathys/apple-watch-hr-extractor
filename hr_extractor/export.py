"""Sonuçları CSV / Excel / JSON olarak dışa aktarma."""
from __future__ import annotations

import io
import json
from typing import Iterable

import pandas as pd

from .core import Result


def _summary_df(result: Result, name: str = "") -> pd.DataFrame:
    s = result.summary()
    flat = {}
    for k, v in s.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    if name:
        flat = {"file": name, **flat}
    return pd.DataFrame([flat])


def to_excel(results: Iterable[tuple[str, Result]]) -> bytes:
    """Her görüntü için ayrı sayfalar + tüm dosyaların özet sayfası."""
    results = list(results)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.concat([_summary_df(r, n) for n, r in results], ignore_index=True) \
            .to_excel(xw, sheet_name="Ozet", index=False)
        for i, (name, r) in enumerate(results, 1):
            p = f"{i}_" if len(results) > 1 else ""
            r.samples().to_excel(xw, sheet_name=f"{p}Olcumler"[:31], index=False)
            r.per_minute().to_excel(xw, sheet_name=f"{p}Dakikalik"[:31], index=False)
            r.per_second(5).to_excel(xw, sheet_name=f"{p}5sn"[:31], index=False)
            r.zones_df().to_excel(xw, sheet_name=f"{p}Bolgeler"[:31], index=False)
            rec = r.recovery_samples()
            if not rec.empty:
                rec.to_excel(xw, sheet_name=f"{p}Toparlanma"[:31], index=False)
    return buf.getvalue()


def to_json(result: Result) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=_default)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, float) and o != o:  # NaN
        return None
    return str(o)
