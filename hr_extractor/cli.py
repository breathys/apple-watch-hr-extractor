"""Komut satırı: hr-extract ekran1.png ekran2.jpg -o cikti/"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .core import annotate, extract
from .export import to_excel, to_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hr-extract",
        description="Apple Watch kalp atım ekran görüntülerinden veri çıkarır.")
    ap.add_argument("images", nargs="+", help="PNG/JPG ekran görüntüleri")
    ap.add_argument("-o", "--out", default="hr_output", help="Çıktı klasörü (varsayılan: hr_output)")
    ap.add_argument("--debug", action="store_true", help="Tespitleri gösteren kontrol görüntüsü de yaz")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for img_path in args.images:
        p = Path(img_path)
        r = extract(str(p))
        results.append((p.name, r))
        stem = out / p.stem
        r.samples().to_csv(f"{stem}_olcumler.csv", index=False)
        r.per_minute().to_csv(f"{stem}_dakikalik.csv", index=False)
        r.zones_df().to_csv(f"{stem}_bolgeler.csv", index=False)
        Path(f"{stem}.json").write_text(to_json(r), encoding="utf-8")
        if args.debug:
            cv2.imwrite(f"{stem}_kontrol.png", annotate(str(p), r))
        print(f"\n== {p.name}")
        print(json.dumps(r.summary(), ensure_ascii=False, indent=2, default=str))
        for w in r.warnings:
            print(f"  ! {w}")
    (out / "hepsi.xlsx").write_bytes(to_excel(results))
    print(f"\nÇıktılar: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
