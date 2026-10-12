"""Apple Watch Nabız Veri Çıkarıcı — web arayüzü.

Çalıştırma:  streamlit run app.py
"""
from __future__ import annotations

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hr_extractor import annotate, extract
from hr_extractor.export import to_excel, to_json

st.set_page_config(page_title="Nabız Veri Çıkarıcı", page_icon="❤️", layout="wide")

ZONE_COLORS = {1: "#3b82f6", 2: "#2dd4bf", 3: "#a3e635", 4: "#f97316", 5: "#ec4899"}


@st.cache_data(show_spinner=False, max_entries=64)
def run(data: bytes):
    r = extract(data)
    ann = cv2.cvtColor(annotate(data, r), cv2.COLOR_BGR2RGB)
    return r, ann


def hr_figure(r) -> go.Figure:
    df = r.samples()
    if "clock" in df:
        x = pd.to_datetime("2000-01-01 " + df["clock"])
        width = df["t_sec"].diff().median() * 600  # ms, aralığın %60'ı
    else:
        x, width = df["t_sec"], df["t_sec"].diff().median() * 0.6
    fig = go.Figure()
    fig.add_bar(x=x, base=df["hr_min"], y=(df["hr_max"] - df["hr_min"]).clip(lower=0.8),
                name="Min–maks aralığı", marker_color="rgba(45,212,191,0.55)", width=width,
                hoverinfo="skip")
    fig.add_scatter(x=x, y=df["hr_mid"], mode="lines+markers", name="Orta değer",
                    line=dict(color="#ef4444", width=2), marker=dict(size=4))
    for z in r.zones:
        if z.low and z.zone > 1:
            fig.add_hline(y=z.low, line=dict(color=ZONE_COLORS.get(z.zone, "#888"), width=1, dash="dot"),
                          annotation_text=f"Z{z.zone}", annotation_position="right")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickformat="%H:%M",
                      yaxis_title="BPM", legend=dict(orientation="h", y=1.08), hovermode="x unified")
    return fig


def zones_table(r) -> pd.DataFrame:
    est = r.summary().get("zone_est_from_chart_min", {})
    rows = []
    for z in r.zones:
        rows.append({"Bölge": f"Z{z.zone}", "Aralık": z.range_text,
                     "Apple süresi": z.duration, "Apple (dk)": round(z.seconds / 60, 1),
                     "Grafikten tahmin (dk)": est.get(f"Z{z.zone}")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
st.title("❤️ Nabız Veri Çıkarıcı")
st.caption("Apple Watch / Fitness uygulamasındaki antrenman **Kalp Atış Hızı** ekranının "
           "görüntüsünü yükleyin; grafikteki her ölçüm, bölge süreleri ve toparlanma nabzı tabloya dönüşür.")

files = st.file_uploader("Ekran görüntüleri (PNG / JPG)", type=["png", "jpg", "jpeg"],
                         accept_multiple_files=True)

if not files:
    st.info("Başlamak için bir veya birden fazla ekran görüntüsü sürükleyip bırakın.")
    st.stop()

results = []
for f in files:
    with st.spinner(f"{f.name} işleniyor…"):
        try:
            r, ann = run(f.getvalue())
        except Exception as e:  # noqa: BLE001
            st.error(f"{f.name}: okunamadı ({e})")
            continue
    results.append((f.name, r, ann))

if not results:
    st.stop()

if len(results) > 1:
    st.download_button("⬇️ Tümünü Excel olarak indir", to_excel([(n, r) for n, r, _ in results]),
                       file_name="nabiz_verileri.xlsx", type="primary",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

for name, r, ann in results:
    st.divider()
    st.subheader(name)
    s = r.summary()
    for w in r.warnings:
        st.warning(w)

    c = st.columns(6)
    c[0].metric("Ortalama (Apple)", f"{s.get('apple_avg_hr') or '–'} BPM")
    c[1].metric("Ortalama (grafik)", f"{s.get('hr_avg_est', '–')} BPM")
    c[2].metric("En yüksek", f"{s.get('hr_max_est', '–')} BPM")
    c[3].metric("En düşük", f"{s.get('hr_min_est', '–')} BPM")
    c[4].metric("Süre", f"{s.get('duration_min', '–')} dk")
    c[5].metric("Ölçüm çubuğu", s.get("bars_detected", "–"))

    if not r.samples().empty:
        st.plotly_chart(hr_figure(r), width="stretch")

    t1, t2, t3, t4, t5, t6 = st.tabs(["Ölçümler", "Dakikalık", "Bölgeler", "Toparlanma",
                                      "Kontrol görüntüsü", "JSON"])
    with t1:
        st.caption("Her satır grafikteki bir çubuktur: o zaman dilimindeki en düşük / en yüksek nabız.")
        st.dataframe(r.samples().drop(columns=["x_px", "top_px", "bottom_px", "hue"], errors="ignore"),
                     width="stretch", hide_index=True)
    with t2:
        st.dataframe(r.per_minute(), width="stretch", hide_index=True)
    with t3:
        st.dataframe(zones_table(r), width="stretch", hide_index=True)
        st.caption("Grafikten tahmin: her çubuğun süresi min–maks aralığına eşit dağıtılarak hesaplanır; "
                   "Apple'ın saniyelik verisinden küçük farklar olması normaldir.")
    with t4:
        rec = r.recovery_samples()
        if rec.empty:
            st.write("Toparlanma grafiği bulunamadı.")
        else:
            st.dataframe(rec.drop(columns=["x_px", "top_px", "bottom_px", "hue"], errors="ignore"),
                         width="stretch", hide_index=True)
            st.write(f"İlk değer **{s.get('recovery_start_hr')}**, son değer **{s.get('recovery_last_hr')}** BPM; "
                     f"etiket: **{s.get('recovery_label_bpm', '–')} BPM**")
    with t5:
        st.caption("Sarı/mor çerçeve = bulunan grafik, beyaz çizgi = okunan çubuk (kırmızı: maks, mavi: min).")
        st.image(ann, width="stretch")
    with t6:
        st.json(r.to_dict()["summary"])

    d1, d2, d3 = st.columns(3)
    base = name.rsplit(".", 1)[0]
    d1.download_button("⬇️ Excel", to_excel([(name, r)]), file_name=f"{base}.xlsx", key=f"x{name}",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("⬇️ CSV (ölçümler)", r.samples().to_csv(index=False).encode("utf-8"),
                       file_name=f"{base}_olcumler.csv", mime="text/csv", key=f"c{name}")
    d3.download_button("⬇️ JSON", to_json(r).encode("utf-8"), file_name=f"{base}.json",
                       mime="application/json", key=f"j{name}")
