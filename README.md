# ❤️ Apple Watch Nabız Veri Çıkarıcı

Apple Watch / iPhone Fitness uygulamasındaki antrenman **Kalp Atış Hızı (Heart Rate)** ekranının
görüntüsünden ayrıntılı veriyi çıkaran araç. Görüntü işleme **OpenCV**, metin okuma **Tesseract OCR** ile yapılır.

Ekran görüntüsü yükle → tablo, grafik, Excel/CSV/JSON indir.

## Neler çıkarılır?

| Veri | Kaynak | Açıklama |
|---|---|---|
| Her ölçüm çubuğu | Antrenman grafiği | Saat (`14:52:38`), o dilimdeki **min / maks / orta** nabız |
| Dakikalık ve 5 sn'lik seri | Hesaplanır | Çubuklardan doğrusal enterpolasyon |
| Ortalama nabız | Metin (OCR) | Apple'ın gösterdiği değer + grafikten hesaplanan |
| En yüksek / en düşük nabız | Grafik | ±1 BPM çözünürlük (≈0,9 BPM/piksel) |
| Bölge 1–5 süreleri ve aralıkları | Metin (OCR) | `40:07`, `132–142 BPM` … + grafikten tahmini süre |
| Toparlanma (Post-Workout) nabzı | Alt grafik | Her çubuk + etiket değeri |

## Nasıl çalışır?

1. Arka plan rengi bulunur; renkli/gri tüm yazı ve işaretler kontrast görüntüsüne ayrılır.
2. **Grafik tespiti:** tam genişlikte yatay taban çizgisi + üzerinden yükselen dikey ızgara çizgileri.
3. **Eksen kalibrasyonu:** `165` / `75` (nabız) ve `14:02 · 14:34 · 15:06` ya da `1 MIN` (zaman) etiketleri
   OCR ile okunur; ızgara konumlarına doğrusal uydurma ile piksel → saat / BPM dönüşümü kurulur.
4. **Çubuk çıkarımı:** sütun profili taranır, bitişik çubuklar kenar-yumuşatma vadilerinden ayrılır,
   koyu pembe Bölge 5 uçları gövdeye bitişikse eklenir (JPEG gürültüsü elenir).
5. Bölge, ortalama ve toparlanma metinleri tek tek kelime kutularında OCR ile okunur.

**Kontrol görüntüsü** sekmesi neyin okunduğunu görüntü üzerine çizer; sonuçları gözle doğrulamak kolaydır.

## Kurulum

```bash
git clone https://github.com/breathys/apple-watch-hr-extractor.git
cd apple-watch-hr-extractor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**OCR motoru:** Linux'ta `tesserocr` paketi Tesseract'ı kendi içinde getirir, ayrıca kurulum gerekmez
(dil verisi `hr_extractor/tessdata/` içinde). macOS / Windows'ta sistem Tesseract'ı kullanılır:
`brew install tesseract` (macOS) veya `choco install tesseract` (Windows).

## Kullanım

### Web arayüzü + API (yerel)

```bash
uvicorn hr_extractor.web:app --reload --port 8000
```

http://localhost:8000 adresini açın ve ekran görüntülerini sürükleyip bırakın. API belgeleri: http://localhost:8000/api/docs

### Alternatif Streamlit arayüzü

```bash
pip install -r requirements-ui.txt
streamlit run streamlit_app.py
```

### Komut satırı

```bash
pip install -e .
hr-extract ekran1.jpg ekran2.png -o cikti/ --debug
```

Her görüntü için `_olcumler.csv`, `_dakikalik.csv`, `_bolgeler.csv`, `.json` ve tümü için `hepsi.xlsx` üretir.

### Python

```python
from hr_extractor import extract

r = extract("ekran.jpg")
r.samples()      # çubuk başına min/maks/orta nabız (DataFrame)
r.per_minute()   # dakikalık tablo
r.zones_df()     # bölge süreleri
r.summary()      # özet sözlük
```

### REST API

| Uç nokta | Döndürür |
|---|---|
| `POST /api/extract` | Tüm tablolar tek JSON'da (`?debug=true` ile kontrol görüntüsü de) |
| `POST /api/extract.xlsx` | Excel dosyası |
| `POST /api/extract.csv` | Ölçümler CSV |
| `GET /api/health` | Durum |

```bash
curl -F "file=@ekran.jpg" https://<alan-adınız>/api/extract
```

`HR_API_KEY` ortam değişkeni tanımlanırsa her istekte `X-API-Key` başlığı zorunlu olur.

## Yayına alma

### Vercel (önerilen)

Repo Vercel'e hazır: arayüz `public/index.html` (CDN), API tek bir Python fonksiyonu
(`hr_extractor.web:app`, `pyproject.toml` içindeki `[tool.vercel]`).

1. vercel.com → **Add New → Project** → bu GitHub reposunu içe aktarın.
2. Ayarları değiştirmeden **Deploy**.

Sonraki her `git push` otomatik yayınlanır. Vercel istek gövdesi ~4.5 MB ile sınırlı olduğundan arayüz büyük
görüntüleri yüklemeden önce tarayıcıda küçültür.

### Docker / kendi sunucunuz

```bash
docker compose up -d --build    # http://localhost:8000
```

## Desteklenen ekran

iPhone **Fitness** uygulaması → antrenman → **Heart Rate** ayrıntı ekranı (koyu tema), İngilizce arayüz.
Farklı iPhone çözünürlükleri otomatik ölçeklenir. Açık tema ve Türkçe arayüz büyük olasılıkla çalışır ama
test edilmedi; okunamayan alanlar uyarı olarak gösterilir.

## Doğruluk notları

- Grafik bir **özettir**: Apple her çubukta ~1 dakikalık aralığın min–maks değerini çizer. Bu araç ekrandaki
  her çubuğu okur, saniyelik ham veriyi geri getiremez. Ham veri için Sağlık uygulamasından XML dışa aktarımı gerekir.
- Nabız çözünürlüğü yaklaşık ±1 BPM, zaman çözünürlüğü yaklaşık ±10 sn'dir.
- "Grafikten tahmin" bölge süreleri yaklaşıktır; Apple'ın OCR ile okunan bölge süreleri esas alınmalıdır.

## Test

```bash
pytest -q
```

`tests/test_synthetic.py` bilinen değerlerle çizilen sentetik bir grafikle uçtan uca doğrulama yapar,
`tests/test_api.py` API'yi test eder.
Kendi ekran görüntülerini `tests/fixtures/` klasörüne koyarsan gerçek görüntü testi de çalışır
(bu klasördeki görüntüler `.gitignore` ile repodan hariç tutulur).

## Lisans

MIT
