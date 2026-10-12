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

Tesseract gerekli:

```bash
# macOS
brew install tesseract
# Ubuntu / Debian
sudo apt install tesseract-ocr
```

```bash
git clone https://github.com/<kullanici>/apple-watch-hr-extractor.git
cd apple-watch-hr-extractor
pip install -r requirements.txt
```

## Kullanım

### Web arayüzü

```bash
streamlit run app.py
```

Tarayıcıda açılan sayfaya bir veya birden çok ekran görüntüsünü sürükleyin. Birden çok dosya
yüklenirse hepsi tek Excel dosyasında (özet + dosya başına sayfalar) indirilebilir.

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

```bash
uvicorn api:app --port 8000
curl -F "file=@ekran.jpg" http://localhost:8000/extract          # JSON
curl -F "file=@ekran.jpg" http://localhost:8000/extract.xlsx -o nabiz.xlsx
```

`HR_API_KEY` ortam değişkeni tanımlanırsa her istekte `X-API-Key` başlığı zorunlu olur.

## Yayına alma (SaaS)

- **Streamlit Community Cloud:** repoyu bağla, ana dosya `app.py`. `packages.txt` Tesseract'ı otomatik kurar.
- **Docker / kendi sunucun:**
  ```bash
  docker compose up -d --build    # arayüz :8501, API :8000
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

`tests/test_synthetic.py` bilinen değerlerle çizilen sentetik bir grafikle uçtan uca doğrulama yapar.
Kendi ekran görüntülerini `tests/fixtures/` klasörüne koyarsan gerçek görüntü testi de çalışır
(bu klasördeki görüntüler `.gitignore` ile repodan hariç tutulur).

## Lisans

MIT
