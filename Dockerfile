FROM python:3.12-slim

# tesserocr Linux'ta Tesseract'ı kendi içinde getirir; libglib OpenCV için
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements-ui.txt
COPY . .

EXPOSE 8000 8501
# Varsayılan: web arayüzü + API (http://localhost:8000)
# Streamlit arayüzü için: docker run -p 8501:8501 IMAGE streamlit run streamlit_app.py --server.address=0.0.0.0
CMD ["uvicorn", "hr_extractor.web:app", "--host", "0.0.0.0", "--port", "8000"]
