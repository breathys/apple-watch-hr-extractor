FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT=8501
EXPOSE 8501 8000
# Varsayılan: web arayüzü. API için:  docker run -p 8000:8000 IMAGE uvicorn api:app --host 0.0.0.0 --port 8000
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
