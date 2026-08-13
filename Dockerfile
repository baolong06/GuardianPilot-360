# GuardianPilot — CPU development image (Flask DMS API)
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV SAVE_FACE_SNAPSHOTS=false
ENV MEDIAPIPE_DISABLE_GPU=1
ENV PORT=5000
EXPOSE 5000

# Render injects $PORT; default 5000 for local/docker-compose.
CMD ["sh", "-c", "python app.py --host 0.0.0.0 --port ${PORT:-5000}"]
