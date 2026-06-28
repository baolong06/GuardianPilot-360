# ─────────────────────────────────────────────────────────────────────────────
# GuardianPilot 360 — Dockerfile (multi-stage)
# ─────────────────────────────────────────────────────────────────────────────
# Build: docker build -t guardian-pilot:latest .
# Test:  docker run --rm guardian-pilot:latest test
# Video: docker run --rm -v "$PWD:/app/models" -v "$PWD/data:/data" \
#               -e VIDEO_PATH=/data/clip.mp4 guardian-pilot:latest video
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — builder: cài pip, build wheels, KHÔNG có trong image cuối
# ══════════════════════════════════════════════════════════════════════════════
FROM tensorflow/tensorflow:2.13.0 AS builder

WORKDIR /build

# Build tools cần cho mediapipe native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker layer cache.
# Nếu requirements.txt không đổi → layer này không rebuild.
COPY requirements.txt .

# Build tất cả thành wheel files → sẽ copy sang stage runtime
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — runtime: image gọn gàng, chỉ chứa những gì cần để chạy
# ══════════════════════════════════════════════════════════════════════════════
FROM tensorflow/tensorflow:2.13.0 AS runtime

LABEL maintainer="GuardianPilot Team"
LABEL description="GuardianPilot 360 — DMS Multi-Agent System"
LABEL version="1.0.0-mvp"

# Runtime OS libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OpenCV headless runtime
        libgl1-mesa-glx \
        libglib2.0-0 \
        # OpenMP (numpy, TF parallel)
        libgomp1 \
        # OpenCV display libs (cần khi X11 forwarding)
        libsm6 \
        libxext6 \
        libxrender-dev \
        # Decode video files
        ffmpeg \
        # find command dùng trong entrypoint
        findutils \
    && rm -rf /var/lib/apt/lists/*

# Cài wheels từ builder — không cần internet, không cần pip install lại
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels \
        networkx \
        scikit-learn \
        mediapipe \
        "opencv-python-headless>=4.8.0" \
        pytest \
    && rm -rf /wheels

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# ── Copy source code (không copy model files — mount qua volume) ──────────────
COPY guardian_pilot/  ./guardian_pilot/
COPY run.py           ./run.py
COPY tests/           ./tests/

# ── Entrypoint script ─────────────────────────────────────────────────────────
COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Tạo thư mục cho volume mounts
RUN mkdir -p /app/models /data /audit \
    && chown -R appuser:appuser /app /data /audit

USER appuser

# ── Environment defaults ──────────────────────────────────────────────────────
# Giảm TF log spam
ENV TF_CPP_MIN_LOG_LEVEL=2
# Tắt GPU cho mediapipe (dùng CPU pipeline)
ENV MEDIAPIPE_DISABLE_GPU=1
# Đường dẫn models (override bằng -e MODEL_DIR=...)
ENV MODEL_DIR=/app/models
# Video path mặc định
ENV VIDEO_PATH=/data/input.mp4
# FPS mặc định
ENV TARGET_FPS=15
# Audit log
ENV AUDIT_LOG=/audit/guardian_pilot_audit.log
# Camera index
ENV CAMERA_INDEX=0

# ── Expose nothing (không phải web server) ────────────────────────────────────

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command: chạy unit tests
CMD ["test"]
