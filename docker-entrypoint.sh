#!/bin/bash
# docker-entrypoint.sh
# ---------------------
# Xử lý MODEL_DIR và symlink các thư mục model về đúng vị trí
# mà system.py -> from_model_dir() mong đợi.
#
# Cấu trúc volume mount mong đợi (/app/models/):
#   task_1/
#     dcnn_drowsiness_task1_baseline.keras
#   Task 2 — .../
#     cnn_16s_best.keras
#   Task_3/
#     dbmnet_full_task3.keras
#     baseline_ghostnetlike_task3.keras
#   Task_4/
#     lstm_landmark_task4_fixed.keras
#     mlp_landmark_task4_fixed.keras
#     landmark_scaler_task4.pkl
#     face_landmarker.task

set -e

MODEL_DIR="${MODEL_DIR:-/app/models}"
WORK_DIR="/app"
CMD="${1:-test}"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║       GuardianPilot 360 — Docker Container           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  MODEL_DIR : $MODEL_DIR"
echo "  Command   : $CMD"
echo ""

# ── Symlink model directories ─────────────────────────────────────────────────
setup_models() {
    if [ ! -d "$MODEL_DIR" ]; then
        echo "⚠  WARNING: MODEL_DIR '$MODEL_DIR' không tồn tại."
        echo "   Mount volume: -v /path/to/Model:/app/models"
        return 0
    fi

    # task_1, Task_3, Task_4 — tên cố định
    for dir in "task_1" "Task_3" "Task_4"; do
        src="$MODEL_DIR/$dir"
        dst="$WORK_DIR/$dir"
        if [ -d "$src" ] && [ ! -e "$dst" ]; then
            ln -s "$src" "$dst"
            echo "  ✓ Linked: $dir"
        fi
    done

    # Task 2 — tên thư mục chứa dấu "—" đặc biệt
    TASK2_SRC=$(find "$MODEL_DIR" -maxdepth 1 -type d -name "Task 2*" 2>/dev/null | head -1)
    if [ -n "$TASK2_SRC" ]; then
        TASK2_NAME=$(basename "$TASK2_SRC")
        TASK2_DST="$WORK_DIR/$TASK2_NAME"
        if [ ! -e "$TASK2_DST" ]; then
            ln -s "$TASK2_SRC" "$TASK2_DST"
            echo "  ✓ Linked: Task 2"
        fi
    fi

    echo "  ✓ Models ready."
}

# Tạo thư mục logs nếu chưa có
mkdir -p "$(dirname "${AUDIT_LOG:-/tmp/gp_audit.log}")" 2>/dev/null || true

# ── Dispatch theo command ─────────────────────────────────────────────────────
case "$CMD" in

    test)
        echo "▶  Running unit tests..."
        echo ""
        exec python -m pytest tests/ -v --tb=short --no-header
        ;;

    video)
        setup_models
        shift   # bỏ chữ "video" khỏi "$@"
        VIDEO="${VIDEO_PATH:-/data/input.mp4}"
        FPS="${TARGET_FPS:-15}"
        AUDIT="${AUDIT_LOG:-/tmp/guardian_pilot_audit.log}"
        EEG_FLAG=""
        [ "${SENSOR_EEG:-0}" = "1" ] && EEG_FLAG="--eeg"
        echo "▶  Video: $VIDEO (fps=$FPS)"
        exec python run.py \
            --video     "$VIDEO" \
            --fps       "$FPS" \
            --audit-log "$AUDIT" \
            --no-display \
            $EEG_FLAG \
            "$@"
        ;;

    camera)
        setup_models
        shift
        CAM="${CAMERA_INDEX:-0}"
        FPS="${TARGET_FPS:-15}"
        AUDIT="${AUDIT_LOG:-/tmp/guardian_pilot_audit.log}"
        EEG_FLAG=""
        [ "${SENSOR_EEG:-0}" = "1" ] && EEG_FLAG="--eeg"
        echo "▶  Camera index=$CAM (fps=$FPS)"
        exec python run.py \
            --camera    "$CAM" \
            --fps       "$FPS" \
            --audit-log "$AUDIT" \
            $EEG_FLAG \
            "$@"
        ;;

    shell)
        echo "▶  Interactive shell"
        exec /bin/bash
        ;;

    *)
        # Cho phép override hoàn toàn: docker run ... python run.py --help
        exec "$@"
        ;;
esac
