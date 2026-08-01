import os
import sys

# Fix encoding cho Windows terminal (tránh UnicodeEncodeError với emoji)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import tempfile
import shutil
import time
import cv2
import numpy as np
import gradio as gr
from pipeline.pipeline import PedestrianCVPipeline

# === THIẾT LẬP THƯ MỤC TẠM ===
WORK_DIR = os.path.join(os.getcwd(), "temp_uploads")
os.makedirs(WORK_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = WORK_DIR
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["GRADIO_MAX_FILE_SIZE"] = "2000"

print(f"📁 Thư mục làm việc: {WORK_DIR}")

pipeline = PedestrianCVPipeline()

def process_video(video_file):
    if video_file is None:
        return None
    try:
        # Reset các stage
        for stage in pipeline.video_pipeline.stages:
            if hasattr(stage, 'reset'):
                stage.reset()
        
        # Lấy đường dẫn file từ Gradio (file đã được upload)
        input_path = video_file
        print(f"📹 File gốc: {input_path}")
        
        # Copy file vào thư mục làm việc để tránh lỗi permission
        file_name = os.path.basename(input_path)
        safe_path = os.path.join(WORK_DIR, f"input_{file_name}")
        shutil.copy2(input_path, safe_path)
        print(f"📹 Đã copy vào: {safe_path}")
        
        # Tạo output path
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}_processed.mp4")
        
        # Xóa output cũ nếu có
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        
        # Chạy pipeline
        pipeline.run_on_video(safe_path, output_path)
        
        if not os.path.exists(output_path):
            print("❌ Không tạo được video output.")
            return None
        
        return output_path
    except Exception as e:
        print(f"❌ LỖI: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_image(image):
    if image is None:
        return None
    start = time.time()
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    frame_out = pipeline.run_on_image(frame)
    frame_out_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
    print(f"⏱️ Ảnh xử lý trong {time.time()-start:.2f}s")
    return frame_out_rgb

# Giao diện Gradio
with gr.Blocks(title="Vehicle Warning System") as demo:
    gr.Markdown("""
    # 🚦 Hệ thống Cảnh báo Nguy hiểm cho Phương tiện
    **Upload video hoặc ảnh** để phát hiện và dự đoán nguy cơ.
    """)

    with gr.Tab("📹 Xử lý Video"):
        with gr.Row():
            video_input = gr.Video(label="📂 Upload video", height=300)
            video_output = gr.Video(label="📊 Kết quả sau xử lý", height=300)
        video_btn = gr.Button("▶️ Bắt đầu xử lý", variant="primary", size="lg")
        video_btn.click(
            fn=process_video,
            inputs=video_input,
            outputs=video_output
        )
        gr.Markdown("💡 **Hỗ trợ:** MP4, AVI, MOV (tối đa 2GB)")

    with gr.Tab("🖼️ Xử lý Ảnh"):
        with gr.Row():
            image_input = gr.Image(label="📂 Upload ảnh", type="numpy")
            image_output = gr.Image(label="📊 Kết quả sau xử lý")
        image_btn = gr.Button("▶️ Bắt đầu xử lý", variant="primary", size="lg")
        image_btn.click(
            fn=process_image,
            inputs=image_input,
            outputs=image_output
        )
        gr.Markdown("💡 **Hỗ trợ:** JPG, PNG, WEBP")

if __name__ == "__main__":
    demo.launch(
        share=False,
        max_file_size="2gb",
        server_name="127.0.0.1",
        server_port=7860
    )