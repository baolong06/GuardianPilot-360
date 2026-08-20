# Camera calibration & head pose

## Vấn đề (M9)

`src/landmarks.py::compute_head_pose()` dùng `cv2.solvePnP` để ước lượng
pitch/yaw/roll từ 6 điểm landmark trên khuôn mặt. solvePnP cần **ma trận nội
tại camera** (camera intrinsics). Trước đây giá trị này được hard-code:

```python
focal_length = float(img_w)      # giả định focal = chiều rộng ảnh
cx, cy       = img_w / 2, img_h / 2
dist_coef    = np.zeros((4, 1))  # giả định không méo ống kính
```

Đây là giả định "pinhole chưa calib" thường gặp trong demo. Nó **chạy được**,
nhưng có hai hệ quả mà cần biết rõ:

1. **Góc trả về là góc tương đối, không phải góc tuyệt đối.** Giá trị `pitch`
   khi ngồi thẳng có thể là −24° chứ không phải 0°. Vì vậy toàn bộ logic
   head-nod trong `src/fusion.py` dùng **delta so với baseline EMA**, không
   dùng giá trị thô — đó là thiết kế đúng cho tình huống chưa calib.

2. **Ngưỡng theo độ gắn chặt với ống kính đang dùng.** `yaw_thresh_deg = 25`
   (looking-away), `PITCH_NOD_PEAK_DEG = 12` được chỉnh trên một webcam cụ thể.
   Đổi camera (góc rộng hơn/hẹp hơn) là phải chỉnh lại.

## Cách nạp thông số calib thật

`src/camera.py` đọc các biến môi trường sau. **Không set gì cả → giữ nguyên
giả định cũ**, kết quả không đổi so với trước.

| Biến | Ý nghĩa | Ví dụ |
|---|---|---|
| `CAMERA_FOCAL_PX` | focal length (pixel), dùng cho cả fx và fy | `640` |
| `CAMERA_FOCAL_X` | focal trục x (ưu tiên hơn `CAMERA_FOCAL_PX`) | `638.2` |
| `CAMERA_FOCAL_Y` | focal trục y | `641.7` |
| `CAMERA_CX` | principal point x (pixel) | `319.5` |
| `CAMERA_CY` | principal point y (pixel) | `239.5` |
| `CAMERA_DIST_COEFFS` | `k1,k2,p1,p2[,k3]` | `0.12,-0.25,0.001,0.0,0.1` |

```bash
# Linux/macOS
export CAMERA_FOCAL_X=638.2 CAMERA_FOCAL_Y=641.7 CAMERA_CX=319.5 CAMERA_CY=239.5
python app.py

# Windows PowerShell
$env:CAMERA_FOCAL_X="638.2"; $env:CAMERA_CX="319.5"; python app.py
```

Kiểm tra đã nhận chưa:

```bash
curl http://127.0.0.1:5000/api/runtime-profile
# → "camera": { "calibrated": true, ... }
```

## Lấy thông số calib ở đâu

Cách chuẩn: chụp 15–20 ảnh bàn cờ (checkerboard) ở nhiều góc rồi chạy
`cv2.calibrateCamera`. OpenCV có sẵn tutorial *Camera Calibration*. Kết quả trả
về đúng `cameraMatrix` (chứa fx, fy, cx, cy) và `distCoeffs` cần điền ở trên.

**Lưu ý quan trọng:** intrinsics phụ thuộc **độ phân giải**. Nếu calib ở
1280×720 nhưng pipeline resize xuống 480×360 thì phải scale fx, fy, cx, cy theo
đúng tỷ lệ. `src/pipeline.py` resize mọi frame về `MEDIAPIPE_INPUT_SIZE`, nhưng
landmark trả về là toạ độ normalized nên `compute_head_pose` nhận **kích thước
ảnh gốc** — hãy calib theo độ phân giải gốc của camera.

## Sau khi calib

Chỉnh lại các ngưỡng theo độ, vì chúng được tune trên hệ chưa calib:

- `src/thresholds.py` → `yaw_thresh_deg`
- `src/fusion.py` → `PITCH_NOD_PEAK_DEG`, `PITCH_NOD_CURRENT_DEG`,
  `NECK_TILT_ALARM_DEG`

Có thể chỉnh `yaw_thresh_deg` lúc chạy qua `PUT /api/thresholds` mà không cần
restart.
