# Dataset — GuardianPilot 360

Cấu trúc theo PRD mục 10.4. Đặt dữ liệu thô / đã gán nhãn vào đây (không commit file lớn).

```
data/
├── face_eye/
│   ├── train/
│   └── test/
├── yawn/
│   ├── train/
│   └── test/
├── head_pose/
│   ├── train/
│   └── test/
├── drowsiness_video/
│   ├── train/
│   └── test/
├── snapshots/          # runtime JPEG từ Event Logger (gitignored)
└── events.db           # runtime SQLite (gitignored)
```

## Annotation guideline (D1–D6)

| Code | Nội dung | Ghi chú |
|------|----------|---------|
| D1 | Face bbox / visibility | Có/không mặt, bbox nếu có |
| D2 | Eye state | open / closed / partial |
| D3 | Yawn | none / yawn (MAR + duration) |
| D4 | Head pose / nod | pitch/yaw/roll hoặc nod event |
| D5 | Drowsiness label | NORMAL / FATIGUE / DROWSY / MICROSLEEP / CRITICAL |
| D6 | PERCLOS window | optional meta cho clip 30s |

## Naming

- Ảnh: `{subject_id}_{seq}_{frame}.jpg`
- Label: cùng tên `.json` hoặc CSV chung thư mục `labels.csv`
- Video: `{subject_id}_{scenario}.mp4` + `{subject_id}_{scenario}.json` (timeline)

## Lưu ý

- Không commit video/ảnh lớn vào git (xem `.gitignore`)
- Snapshot runtime nằm ở `data/snapshots/`, chỉ dùng để debug/event log
