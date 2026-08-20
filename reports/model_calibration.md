# Model calibration — MLP / LSTM theo EAR

Sinh tự động bởi `tools/model_calibration.py` lúc 2026-08-20 11:15 (load_mode=`weights`).

> **Đây KHÔNG phải evaluation.** Không có nhãn người thật ở đây — chỉ là
> phép quét feature tổng hợp để theo dõi hành vi model. Đánh giá đúng
> nghĩa cần test set có nhãn (vấn đề C3).

Ngưỡng tham chiếu: `EYE_CLOSED_THRESH = 0.16` (mắt nhắm), `HYSTERESIS_ON = 0.65` (bật cảnh báo).

## Có pose vai (neck_tilt hợp lệ)

| EAR | Trạng thái mắt | p_mlp_drowsy | p_lstm_drowsy | Vượt HYSTERESIS_ON? |
|---|---|---|---|---|
| 0.08 | nhắm | 0.950 | 0.601 | ⚠️ CÓ |
| 0.10 | nhắm | 0.931 | 0.600 | ⚠️ CÓ |
| 0.12 | nhắm | 0.903 | 0.598 | ⚠️ CÓ |
| 0.14 | nhắm | 0.864 | 0.595 | ⚠️ CÓ |
| 0.16 | mở | 0.793 | 0.590 | ⚠️ CÓ |
| 0.18 | mở | 0.689 | 0.583 | ⚠️ CÓ |
| 0.20 | mở | 0.560 | 0.576 | không |
| 0.25 | mở | 0.249 | 0.557 | không |
| 0.30 | mở | 0.148 | 0.543 | không |
| 0.35 | mở | 0.143 | 0.534 | không |

## Mất pose vai (neck_tilt = NaN)

_Đây là trường hợp gây bug C2 trước khi sửa._

| EAR | Trạng thái mắt | p_mlp_drowsy | p_lstm_drowsy | Vượt HYSTERESIS_ON? |
|---|---|---|---|---|
| 0.08 | nhắm | 0.951 | 0.596 | ⚠️ CÓ |
| 0.10 | nhắm | 0.931 | 0.595 | ⚠️ CÓ |
| 0.12 | nhắm | 0.904 | 0.593 | ⚠️ CÓ |
| 0.14 | nhắm | 0.866 | 0.589 | ⚠️ CÓ |
| 0.16 | mở | 0.806 | 0.583 | ⚠️ CÓ |
| 0.18 | mở | 0.719 | 0.576 | ⚠️ CÓ |
| 0.20 | mở | 0.602 | 0.568 | không |
| 0.25 | mở | 0.293 | 0.549 | không |
| 0.30 | mở | 0.151 | 0.536 | không |
| 0.35 | mở | 0.140 | 0.529 | không |

## Nhận xét tự động

- Mắt mở rõ (EAR ≥ 0.25): `p_mlp_drowsy` cao nhất **0.249** (EAR=0.25); `p_lstm_drowsy` cao nhất **0.557** (EAR=0.25).
- MLP không có bias rõ rệt ở dải mắt mở.
- ⚠️ **LSTM** vượt ngưỡng FATIGUE (0.4) DÙ MẮT ĐANG MỞ.
- MLP phản ứng theo EAR với biên độ 0.808. OK.
- ⚠️ **LSTM gần như KHÔNG phản ứng với EAR** (biên độ chỉ 0.067 trên toàn dải 0.08→0.35). Model này không đóng góp tín hiệu hữu ích; `src/fusion.py` đã có guard bỏ qua LSTM khi nó lệch MLP > 0.15, nhưng nó vẫn kéo `drowsiness_score` lên nền cao.

**Cần retrain (C3) mới xử lý được gốc.** Trong lúc chờ, chạy `FORCE_RULE_ONLY=true` để vận hành thuần rule engine.
