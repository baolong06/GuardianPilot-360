# import cv2
# import torch
# import numpy as np
# from collections import deque, Counter
# from pipeline.core.base import Stage
# import logging
# from torchvision import transforms

# logger = logging.getLogger(__name__)


# class TrafficLightStage(Stage):
#     def __init__(self, history_len=7, expand_ratio=0.10):
#         self.history_len = history_len
#         self.expand_ratio = expand_ratio
#         self.history = {}
#         self.last_seen = {}
#         self.frame_count = 0
        
#         # Load ML model
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.use_ml = True
#         if self.use_ml:
#             try:
#                 from models.traffic_light.model import TrafficLightClassifier
#                 self.ml_model = TrafficLightClassifier(num_classes=3)
#                 self.ml_model.load_state_dict(
#                     torch.load("models/traffic_light/weights/best.pth", map_location=self.device)
#                 )
#                 self.ml_model.to(self.device)
#                 self.ml_model.eval()
#                 self.transform = transforms.Compose([
#                     transforms.ToPILImage(),
#                     transforms.Resize((128, 128)),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                        std=[0.229, 0.224, 0.225])
#                 ])
#                 print("✅ Traffic Light ML model loaded")
#             except Exception as e:
#                 print(f"⚠️ Failed to load ML model: {e}, falling back to HSV")
#                 self.use_ml = False
        
#         self.SMALL_BOX_THRESH = 1500
#         self.LARGE_BOX_THRESH = 4000

#     # ── HSV fallback methods ──────────────────────────────────────────
#     def _color_masks(self, hsv, bright_mask):
#         hue = hsv[:, :, 0]
#         sat = hsv[:, :, 1]
#         red_mask    = ((hue < 15) | (hue > 160)) & bright_mask & (sat > 80)
#         yellow_mask = ((hue >= 15) & (hue < 40)) & bright_mask & (sat > 60)
#         green_mask  = ((hue >= 40) & (hue < 95)) & bright_mask & (sat > 50)
#         return int(np.sum(red_mask)), int(np.sum(yellow_mask)), int(np.sum(green_mask))

#     def analyze_region(self, region):
#         if region is None or region.size == 0:
#             return 'unknown', 0.0, 0.0
#         try:
#             hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
#         except Exception:
#             return 'unknown', 0.0, 0.0

#         val    = hsv[:, :, 2]
#         mean_v = float(np.mean(val))
#         bright_mask = (val > 90) & (hsv[:, :, 1] > 50)
#         n_bright    = int(np.sum(bright_mask))
#         if n_bright < 15:
#             return 'unknown', 0.0, mean_v

#         red_px, yellow_px, green_px = self._color_masks(hsv, bright_mask)
#         total = n_bright

#         red_r    = red_px    / total
#         yellow_r = yellow_px / total
#         green_r  = green_px  / total

#         RED_THRESH    = 0.18
#         YELLOW_THRESH = 0.22
#         GREEN_THRESH  = 0.15

#         if red_r >= RED_THRESH and red_r >= green_r and red_r >= yellow_r:
#             return 'red', red_r, mean_v
#         if yellow_r >= YELLOW_THRESH and yellow_r > red_r and yellow_r >= green_r:
#             return 'yellow', yellow_r, mean_v
#         if green_r >= GREEN_THRESH:
#             return 'green', green_r, mean_v
#         return 'unknown', 0.0, mean_v

#     # ── ML classification ─────────────────────────────────────────────
#     def classify_with_ml(self, crop):
#         """Phân loại màu đèn bằng ML model"""
#         try:
#             img = self.transform(crop).unsqueeze(0).to(self.device)
#             with torch.no_grad():
#                 output = self.ml_model(img)
#                 probs = torch.softmax(output, dim=1).cpu().numpy()[0]
#             label_map = {0: 'red', 1: 'yellow', 2: 'green'}
#             pred = np.argmax(probs)
#             return label_map[pred], probs[pred]
#         except Exception as e:
#             logger.warning(f"ML classification failed: {e}, falling back to HSV")
#             return None, 0.0

#     # ── Classify traffic light ────────────────────────────────────────
#     def classify_traffic_light(self, crop, box_area):
#         h, w = crop.shape[:2]
#         if h < 20 or w < 10:
#             return 'unknown', 0.0
        
#         # Ưu tiên ML nếu khả dụng
#         if self.use_ml:
#             state, conf = self.classify_with_ml(crop)
#             if state is not None and conf > 0.7:
#                 return state, conf
        
#         # Fallback: HSV
#         if box_area < self.SMALL_BOX_THRESH:
#             return self._classify_small(crop)
#         if box_area > self.LARGE_BOX_THRESH:
#             return self._classify_large(crop)
#         s1, c1 = self._classify_small(crop)
#         s2, c2 = self._classify_large(crop)
#         if s1 == s2:
#             return s1, max(c1, c2)
#         return (s1, c1) if c1 >= c2 else (s2, c2)

#     def _classify_small(self, crop):
#         state, ratio, v = self.analyze_region(crop)
#         if state != 'unknown' and v > 70:
#             return state, ratio
#         return 'unknown', 0.0

#     def _classify_large(self, crop):
#         h, w = crop.shape[:2]
#         top_frac  = 0.38
#         bot_start = 0.62

#         top = crop[0 : int(h * top_frac), :]
#         mid = crop[int(h * top_frac) : int(h * bot_start), :]
#         bot = crop[int(h * bot_start):, :]

#         top_s, top_r, top_v = self.analyze_region(top)
#         mid_s, mid_r, mid_v = self.analyze_region(mid)
#         bot_s, bot_r, bot_v = self.analyze_region(bot)

#         # GREEN bot first
#         if bot_s == 'green' and bot_v > 70 and bot_r > 0.15:
#             return 'green', bot_r

#         # RED top with higher threshold
#         if top_s == 'red' and top_v > 80 and top_r > 0.28:
#             return 'red', top_r

#         if mid_s == 'yellow' and mid_v > 80 and mid_r > 0.20:
#             return 'yellow', mid_r

#         candidates = [
#             (top_s, top_r, top_v),
#             (mid_s, mid_r, mid_v),
#             (bot_s, bot_r, bot_v),
#         ]
#         valid = [(s, r, v) for s, r, v in candidates if s != 'unknown' and v > 70]
#         if valid:
#             best = max(valid, key=lambda x: x[2])
#             return best[0], best[1]

#         return self._classify_small(crop)

#     def get_color_counts(self, crop):
#         if crop.size == 0:
#             return 0, 0, 0
#         hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#         mask = (hsv[:, :, 2] > 90) & (hsv[:, :, 1] > 50)
#         return self._color_masks(hsv, mask)

#     def _cleanup_stale_tracks(self):
#         timeout  = 30
#         stale    = [t for t, last in self.last_seen.items()
#                     if self.frame_count - last > timeout]
#         for t in stale:
#             self.history.pop(t, None)
#             self.last_seen.pop(t, None)

#     def process(self, data):
#         frame = data.get("frame")
#         if frame is None:
#             return data

#         self.frame_count += 1
#         self._cleanup_stale_tracks()

#         detections     = data.get("detections",  [])
#         class_names    = data.get("class_names", [])
#         track_ids      = data.get("track_ids",   [])
#         traffic_lights = {}

#         for i, class_name in enumerate(class_names):
#             if class_name != 'traffic light' or i >= len(detections):
#                 continue
#             try:
#                 bbox     = detections[i]
#                 tid      = int(track_ids[i]) if i < len(track_ids) else i
#                 x1, y1, x2, y2 = map(int, bbox)
#                 w_box    = x2 - x1
#                 h_box    = y2 - y1
#                 box_area = w_box * h_box

#                 if box_area < 150 or box_area > 20000:
#                     continue
#                 if w_box == 0 or h_box / float(w_box) < 0.5:
#                     continue

#                 pad_w = max(2, int(w_box * self.expand_ratio))
#                 pad_h = max(2, int(h_box * self.expand_ratio))
#                 x1e = max(0, x1 - pad_w)
#                 y1e = max(0, y1 - pad_h)
#                 x2e = min(frame.shape[1], x2 + pad_w)
#                 y2e = min(frame.shape[0], y2 + pad_h)

#                 crop = frame[y1e:y2e, x1e:x2e]
#                 if crop.size == 0:
#                     continue

#                 r, y, g = self.get_color_counts(crop)
#                 logger.debug(f"TL bbox_area={box_area} R={r} G={g} Y={y}")

#                 state, confidence = self.classify_traffic_light(crop, box_area)

#                 if tid not in self.history:
#                     self.history[tid] = deque(maxlen=self.history_len)
#                 self.history[tid].append(state)
#                 self.last_seen[tid] = self.frame_count

#                 history_list = list(self.history[tid])
#                 non_unknown  = [s for s in history_list if s != 'unknown']

#                 if non_unknown:
#                     final_state = Counter(non_unknown).most_common(1)[0][0]
#                     final_count = sum(1 for s in history_list if s == final_state)
#                     avg_conf    = final_count / len(history_list)
#                 else:
#                     final_state = 'unknown'
#                     avg_conf    = 0.0

#                 traffic_lights[tid] = {'state': final_state, 'confidence': avg_conf}
#                 print(f"TL {tid}: {final_state} ({avg_conf*100:.0f}%)")

#             except Exception as e:
#                 logger.error(f"Error TL detection {i}: {e}", exc_info=True)

#         data["traffic_lights"] = traffic_lights
#         return data


import cv2
import torch
import numpy as np
from collections import deque, Counter
from pipeline.core.base import Stage
import logging
from torchvision import transforms
import os

logger = logging.getLogger(__name__)


class TrafficLightStage(Stage):
    def __init__(self, history_len=7, expand_ratio=0.10):
        self.history_len = history_len
        self.expand_ratio = expand_ratio
        self.history = {}
        self.last_seen = {}
        self.frame_count = 0
        
        # Load ML model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ml = True
        if self.use_ml:
            try:
                from models.traffic_light.model import TrafficLightClassifier
                self.ml_model = TrafficLightClassifier(num_classes=3)
                model_path = "models/traffic_light/weights/best.pth"
                if os.path.exists(model_path):
                    print(f"Loading Traffic Light model ({os.path.getsize(model_path)//1024//1024}MB)...")
                    # weights_only=False bắt buộc vì checkpoint có thể chứa object Python
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                    self.ml_model.load_state_dict(state_dict, strict=True)
                    self.ml_model.to(self.device)
                    self.ml_model.eval()
                    self.transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((128, 128)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
                    ])
                    print("[OK] Traffic Light ResNet18 ML model loaded successfully")
                    logger.info("Traffic Light ML model loaded (strict=True)")
                else:
                    logger.warning(f"Weights not found at {model_path}, using HSV fallback")
                    self.use_ml = False
            except Exception as e:
                logger.warning(f"Failed to load Traffic Light ML model: {e}, using HSV fallback")
                self.use_ml = False

        
        self.SMALL_BOX_THRESH = 1500
        self.LARGE_BOX_THRESH = 4000

    # ── HSV fallback methods ──────────────────────────────────────────
    def _color_masks(self, hsv, bright_mask):
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        red_mask    = ((hue < 15) | (hue > 160)) & bright_mask & (sat > 100)
        yellow_mask = ((hue >= 15) & (hue < 38)) & bright_mask & (sat > 80)
        # FIX BUG: thu hẹp dải xanh (60–92 thay vì 40–95) tránh lẫn tán lá/màu trời
        green_mask  = ((hue >= 60) & (hue < 92)) & bright_mask & (sat > 80)
        return int(np.sum(red_mask)), int(np.sum(yellow_mask)), int(np.sum(green_mask))

    def analyze_region(self, region):
        if region is None or region.size == 0:
            return 'unknown', 0.0, 0.0
        try:
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        except Exception:
            return 'unknown', 0.0, 0.0

        val    = hsv[:, :, 2]
        mean_v = float(np.mean(val))
        bright_mask = (val > 90) & (hsv[:, :, 1] > 50)
        n_bright    = int(np.sum(bright_mask))
        if n_bright < 15:
            return 'unknown', 0.0, mean_v

        red_px, yellow_px, green_px = self._color_masks(hsv, bright_mask)
        total = n_bright

        red_r    = red_px    / total
        yellow_r = yellow_px / total
        green_r  = green_px  / total

        RED_THRESH    = 0.18
        YELLOW_THRESH = 0.22
        GREEN_THRESH  = 0.15

        if red_r >= RED_THRESH and red_r >= green_r and red_r >= yellow_r:
            return 'red', red_r, mean_v
        if yellow_r >= YELLOW_THRESH and yellow_r > red_r and yellow_r >= green_r:
            return 'yellow', yellow_r, mean_v
        if green_r >= GREEN_THRESH:
            return 'green', green_r, mean_v
        return 'unknown', 0.0, mean_v

    # ── ML classification ─────────────────────────────────────────────
    def classify_with_ml(self, crop):
        """Phân loại màu đèn bằng ML model"""
        try:
            img = self.transform(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.ml_model(img)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            label_map = {0: 'red', 1: 'yellow', 2: 'green'}
            pred = np.argmax(probs)
            return label_map[pred], probs[pred]
        except Exception as e:
            logger.warning(f"ML classification failed: {e}, falling back to HSV")
            return None, 0.0

    # ── Classify traffic light ────────────────────────────────────────
    def classify_traffic_light(self, crop, box_area):
        h, w = crop.shape[:2]
        if h < 20 or w < 10:
            return 'unknown', 0.0
        
        # Ưu tiên ML nếu khả dụng
        if self.use_ml:
            state, conf = self.classify_with_ml(crop)
            if state is not None and conf > 0.7:
                return state, conf
        
        # Fallback: HSV
        if box_area < self.SMALL_BOX_THRESH:
            return self._classify_small(crop)
        if box_area > self.LARGE_BOX_THRESH:
            return self._classify_large(crop)
        s1, c1 = self._classify_small(crop)
        s2, c2 = self._classify_large(crop)
        if s1 == s2:
            return s1, max(c1, c2)
        return (s1, c1) if c1 >= c2 else (s2, c2)

    def _classify_small(self, crop):
        state, ratio, v = self.analyze_region(crop)
        if state != 'unknown' and v > 70:
            return state, ratio
        return 'unknown', 0.0

    def _classify_large(self, crop):
        h, w = crop.shape[:2]
        top_frac  = 0.38
        bot_start = 0.62

        top = crop[0 : int(h * top_frac), :]
        mid = crop[int(h * top_frac) : int(h * bot_start), :]
        bot = crop[int(h * bot_start):, :]

        top_s, top_r, top_v = self.analyze_region(top)
        mid_s, mid_r, mid_v = self.analyze_region(mid)
        bot_s, bot_r, bot_v = self.analyze_region(bot)

        # GREEN bot first — tăng threshold tránh nhiễu lá cây
        if bot_s == 'green' and bot_v > 80 and bot_r > 0.25:
            return 'green', bot_r

        # FIX BUG: Tăng threshold RED (0.28 → 0.40) tránh khung vỏ đèn/biển báo
        if top_s == 'red' and top_v > 90 and top_r > 0.40:
            return 'red', top_r

        if mid_s == 'yellow' and mid_v > 80 and mid_r > 0.20:
            return 'yellow', mid_r

        candidates = [
            (top_s, top_r, top_v),
            (mid_s, mid_r, mid_v),
            (bot_s, bot_r, bot_v),
        ]
        valid = [(s, r, v) for s, r, v in candidates if s != 'unknown' and v > 70]
        if valid:
            best = max(valid, key=lambda x: x[2])
            return best[0], best[1]

        return self._classify_small(crop)

    def get_color_counts(self, crop):
        if crop.size == 0:
            return 0, 0, 0
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 2] > 90) & (hsv[:, :, 1] > 50)
        return self._color_masks(hsv, mask)

    def _cleanup_stale_tracks(self):
        timeout  = 30
        stale    = [t for t, last in self.last_seen.items()
                    if self.frame_count - last > timeout]
        for t in stale:
            self.history.pop(t, None)
            self.last_seen.pop(t, None)

    def process(self, data):
        frame = data.get("frame")
        if frame is None:
            return data

        self.frame_count += 1
        self._cleanup_stale_tracks()

        detections     = data.get("detections",  [])
        class_names    = data.get("class_names", [])
        track_ids      = data.get("track_ids",   [])
        traffic_lights = {}

        for i, class_name in enumerate(class_names):
            if class_name != 'traffic light' or i >= len(detections):
                continue
            try:
                bbox     = detections[i]
                tid      = int(track_ids[i]) if i < len(track_ids) else i
                x1, y1, x2, y2 = map(int, bbox)
                w_box    = x2 - x1
                h_box    = y2 - y1
                box_area = w_box * h_box

                if box_area < 150 or box_area > 20000:
                    continue
                if w_box == 0 or h_box / float(w_box) < 0.5:
                    continue

                pad_w = max(2, int(w_box * self.expand_ratio))
                pad_h = max(2, int(h_box * self.expand_ratio))
                x1e = max(0, x1 - pad_w)
                y1e = max(0, y1 - pad_h)
                x2e = min(frame.shape[1], x2 + pad_w)
                y2e = min(frame.shape[0], y2 + pad_h)

                crop = frame[y1e:y2e, x1e:x2e]
                if crop.size == 0:
                    continue

                r, y, g = self.get_color_counts(crop)
                logger.debug(f"TL bbox_area={box_area} R={r} G={g} Y={y}")

                state, confidence = self.classify_traffic_light(crop, box_area)

                if tid not in self.history:
                    self.history[tid] = deque(maxlen=self.history_len)
                self.history[tid].append(state)
                self.last_seen[tid] = self.frame_count

                history_list = list(self.history[tid])
                non_unknown  = [s for s in history_list if s != 'unknown']

                # FIX BUG: Yêu cầu ít nhất 3 frame hợp lệ để tránh nhiễm bởi 1-2 frame đầu sai
                if len(non_unknown) >= 3:
                    final_state = Counter(non_unknown).most_common(1)[0][0]
                    final_count = sum(1 for s in history_list if s == final_state)
                    avg_conf    = final_count / len(history_list)
                else:
                    final_state = 'unknown'
                    avg_conf    = 0.0

                traffic_lights[tid] = {'state': final_state, 'confidence': avg_conf}
                # Sử dụng logger.debug thay vì print để không flood console
                if final_state != 'unknown':
                    logger.debug(f"TL {tid}: {final_state} ({avg_conf*100:.0f}%)")

            except Exception as e:
                logger.error(f"Error TL detection {i}: {e}", exc_info=True)

        data["traffic_lights"] = traffic_lights
        return data