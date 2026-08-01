import cv2
import numpy as np
from pipeline.core.base import Stage
import logging

logger = logging.getLogger(__name__)

class EgoMotionStage(Stage):
    def __init__(self, method="sparse_optical_flow", max_resolution=720):
        """
        max_resolution: kích thước tối đa của cạnh để resize frame (tránh lỗi memory)
        """
        self.method = method
        self.max_resolution = max_resolution
        self.prev_gray = None
        self.prev_pts = None
        self.ego_dx = 0.0
        self.ego_dy = 0.0
        
        # Lucas-Kanade params
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Shi-Tomasi params for feature detection
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.ego_dx = 0.0
        self.ego_dy = 0.0

    def _resize_frame(self, frame):
        """Resize frame nếu quá lớn để tránh lỗi memory"""
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        scale = 1.0
        if max_dim > self.max_resolution:
            scale = self.max_resolution / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame, scale

    def process(self, data):
        frame = data.get("frame")
        if frame is None:
            data.setdefault("ego_dx", 0.0)
            data.setdefault("ego_dy", 0.0)
            return data

        try:
            # Resize frame nếu quá lớn
            frame_resized, scale = self._resize_frame(frame)
            h, w = frame_resized.shape[:2]
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logger.error(f"EgoMotionStage: Failed to process frame: {e}")
            data["ego_dx"] = 0.0
            data["ego_dy"] = 0.0
            return data

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            data["ego_dx"] = 0.0
            data["ego_dy"] = 0.0
            return data

        # Nếu mất điểm đặc trưng, tìm lại
        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)

        if self.prev_pts is not None and len(self.prev_pts) > 0:
            try:
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_pts, None, **self.lk_params
                )

                good_new = curr_pts[status == 1]
                good_old = self.prev_pts[status == 1]

                if len(good_new) > 5:
                    dx_pts = good_new[:, 0] - good_old[:, 0]
                    dy_pts = good_new[:, 1] - good_old[:, 1]
                    u_median = np.median(dx_pts)
                    v_median = np.median(dy_pts)
                    self.ego_dx = u_median / w
                    self.ego_dy = v_median / h
                else:
                    self.ego_dx = 0.0
                    self.ego_dy = 0.0

                self.prev_pts = good_new.reshape(-1, 1, 2)
            except Exception as e:
                logger.error(f"EgoMotionStage: Optical flow failed: {e}")
                self.ego_dx = 0.0
                self.ego_dy = 0.0
        else:
            self.ego_dx = 0.0
            self.ego_dy = 0.0

        data["ego_dx"] = self.ego_dx
        data["ego_dy"] = self.ego_dy

        self.prev_gray = gray
        return data