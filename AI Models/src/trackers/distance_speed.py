import numpy as np
import cv2
from collections import defaultdict
from src.config import PITCH_LENGTH, PITCH_WIDTH, RADAR_WIDTH, RADAR_HEIGHT

class SpeedDistanceTracker:
    def __init__(self, fps, homography_matrix=None, pixel_to_meter=0.02):
        self.fps = fps
        self.homography_matrix = homography_matrix
        self.pixel_to_meter_ratio = pixel_to_meter

        self.prev_positions = {}
        self.total_distance = defaultdict(float)
        self.speeds = {}
        self.top_speeds = defaultdict(float)

        # 🔥 history للـ smoothing
        self.speed_history = defaultdict(list)

    def convert_position(self, point, homography_matrix=None, dx=0, dy=0, radar_w=RADAR_WIDTH, radar_h=RADAR_HEIGHT):
        # 🔥 dynamic homography if provided (moving camera compensation)
        if homography_matrix is not None:
            x, y = point
            pt = np.array([[[x, y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, homography_matrix)
            rx = transformed[0][0][0] + dx
            ry = transformed[0][0][1] + dy
            px = (rx / float(radar_w)) * PITCH_LENGTH
            py = (ry / float(radar_h)) * PITCH_WIDTH
            return (float(np.clip(px, 0.0, PITCH_LENGTH)), float(np.clip(py, 0.0, PITCH_WIDTH)))

        # Fallback to static homography
        if self.homography_matrix is not None:
            px = np.array([[point]], dtype='float32')
            transformed = cv2.perspectiveTransform(px, self.homography_matrix)
            return transformed[0][0]

        x, y = point
        return (x * self.pixel_to_meter_ratio,
                y * self.pixel_to_meter_ratio)

    def update(self, tracks, homography_matrix=None, dx=0, dy=0):
        """
        tracks: dict -> {track_id: (x, y)}
        """

        for track_id, current_pos in tracks.items():

            current_pos = self.convert_position(current_pos, homography_matrix, dx, dy)

            if track_id in self.prev_positions:
                prev_pos = self.prev_positions[track_id]

                # 🔥 المسافة
                distance = np.linalg.norm(
                    np.array(current_pos) - np.array(prev_pos)
                )

                # ❌ ignore noise صغير جدًا
                if distance < 0.01:
                    distance = 0

                # إجمالي المسافة
                self.total_distance[track_id] += distance

                # 🔥 السرعة اللحظية
                speed = distance * self.fps  # m/s

                # ❌ limit غير منطقي
                if speed > 12:   # 12 m/s ≈ 43 km/h (max sprint)
                    speed = 12

                # 🔥 smoothing باستخدام history
                self.speed_history[track_id].append(speed)

                if len(self.speed_history[track_id]) > 5:
                    self.speed_history[track_id].pop(0)

                smooth_speed = np.mean(self.speed_history[track_id])

                self.speeds[track_id] = smooth_speed
                if smooth_speed > self.top_speeds[track_id]:
                    self.top_speeds[track_id] = smooth_speed

            self.prev_positions[track_id] = current_pos

        return self.total_distance, self.speeds