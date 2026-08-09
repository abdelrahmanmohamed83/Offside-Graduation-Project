"""
KeypointHomographyMapper
════════════════════════
Replaces SemanticPitchMapper.

Uses the 32-keypoint football-field pose model to compute a full perspective
homography every frame, then calls radar.set_matrix(H) so player dots are
placed with real geometric accuracy regardless of camera pan/tilt/zoom.

Pitch coordinate system
───────────────────────
  (0, 0) = top-left corner of the pitch
  x increases rightward  (105 m total)
  y increases downward   ( 68 m total)

Keypoint index → real-world (x, y) in metres.
Based on the 32-point schema shown in the project annotation image.
"""

import cv2
import numpy as np
from src.config import FIELD_DETECTOR_CONFIDENCE, RADAR_SMOOTHING

# ── Real-world pitch coordinates for each of the 32 keypoints ─────────
# Index 0-31 (model outputs 0-based).
# Adjust any values here if the model's numbering differs from your
# annotation reference — only the correctly matched points matter for
# RANSAC; wrong ones are automatically rejected.
PITCH_KP_WORLD: dict[int, tuple[float, float]] = {
    0:  (0.0,    0.0),    # top-left corner
    1:  (0.0,   13.84),   # left penalty box top-left
    2:  (0.0,   24.84),   # left 6-yard box top-left
    3:  (0.0,   43.16),   # left 6-yard box bottom-left
    4:  (0.0,   54.16),   # left penalty box bottom-left
    5:  (0.0,   68.0),    # bottom-left corner
    6:  (5.5,   24.84),   # left 6-yard box top-right
    7:  (5.5,   43.16),   # left 6-yard box bottom-right
    8:  (11.0,  20.16),   # left penalty arc top
    9:  (16.5,  13.84),   # left penalty box top-right
    10: (16.5,  54.16),   # left penalty box bottom-right
    11: (11.0,  47.84),   # left penalty arc bottom
    12: (11.0,  34.0),    # left penalty spot
    13: (43.35, 34.0),    # centre circle left tangent on halfway line
    14: (52.5,   0.0),    # halfway line top (touchline)
    15: (52.5,  24.85),   # centre circle top
    16: (52.5,  34.0),    # centre spot
    17: (52.5,  68.0),    # halfway line bottom (touchline)
    18: (61.65, 34.0),    # centre circle right tangent on halfway line
    19: (88.5,  13.84),   # right penalty box top-left
    20: (99.5,  24.84),   # right 6-yard box top-left
    21: (105.0, 13.84),   # right penalty box top-right
    22: (88.5,  54.16),   # right penalty box bottom-left
    23: (94.0,  34.0),    # right penalty spot
    24: (105.0, 24.84),   # right 6-yard box top-right
    25: (105.0, 54.16),   # right penalty box bottom-right
    26: (105.0,  0.0),    # top-right corner
    27: (105.0, 43.16),   # right 6-yard box bottom-right
    28: (99.5,  43.16),   # right 6-yard box bottom-left
    29: (88.5,   0.0),    # right penalty area top (touchline)
    30: (88.5,  68.0),    # right penalty area bottom (touchline)
    31: (105.0, 68.0),    # bottom-right corner
}

PITCH_W = 105.0   # metres
PITCH_H =  68.0   # metres


class KeypointHomographyMapper:
    """
    Compute a homography matrix from camera frame → radar image every frame.

    Usage (same interface as SemanticPitchMapper):
        mapper = KeypointHomographyMapper(radar_w, radar_h)
        H = mapper.get_homography(field_results)
        if H is not None:
            radar.set_matrix(H)
    """

    def __init__(self, radar_w: int = 500, radar_h: int = 300,
                 smoothing: float = RADAR_SMOOTHING,
                 min_points: int = 6,
                 kp_conf_thresh: float = FIELD_DETECTOR_CONFIDENCE):
        self.radar_w = radar_w
        self.radar_h = radar_h
        self.smoothing = smoothing
        self.min_points = min_points          # Min matched keypoints for RANSAC
        self.kp_conf_thresh = kp_conf_thresh

        self._scale_x = radar_w / PITCH_W     # pixels per metre (x)
        self._scale_y = radar_h / PITCH_H     # pixels per metre (y)

        self._last_H: np.ndarray | None = None   # Smoothed homography

    # ── private helpers ────────────────────────────────────────────────

    def _world_to_radar(self, wx: float, wy: float):
        """Convert real-world pitch metres → radar pixel coordinates."""
        return wx * self._scale_x, wy * self._scale_y

    def _smooth_H(self, H_new: np.ndarray) -> np.ndarray:
        """Exponential moving average on homography coefficients."""
        if self._last_H is None:
            self._last_H = H_new
            return H_new
        H_smooth = (1 - self.smoothing) * self._last_H + self.smoothing * H_new
        self._last_H = H_smooth
        return H_smooth

    # ── public API ─────────────────────────────────────────────────────

    def get_homography(self, field_results) -> np.ndarray | None:
        """
        Parameters
        ----------
        field_results : YOLO results list from a pose model

        Returns
        -------
        H : 3×3 homography (image → radar) or None if not enough points
        """
        if not field_results:
            return self._last_H

        result = field_results[0]
        if result.keypoints is None:
            return self._last_H

        kps_xy   = result.keypoints.xy.cpu().numpy()    # (N, 32, 2)
        kps_conf = result.keypoints.conf                # (N, 32) or None
        if kps_conf is not None:
            kps_conf = kps_conf.cpu().numpy()

        src_pts: list[list[float]] = []  # image pixel coords
        dst_pts: list[list[float]] = []  # radar pixel coords

        for inst_idx, inst_kps in enumerate(kps_xy):
            for kp_idx, (kx, ky) in enumerate(inst_kps):
                if kp_idx not in PITCH_KP_WORLD:
                    continue
                # Skip invisible/low-confidence keypoints
                if kx == 0.0 and ky == 0.0:
                    continue
                if kps_conf is not None:
                    if kps_conf[inst_idx][kp_idx] < self.kp_conf_thresh:
                        continue

                wx, wy = PITCH_KP_WORLD[kp_idx]
                rx, ry = self._world_to_radar(wx, wy)

                src_pts.append([kx, ky])
                dst_pts.append([rx, ry])

        if len(src_pts) < self.min_points:
            # Not enough correspondences — keep last known homography
            return self._last_H

        src_np = np.array(src_pts, dtype=np.float32)
        dst_np = np.array(dst_pts, dtype=np.float32)

        H, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, ransacReprojThreshold=15.0)

        if H is None:
            return self._last_H

        # Count inliers
        n_inliers = int(mask.sum()) if mask is not None else 0
        if n_inliers < 4:
            return self._last_H

        return self._smooth_H(H)
