import cv2
import numpy as np

class PitchRadar:
    # 🔴 التعديل هنا: قمنا بتغيير الـ Ratio الافتراضي ليكون أفقيًا (Radar_w > Radar_h)
    def __init__(self, frame_w, frame_h, radar_w=500, radar_h=300):
        self.radar_w = radar_w
        self.radar_h = radar_h

        # نقاط افتراضية مبدئية لكي يعمل الرادار إذا فشل الموديل في أول فريم
        self.src_pts = np.float32([
            [415, 320],   
            [1505, 320],  
            [1880, 950],  
            [40, 950]     
        ])

        # 2. نقاط الرادار (المستطيل المثالي)
        # تم تعديلها لتربط الشاشة بـ 55% من مساحة الرادار لتعكس البث التلفزيوني بدقة (تجنب الشد الأفقية)
        pitch_view_width = radar_w * 0.55
        offset = (radar_w - pitch_view_width) / 2.0
        
        self.dst_pts = np.float32([
            [offset, 0],
            [offset + pitch_view_width, 0],
            [offset + pitch_view_width, radar_h],
            [offset, radar_h]
        ])

        # حساب المصفوفة المبدئية للرادار العادي لل fallback
        self.matrix = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.dx = 0
        self.dy = 0

    def set_matrix(self, matrix):
        """تحديث المصفوفة بالكامل (مثل هوموجرافي المفاصل)"""
        if matrix is not None:
            self.matrix = matrix
            self.dx = 0 # تصفير الإزاحة لأن المصفوفة الجديدة تشمل كل شيء
            self.dy = 0

    def update_matrix(self, dx, dy):
        """تحديث زاوية الرادار ديناميكياً باستخدام إزاحة الكاميرا (Panning)"""
        if dx is not None and dy is not None:
            self.dx = dx
            self.dy = dy

    def draw_radar(self, frame, players_data, ball_data,
                   position="bottom-right", title=None,
                   team_1_color=None, team_2_color=None,
                   referee_color=None,
                   team_1_name="Team 1", team_2_name="Team 2"):
        # ── Background ────────────────────────────────────────────────
        radar_img = np.zeros((self.radar_h, self.radar_w, 3), dtype=np.uint8)
        radar_img[:] = (34, 139, 34)  # Forest Green

        # ── Pitch lines ───────────────────────────────────────────────
        cv2.rectangle(radar_img, (0, 0), (self.radar_w, self.radar_h), (255, 255, 255), 2)
        cv2.line(radar_img, (self.radar_w // 2, 0), (self.radar_w // 2, self.radar_h), (255, 255, 255), 1)
        cv2.circle(radar_img, (self.radar_w // 2, self.radar_h // 2), int(self.radar_h * 0.134), (255, 255, 255), 1)

        # Default team colors if not provided
        c1 = team_1_color if team_1_color else (255,  50,  50)   # blue-ish
        c2 = team_2_color if team_2_color else (255, 255, 255)   # white

        # If Team 1 color is very dark (black), swap radar dot to bright orange
        # so it's visible on the dark green pitch background
        def _radar_color(bgr):
            """Return a visible radar colour: if bgr is too dark, use bright orange."""
            if bgr is None:
                return (255, 165, 0)
            b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
            if (b + g + r) < 120:          # too dark for green background
                return (0, 140, 255)       # vivid orange-amber (BGR)
            if (b + g + r) > 700:          # very white → keep white but add outline
                return bgr
            return bgr

        c1_radar = _radar_color(team_1_color)
        c2_radar = _radar_color(team_2_color)

        # ── Players ───────────────────────────────────────────────────
        for player in players_data:
            p_bbox = player['bbox']
            team   = player['team']

            feet_x = (p_bbox[0] + p_bbox[2]) / 2
            feet_y = p_bbox[3]

            pt = np.array([[[feet_x, feet_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.matrix)
            rx = int(transformed[0][0][0]) + self.dx
            ry = int(transformed[0][0][1]) + self.dy

            if not (0 <= rx <= self.radar_w and 0 <= ry <= self.radar_h):
                continue

            if team == "Referee":
                # Referee → use configured referee color (default: yellow)
                ref_c = referee_color if referee_color is not None else (0, 235, 235)
                cv2.circle(radar_img, (rx, ry), 3, ref_c, -1, cv2.LINE_AA)
                cv2.circle(radar_img, (rx, ry), 4, (0, 0, 0), 1, cv2.LINE_AA)
            elif team == team_1_name:
                cv2.circle(radar_img, (rx, ry), 4, c1_radar, -1, cv2.LINE_AA)
                cv2.circle(radar_img, (rx, ry), 5, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.circle(radar_img, (rx, ry), 6, (0, 0, 0), 1, cv2.LINE_AA)
            elif team == team_2_name:
                cv2.circle(radar_img, (rx, ry), 4, c2_radar, -1, cv2.LINE_AA)
                cv2.circle(radar_img, (rx, ry), 5, (80, 80, 80), 1, cv2.LINE_AA)
                cv2.circle(radar_img, (rx, ry), 6, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                # Unknown → grey
                cv2.circle(radar_img, (rx, ry), 3, (150, 150, 150), -1, cv2.LINE_AA)

        # ── Ball ──────────────────────────────────────────────────────
        if ball_data is not None:
            bbox_ball, is_interpolated = ball_data
            if bbox_ball is not None:
                bx = (bbox_ball[0] + bbox_ball[2]) / 2
                by = (bbox_ball[1] + bbox_ball[3]) / 2
                pt = np.array([[[bx, by]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(pt, self.matrix)
                rx = int(transformed[0][0][0]) + self.dx
                ry = int(transformed[0][0][1]) + self.dy
                if 0 <= rx <= self.radar_w and 0 <= ry <= self.radar_h:
                    ball_color = (0, 200, 255) if is_interpolated else (0, 255, 255)
                    cv2.circle(radar_img, (rx, ry), 3, ball_color, -1, cv2.LINE_AA)
                    cv2.circle(radar_img, (rx, ry), 4, (0, 0, 0), 1, cv2.LINE_AA)

        # ── Legend ────────────────────────────────────────────────────
        leg_y = self.radar_h - 12
        cv2.circle(radar_img, (8, leg_y), 4, c1_radar, -1)
        cv2.circle(radar_img, (8, leg_y), 5, (200, 200, 200), 1)
        cv2.putText(radar_img, team_1_name[:10], (17, leg_y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.circle(radar_img, (8 + self.radar_w // 2, leg_y), 4, c2_radar, -1)
        cv2.circle(radar_img, (8 + self.radar_w // 2, leg_y), 5, (60, 60, 60), 1)
        cv2.putText(radar_img, team_2_name[:10], (17 + self.radar_w // 2, leg_y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # ── Title ─────────────────────────────────────────────────────
        if title:
            cv2.putText(radar_img, title, (10, 15), cv2.FONT_HERSHEY_DUPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Overlay on frame ──────────────────────────────────────────
        h, w = frame.shape[:2]
        radar_x = (w - self.radar_w - 20) if position == "bottom-right" else 20
        radar_y = h - self.radar_h - 20

        cv2.rectangle(radar_img, (0, 0), (self.radar_w - 1, self.radar_h - 1),
                      (220, 220, 220), 1, cv2.LINE_AA)
        roi = frame[radar_y:radar_y + self.radar_h, radar_x:radar_x + self.radar_w]
        cv2.addWeighted(radar_img, 0.88, roi, 0.12, 0, roi)

        return frame