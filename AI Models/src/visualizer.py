import cv2
import numpy as np
import supervision as sv
from collections import deque

from src.config import (
    BALL_TRAIL_ENABLED,
    BALL_TRAIL_LENGTH,
    BALL_TRAIL_COLOR,
    BALL_TRAIL_THICKNESS,
)


class Visualizer:

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _players_to_detections(players_data):
        """Convert our player dicts into a sv.Detections object."""
        if not players_data:
            return sv.Detections.empty()

        bboxes, labels, colors = [], [], []
        for p in players_data:
            x1, y1, x2, y2 = map(int, p['bbox'])
            bboxes.append([x1, y1, x2, y2])
            labels.append(p.get('name') or "")
            colors.append(p.get('color', (128, 128, 128)))

        xyxy = np.array(bboxes, dtype=np.float32)
        det  = sv.Detections(xyxy=xyxy)
        det.data['labels'] = labels
        det.data['colors'] = colors
        return det

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def draw_annotations(frame, players_data, ball_data=None,
                         possessor_id: int | None = None,
                         ball_trail: deque | None = None,
                         closest_player_id: int | None = None):
        """
        Draw:
        - supervision EllipseAnnotator for every detected player
        - dark-background label above each player (always readable)
        - gold triangle above the ball possessor
        - green arrow above the player closest to the ball
        - ball trail (fading colored line)
        - ball marker
        """

        # ── 1. Build sv.Detections ─────────────────────────────────────
        det = Visualizer._players_to_detections(players_data)

        if len(det) > 0:
            labels  = det.data['labels']
            raw_bgr = det.data['colors']

            # Ellipse: per-player team colour (RGB for supervision)
            sv_colors = [
                sv.Color(r=int(c[2]), g=int(c[1]), b=int(c[0]))
                for c in raw_bgr
            ]
            palette = sv.ColorPalette(colors=sv_colors)

            ellipse_ann = sv.EllipseAnnotator(
                color=palette,
                thickness=2,
                start_angle=-45,
                end_angle=235,
                color_lookup=sv.ColorLookup.INDEX,
            )
            frame = ellipse_ann.annotate(scene=frame, detections=det)

            # Add a white outline ellipse behind dark-coloured teams so the ellipse
            # is always visible (e.g., black jerseys on dark background)
            needs_outline = [
                i for i, c in enumerate(raw_bgr)
                if (int(c[0]) + int(c[1]) + int(c[2])) < 150  # dark colour threshold
            ]
            if needs_outline:
                outline_det = det[needs_outline]
                white_palette = sv.ColorPalette(
                    colors=[sv.Color(r=240, g=240, b=240)] * len(needs_outline)
                )
                outline_ann = sv.EllipseAnnotator(
                    color=white_palette,
                    thickness=1,
                    start_angle=-45,
                    end_angle=235,
                    color_lookup=sv.ColorLookup.INDEX,
                )
                frame = outline_ann.annotate(scene=frame, detections=outline_det)

            # ── 2. Labels: always dark background + white text ──────────
            # Build a dark-grey palette so labels are always readable,
            # regardless of team colour (fixes white-on-white issue).
            dark_palette = sv.ColorPalette(
                colors=[sv.Color(r=30, g=30, b=30)] * len(det)
            )
            label_ann = sv.LabelAnnotator(
                color=dark_palette,
                text_scale=0.4,
                text_thickness=1,
                text_padding=5,
                border_radius=4,
                text_position=sv.Position.TOP_CENTER,
                text_color=sv.Color.WHITE,
                color_lookup=sv.ColorLookup.INDEX,
            )
            frame = label_ann.annotate(
                scene=frame,
                detections=det,
                labels=list(labels),
            )

            # ── 3. Triangle above ball possessor ───────────────────────
            if possessor_id is not None:
                poss_indices = [
                    i for i, p in enumerate(players_data)
                    if p.get('track_id') == possessor_id
                ]
                if poss_indices:
                    poss_det = det[poss_indices]
                    tri_ann = sv.TriangleAnnotator(
                        color=sv.Color(r=255, g=215, b=0),  # Gold
                        base=18,
                        height=16,
                        position=sv.Position.TOP_CENTER,
                        outline_thickness=1,
                        outline_color=sv.Color.BLACK,
                        color_lookup=sv.ColorLookup.INDEX,
                    )
                    frame = tri_ann.annotate(scene=frame, detections=poss_det)

            # ── 3b. Downward Arrow above player closest to the ball ──
            if closest_player_id is not None:
                closest_p = next((p for p in players_data if p.get('track_id') == closest_player_id), None)
                if closest_p:
                    x1, y1, x2, y2 = map(int, closest_p['bbox'])
                    cx = (x1 + x2) // 2
                    cy = y1 - 8
                    # Draw a neat downward green arrow pointing to the player's head (with black outline)
                    cv2.arrowedLine(frame, (cx, cy - 25), (cx, cy), (0, 0, 0), 4, tipLength=0.35, line_type=cv2.LINE_AA)
                    cv2.arrowedLine(frame, (cx, cy - 25), (cx, cy), (50, 255, 50), 2, tipLength=0.35, line_type=cv2.LINE_AA)

        # ── 4. Ball trail ───────────────────────────────────────────────
        # Disabled as per user request to remove trail/path prediction.

        # ── 5. Ball marker ──────────────────────────────────────────────
        if ball_data is not None:
            bbox, is_interpolated = ball_data
            if bbox is not None and not is_interpolated:
                x1, y1, x2, y2 = map(int, bbox)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = max((x2 - x1), (y2 - y1)) // 2 + 3

                # Draw a simple circle on the ball and that's it!
                cv2.circle(frame, (cx, cy), radius, (0, 215, 255), 2, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Action Banners (left = AI model, right = physics)
    # ------------------------------------------------------------------

    @staticmethod
    def draw_action_banners(frame,
                            model_action=None,   model_conf=0.0,   model_frames=0,   model_player="",  model_team_color=None,
                            physics_action=None, physics_conf=0.0, physics_frames=0, physics_player="", physics_team_color=None):
        """
        Draw two sliding toast banners:
          - Left side  → AI Model action
          - Right side → Physics / Rule-based action

        Each banner only appears when frames > 0.
        """
        fh, fw = frame.shape[:2]

        # ── Emoji / icon map ──────────────────────────────────────────
        ACTION_ICON = {
            'PASS':                    'PASS',
            'HIGH_PASS':               'HIGH PASS',
            'CROSS':                   'CROSS',
            'SHOT':                    'SHOT',
            'CLEARANCE':               'CLEARANCE',
            'HEADER':                  'HEADER',
            'THROW_IN':                'THROW IN',
            'PLAYER_SUCCESSFUL_TACKLE':'TACKLE',
            'BALL_PLAYER_BLOCK':       'BLOCK',
            'INTERCEPTION':            'INTERCEPTION',
            'No Action':               'NO ACTION',
        }

        # ── Color per action type ─────────────────────────────────────
        ACTION_COLOR = {
            'PASS':                    (60, 220, 60),
            'HIGH_PASS':               (180, 255, 60),
            'CROSS':                   (255, 180, 40),
            'SHOT':                    (0,  80, 255),
            'CLEARANCE':               (0, 180, 255),
            'HEADER':                  (200, 80, 255),
            'THROW_IN':                (255, 220, 0),
            'PLAYER_SUCCESSFUL_TACKLE': (0, 200, 120),
            'BALL_PLAYER_BLOCK':       (255, 140, 0),
            'INTERCEPTION':            (0, 120, 255),
            'No Action':               (100, 100, 100),
        }

        def _accent(action):
            c = ACTION_COLOR.get(action, (0, 215, 255))
            return c if isinstance(c, tuple) else (0, 215, 255)

        def _draw_banner(frame, action, conf, frames_left, player, team_color, side):
            """Draw one banner. side='left' or 'right'."""
            if frames_left <= 0 or not action or action == 'No Action':
                return frame

            # Fade alpha: full for first half of life, fades in last 15 frames
            max_frames  = 90
            alpha       = min(1.0, frames_left / 15.0)

            # Banner geometry
            BW, BH  = 300, 80
            margin  = 12
            BY      = fh // 2 - BH // 2   # vertically centred

            if side == 'left':
                BX = margin
            else:
                BX = fw - BW - margin

            # Slide-in animation: start off-screen, slide to final position
            slide_frames = 12
            progress     = min(1.0, (max_frames - frames_left + 1) / slide_frames)
            if side == 'left':
                BX_draw = int(BX - BW + BX * progress + BW * progress)
                BX_draw = max(margin, min(BX, BX_draw))
            else:
                start_x  = fw
                BX_draw  = int(start_x - (start_x - BX) * progress)
                BX_draw  = max(BX, min(fw - margin, BX_draw))

            # Clamp drawing region to frame
            x1 = max(0, BX_draw)
            y1 = max(0, BY)
            x2 = min(fw, BX_draw + BW)
            y2 = min(fh, BY + BH)
            if x2 <= x1 or y2 <= y1:
                return frame

            accent = _accent(action)
            label  = ACTION_ICON.get(action, action)

            # Translucent dark background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
            cv2.addWeighted(overlay, alpha * 0.80, frame, 1 - alpha * 0.80, 0, frame)

            # Accent side stripe (4px)
            stripe_x = x1 if side == 'left' else x2 - 4
            cv2.rectangle(frame, (stripe_x, y1), (stripe_x + 4, y2), accent, -1)

            # Source label  (small, top)
            source_lbl = "AI MODEL" if side == 'left' else "PHYSICS"
            src_color  = (120, 220, 255) if side == 'left' else (255, 200, 80)
            tx = x1 + 14 if side == 'left' else x1 + 10
            cv2.putText(frame, source_lbl,
                        (tx, y1 + 16),
                        cv2.FONT_HERSHEY_DUPLEX, 0.35, src_color, 1, cv2.LINE_AA)

            # Action label (large, bold)
            cv2.putText(frame, label,
                        (tx, y1 + 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, accent, 2, cv2.LINE_AA)

            # Player name (small, below action)
            if player:
                pname = player[:22] + '..' if len(player) > 22 else player
                cv2.putText(frame, pname,
                            (tx, y1 + 58),
                            cv2.FONT_HERSHEY_DUPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

            # Confidence bar
            bar_y   = y2 - 6
            bar_x1  = x1 + 10
            bar_x2  = x2 - 10
            bar_len = int((bar_x2 - bar_x1) * min(1.0, conf))
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 3), (50, 50, 50), -1)
            if bar_len > 0:
                cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_len, bar_y + 3), accent, -1)

            # Team colour dot (top-right corner of banner)
            if team_color is not None:
                dot_x = x2 - 14 if side == 'left' else x2 - 14
                cv2.circle(frame, (dot_x, y1 + 14), 6, team_color, -1)
                cv2.circle(frame, (dot_x, y1 + 14), 6, (255,255,255), 1)

            return frame

        # Draw left banner (AI model)
        frame = _draw_banner(frame,
                             model_action, model_conf, model_frames,
                             model_player, model_team_color, 'left')

        # Draw right banner (physics)
        frame = _draw_banner(frame,
                             physics_action, physics_conf, physics_frames,
                             physics_player, physics_team_color, 'right')

        return frame

    # ------------------------------------------------------------------
    # Speed / distance overlay
    # ------------------------------------------------------------------

    @staticmethod
    def draw_speed_distance(frame, players_data, speeds, total_dist):
        """Draw speed and total-distance badge below each player."""
        for p in players_data:
            tid = p.get('track_id')
            if tid is None:
                continue

            x1, y1, x2, y2 = map(int, p['bbox'])
            color = p.get('color', (200, 200, 200))

            speed_ms  = speeds.get(tid, 0.0)
            speed_kmh = speed_ms * 3.6
            dist_m    = total_dist.get(tid, 0.0)

            line1 = f"{speed_kmh:.1f} km/h"
            line2 = f"{dist_m:.0f} m"

            font      = cv2.FONT_HERSHEY_DUPLEX
            scale     = 0.4
            thickness = 1

            (w1, h1), _ = cv2.getTextSize(line1, font, scale, thickness)
            (w2, h2), _ = cv2.getTextSize(line2, font, scale, thickness)

            box_w    = max(w1, w2) + 14
            box_h    = h1 + h2 + 18
            center_x = (x1 + x2) // 2
            bx1 = center_x - box_w // 2
            bx2 = center_x + box_w // 2
            by1 = y2 + 6
            by2 = by1 + box_h

            sub = frame[by1:by2, bx1:bx2]
            if sub.shape[0] > 0 and sub.shape[1] > 0:
                black = np.full_like(sub, (20, 20, 20))
                cv2.addWeighted(black, 0.75, sub, 0.25, 0, sub)
                frame[by1:by2, bx1:bx2] = sub

            cv2.line(frame, (bx1, by1), (bx2, by1), color, 2)
            cv2.putText(frame, line1, (bx1 + 7, by1 + h1 + 5),
                        font, scale, (0, 230, 255), thickness, cv2.LINE_AA)
            cv2.putText(frame, line2, (bx1 + 7, by1 + h1 + h2 + 12),
                        font, scale, (200, 200, 200), thickness, cv2.LINE_AA)

        return frame