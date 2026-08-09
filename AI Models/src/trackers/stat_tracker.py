import cv2
import numpy as np
from collections import Counter, defaultdict
from src.config import (
    STICKY_FRAMES,
    REQUIRED_POSSESSION_FRAMES,
    FEET_ZONE_HEIGHT_RATIO,
    FEET_ZONE_WIDTH_EXPANSION,
    TEAM_1_NAME,
    TEAM_2_NAME,
    TEAM_1_DISPLAY_COLOR,
    TEAM_2_DISPLAY_COLOR,
    PITCH_LENGTH,
    PITCH_WIDTH,
    RADAR_WIDTH,
    RADAR_HEIGHT,
)


class MatchStats:
    """
    Possession & event tracking based on bbox feet-zone intersection.

    Logic:
    ─────
    • Ball is "at" a player if its centre falls inside the player's FEET ZONE
      (bottom FEET_ZONE_HEIGHT_RATIO of their bounding box, slightly widened).
    • A PASS is recorded when the ball moves from player A's feet zone →
      player B's feet zone and both are on the SAME team.
    • An INTERCEPTION is recorded when the ball moves from A → B and they are
      on DIFFERENT teams.
    • If the ball leaves A's zone and returns to A's zone (dribble), no event.
    • Possession is "sticky": the last possessor keeps credit for STICKY_FRAMES
      frames after the ball leaves their feet.
    """

    def __init__(self, team_1_name=TEAM_1_NAME, team_2_name=TEAM_2_NAME,
                 team_1_color=TEAM_1_DISPLAY_COLOR, team_2_color=TEAM_2_DISPLAY_COLOR):
        self.required_frames = REQUIRED_POSSESSION_FRAMES
        self.sticky_frames   = STICKY_FRAMES

        # ── Dynamic team names ─────────────────────────────────────────
        self.team_1_name  = team_1_name
        self.team_2_name  = team_2_name
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

        # ── Current confirmed possessor ────────────────────────────────
        self.possessor_tid   = None   # track_id
        self.possessor_team  = None
        self.possessor_label = "Free Ball"   # display string
        self.frames_without_contact = 0

        # ── Candidate (building up consecutive frames) ─────────────────
        self.candidate_tid   = None
        self.candidate_team  = None
        self.candidate_name  = None
        self.candidate_count = 0

        # ── Statistics ─────────────────────────────────────────────────
        self.team_possession_frames = {team_1_name: 0, team_2_name: 0}
        self.total_possession_frames = 0
        self.current_possessor = "Free Ball"   # shown in HUD

        self.event_counts = {
            "passes_team1":        0,
            "passes_team2":        0,
            "interceptions_team1": 0,
            "interceptions_team2": 0,
        }

        # ── UI alerts ──────────────────────────────────────────────────
        self.current_alert = None
        self.alert_frames  = 0
        self.alert_color   = (255, 255, 255)

        # ── Action tracking per player & team ──────────────────────────
        self.player_actions: dict = defaultdict(Counter)
        self.player_names:   dict = {}
        self.player_teams:   dict = {}
        self.team_actions:   dict = {team_1_name: Counter(), team_2_name: Counter()}
        self.current_action:       str   = ""
        self.current_action_conf:  float = 0.0
        self.action_display_frames: int  = 0

        # ── Rule-based tracking states ─────────────────────────────────
        self.frame_counter = 0
        self.last_possessor_tid = None
        self.last_possessor_name = None
        self.last_possessor_team = None
        self.kick_point = None  # (px, py) in meters
        self.free_ball_positions = []  # list of (px, py, frame)
        self.free_ball_speeds = []     # list of speeds in m/s
        self.player_possession_frames = defaultdict(int)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _feet_zone(bbox):
        """Return (x1, fy1, x2, y2) — the feet area of a player bbox."""
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        fy1 = y2 - int(h * FEET_ZONE_HEIGHT_RATIO)
        # Expand width slightly so ball touching the side is also captured
        w_margin = int((x2 - x1) * FEET_ZONE_WIDTH_EXPANSION)
        return (x1 - w_margin, fy1, x2 + w_margin, y2)

    @staticmethod
    def _ball_in_feet_zone(ball_cx, ball_cy, bbox):
        """True if the ball centre falls inside the player's feet zone."""
        fx1, fy1, fx2, fy2 = MatchStats._feet_zone(bbox)
        return fx1 <= ball_cx <= fx2 and fy1 <= ball_cy <= fy2

    def _to_pitch_coords(self, x, y, matrix, dx, dy, radar_w=RADAR_WIDTH, radar_h=RADAR_HEIGHT):
        if matrix is None:
            # Fallback mapping (image relative coordinates to pitch size)
            return float((x / 1920.0) * PITCH_LENGTH), float((y / 1080.0) * PITCH_WIDTH)
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, matrix)
        rx = transformed[0][0][0] + dx
        ry = transformed[0][0][1] + dy
        px = (rx / float(radar_w)) * PITCH_LENGTH
        py = (ry / float(radar_h)) * PITCH_WIDTH
        return float(np.clip(px, 0.0, PITCH_LENGTH)), float(np.clip(py, 0.0, PITCH_WIDTH))

    def _classify_and_record_action(self, from_tid, from_team, from_name, to_tid, to_team, to_name, curr_x, curr_y):
        """Classifies the possession transition between two players using rules."""
        kx, ky = self.kick_point if self.kick_point is not None else (curr_x, curr_y)
        dist = np.sqrt((curr_x - kx)**2 + (curr_y - ky)**2)
        max_speed = max(self.free_ball_speeds) if self.free_ball_speeds else 0.0
        if max_speed == 0.0:
            max_speed = dist * 25.0

        action = "PASS"
        confidence = 0.90
        is_team1 = (from_team == self.team_1_name)

        # 1. SHOT: high speed towards opponent's goal
        is_shot_trajectory = False
        if is_team1:
            if curr_x > kx and kx > (PITCH_LENGTH / 2.0):
                is_shot_trajectory = True
        else:
            if curr_x < kx and kx < (PITCH_LENGTH / 2.0):
                is_shot_trajectory = True

        if is_shot_trajectory and max_speed > 10.0:
            action = "SHOT"
            confidence = min(0.98, 0.70 + (max_speed - 10.0) * 0.02)
        
        # 2. CLEARANCE: high speed from defensive area
        elif max_speed > 8.0 and dist > 8.0:
            is_defensive_zone = (kx < (PITCH_LENGTH / 3.0)) if is_team1 else (kx > (2.0 * PITCH_LENGTH / 3.0))
            if is_defensive_zone:
                action = "CLEARANCE"
                confidence = 0.85

        # 3. CROSS: from wide area to opponent's penalty box
        elif (ky < 4.5 or ky > 15.5) and dist > 7.0:
            in_opponent_box = False
            if is_team1:
                if curr_x > 34.0 and 4.0 < curr_y < 16.0:
                    in_opponent_box = True
            else:
                if curr_x < 6.0 and 4.0 < curr_y < 16.0:
                    in_opponent_box = True
            if in_opponent_box:
                action = "CROSS"
                confidence = 0.88

        # 4. HIGH_PASS: long pass
        elif dist > 10.0:
            action = "HIGH_PASS"
            confidence = 0.85

        if from_team == to_team:
            # Same team -> PASS/SHOT/CROSS/HIGH_PASS/CLEARANCE
            self.record_action(from_tid, from_team, action, confidence, from_name)
            key = "passes_team1" if to_team == self.team_1_name else "passes_team2"
            self.event_counts[key] += 1
            if action == "SHOT":
                self.current_alert = "WHAT A SHOT!"
                self.alert_color   = (0, 0, 255)
                self.alert_frames  = 60
            elif action == "CROSS":
                self.current_alert = "BEAUTIFUL CROSS!"
                self.alert_color   = (255, 150, 0)
                self.alert_frames  = 45
            elif action == "HIGH_PASS":
                self.current_alert = "LONG PASS!"
                self.alert_color   = (200, 200, 0)
                self.alert_frames  = 45
            else:
                self.current_alert = "NICE PASS!"
                self.alert_color   = (0, 220, 0)
                self.alert_frames  = 45
        else:
            # Different team -> INTERCEPTION (or TACKLE by intercepting player)
            if action == "SHOT":
                self.record_action(from_tid, from_team, "SHOT", confidence, from_name)
                self.current_alert = "SHOT BLOCKED/SAVED!"
                self.alert_color   = (0, 100, 255)
                self.alert_frames  = 50
            else:
                self.record_action(to_tid, to_team, "PLAYER_SUCCESSFUL_TACKLE", 0.90, to_name)
                key = "interceptions_team1" if to_team == self.team_1_name else "interceptions_team2"
                self.event_counts[key] += 1
                self.current_alert = "INTERCEPTION!"
                self.alert_color   = (0, 80, 255)
                self.alert_frames  = 50

    def _fire_event(self, from_team, to_team, to_team_name):
        """Deprecated: superseded by _classify_and_record_action."""
        pass

    def _carry_possession(self):
        """Continue crediting the last possessor for sticky_frames."""
        if self.possessor_team is not None:
            self.frames_without_contact += 1
            if self.frames_without_contact <= self.sticky_frames:
                self.team_possession_frames[self.possessor_team] += 1
                self.total_possession_frames += 1
                self.current_possessor = self.possessor_label
            else:
                self.current_possessor = "Free Ball"
        else:
            self.current_possessor = "Free Ball"

    # ──────────────────────────────────────────────────────────────────
    # Main update — called every frame
    # ──────────────────────────────────────────────────────────────────

    def update(self, players_data, ball_data, matrix=None, dx=0, dy=0):
        self.frame_counter += 1

        # Countdown alert display
        if self.alert_frames > 0:
            self.alert_frames -= 1
        else:
            self.current_alert = None

        if self.action_display_frames > 0:
            self.action_display_frames -= 1

        # ── Ball must be a real detection (not interpolated) ───────────
        if not ball_data or ball_data[0] is None or ball_data[1] is True:
            self._carry_possession()
            if self.possessor_tid is not None:
                p_name = self.player_names.get(self.possessor_tid, f"Player #{self.possessor_tid}")
                self.player_possession_frames[p_name] += 1
            return

        bbox_ball = ball_data[0]
        ball_cx = (bbox_ball[0] + bbox_ball[2]) // 2
        ball_cy = (bbox_ball[1] + bbox_ball[3]) // 2
        ball_px, ball_py = self._to_pitch_coords(ball_cx, ball_cy, matrix, dx, dy)

        # ── Find which player's feet zone contains the ball ───────────
        contact_tid   = None
        contact_team  = None
        contact_name  = None

        closest_p = None
        min_dist = float('inf')

        for p in players_data:
            if p['team'] in ('Referee', 'Unknown'):
                continue
            if self._ball_in_feet_zone(ball_cx, ball_cy, p['bbox']):
                # Calculate distance between player's feet center and ball center
                x1, y1, x2, y2 = p['bbox']
                feet_cx = (x1 + x2) / 2
                feet_cy = y2
                dist = np.sqrt((feet_cx - ball_cx)**2 + (feet_cy - ball_cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_p = p

        if closest_p is not None:
            contact_tid  = closest_p['track_id']
            contact_team = closest_p['team']
            contact_name = closest_p.get('name') or f"Player #{contact_tid}"
            self.player_names[contact_tid] = contact_name
            self.player_teams[contact_tid] = contact_team

        # ── Process contact ────────────────────────────────────────────
        if contact_tid is not None:
            # Accumulate candidate frames for this track_id
            if contact_tid == self.candidate_tid:
                self.candidate_count += 1
            else:
                # New candidate — reset counter
                self.candidate_tid   = contact_tid
                self.candidate_team  = contact_team
                self.candidate_name  = contact_name
                self.candidate_count = 1

            # Enough frames to confirm possession?
            if self.candidate_count >= self.required_frames:
                prev_tid  = self.possessor_tid
                prev_team = self.possessor_team
                prev_name = self.player_names.get(prev_tid) if prev_tid is not None else None

                # Confirm new possessor
                self.possessor_tid   = contact_tid
                self.possessor_team  = contact_team
                self.possessor_label = f"{contact_name} ({contact_team})"
                self.frames_without_contact = 0

                # Make sure team key exists (handles Unknown/Referee edge cases)
                if contact_team in self.team_possession_frames:
                    self.team_possession_frames[contact_team] += 1
                self.total_possession_frames += 1
                self.current_possessor = self.possessor_label
                self.player_possession_frames[contact_name] += 1

                # ── Detect pass / interception / shot / clearance ────────────────────────
                if prev_tid is not None and prev_tid != contact_tid:
                    self._classify_and_record_action(
                        prev_tid, prev_team, prev_name,
                        contact_tid, contact_team, contact_name,
                        ball_px, ball_py
                    )

                self.last_possessor_tid = contact_tid
                self.last_possessor_name = contact_name
                self.last_possessor_team = contact_team
                self.kick_point = (ball_px, ball_py)
                self.free_ball_positions = []
                self.free_ball_speeds = []

        else:
            # No player's feet zone contains the ball
            self.candidate_count = 0   # Reset candidate streak
            self._carry_possession()
            if self.possessor_tid is not None:
                p_name = self.player_names.get(self.possessor_tid, f"Player #{self.possessor_tid}")
                self.player_possession_frames[p_name] += 1

            if self.last_possessor_tid is not None:
                self.free_ball_positions.append((ball_px, ball_py, self.frame_counter))
                if len(self.free_ball_positions) >= 2:
                    x1, y1, f1 = self.free_ball_positions[-2]
                    x2, y2, f2 = self.free_ball_positions[-1]
                    dt = (f2 - f1) / 25.0
                    if dt > 0:
                        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        speed = dist / dt
                        self.free_ball_speeds.append(speed)

    # ──────────────────────────────────────────────────────────────────
    # Stats accessors
    # ──────────────────────────────────────────────────────────────────

    def get_possession_stats(self):
        if self.total_possession_frames == 0:
            return {self.team_1_name: 50, self.team_2_name: 50}
        t1 = self.team_possession_frames.get(self.team_1_name, 0)
        t2 = self.team_possession_frames.get(self.team_2_name, 0)
        total = self.total_possession_frames
        return {self.team_1_name: int(t1 / total * 100),
                self.team_2_name: int(t2 / total * 100)}

    def record_action(self, track_id: int, team: str, action: str,
                       confidence: float, player_name: str = ""):
        """Record a detected action for a player and their team."""
        self.player_actions[track_id][action] += 1
        self.player_names[track_id] = player_name or f"Player #{track_id}"
        self.player_teams[track_id] = team
        if team in self.team_actions:
            self.team_actions[team][action] += 1
        # Show on HUD for 60 frames
        self.current_action       = action
        self.current_action_conf  = confidence
        self.action_display_frames = 60

    def get_player_action_stats(self) -> list:
        """Returns list of dicts with per-player action counts."""
        result = []
        for tid, counter in self.player_actions.items():
            result.append({
                "track_id":    int(tid),
                "player_name": self.player_names.get(tid, f"Player #{tid}"),
                "team":        self.player_teams.get(tid, "Unknown"),
                "actions":     dict(counter),
                "total_actions": sum(counter.values()),
            })
        return sorted(result, key=lambda x: x["total_actions"], reverse=True)

    def get_team_action_stats(self) -> dict:
        """Returns dict of team → action counts."""
        return {team: dict(counter) for team, counter in self.team_actions.items()}

    def get_event_stats(self):
        return {
            "passes_t1": self.event_counts["passes_team1"],
            "passes_t2": self.event_counts["passes_team2"],
            "inter_t1":  self.event_counts["interceptions_team1"],
            "inter_t2":  self.event_counts["interceptions_team2"],
        }

    # ──────────────────────────────────────────────────────────────────
    # HUD drawing
    # ──────────────────────────────────────────────────────────────────

    def draw_stats(self, frame):
        stats  = self.get_possession_stats()
        t1_pct = stats[self.team_1_name]
        t2_pct = stats[self.team_2_name]

        # Dynamically size HUD height based on whether action is active
        has_action = (self.action_display_frames > 0 and self.current_action)
        w = 260
        h = 102 if has_action else 82
        x, y = 20, 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (x, y), (x + 4, y + h), (0, 215, 255), -1)

        cv2.putText(frame, "LIVE MATCH STATS",
                    (x + 15, y + 18), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        poss_text = f"{self.team_1_name[:10]} {t1_pct}% | {self.team_2_name[:10]} {t2_pct}%"
        cv2.putText(frame, poss_text,
                    (x + 15, y + 38), cv2.FONT_HERSHEY_DUPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        bar_y = y + 46
        cv2.rectangle(frame, (x + 15, bar_y), (x + w - 15, bar_y + 4), (220, 220, 220), -1)
        t1_w = int((w - 30) * t1_pct / 100)
        if t1_w > 0:
            cv2.rectangle(frame, (x + 15, bar_y), (x + 15 + t1_w, bar_y + 4), self.team_1_color, -1)

        cv2.putText(frame, f"Ball: {self.current_possessor}",
                    (x + 15, y + 70), cv2.FONT_HERSHEY_DUPLEX, 0.35, (0, 215, 255), 1, cv2.LINE_AA)

        # Action Display
        if has_action:
            action_text = f"Action: {self.current_action} ({self.current_action_conf:.2f})"
            cv2.putText(frame, action_text,
                        (x + 15, y + 90), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 255), 1, cv2.LINE_AA)


        # Pop-up alert
        if self.alert_frames > 0 and self.current_alert:
            alpha = min(1.0, self.alert_frames / 15.0)
            alert_overlay = frame.copy()
            fh, fw = frame.shape[:2]
            text_size = cv2.getTextSize(self.current_alert, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0]
            bx1 = int((fw - text_size[0]) / 2) - 20
            by1 = fh - 110 - text_size[1] - 10
            bx2 = bx1 + text_size[0] + 40
            by2 = fh - 110 + 10
            cv2.rectangle(alert_overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.addWeighted(alert_overlay, alpha * 0.65, frame, 1 - alpha * 0.65, 0, frame)
            cv2.putText(frame, self.current_alert,
                        (bx1 + 20, fh - 110), cv2.FONT_HERSHEY_DUPLEX, 1.1,
                        self.alert_color, 2, cv2.LINE_AA)

        return frame