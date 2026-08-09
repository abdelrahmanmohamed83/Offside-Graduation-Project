import os
import requests
import json
import cv2
import numpy as np
from src.config import API_BASE_URL, SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET

def hex_to_hsv(hex_color):
    """
    Converts a hex color string (e.g. '#ffffff' or 'ffffff') to a LIST of OpenCV
    HSV ranges covering all lighting variants of that color:
      - Normal light
      - Shadow (darker, less saturated)
      - Overexposed highlight (brighter, lower saturation)

    Returns: [{"lower": [...], "upper": [...]}, ...] or None on failure.
    """
    if hex_color is None or hex_color == "NULL" or not isinstance(hex_color, str):
        return None

    hex_color = hex_color.strip().lstrip('#')
    if len(hex_color) != 6:
        return None

    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return None

    # Convert RGB to HSV via cv2 (input must be BGR)
    color_bgr = np.uint8([[[b, g, r]]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(color_hsv[0]), int(color_hsv[1]), int(color_hsv[2])

    # ── Special cases: achromatic colours ─────────────────────────────
    if s < 30 and v > 200:  # White
        return [
            {"lower": [0,   0,  175], "upper": [180, 50,  255]},  # Normal white
            {"lower": [0,   0,  130], "upper": [180, 80,  200]},  # White in shadow
        ]

    if v < 60:  # Black / very dark
        return [
            {"lower": [0,   0,   0],  "upper": [15,  255, 60]},   # Dark (reddish-black)
            {"lower": [160, 0,   0],  "upper": [180, 255, 60]},   # Dark (wrap-around)
            {"lower": [86,  0,   0],  "upper": [159, 255, 60]},   # Dark blues/purples
            {"lower": [0,   0,   0],  "upper": [180, 40,  60]},   # Desaturated darks
        ]

    # ── Normal / saturated colour ─────────────────────────────────────
    H_TOL   = 18   # Hue tolerance (degrees) – covers camera/WB shifts
    lower_h = max(0,   h - H_TOL)
    upper_h = min(180, h + H_TOL)

    ranges = [
        # Normal lighting
        {"lower": [lower_h, 50,  60],  "upper": [upper_h, 255, 255]},
        # Deep shadow – same hue, low V and low S
        {"lower": [lower_h, 25,  20],  "upper": [upper_h, 180, 110]},
        # Overexposed / washed-out – very low S, high V
        {"lower": [lower_h, 15,  160], "upper": [upper_h,  90, 255]},
    ]

    # Handle hue wrap-around (e.g. red: h near 0 or 180)
    if lower_h < H_TOL:  # hue wraps below 0
        wrap_h = 180 + lower_h
        ranges += [
            {"lower": [wrap_h, 50,  60],  "upper": [180,    255, 255]},
            {"lower": [wrap_h, 25,  20],  "upper": [180,    180, 110]},
        ]
    if upper_h > 180 - H_TOL:  # hue wraps above 180
        wrap_h = upper_h - 180
        ranges += [
            {"lower": [0, 50,  60],  "upper": [wrap_h, 255, 255]},
            {"lower": [0, 25,  20],  "upper": [wrap_h, 180, 110]},
        ]

    return ranges

def hex_to_bgr(hex_color):
    """
    Converts hex color to BGR tuple.
    """
    if not hex_color or hex_color == "NULL" or not isinstance(hex_color, str):
        return (128, 128, 128)
    hex_color = hex_color.strip().lstrip('#')
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)
    except ValueError:
        return (128, 128, 128)

class APIClient:
    def __init__(self):
        self.api_base = API_BASE_URL
        self.supabase_url = SUPABASE_URL
        self.headers = {
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "apikey": SUPABASE_KEY,
            "Content-Type": "application/json"
        }

    def fetch_match_data(self, match_id):
        print(f"Fetching data for match_id: {match_id} from API...")
        # 1. Fetch match info
        match_res = requests.get(f"{self.api_base}/matches/{match_id}")
        if not match_res.ok:
            raise Exception(f"Failed to fetch match {match_id}: {match_res.text}")
        
        match_data = match_res.json().get("match", {})
        home_team_id = match_data.get("home_team_id")
        away_team_id = match_data.get("away_team_id")
        
        # 2. Fetch team info
        home_res = requests.get(f"{self.api_base}/teams/{home_team_id}")
        away_res = requests.get(f"{self.api_base}/teams/{away_team_id}")
        
        home_team = home_res.json().get("team", {}) if home_res.ok else {}
        away_team = away_res.json().get("team", {}) if away_res.ok else {}
        
        print(f"Home Team: {home_team.get('team_name')} - Away Team: {away_team.get('team_name')}")
        
        # 3. Fetch players directly from Supabase REST API
        players_db = {}
        
        for t_id in [home_team_id, away_team_id]:
            if not t_id: continue
            # Try PLAYERS table
            p_res = requests.get(
                f"{self.supabase_url}/rest/v1/PLAYERS?team_id=eq.{t_id}&select=jersey_number,full_name",
                headers=self.headers
            )
            # if not found, try players table
            if not p_res.ok:
                p_res = requests.get(
                    f"{self.supabase_url}/rest/v1/players?team_id=eq.{t_id}&select=jersey_number,full_name",
                    headers=self.headers
                )
            
            if p_res.ok:
                for player in p_res.json():
                    jn = str(player.get("jersey_number"))
                    players_db[jn] = player.get("full_name")
        
        print(f"Loaded {len(players_db)} players from Database.")
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "players_db": players_db
        }
        
    def upload_heatmap(self, player_name, image_path):
        """Uploads an image to Supabase storage and returns the public URL."""
        if not os.path.exists(image_path):
            return None
            
        filename = f"{player_name.replace(' ', '_')}_heatmap.png"
        upload_url = f"{self.supabase_url}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
        
        upload_headers = self.headers.copy()
        upload_headers["Content-Type"] = "image/png"
        
        print(f"Uploading heatmap for {player_name}...")
        with open(image_path, "rb") as f:
            res = requests.post(upload_url, headers=upload_headers, data=f)
            
        if res.ok or res.status_code == 400: # 400 might be 'already exists' or similar but let's assume it works
            public_url = f"{self.supabase_url}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
            return public_url
        else:
            print(f"Warning: Failed to upload heatmap for {player_name}: {res.text}")
            return None

    def submit_ai_results(self, match_id, final_stats, event_stats, player_stats,
                          heatmap_urls, team_stats=None):
        """Submits the final JSON back to FastAPI."""
        print(f"Submitting final AI analysis for match {match_id}...")

        payload = {
            "match_id": match_id,
            "team_stats": {
                "possession":           final_stats,
                "passes_red":           event_stats.get("passes_t1", 0),
                "passes_green":         event_stats.get("passes_t2", 0),
                "interceptions_red":    event_stats.get("inter_t1", 0),
                "interceptions_green":  event_stats.get("inter_t2", 0),
                # Full per-team breakdown (distance, speed, actions, etc.)
                "team_breakdown":       team_stats or {},
            },
            "player_stats": player_stats,
            "heatmap_urls": heatmap_urls,
        }

        res = requests.post(f"{self.api_base}/ai/analyze-match/{match_id}", json=payload)
        if res.ok:
            print("Successfully submitted AI analysis to Database!")
            return res.json()
        else:
            print(f"Failed to submit analysis: {res.text}")
            return None
