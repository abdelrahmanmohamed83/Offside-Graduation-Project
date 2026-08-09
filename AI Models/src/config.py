import os

# ─────────────────────────────────────────────────────────
# 1. Base Paths
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────
# 2. Model Weights Paths
# ─────────────────────────────────────────────────────────
PLAYER_DETECTOR_WEIGHTS   = os.path.join(BASE_DIR, "yolo26x.pt")
BALL_DETECTOR_WEIGHTS     = os.path.join(BASE_DIR, "weights", "football ball detection", "weights", "best.pt")
NUMBER_RECOGNIZER_WEIGHTS = os.path.join(BASE_DIR, "weights", "jersey_recognition.pt")
STADIUM_SEGMENTER_WEIGHTS = os.path.join(BASE_DIR, "weights", "Studiam_seg.pt")
FIELD_DETECTOR_WEIGHTS    = os.path.join(BASE_DIR, "weights", "football-field-detection-15", "weights", "best.pt")
ACTION_RECOGNIZER_WEIGHTS = os.path.join(BASE_DIR, "weights", "action", "best_soccer_r3d_model.pt")


# ─────────────────────────────────────────────────────────
# 3. Confidence Thresholds (Individual for each model)
# ─────────────────────────────────────────────────────────
PLAYER_CONFIDENCE = 0.15
BALL_CONFIDENCE   = 0.75
NUMBER_CONFIDENCE = 0.2
STADIUM_CONFIDENCE = 0.4
FIELD_DETECTOR_CONFIDENCE = 0.5   # Keypoint confidence for homography

# Ball class ID: 0 = custom model (ball only), 32 = COCO pre-trained model
BALL_CLASS_ID = 0

# ─────────────────────────────────────────────────────────
# 4. Video Input/Output Settings
# ─────────────────────────────────────────────────────────
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "input_data", "7.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "output_data", "اخر فيديو عشان زهقت 3.mp4")

# ─────────────────────────────────────────────────────────
# 5. Asset Paths (Database & Pitch Image)
# ─────────────────────────────────────────────────────────
PLAYER_DB_PATH = os.path.join(BASE_DIR, "data", "players_database.json")
PITCH_IMAGE_PATH = os.path.join(BASE_DIR, "data", "pitch_topdown.png")

# ─────────────────────────────────────────────────────────
# 6. Possession & Game Logic Settings
# ─────────────────────────────────────────────────────────
POSSESSION_CONTACT_RADIUS = 0.55  # Legacy — kept for reference only
POSSESSION_DRIBBLE_RADIUS = 1.2   # Legacy — kept for reference only
POSSESSION_RADIUS = POSSESSION_CONTACT_RADIUS  # Legacy alias
STICKY_FRAMES = 45              # Frames to hold possession after ball leaves feet zone
REQUIRED_POSSESSION_FRAMES = 3  # Frames of consecutive contact to confirm possession
MAX_BALL_SPEED_FOR_DRIBBLING = 18

# Feet zone for bbox-based possession detection
FEET_ZONE_HEIGHT_RATIO = 0.40   # Bottom 40% of player bbox = feet/legs area
FEET_ZONE_WIDTH_EXPANSION = 0.15 # Expand bbox width by 15% on each side for ball tolerance

# 5-a-side Pitch and Radar Dimensions
PITCH_LENGTH = 40.0             # Real-world pitch length (meters)
PITCH_WIDTH = 20.0              # Real-world pitch width (meters)
RADAR_WIDTH = 280               # Width of the drawn radar
RADAR_HEIGHT = 168              # Height of the drawn radar

# ─────────────────────────────────────────────────────────
# 7. Smoothing & Filtering Settings
# ─────────────────────────────────────────────────────────
RADAR_SMOOTHING = 0.15      # Homography smoothing factor
NUMBER_VOTING_FRAMES = 10   # Frames required to lock a jersey number (was 30)
BALL_INTERPOLATION_MAX = 7  # Max frames to predict ball position when hidden

# ─────────────────────────────────────────────────────────
# 8. Ball Trail Settings
# ─────────────────────────────────────────────────────────
BALL_TRAIL_ENABLED = True
BALL_TRAIL_LENGTH = 25       # Number of historical positions to draw
BALL_TRAIL_COLOR = (0, 200, 255)  # BGR color of trail (yellow-orange)
BALL_TRAIL_THICKNESS = 3     # Max thickness of trail line at newest point


# ─────────────────────────────────────────────────────────
# 7. Team Classification Settings
# ─────────────────────────────────────────────────────────
TEAM_PIXEL_THRESHOLD = 20   # Min pixels to identify a team color (raised to reduce noise misclassification)
SHIRT_CROP_HEIGHT_RATIO = (0.10, 0.45) # Top/Bottom ratio for shirt crop (tighter to focus on torso/shirt)
SHIRT_CROP_WIDTH_RATIO = (0.25, 0.75)  # Left/Right ratio for shirt crop (tighter to avoid background bleed)

# ─────────────────────────────────────────────────────────
# 8. Team Color Ranges (HSV)
# ─────────────────────────────────────────────────────────
# Team 1 (Red) - Bright red shirts
TEAM_1_HSV = [
    {"lower": [0,   70,  70],  "upper": [10,  255, 255]},  # Bright reds lower
    {"lower": [170, 70,  70],  "upper": [180, 255, 255]},  # Bright reds upper
]

# Team 2 (Teal/Mint Green) - Mint green/teal shirts
TEAM_2_HSV = [
    {"lower": [65,  40,  100], "upper": [95,  255, 255]}  # Teal / Mint Green
]

# Referee (Yellow) - Yellow shirt
REFEREE_HSV = [
    {"lower": [18,  100, 100],  "upper": [35,  255, 255]}   # Yellow
]

# Team Names
TEAM_1_NAME = "Red Team"
TEAM_2_NAME = "Teal Team"

# Team Display Colors (BGR)
TEAM_1_DISPLAY_COLOR = (40, 40, 220)       # Red BGR
TEAM_2_DISPLAY_COLOR = (180, 250, 100)     # Mint Green / Teal BGR

REFEREE_DISPLAY_COLOR = (0, 235, 235)       # Bright Yellow BGR

# ─────────────────────────────────────────────────────────
# 9. Visualization Settings
# ─────────────────────────────────────────────────────────
SHOW_RADAR = True
SHOW_HEATMAPS = True
MIN_FRAMES_FOR_HEATMAP = 30

# ─────────────────────────────────────────────────────────
# 10. API & Database Settings
# ─────────────────────────────────────────────────────────
API_BASE_URL = "http://107.21.186.172:8080/api/v1"
SUPABASE_URL = "https://gsvowvzdxphlguclawur.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdzdm93dnpkeHBobGd1Y2xhd3VyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI3MzU2NDgsImV4cCI6MjA4ODMxMTY0OH0.jORXG6_6LjDP07EAhJvYb9G10AKRuKaDkCjy0SfhQe8"
SUPABASE_BUCKET = "heatmaps"  # Assuming heatmaps are uploaded here
MATCH_ID = 10  # Change this to the ID of the match to analyze