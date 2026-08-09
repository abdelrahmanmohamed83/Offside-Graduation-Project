"""
ActionRecognizer
================
Runs R3D-18 action recognition on the ball possessor only.

Usage in pipeline:
    recognizer = ActionRecognizer()
    result = recognizer.update(track_id, frame, bbox)
    # result → ('PASS', 0.91) or None
"""
import cv2
import torch
import torch.nn as nn
import torchvision.models.video as video_models
import numpy as np
from collections import deque

ACTION_CLASSES = [
    'BALL_PLAYER_BLOCK', 'CROSS', 'HEADER', 'HIGH_PASS',
    'PASS', 'PLAYER_SUCCESSFUL_TACKLE', 'SHOT', 'THROW_IN'
]

NUM_FRAMES      = 16          # frames fed to the model
BUFFER_SIZE     = 32          # collect 32 raw frames, sample every 2nd
CROP_PADDING    = 0.4         # expand player bbox by this ratio on each side
FRAME_SIZE      = 112
CONF_THRESHOLD  = 0.35        # minimum confidence to report an action

MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD  = np.array([0.22803, 0.22145,  0.216989], dtype=np.float32)


class ActionRecognizer:
    def __init__(self, weights_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model  = self._load_model(weights_path)
        self.enabled = (self.model is not None)

        # Per-player frame buffers: track_id → deque of cropped frames
        self._buffers: dict[int, deque] = {}

        # Last prediction per player (persists until next inference)
        self._last_action: dict[int, tuple] = {}   # tid → (action, conf)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def update(self, track_id: int, frame: np.ndarray, bbox: tuple, player_speed: float = 0.0, ball_speed: float = 0.0):
        """
        Call every frame for the ball possessor / closest player.
        Returns (action_label, confidence) ONLY when a brand-new inference
        completes (i.e. every BUFFER_SIZE frames of possession).
        Returns None otherwise — use get_last_action() for HUD display.
        """
        if not self.enabled:
            return None

        # Check if player is inactive (standing still / doing nothing)
        # Threshold: player speed < 1.0 m/s and ball speed < 3.0 m/s
        if player_speed < 1.0 and ball_speed < 3.0:
            self.clear_player(track_id)
            self._last_action[track_id] = ("No Action", 1.0)
            return ("No Action", 1.0)

        crop = self._crop_player(frame, bbox)
        if crop is None:
            return None  # No crop available; don't count, but last action still accessible

        # Init buffer for new track_id
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=BUFFER_SIZE)

        self._buffers[track_id].append(crop)

        # Run inference ONLY when buffer fills up
        if len(self._buffers[track_id]) == BUFFER_SIZE:
            result = self._infer(self._buffers[track_id])
            self._buffers[track_id].clear()   # always reset after inference
            if result[1] >= CONF_THRESHOLD:
                self._last_action[track_id] = result
                return result          # <-- NEW inference: caller should count this

        return None  # Buffer still filling; return None (use get_last_action for display)

    def get_last_action(self, track_id: int):
        """Return the most recent confirmed action for a player (for HUD display only)."""
        return self._last_action.get(track_id)

    def clear_player(self, track_id: int):
        """Call when a player loses possession to reset their buffer."""
        self._buffers.pop(track_id, None)

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _load_model(self, weights_path: str):
        import os
        if not os.path.isfile(weights_path):
            print(f"[WARNING] [ActionRecognizer] Weights file not found: {weights_path}")
            print("    Model-based action recognition will be DISABLED for this run.")
            return None
        try:
            model = video_models.r3d_18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(model.fc.in_features, len(ACTION_CLASSES))
            )
            
            # Ensure the weights zip file has the correct structure for PyTorch
            weights_path = self._ensure_valid_zip_structure(weights_path)
            
            sd = torch.load(weights_path, map_location='cpu')
            # Handle both raw state-dict and checkpoint dicts
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            elif isinstance(sd, dict) and 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            model.load_state_dict(sd)
            model.eval().to(self.device)
            print(f"[SUCCESS] [ActionRecognizer] Loaded weights from {weights_path}")
            return model
        except Exception as e:
            print(f"[WARNING] [ActionRecognizer] Failed to load weights: {e}")
            print("    Model-based action recognition will be DISABLED for this run.")
            return None

    def _ensure_valid_zip_structure(self, filepath: str) -> str:
        """
        Detects if the weights zip file lacks a top-level directory structure,
        which causes PyTorch's zip reader to fail with:
        'Expected hasRecord("version") to be true, but got false'.
        If detected, it automatically repackages the zip file to include a prefix directory.
        """
        import os
        import zipfile
        import shutil

        if not zipfile.is_zipfile(filepath):
            return filepath

        has_root_files = False
        try:
            with zipfile.ZipFile(filepath, 'r') as z:
                namelist = z.namelist()
                if 'version' in namelist or 'data.pkl' in namelist:
                    has_root_files = True
        except Exception:
            return filepath

        if not has_root_files:
            return filepath

        fixed_path = filepath.replace(".pt", "_fixed.pt")
        # If fixed file already exists, return it
        if os.path.isfile(fixed_path):
            return fixed_path

        print(f"[WARNING] [ActionRecognizer] Detected root-level zip structure in weights: {filepath}")
        print("   This causes PyTorch to fail loading the weights. Auto-repackaging weights...")

        try:
            temp_dir = filepath + "_temp_extract"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)

            with zipfile.ZipFile(filepath, 'r') as z:
                z.extractall(temp_dir)

            # Re-zip with prefix 'archive/'
            with zipfile.ZipFile(fixed_path, 'w', zipfile.ZIP_DEFLATED) as z_out:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, temp_dir)
                        rel_path = rel_path.replace(os.path.sep, '/')
                        archive_path = "archive/" + rel_path
                        z_out.write(full_path, archive_path)

            shutil.rmtree(temp_dir)

            # Try to overwrite original file
            try:
                shutil.copy2(fixed_path, filepath)
                print(f"[SUCCESS] [ActionRecognizer] Successfully repackaged and overwrote original {filepath}")
                try:
                    os.remove(fixed_path)
                except Exception:
                    pass
                return filepath
            except Exception:
                print(f"[INFO] [ActionRecognizer] Loaded repackaged weights from {fixed_path}")
                return fixed_path

        except Exception as e:
            print(f"[ERROR] [ActionRecognizer] Failed to auto-repackage zip: {e}")
            return filepath

    def _crop_player(self, frame: np.ndarray, bbox: tuple):
        """Crop frame around player bbox with padding. Returns resized crop or None."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        bw, bh = x2 - x1, y2 - y1

        pad_x = int(bw * CROP_PADDING)
        pad_y = int(bh * CROP_PADDING)
        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(w, x2 + pad_x)
        ny2 = min(h, y2 + pad_y)

        crop = frame[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (FRAME_SIZE, FRAME_SIZE))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop

    def _infer(self, buffer: deque) -> tuple:
        """Sample 16 frames from buffer, run model, return (action, conf)."""
        frames = list(buffer)
        # Sample evenly: every 2nd frame from 32 → 16 frames
        sampled = frames[::2][:NUM_FRAMES]
        if len(sampled) < NUM_FRAMES:
            sampled += [sampled[-1]] * (NUM_FRAMES - len(sampled))

        # Preprocess: (T, H, W, C) → (1, C, T, H, W)
        clip = np.stack(sampled, axis=0).astype(np.float32) / 255.0  # (T,H,W,C)
        clip = (clip - MEAN) / STD
        clip = np.transpose(clip, (3, 0, 1, 2))   # (C, T, H, W)
        tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device)  # (1,C,T,H,W)

        with torch.no_grad():
            out   = self.model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            idx   = probs.argmax().item()

        return ACTION_CLASSES[idx], float(probs[idx])
