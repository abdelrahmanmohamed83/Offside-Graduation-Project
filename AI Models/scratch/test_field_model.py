"""
Quick test script for the football field detection model.
Runs on a single frame from the video and saves the result.
"""
import cv2
import numpy as np
from ultralytics import YOLO

# ─── Config ───────────────────────────────────────────────────────────
MODEL_PATH  = r"D:\offside\weights\football-field-detection-15\weights\best.pt"
VIDEO_PATH  = r"D:\offside\data\input_data\4.mp4"
OUT_IMAGE   = r"D:\offside\data\output_data\field_detection_test.jpg"
FRAME_SEC   = 3          # which second of the video to test on
CONF        = 0.7
# ──────────────────────────────────────────────────────────────────────

# Load model
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Print model info
print(f"Task   : {model.task}")
print(f"Names  : {model.names}")

# Grab a frame
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.set(cv2.CAP_PROP_POS_FRAMES, int(FRAME_SEC * fps))
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Could not read frame from video")
    exit(1)

print(f"Frame shape: {frame.shape}")

# ── Run inference ──────────────────────────────────────────────────────
results = model(frame, conf=CONF, verbose=True)
result  = results[0]

# ── Draw output depending on task type ────────────────────────────────
vis = frame.copy()

# --- Keypoints (pose estimation) ---
if result.keypoints is not None:
    kps = result.keypoints.xy.cpu().numpy()       # (N_instances, K, 2)
    kp_conf = result.keypoints.conf              
    if kp_conf is not None:
        kp_conf = kp_conf.cpu().numpy()           # (N_instances, K)

    colors = [
        (0, 165, 255),   # orange  (left side)
        (255, 50,  50),  # pink/red (centre)
        (50,  200, 255), # cyan    (right side)
    ]

    for inst_idx, inst_kps in enumerate(kps):
        for kp_idx, (kx, ky) in enumerate(inst_kps):
            if kx == 0 and ky == 0:
                continue
            # Confidence gate
            if kp_conf is not None and kp_conf[inst_idx][kp_idx] < 0.3:
                continue
            color = colors[kp_idx % len(colors)]
            cv2.circle(vis, (int(kx), int(ky)), 6, color, -1, cv2.LINE_AA)
            cv2.circle(vis, (int(kx), int(ky)), 7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, str(kp_idx),
                        (int(kx) + 8, int(ky) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    print(f"✅ Keypoints detected: {len(kps)} instance(s)")

# --- Bounding boxes (detection) ---
elif result.boxes is not None and len(result.boxes) > 0:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = model.names.get(cls_id, str(cls_id))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 100), 2)
        cv2.putText(vis, f"{label} {conf:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
    print(f"✅ Boxes detected: {len(result.boxes)}")

else:
    print("⚠️  No detections found — try lowering CONF or checking the frame.")

# ── Save result ────────────────────────────────────────────────────────
cv2.imwrite(OUT_IMAGE, vis)
print(f"\n📸 Result saved to: {OUT_IMAGE}")
