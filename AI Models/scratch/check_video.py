import cv2
import os
from src.config import INPUT_VIDEO_PATH

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print("Cannot open video")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps > 0 else 0
    print(f"Video: {INPUT_VIDEO_PATH}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
cap.release()
