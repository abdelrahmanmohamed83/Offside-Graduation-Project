import os
import zipfile
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ZIP = os.path.join(BASE_DIR, "colab_package.zip")

# Files to include
files_to_zip = {
    # Models
    "yolo26x.pt": os.path.join(BASE_DIR, "yolo26x.pt"),
    "weights/jersey_recognition.pt": os.path.join(BASE_DIR, "weights", "jersey_recognition.pt"),
    "weights/Studiam_seg.pt": os.path.join(BASE_DIR, "weights", "Studiam_seg.pt"),
    "weights/football ball detection/weights/best.pt": os.path.join(BASE_DIR, "weights", "football ball detection", "weights", "best.pt"),
    "weights/football-field-detection-15/weights/best.pt": os.path.join(BASE_DIR, "weights", "football-field-detection-15", "weights", "best.pt"),
}

# Find the video file
video_candidates = [
    os.path.join(BASE_DIR, "data", "input_data", "1_720p.mp4"),
]
# Also search recursively
for root, dirs, files in os.walk(os.path.join(BASE_DIR, "data")):
    for f in files:
        if "1_720p" in f:
            video_candidates.append(os.path.join(root, f))

video_path = None
for v in video_candidates:
    if os.path.exists(v):
        video_path = v
        break

if video_path:
    files_to_zip["data/input_data/1_720p.mp4"] = video_path
    print(f"[+] Found video: {video_path}")
else:
    print("[-] Video 1_720p.mp4 not found!")

print(f"\nCreating zip: {OUTPUT_ZIP}")
print("-" * 50)

with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for arcname, filepath in files_to_zip.items():
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"[+] Adding: {arcname} ({size_mb:.1f} MB)")
            zf.write(filepath, arcname)
        else:
            print(f"[-] NOT FOUND: {filepath}")

total_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
print("-" * 50)
print(f"\n✅ Done! colab_package.zip created ({total_mb:.1f} MB)")
print(f"   Location: {OUTPUT_ZIP}")
