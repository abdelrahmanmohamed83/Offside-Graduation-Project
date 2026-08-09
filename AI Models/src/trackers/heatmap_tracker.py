import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import scipy.ndimage as ndimage
from matplotlib.colors import LinearSegmentedColormap

class HeatmapTracker:
    def __init__(self, pitch_image_path, output_dir):
        self.pitch_image_path = pitch_image_path
        self.output_dir = output_dir
        self.player_positions = defaultdict(list)
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.pitch_img = plt.imread(self.pitch_image_path)
        self.pitch_h, self.pitch_w = self.pitch_img.shape[:2]

        # 🎨 اختراع التدرج الحراري (شفاف -> أخضر فاتح -> أصفر -> أحمر)
        colors = [(0, 0, 0, 0), (0.2, 0.8, 0.2, 0.3), (1, 1, 0, 0.6), (1, 0, 0, 0.9)] 
        self.custom_cmap = LinearSegmentedColormap.from_list('custom_thermal', colors, N=256)

    def update(self, players_2d_data):
        for player_identifier, (norm_x, norm_y) in players_2d_data.items():
            self.player_positions[player_identifier].append((norm_x, norm_y))

    def generate_heatmaps(self, min_frames=30):
        print(f"\n Generating Thermal Heatmaps in: {self.output_dir}")
        generated_paths = {}  # {player_name: file_path}
        
        for player_id, positions in self.player_positions.items():
            if len(positions) < min_frames:
                continue 
            
            xs = np.array([p[0] * self.pitch_w for p in positions])
            ys = np.array([p[1] * self.pitch_h for p in positions])

            # 1. تقسيم الملعب لشبكة دقيقة جداً (100x100) وتوزيع النقط
            heatmap, xedges, yedges = np.histogram2d(
                xs, ys, bins=100, range=[[0, self.pitch_w], [0, self.pitch_h]]
            )

            # 2. سر الصنعة: تسييح النقط في بعضها لعمل "توهج حراري"
            # كل ما زاد الـ sigma، التوهج بقى أوسع وأنعم
            heatmap = ndimage.gaussian_filter(heatmap, sigma=6)

            # 3. تظبيط أبعاد المصفوفة عشان تركب على الملعب صح
            heatmap = heatmap.T 

            # 4. تضخيم الألوان عشان الأحمر والأصفر يظهروا بقوة
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

            # --- الرسم ---
            plt.figure(figsize=(12, 8), dpi=150)
            
            # رسم صورة الملعب
            plt.imshow(self.pitch_img, extent=[0, self.pitch_w, self.pitch_h, 0])
            
            # رسم التوهج الحراري فوق الملعب
            plt.imshow(heatmap, extent=[0, self.pitch_w, self.pitch_h, 0], 
                       cmap=self.custom_cmap, origin='upper', aspect='auto', interpolation='bilinear')

            safe_name = str(player_id).replace(" ", "_").replace("/", "_")
            output_file = os.path.join(self.output_dir, f"{safe_name}_Heatmap.png")

            plt.title(f"Thermal Heatmap: {player_id}", fontsize=20, fontweight='bold', color='black', pad=15)
            plt.axis('off')

            plt.savefig(output_file, bbox_inches='tight', transparent=True)
            plt.close()
            
            # Save the path for uploading later
            generated_paths[str(player_id)] = output_file

        print(f"All Thermal Heatmaps generated! ({len(generated_paths)} players)")
        return generated_paths  # {player_name: file_path}

