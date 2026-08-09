# مشروع تخرج: نظام تحليل مباريات كرة القدم بالذكاء الاصطناعي
## Football Match Analysis System — Graduation Thesis Content

---

## أولاً: نظرة عامة على النظام (System Overview)

النظام عبارة عن **Pipeline متكاملة** لتحليل فيديوهات مباريات كرة القدم في الوقت الفعلي، تجمع بين:
- **Computer Vision** لرؤية وتتبع اللاعبين والكرة
- **Deep Learning** للتعرف على الأكشنز وتصنيف الفرق
- **Classical Algorithms** لحساب الإحصائيات والرسم
- **Backend Integration** لتخزين وعرض النتائج

---

## ثانياً: الموديلات المستخدمة (Models Used)

### 1. YOLOv8/v10 — Object Detection & Tracking
**الاستخدام:** كشف اللاعبين والكرة في كل فريم

**المعادلة الأساسية (Loss Function):**
```
L_total = λ_cls · L_cls + λ_box · L_box + λ_dfl · L_dfl
```
حيث:
- `L_cls` = Classification Loss (Binary Cross-Entropy)
- `L_box` = Bounding Box Regression Loss (IoU-based)
- `L_dfl` = Distribution Focal Loss

**IoU (Intersection over Union):**
```
IoU = Area(Prediction ∩ Ground Truth) / Area(Prediction ∪ Ground Truth)
```

**NMS (Non-Maximum Suppression):**
```
Keep box_i if: IoU(box_i, box_j) < threshold,  ∀j where score_j > score_i
```

**المدخلات/المخرجات:**
- Input: فريم فيديو (H × W × 3)
- Output: [(x1,y1,x2,y2, confidence, class_id)] لكل كيان مكتشف

**الإعدادات المستخدمة في المشروع:**
- `PLAYER_CONFIDENCE = 0.15` (حساسية عالية لاكتشاف اللاعبين)
- `BALL_CONFIDENCE = 0.75` (دقة عالية لتجنب False Positives للكرة)
- النموذج: `yolo26x.pt` (أكبر نسخة للدقة القصوى)

---

### 2. ByteTrack / BoT-SORT — Multi-Object Tracking
**الاستخدام:** تتبع هوية كل لاعب عبر الفريمات (Track ID ثابت)

**الخوارزمية:**
1. لكل فريم جديد: يربط الـ detections الجديدة بالـ tracks الموجودة
2. **Hungarian Algorithm** لحل مشكلة Assignment:
```
min Σ_ij  C_ij · X_ij
subject to: Σ_j X_ij = 1  (كل detection لـ track واحد)
           Σ_i X_ij = 1  (كل track من detection واحد)
           X_ij ∈ {0, 1}
```
3. `C_ij` = تكلفة ربط Detection i بـ Track j (بناءً على IoU + Appearance)

**Kalman Filter للتنبؤ بالموضع:**
```
State vector: x = [cx, cy, w, h, vx, vy, vw, vh]
Prediction:   x̂_k = F · x_{k-1}
Update:       x_k = x̂_k + K · (z_k - H · x̂_k)
```
حيث `K` هو Kalman Gain و`z_k` هو القياس الحقيقي.

---

### 3. R3D-18 — 3D CNN for Action Recognition
**الاستخدام:** التعرف على أكشن اللاعب صاحب الكرة (PASS, SHOT, HEADER, ...)

**Architecture:**
```
Input: (B=1, C=3, T=16, H=112, W=112)
  ↓
3D Conv (3×3×3) × 4 stages (ResNet blocks)
  ↓
Global Average Pooling
  ↓
Dropout(0.5)
  ↓
Linear(512 → 8 classes)
  ↓
Softmax → probabilities
```

**الـ 3D Convolution (vs 2D):**
```
2D Conv: f(x,y)     — يفهم الشكل فقط
3D Conv: f(x,y,t)   — يفهم الشكل + الحركة عبر الوقت
```

**Softmax:**
```
P(class_i) = exp(z_i) / Σ_j exp(z_j)
```

**Preprocessing:**
```
x_normalized = (x_raw / 255.0 - μ) / σ
μ = [0.43216, 0.394666, 0.37645]
σ = [0.22803, 0.22145,  0.216989]
```

**Classes (8 أكشن):**
`PASS, SHOT, HEADER, CROSS, HIGH_PASS, THROW_IN, BALL_PLAYER_BLOCK, PLAYER_SUCCESSFUL_TACKLE`

**Training Details:**
- Transfer Learning من Kinetics-400
- Data Augmentation: Flip, Speed Change, Brightness, Rotation, Gaussian Noise
- Optimizer: Adam  |  LR: 1e-4  |  Epochs: 200+
- Loss: Cross-Entropy

**Frame Sampling Strategy:**
```
Buffer Size = 32 frames → Sample every 2nd frame → 16 frames للموديل
Inference يحصل كل ~1 ثانية (32 frames ÷ 30fps)
```

---

### 4. YOLOv8-Pose — Field Keypoint Detection (Homography)
**الاستخدام:** تحديد نقاط معروفة في الملعب لحساب مصفوفة الـ Homography

**32 Keypoints مُعرَّفة في إحداثيات الملعب الحقيقي (metres):**
```
KP_0  = (0.0,   0.0)    # زاوية يسار أعلى
KP_14 = (52.5,  0.0)    # منتصف خط النص (أعلى)
KP_16 = (52.5, 34.0)    # مركز الملعب
KP_26 = (105.0, 0.0)    # زاوية يمين أعلى
KP_31 = (105.0, 68.0)   # زاوية يمين أسفل
... (32 نقطة إجمالاً)
```

---

### 5. YOLOv8-Seg — Stadium Segmentation (Semantic Radar Mapping)
**الاستخدام:** تحديد مناطق الملعب (منطقة الجزاء، دائرة الوسط) لتصحيح إزاحة الكاميرا

**Classes:**
- Class 0: 18-yard box (منطقة الجزاء)
- Class 3: Central Circle - First Half
- Class 5: Central Circle - Second Half

---

### 6. YOLOv8 (Custom) — Jersey Number Recognition
**الاستخدام:** قراءة رقم القميص لكل لاعب وربطه باسمه من قاعدة البيانات

**Voting System:**
```
لو اللاعب ظهر في 10 فريمات → نأخذ الرقم الأكثر تكراراً (Majority Vote)
Final_Number = argmax(count[number])
```

---

## ثالثاً: الخوارزميات الكلاسيكية (Classical Algorithms)

### 7. Perspective Homography (إسقاط المنظور)
**الاستخدام:** تحويل مواضع اللاعبين من إحداثيات الكاميرا → إحداثيات الرادار الحقيقية

**المعادلة:**
```
[x']     [h11  h12  h13]   [x]
[y']  =  [h21  h22  h23] · [y]
[w']     [h31  h32  h33]   [1]

x_radar = x' / w'
y_radar = y' / w'
```

**H** = 3×3 Homography Matrix

**RANSAC لحساب H من نقاط الملعب:**
```
RANSAC Algorithm:
  Repeat N times:
    1. Sample 4 point correspondences (src → dst)
    2. Compute H_candidate
    3. Count inliers: ||H·src_i - dst_i|| < threshold
  Return H with most inliers
```

**Exponential Smoothing للـ Homography:**
```
H_smooth(t) = α · H_new(t) + (1-α) · H_smooth(t-1)
α = 0.15  (smoothing factor)
```

---

### 8. Ball Interpolation (تقدير موضع الكرة)
**الاستخدام:** عندما تختفي الكرة وراء لاعب أو تخرج عن الكاميرا

**الخوارزمية:**
```
Velocity = Center(t) - Center(t-1) = (vx, vy)

إذا لم تُكتشف الكرة في الفريم t:
  Position_estimated(t) = Position(t-1) + (vx, vy)

تستمر لـ max_missing_frames = 7 فريمات كحد أقصى
```

---

### 9. HSV Color Classification (تصنيف الفرق)
**الاستخدام:** تحديد الفريق من لون قميص اللاعب

**خطوات الخوارزمية:**
```
1. قص الـ Bounding Box للاعب
2. اخذ الجزء العلوي (10%-50%) = منطقة القميص فقط
3. تحويل BGR → HSV
4. لكل فريق: عدد البكسلات داخل نطاق اللون:
   mask = inRange(img_hsv, lower_bound, upper_bound)
   pixels = countNonZero(mask)
5. الفريق = argmax(pixels_team1, pixels_team2, pixels_referee)
```

**تمثيل HSV:**
```
H (Hue):        0°  → 360°  (اللون)
S (Saturation): 0%  → 100%  (التشبع)
V (Value):      0%  → 100%  (السطوع)

OpenCV Scale: H: 0-180, S: 0-255, V: 0-255
```

---

### 10. Speed & Distance Computation
**الاستخدام:** حساب سرعة ومسافة كل لاعب

**المسافة الفيزيائية:**
```
distance(t) = ||position(t) - position(t-1)||
            = √[(Δx)² + (Δy)²]  (Euclidean Distance)

total_distance = Σ_t distance(t)
```

**السرعة اللحظية:**
```
speed(t) = distance(t) × fps    [m/s]
speed_kmh = speed(t) × 3.6      [km/h]

Cap: speed_max = 12 m/s  (≈ 43 km/h)
```

**Temporal Smoothing (Moving Average):**
```
speed_smooth(t) = (1/N) × Σ_{i=t-N+1}^{t} speed(i)
N = 5 frames window
```

**Pixel-to-Meter Ratio:**
```
pixel_to_meter = 0.02  m/pixel  (تقريباً على ملعب 105×68 متر)
OR عبر الـ Homography Matrix لدقة أعلى
```

---

### 11. Ball Possession Detection
**الاستخدام:** تحديد اللاعب الحامل للكرة

**منطقة القدم (Feet Zone):**
```
Player BBox = (x1, y1, x2, y2)
Feet Zone:
  fy1 = y1 + (y2-y1) × (1 - FEET_ZONE_HEIGHT_RATIO)   = y1 + 60% × height
  fy2 = y2
  fx1 = x1 - EXPANSION × width
  fx2 = x2 + EXPANSION × width
  EXPANSION = 0.15
```

**Possession Confirmation:**
```
إذا كانت الكرة داخل Feet Zone:
  candidate_count += 1
  إذا candidate_count ≥ REQUIRED_FRAMES (=3):
    تأكيد الحيازة

Sticky Possession (45 frames):
  بعد مغادرة الكرة → تظل الحيازة للاعب لـ 45 فريم
```

---

### 12. Heatmap Generation
**الاستخدام:** توليد خريطة حرارية لمناطق حركة كل لاعب

**الخوارزمية:**
```
1. لكل فريم: تسجيل موضع اللاعب (x, y) مُطبَّع على [0,1]
2. تحويل للإحداثيات على صورة الملعب
3. Gaussian Kernel لكل نقطة:
   K(x,y) = exp(-(x² + y²) / (2σ²))
4. تجميع كل الـ Kernels → Heatmap matrix
5. Normalize + Apply colormap (COLORMAP_JET)
```

---

## رابعاً: الـ Pipeline الكاملة (System Architecture)

```
┌──────────────────────────────────────────────────────────────────┐
│                     INPUT VIDEO FRAME                            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
   │ YOLO Player │   │  YOLO Ball  │   │ YOLO Stadium │
   │  Detector   │   │  Detector   │   │  Segmenter   │
   └──────┬──────┘   └──────┬──────┘   └──────┬───────┘
          │                 │                  │
          ▼                 ▼                  ▼
   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
   │  ByteTrack  │   │    Ball     │   │   Semantic   │
   │  (Track ID) │   │Interpolation│   │    Mapper    │
   └──────┬──────┘   └──────┬──────┘   └──────┬───────┘
          │                 │                  │
          ▼                 │                  ▼
   ┌─────────────┐          │          ┌──────────────┐
   │   Jersey #  │          │          │  Homography  │
   │  Recognizer │          │          │   Matrix H   │
   └──────┬──────┘          │          └──────┬───────┘
          │                 │                  │
   ┌──────▼──────┐          │                  │
   │    Team     │          │                  │
   │  Classifier │          │                  │
   │  (HSV-based)│          │                  │
   └──────┬──────┘          │                  │
          │                 │                  │
          └────────┬─────────┘                  │
                   ▼                            │
           ┌───────────────┐                    │
           │  Possession   │                    │
           │  Detection    │                    │
           │ (Feet Zone)   │                    │
           └───────┬───────┘                    │
                   │                            │
                   ▼                            │
           ┌───────────────┐                    │
           │    R3D-18     │                    │
           │    Action     │                    │
           │  Recognition  │                    │
           └───────┬───────┘                    │
                   │                            │
          ┌────────┴──────────────┐             │
          ▼                       ▼             ▼
   ┌─────────────┐         ┌─────────────────────┐
   │  Speed &    │         │    Pitch Radar       │
   │  Distance   │         │  Visualization       │
   │  Tracking   │         │  (Team Colors+Ball)  │
   └──────┬──────┘         └──────────┬──────────┘
          │                           │
          └─────────────┬─────────────┘
                        ▼
              ┌──────────────────┐
              │  OUTPUT VIDEO    │
              │  + STATS HUD     │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Backend API     │
              │ (FastAPI + Supa) │
              │  - Possession    │
              │  - Actions/Player│
              │  - Distance/Speed│
              │  - Heatmaps      │
              └──────────────────┘
```

---

## خامساً: الإحصائيات المولَّدة (Generated Statistics)

### Per-Player Statistics:
| الإحصائية | الوحدة | كيفية الحساب |
|-----------|--------|--------------|
| Total Distance | km | Σ Euclidean distances × pixel_to_meter |
| Top Speed | km/h | max(speed_smoothed) × 3.6 |
| Avg Speed | km/h | mean(speed_smoothed) × 3.6 |
| Possession Time | frames | عدد فريمات الحيازة |
| PASS count | - | ActionRecognizer output |
| SHOT count | - | ActionRecognizer output |
| HEADER count | - | ActionRecognizer output |
| CROSS count | - | ActionRecognizer output |

### Per-Team Statistics:
| الإحصائية | الوحدة |
|-----------|--------|
| Possession % | % |
| Total Distance | km |
| Avg Distance/Player | km |
| Top Speed (Best Player) | km/h |
| Total Passes | - |
| Total Shots | - |
| Interceptions | - |

---

## سادساً: التقنيات والمكتبات المستخدمة

| المكتبة | الاستخدام |
|---------|-----------|
| **PyTorch** | تدريب وتشغيل R3D-18 |
| **Torchvision** | R3D-18 pretrained model |
| **Ultralytics (YOLO)** | Player/Ball/Field detection |
| **OpenCV** | معالجة الصور والفيديو |
| **NumPy** | العمليات الحسابية |
| **FastAPI** | Backend REST API |
| **Supabase** | قاعدة البيانات + Storage |
| **Python** | اللغة الأساسية |

---

## سابعاً: مقاييس الأداء (Performance Metrics)

### Action Recognition Model:
```
Accuracy  = (TP + TN) / Total
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

**نتائج التدريب (R3D-18):**
```
                precision  recall  f1-score  support
CROSS              0.44     0.50     0.47       14
HEADER             0.65     0.65     0.65       20
HIGH_PASS          0.86     0.90     0.88       20
PASS               0.62     0.80     0.70       10
SHOT               0.90     0.53     0.67       17
THROW_IN           0.73     0.80     0.76       10

accuracy                             0.69       91
```

**بعد Data Augmentation (×9 dataset):**
```
Epoch 3/40:  train_acc=0.800  val_acc=0.897  ← تحسن كبير
```

### Data Augmentation Techniques:
1. **Horizontal Flip** — عكس الصورة أفقياً
2. **Speed Change** — تغيير سرعة الفيديو (0.75x, 1.25x)
3. **Brightness Change** — تغيير السطوع (±30%)
4. **Rotation** — تدوير (±15°)
5. **Gaussian Noise** — إضافة ضوضاء عشوائية
6. **Color Jitter** — تغيير التشبع والتباين
7. **Temporal Jitter** — تغيير نقطة بداية الكليب
8. **Combined** — مزج أكثر من تقنية معاً

**معادلة Augmentation Rate:**
```
N_augmented = N_original × 9
458 videos → 4,122 videos
```

---

## ثامناً: تحديات المشروع والحلول

| التحدي | الحل |
|--------|-------|
| الكرة بتختفي وراء اللاعبين | Ball Interpolation بالسرعة |
| الكاميرا بتتحرك (Pan/Tilt) | Semantic + Keypoint Homography |
| لاعبَين من نفس الفريق يتشابهوا | Jersey Number Voting System |
| قلة الداتا للتدريب | Data Augmentation ×9 |
| اللاعب بيخرج ويرجع (ID Switch) | ByteTrack + Re-ID |
| الأكشن بتحتاج كليب مش فريم | 3D CNN (R3D-18) بـ 16 فريم |

---

## تاسعاً: المراجع العلمية المقترحة

1. **YOLO (You Only Look Once):** Redmon et al., 2016 — Real-time object detection
2. **ByteTrack:** Zhang et al., 2022 — Multi-object tracking
3. **R3D (Res3D):** Tran et al., 2018 — Video classification with 3D CNNs
4. **Homography:** Hartley & Zisserman, "Multiple View Geometry in Computer Vision"
5. **RANSAC:** Fischler & Bolles, 1981 — Random Sample Consensus
6. **HSV Color Space:** Smith, 1978 — Color Gamut Transform
7. **Kalman Filter:** Kalman, 1960 — A New Approach to Linear Filtering
8. **Data Augmentation for Video:** Shorten & Khoshgoftaar, 2019
