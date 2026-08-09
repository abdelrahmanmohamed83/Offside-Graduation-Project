import cv2
import numpy as np
from src.config import PITCH_LENGTH

class SemanticPitchMapper:
    def __init__(self, radar_w=500, radar_h=300, smoothing=0.15):
        self.radar_w = radar_w
        self.radar_h = radar_h
        
        # الأبعاد الرياضية الحقيقية للرادار
        self.mid_x = radar_w / 2.0        
        self.mid_y = radar_h / 2.0        
        self.box18_w = (6.0/PITCH_LENGTH) * radar_w  
        
        # التخزين المؤقت للإزاحة (النعومة) Temporal Smoothing
        self.current_dx = 0.0
        self.current_dy = 0.0
        self.smoothing = smoothing
        self.initialized = False

    def clean_mask(self, mask, min_area=4000):
        """
        🛡️ تصفية الماسك من التشوهات والقطوعات العشوائية.
        يبقي فقط على أكبر مكون متصل يتجاوز المساحة المحددة.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return np.zeros_like(mask)
        
        # استخراج المساحات للمكونات (مع تخطي الخلفية رقم 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = np.argmax(areas) + 1  # تعويض الإزاحة بسبب الخلفية
        max_area = stats[max_idx, cv2.CC_STAT_AREA]
        
        if max_area < min_area:
            return np.zeros_like(mask)
        
        cleaned = (labels == max_idx).astype(np.uint8) * 255
        return cleaned

    def transform_point(self, point, matrix):
        """تطبيق مصفوفة المنظور على نقطة واحدة"""
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed_pt = cv2.perspectiveTransform(pt, matrix)
        return transformed_pt[0][0][0], transformed_pt[0][0][1]

    def get_camera_offset(self, seg_results, w, h, base_matrix):
        """
        يحسب الإزاحة الدقيقة المطلوبة (dx, dy) على الرادار استناداً 
        إلى دوران كاميرا التلفزيون وموقع النقاط السيمانتيكية.
        """
        if not seg_results or seg_results[0].masks is None:
            return int(self.current_dx), int(self.current_dy)

        masks_data = seg_results[0].masks.data.cpu().numpy()
        boxes_cls = seg_results[0].boxes.cls.cpu().numpy()

        masks_dict = {}
        for i, cls_idx in enumerate(boxes_cls):
            cls_idx = int(cls_idx)
            mask = (masks_data[i] > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (w, h))
            if cls_idx in masks_dict:
                masks_dict[cls_idx] = cv2.bitwise_or(masks_dict[cls_idx], mask)
            else:
                masks_dict[cls_idx] = mask

        # 🛡️ تنظيف الماسكات المدمجة لإزالة أي تشوهات أو مساحات عشوائية صغيرة
        for cls_idx in list(masks_dict.keys()):
            masks_dict[cls_idx] = self.clean_mask(masks_dict[cls_idx], min_area=4000)

        detected_dx, detected_dy = None, None

        # ---------------------------------------------------------
        # 1. حالة المنتصف (أقوى نقطة مرجعية)
        # ---------------------------------------------------------
        center_pts = []
        if 3 in masks_dict: # First Half Central Circle
            y, x = np.where(masks_dict[3] > 0)
            if len(x) > 0:
                idx = np.argmax(x) # أقصى يمين الدائرة هو السنتر تماماً
                center_pts.append((x[idx], y[idx]))
        
        if 5 in masks_dict: # Second Half Central Circle
            y, x = np.where(masks_dict[5] > 0)
            if len(x) > 0:
                idx = np.argmin(x) # أقصى يسار الدائرة هو السنتر تماماً
                center_pts.append((x[idx], y[idx]))

        if len(center_pts) > 0:
            center_pts = np.array(center_pts)
            img_cx, img_cy = np.mean(center_pts[:, 0]), np.mean(center_pts[:, 1])
            
            # نحول مركز الصورة عبر المصفوفة الأساسية لمعرفة أين سيقع افتراضياً
            mapped_cx, mapped_cy = self.transform_point((img_cx, img_cy), base_matrix)
            
            # نحسب الفرق بين موقعه الافتراضي وموقعه الحقيقي على الرادار (منتصف الرادار)
            detected_dx = self.mid_x - mapped_cx
            detected_dy = self.mid_y - mapped_cy

        # ---------------------------------------------------------
        # 2. حالة منطقة الجزاء (لو السنتر مش باين)
        # ---------------------------------------------------------
        elif 0 in masks_dict: # 18Yard box
            y, x = np.where(masks_dict[0] > 0)
            if len(x) > 0:
                # نحدد هل نحن في الشق الأيمن أم الأيسر؟
                area_first = np.sum(masks_dict.get(4, np.zeros((h,w), dtype=np.uint8)) > 0)
                area_second = np.sum(masks_dict.get(6, np.zeros((h,w), dtype=np.uint8)) > 0)
                
                if area_first >= area_second:
                    # المنطقة اليسرى. الـ (Max X) الخاص بالماسك هو الحافة المضمونة التي تفصلها عن باقي الملعب 
                    img_bx = np.max(x)
                    true_bx = self.box18_w # خط الـ 18 يارد الداخلي
                else:
                    # المنطقة اليمنى. الـ (Min X) الخاص بالماسك هو الحافة المضمونة التي تفصلها عن باقي الملعب
                    img_bx = np.min(x)
                    true_bx = self.radar_w - self.box18_w # خط الـ 18 يارد الداخلي من جهة اليمين
                
                # نأخذ Y المتوسطة لأن الانقطاع الجانبي هو المشكلة الأكبر للكاميرا التلفزيونية
                img_by = np.mean(y)
                true_by = self.mid_y 
                
                mapped_bx, mapped_by = self.transform_point((img_bx, img_by), base_matrix)
                detected_dx = true_bx - mapped_bx
                detected_dy = true_by - mapped_by

        # ---------------------------------------------------------
        # 3. توفيق النعومة أو الحجب (Temporal Smoothing)
        # ---------------------------------------------------------
        if detected_dx is not None:
            if not self.initialized:
                self.current_dx = detected_dx
                self.current_dy = detected_dy
                self.initialized = True
            else:
                # 🛡️ فلترة القفزات المفاجئة (Outlier Jump Rejection)
                # إذا كانت القفزة غير طبيعية أو نتيجة تشوه مفاجئ في الموديل، نتجاهلها ونحافظ على نعومة التتبع
                diff_x = abs(detected_dx - self.current_dx)
                diff_y = abs(detected_dy - self.current_dy)
                if diff_x > 80.0 or diff_y > 50.0:
                    detected_dx = self.current_dx
                    detected_dy = self.current_dy
                
                self.current_dx = (self.smoothing * detected_dx) + ((1 - self.smoothing) * self.current_dx)
                self.current_dy = (self.smoothing * detected_dy) + ((1 - self.smoothing) * self.current_dy)
        
        return int(self.current_dx), int(self.current_dy)

    def get_homography(self, pose_results):
        """Added for compatibility during transitions"""
        return None
